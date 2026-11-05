"""Extract typed region interfaces and execute a fixed candidate partition.

Region annotations are supplied algorithm stages. Edges, live boundary values,
rounding points, and reductions are derived from the selected graph. These are
fusion candidates, not proof that a region fits one GPU CTA or kernel.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tilefoundry.analysis import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.evaluator import evaluate

from .graph import constant_tensor
from .lower import deep_ir_visitors, emit_phase, function


def describe(graph, i):
    n = graph.nodes[i]
    return dict(id=i, name=n.name, shape=n.shape, dtype=n.dtype, logical_bytes=n.nbytes)


def partition_phase(graph, plan, phase):
    forward = phase == "forward"
    roots = [*graph.forward_outputs, *plan.saved] if forward else graph.backward_outputs
    available = (
        set(graph.forward_inputs) if forward else graph.external | set(plan.saved)
    )
    available |= {n.id for n in graph.nodes if n.recipe.op in {"scalar", "constant"}}
    recipes = None if forward else plan.recipes
    order = graph.schedule(roots, available=available, recipes=recipes)
    annotations = graph.metadata["regions"]
    owner, members = {}, defaultdict(list)
    for i in order:
        label = annotations.get(i, annotations.get(str(i), "unclassified"))
        if not forward and graph.nodes[i].phase == "forward":
            label = "recompute." + label
        owner[i] = label
        members[label].append(i)
    incoming, outgoing, dependencies = (
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
    )
    for i in order:
        node = graph.nodes[i]
        recipe = node.recipe if forward else node.recipes[plan.recipes[i]]
        for j in recipe.args:
            if owner.get(j) != owner[i]:
                incoming[owner[i]].add(j)
                if j in owner:
                    outgoing[owner[j]].add(j)
                    dependencies[owner[i]].add(owner[j])
        if i in roots:
            outgoing[owner[i]].add(i)
    pending = set(members)
    regions = []
    while pending:
        ready = sorted(k for k in pending if not dependencies[k] & pending)
        if not ready:
            raise ValueError("Supplied regions are not a valid acyclic partition")
        for label in ready:
            pending.remove(label)
            regions.append(
                dict(
                    label=label,
                    nodes=members[label],
                    inputs=sorted(incoming[label]),
                    outputs=sorted(outgoing[label]),
                    dependencies=sorted(dependencies[label]),
                )
            )
    return regions


def boundary_report(graph, plan):
    result = dict(
        scope="derived interfaces of a supplied candidate partition; no fusion search or GPU feasibility claim",
        precision_contract=graph.metadata["precision_contract"],
        phases={},
    )
    for phase in ("forward", "backward"):
        rows = []
        for region in partition_phase(graph, plan, phase):
            casts, reductions, matmuls = [], [], []
            for i in region["nodes"]:
                n = graph.nodes[i]
                r = n.recipe if phase == "forward" else n.recipes[plan.recipes[i]]
                a = dict(r.attrs)
                if r.op == "cast":
                    casts.append(
                        {
                            **describe(graph, i),
                            "from_dtype": graph.nodes[r.args[0]].dtype,
                        }
                    )
                elif r.op == "sum":
                    reductions.append(
                        {
                            **describe(graph, i),
                            "axis": a["axis"],
                            "extent": graph.nodes[r.args[0]].shape[a["axis"]],
                        }
                    )
                elif r.op == "matmul":
                    matmuls.append(
                        {
                            **describe(graph, i),
                            "lhs": graph.nodes[r.args[0]].shape,
                            "rhs": graph.nodes[r.args[1]].shape,
                            "operand_dtype": graph.nodes[r.args[0]].dtype,
                            "contraction_extent": graph.nodes[r.args[0]].shape[-1],
                        }
                    )
            rows.append(
                {
                    **region,
                    "inputs": [describe(graph, i) for i in region["inputs"]],
                    "outputs": [describe(graph, i) for i in region["outputs"]],
                    "casts": casts,
                    "reductions": reductions,
                    "matmuls": matmuls,
                }
            )
        result["phases"][phase] = rows
    needed = graph.ancestors([*graph.forward_outputs, *graph.backward_outputs])
    result["inter_chunk_states"] = [
        describe(graph, n.id)
        for n in graph.nodes
        if n.id in needed and n.name.startswith("core.S_after_chunk_")
    ]
    result["selected_residuals"] = [describe(graph, i) for i in plan.saved]
    result["selected_residual_bytes"] = plan.residual_bytes
    result[
        "boundary_byte_caveat"
    ] = "logical tensor extents, not physical traffic or peak; views can alias"
    return result


@dataclass
class PartitionedProgram:
    graph: object
    plan: object
    path: Path
    module: object
    phases: dict


def lower_partitioned(graph, plan, path):
    """Emit each annotated region as a separately callable native HIR function."""
    phases, sources, count = {}, [], 0
    for phase in ("forward", "backward"):
        phases[phase] = []
        for region in partition_phase(graph, plan, phase):
            if not region["outputs"]:
                continue
            emitted = emit_phase(
                graph,
                region["outputs"],
                name=f"stage_{count}",
                available=set(region["inputs"]),
                recipes=None if phase == "forward" else plan.recipes,
            )
            phases[phase].append((region["label"], emitted))
            sources.append(emitted.source)
            count += 1
    header = (
        '"""Generated fixed candidate regions; no claim of GPU kernel fusion."""\n'
        "from tilefoundry.dsl import Tensor, Topology, func, module, tf\n"
        "from tilefoundry.ir.core.kinds import ReduceKind\n"
        "from tilefoundry_nvidia import CudaTarget\n\n"
        '@module(entry="stage_0", target=CudaTarget("nvidia.h200_sxm"), '
        'topologies=(Topology("cta", 1),))\nclass CandidateRegions:\n'
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n\n".join(sources) + "\n")
    with deep_ir_visitors():
        module = load_authored_ir(str(path))
    return PartitionedProgram(graph, plan, path, module, phases)


def check_partitioned(program):
    with deep_ir_visitors():
        for phases in program.phases.values():
            for _, phase in phases:
                check_program(
                    program.module,
                    function(program.module, phase.name),
                    budget=max(10000, 4 * len(program.graph.nodes)),
                )


def execute_partitioned(program, inputs):
    """Enforce each region's interface and release dead transient boundary values."""
    graph, plan = program.graph, program.plan
    constants = {
        n.id: constant_tensor(n) for n in graph.nodes if n.recipe.op == "constant"
    }

    def phase_execute(label, values, roots):
        values = dict(values)
        uses = Counter(i for _, p in program.phases[label] for i in p.inputs)
        keep = set(roots)
        with deep_ir_visitors():
            for _, p in program.phases[label]:
                outputs = evaluate(
                    function(program.module, p.name), *(values[i] for i in p.inputs)
                )
                values.update(zip(p.outputs, outputs, strict=True))
                for i in p.inputs:
                    uses[i] -= 1
                    if not uses[i] and i not in keep:
                        values.pop(i, None)
        return tuple(values[i] for i in roots)

    roots = [*graph.forward_outputs, *plan.saved]
    forward = phase_execute("forward", {**inputs, **constants}, roots)
    values = {**inputs, **constants, **dict(zip(roots, forward, strict=True))}
    backward = phase_execute("backward", values, graph.backward_outputs)
    assert (
        sum(t.numel() * t.element_size() for t in forward[len(graph.forward_outputs) :])
        == plan.residual_bytes
    )
    return forward[: len(graph.forward_outputs)], backward
