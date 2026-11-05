"""Lower one selected joint-graph plan into separate TileFoundry invocations."""

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tilefoundry.analysis import (
    ComputeCostMetadata,
    RegionMemoryMetadata,
    analyze,
    check_program,
)
from tilefoundry.analysis.liveness import LivenessMetadata
from tilefoundry.analysis.memory import MemoryOptions
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.evaluator import evaluate
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.ir.core import get_metadata

from .graph import constant_tensor


@contextmanager
def deep_ir_visitors():
    """Allow the upstream recursive visitors to walk 128 unrolled chunks.

    Scoped to this single-threaded prototype; it does not replace the visitors.
    Structured chunk/stage modules remain preferable for compiler scalability.
    """
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous, 20000))
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


def expression(recipe, args):
    attrs = dict(recipe.attrs)
    op = recipe.op
    if op == "matmul" and "out_dtype" in attrs:
        return f"tf.matmul({', '.join(args)}, out_dtype={attrs['out_dtype']!r})"
    if op in {
        "add",
        "sub",
        "mul",
        "matmul",
        "neg",
        "exp",
        "relu",
        "rsqrt",
        "sigmoid",
        "silu",
        "softplus",
        "where",
    }:
        return f"tf.{op}({', '.join(args)})"
    if op == "zeros":
        return (
            f"tf.zeros(type=Tensor[{attrs['shape']!r}, {attrs.get('dtype', 'f32')!r}])"
        )
    if op == "reshape":
        return f"tf.reshape({args[0]}, new_shape={attrs['shape']!r})"
    if op == "gt":
        return f"tf.cmp_gt({args[0]}, {args[1]})"
    if op in {"transpose", "permute"}:
        return f"tf.transpose({args[0]}, perm={attrs.get('perm', (1, 0))!r})"
    if op == "cast":
        return f"tf.cast({args[0]}, dtype={attrs['dtype']!r})"
    if op == "sum":
        return f"tf.reduce({args[0]}, axes=({attrs['axis']},), keepdim=True, kind=ReduceKind.SUM)"
    if op == "slice":
        slices = [":"] * attrs.get("rank", attrs.get("axis", 0) + 2)
        slices[attrs.get("axis", 0)] = f"{attrs['start']}:{attrs['end']}"
        return f"{args[0]}[{', '.join(slices)}]"
    if op == "concat":
        return f"tf.concat([{', '.join(args)}], axis={attrs.get('axis', 0)})"
    raise ValueError(f"No TileFoundry lowering for {op}")


@dataclass
class Phase:
    name: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    source: str


@dataclass
class Program:
    graph: object
    plan: object
    path: Path
    forward: Phase
    backward: Phase
    module: object


def emit_phase(graph, roots, *, name, available, recipes=None):
    lines, inputs, known = [], [], {}
    boundary = set(available) | {
        n.id for n in graph.nodes if n.recipe.op in {"constant", "scalar"}
    }

    def label_for(i):
        if i in known:
            return known[i]
        node = graph.nodes[i]
        if node.recipe.op == "scalar":
            return repr(dict(node.recipe.attrs)["value"])
        label = f"v_{i}"
        if i in boundary:
            inputs.append(i)
        else:
            raise ValueError(f"Unavailable input {node.name}")
        known[i] = label
        return label

    for i in graph.schedule(roots, available=boundary, recipes=recipes):
        node = graph.nodes[i]
        if recipes is not None and i not in recipes:
            raise ValueError(f"Solver supplied no source for {node.name}")
        recipe = node.recipe if recipes is None else node.recipes[recipes[i]]
        args = [label_for(j) for j in recipe.args]
        label = f"v_{i}"
        lines.append(f"        {label} = {expression(recipe, args)}")
        known[i] = label
    results = [label_for(i) for i in roots]
    params = [
        f"v_{i}: Tensor[{graph.nodes[i].shape!r}, {graph.nodes[i].dtype!r}]"
        for i in inputs
    ]
    body = ["    @func", f"    def {name}({', '.join(params)}):", *lines]
    body.append(f"        return ({', '.join(results)},)")
    return Phase(name, tuple(inputs), tuple(roots), "\n".join(body))


def lower(graph, plan, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    forward = emit_phase(
        graph,
        [*graph.forward_outputs, *plan.saved],
        name="forward",
        available=set(graph.forward_inputs),
    )
    backward = emit_phase(
        graph,
        graph.backward_outputs,
        name="backward",
        available=graph.external | set(plan.saved),
        recipes=plan.recipes,
    )
    header = (
        '"""Generated from a selected residual plan; edit the joint graph to regenerate."""\n\n'
        "from tilefoundry.dsl import Tensor, Topology, func, module, tf\n"
        "from tilefoundry.ir.core.kinds import ReduceKind\n"
        "from tilefoundry_nvidia import CudaTarget\n\n"
        '@module(entry="forward", target=CudaTarget("nvidia.h200_sxm"), '
        'topologies=(Topology("cta", 1),))\n'
        "class Generated:\n"
    )
    path.write_text(header + forward.source + "\n\n" + backward.source + "\n")
    with deep_ir_visitors():
        module = load_authored_ir(str(path))
    return Program(graph, plan, path, forward, backward, module)


def function(module, name):
    return next(fn for fn in module.functions if fn.name == name)


def execute(program, inputs):
    """Run generated HIR with only its declared phase interface available."""
    values = dict(inputs)
    for node in program.graph.nodes:
        if node.recipe.op == "constant":
            values[node.id] = constant_tensor(node)
    with deep_ir_visitors():
        fwd = evaluate(
            function(program.module, "forward"),
            *(values[i] for i in program.forward.inputs),
        )
        values.update(zip(program.forward.outputs, fwd, strict=True))
        bwd = evaluate(
            function(program.module, "backward"),
            *(values[i] for i in program.backward.inputs),
        )
    actual_saved = sum(
        values[i].numel() * values[i].element_size() for i in program.plan.saved
    )
    if actual_saved != program.plan.residual_bytes:
        raise AssertionError(
            f"Residual accounting differs: {actual_saved}, {program.plan.residual_bytes}"
        )
    return fwd[: len(program.graph.forward_outputs)], bwd


def static_reports(program, directory, *, memory_timeout=15):
    """Run all four analysis families and retain liveness, peaks, and diagnostics."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for phase in (program.forward, program.backward):
        result = analyze(
            *check_program(program.module, function(program.module, phase.name)),
            analysis=("compute-cost", "memory", "roofline", "performance"),
            options=MemoryOptions(
                timeout_seconds=memory_timeout, workers=1, random_seed=0
            ),
        )
        rendered = render_analysis(result)
        (directory / f"{phase.name}.json").write_text(
            json.dumps(rendered.data, indent=2) + "\n"
        )
        (directory / f"{phase.name}.hir.txt").write_text(rendered.annotated + "\n")
        memory = get_metadata(result.function, RegionMemoryMetadata)
        compute = get_metadata(result.function, ComputeCostMetadata)
        live = get_metadata(result.function, LivenessMetadata)
        summaries[phase.name] = {
            "flops": sum(spread.total for _, spread in compute.flops.kinds),
            "other_ops": {
                kind: spread.total for kind, spread in compute.other_ops.kinds
            },
            "traffic_bytes": {
                kind: {"read": spread.total.read, "write": spread.total.write}
                for kind, spread in memory.traffic.storage.kinds
            },
            "peak_bytes": {p.memory_level: p.peak_bytes for p in memory.peaks},
            "allocation_solver_status": memory.solver_status,
            "errors": list(memory.errors),
            "advisories": list(memory.advisories),
            "lifetime_intervals": len(live.intervals),
        }
        if memory.errors:
            raise RuntimeError(f"{phase.name} analysis errors: {memory.errors}")
    return summaries
