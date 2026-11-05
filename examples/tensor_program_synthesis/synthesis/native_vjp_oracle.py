"""Differentiate only the native forward, with explicit FP32 gradient fan-in.

BF16 rounding remains in the forward. Master weight leaves are FP32 and widening
casts through views are shared before fan-out. No candidate backward is reused.
"""

import argparse
import json

import torch
from tilefoundry.evaluator import evaluate

from .full_layer import PARAMETERS, boundary_policy, full_layer_graph
from .graph import Graph, Value, constant_tensor
from .lower import deep_ir_visitors, execute, function, lower
from .mixed_layer_run import DEST
from .mixed_precision import compare_mixed, master_reference_inputs, mixed_inputs
from .multihead import configuration
from .planner import plan_from_saved


def native_forward_oracle(source, extra_outputs=()):
    graph = Graph("native_forward_autograd_oracle")
    mapped = {}

    def promote(value):
        if value.node.dtype == "f32":
            return value
        recipe = value.node.recipe
        if recipe.op in {"slice", "reshape", "permute", "transpose"}:
            return graph.op(
                recipe.op, promote(Value(graph, recipe.args[0])), **dict(recipe.attrs)
            )
        return value.cast("f32")

    for node in source.nodes:
        if node.phase != "forward":
            continue
        recipe, attrs = node.recipe, dict(node.recipe.attrs)
        if recipe.op == "input":
            value = graph.input(
                node.name, node.shape, "f32" if node.name in PARAMETERS else node.dtype
            )
        elif recipe.op == "scalar":
            value = graph.scalar(attrs["value"])
        elif recipe.op == "constant":
            value = graph.constant(attrs["kind"], attrs["size"])
        else:
            args = [mapped[i] for i in recipe.args]
            if recipe.op == "cast" and attrs["dtype"] == "f32":
                value = promote(args[0])
            elif recipe.op == "matmul" and attrs.get("out_dtype") == "f32":
                value = graph.op("matmul", *(promote(v) for v in args))
            else:
                value = graph.op(recipe.op, *args, **attrs)
            if value.node.recipe.op != "input":
                value.named(node.name)
        mapped[node.id] = value
    outputs = [mapped[i] for i in source.forward_outputs]
    outputs += [
        mapped[next(n.id for n in source.nodes if n.name == name)]
        for name in extra_outputs
    ]
    graph.begin_backward(outputs)
    graph.finish(outputs)  # Unused passthrough; autograd reads only forward.
    graph.metadata = {"strict_dtypes": True, "scope": "forward-only gradient oracle"}
    return graph


def evaluate_oracle(graph, path, refs, extra_outputs=()):
    oracle = native_forward_oracle(graph, extra_outputs)
    program = lower(oracle, plan_from_saved(oracle, []), path)
    values = {i: refs[oracle.nodes[i].name] for i in oracle.forward_inputs}
    values.update(
        {n.id: constant_tensor(n) for n in oracle.nodes if n.recipe.op == "constant"}
    )
    with deep_ir_visitors():
        return evaluate(
            function(program.module, "forward"),
            *(values[i] for i in program.forward.inputs),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    graph = full_layer_graph(
        configuration("Qwen3.5-397B-A17B"), args.length, precision="bf16_fp32"
    )
    values = mixed_inputs(graph)
    refs = master_reference_inputs(values)
    torch.manual_seed(909)
    cot = torch.randn(1, args.length, 4096).bfloat16()
    names = graph.metadata["gradient_inputs"]
    print("Native forward with FP32 master-gradient leaves", flush=True)
    out = evaluate_oracle(graph, DEST / "generated/native_forward_oracle.py", refs)
    grad = torch.autograd.grad(out[0], [refs[n] for n in names], cot)
    expected = tuple(t.detach() for t in (*out, *grad))
    del out, grad, refs
    print("Native explicit backward", flush=True)
    inputs = {i: values[graph.nodes[i].name] for i in graph.forward_inputs}
    inputs[graph.backward_inputs[0]] = cot
    program = lower(
        graph,
        plan_from_saved(graph, boundary_policy(graph)),
        DEST / "generated/native_vjp_candidate.py",
    )
    out, grad = execute(program, inputs)
    checks = compare_mixed((*out, *grad), expected, ["output", *names])
    report = dict(
        scope="forward-only native HIR independently differentiated by PyTorch; BF16 rounding retained, FP32 master leaves and fan-in accumulation",
        passed=all(c["passed"] for c in checks),
        output_exact=torch.equal(out[0], expected[0]),
        checks=checks,
    )
    (DEST / "native_vjp_oracle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("Native VJP oracle comparison failed; criteria unchanged")


if __name__ == "__main__":
    main()
