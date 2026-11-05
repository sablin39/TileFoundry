"""Incomplete 8k GDN example with BF16 storage and FP32 accumulation."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .full_layer import boundary_policy, full_layer_graph, load_official_layer
from .fusion_boundaries import (
    boundary_report,
    check_partitioned,
    execute_partitioned,
    lower_partitioned,
)
from .lower import check_program, deep_ir_visitors, execute, function, lower
from .mixed_precision import (
    compare_mixed,
    master_reference_inputs,
    mixed_inputs,
    reference_mixed_layer,
)
from .multihead import configuration
from .planner import plan_from_saved

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/full_gdn_bf16_8k"


def write_summary(summary):
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def literal_official_check(config, tensors, cot, names, expected):
    """Observe the unchanged official module's BF16 leaf-gradient rounding.

    This uses BF16 projection/conv parameters and FP32 gate/norm parameters.
    Its BF16 parameter-gradient buffers and shared-head adjoint rounding differ
    from the FP32 accumulation contract tested by the independent reference.
    """
    block = load_official_layer()(SimpleNamespace(**config), layer_idx=0).float()
    for name, parameter in block.named_parameters():
        parameter.data = tensors[name]
    hidden = tensors["hidden_states"].detach().requires_grad_()
    values = {"hidden_states": hidden, **dict(block.named_parameters())}
    output = block(hidden, attention_mask=tensors["attention_mask"].squeeze(-1))
    grads = torch.autograd.grad(output, [values[n] for n in names], cot)
    checks = compare_mixed(
        (output, *grads), expected, ["output", *names], relative_l2_max=0.02
    )
    return dict(
        scope="unchanged pinned BF16 module, with BF16 parameter-gradient storage; numerical rounding comparison, not identical gradient-storage semantics",
        checks=checks,
        passed=all(r["passed"] for r in checks),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen3.5-397B-A17B")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-native-rounding-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    DEST.mkdir(exist_ok=True, parents=True)
    (DEST / "complete.txt").unlink(missing_ok=True)
    config = configuration(args.model)
    print(
        "Build mixed full-layer graph", args.model, args.batch, args.length, flush=True
    )
    graph = full_layer_graph(config, args.length, args.batch, precision="bf16_fp32")
    plan = plan_from_saved(graph, boundary_policy(graph))
    (DEST / "graph.json").write_text(json.dumps(graph.dump(), indent=2) + "\n")
    (DEST / "fusion_boundaries.json").write_text(
        json.dumps(boundary_report(graph, plan), indent=2) + "\n"
    )
    tensors = mixed_inputs(graph)
    refs = master_reference_inputs(tensors)
    names = graph.metadata["gradient_inputs"]
    torch.manual_seed(909)
    cot = torch.randn(args.batch, args.length, config["hidden_size"]).bfloat16()
    summary = dict(
        model=args.model,
        batch=args.batch,
        length=args.length,
        key_heads=config["linear_num_key_heads"],
        value_heads=config["linear_num_value_heads"],
        hidden_size=config["hidden_size"],
        chunk_size=64,
        weights="synthetic weights, BF16 compute copies, weak decay; no checkpoint weights loaded",
        precision_contract=graph.metadata["precision_contract"],
        gradient_contract="nine FP32 parameter-gradient tensors; BF16 hidden-state gradient",
        reference="independent PyTorch full-layer formulas with explicit BF16 rounding, FP32 master-gradient accumulation, and pinned official triangular-solve chunk core",
        reference_revision="fd3cf0140de03f7619ac95b5cfed05b711ff9e22",
        policy="one fixed selected-boundary residual policy; no solver or optimization",
        residual_bytes=plan.residual_bytes,
        residuals=plan.dump(graph)["saved"],
        acceptance="per tensor: relative L2 <= 0.005 and elementwise atol=0.002*max(reference_RMS,1), rtol=0.02; all values finite",
        executions=[],
    )
    write_summary(summary)
    print("Independent FP32-accumulation reference forward/backward", flush=True)
    y = reference_mixed_layer(refs, config)
    gradients = torch.autograd.grad(y, [refs[n] for n in names], cot)
    expected = tuple(x.detach() for x in (y, *gradients))
    assert all(g.dtype == torch.float32 for g in expected[2:])
    del y, gradients, refs
    gc.collect()
    inputs = {i: tensors[graph.nodes[i].name] for i in graph.forward_inputs}
    inputs[graph.backward_inputs[0]] = cot
    whole_results = None
    for kind in ("whole_layer", "candidate_regions"):
        print("Emit and check native HIR", kind, flush=True)
        path = DEST / "generated" / f"{kind}.py"
        if kind == "whole_layer":
            program = lower(graph, plan, path)
            with deep_ir_visitors():
                for phase in ("forward", "backward"):
                    check_program(
                        program.module,
                        function(program.module, phase),
                        budget=max(10000, 4 * len(graph.nodes)),
                    )
            print("Execute full HIR forward/backward", flush=True)
            out, grad = execute(program, inputs)
        else:
            program = lower_partitioned(graph, plan, path)
            check_partitioned(program)
            print("Execute candidate-region HIR forward/backward", flush=True)
            out, grad = execute_partitioned(program, inputs)
        actual = (*out, *grad)
        checks = compare_mixed(actual, expected, ["output", *names])
        assert out[0].dtype == grad[0].dtype == torch.bfloat16
        assert all(g.dtype == torch.float32 for g in grad[1:])
        entry = dict(
            backend=kind,
            native_program_check=True,
            checks=checks,
            passed=all(r["passed"] for r in checks),
            source=str(path.relative_to(ROOT)),
        )
        if whole_results is None:
            whole_results = actual
        else:
            entry["bitwise_equal_to_whole_layer"] = all(
                torch.equal(x, y) for x, y in zip(actual, whole_results, strict=True)
            )
            entry["passed"] &= entry["bitwise_equal_to_whole_layer"]
        summary["executions"].append(entry)
        write_summary(summary)
        print(
            json.dumps(
                {
                    "backend": kind,
                    "passed": entry["passed"],
                    "max_relative_l2": max(r["relative_l2"] for r in checks),
                    "failed": [r for r in checks if not r["passed"]],
                    "partition_exact": entry.get("bitwise_equal_to_whole_layer"),
                }
            ),
            flush=True,
        )
        del program, out, grad, actual
        gc.collect()
    if not args.skip_native_rounding_check:
        print("Unchanged official BF16 module rounding comparison", flush=True)
        summary["literal_official_rounding"] = literal_official_check(
            config, tensors, cot, names, expected
        )
        write_summary(summary)
        print(
            json.dumps(
                {
                    "literal_official_rounding_passed": summary[
                        "literal_official_rounding"
                    ]["passed"],
                    "max_relative_l2": max(
                        r["relative_l2"]
                        for r in summary["literal_official_rounding"]["checks"]
                    ),
                }
            ),
            flush=True,
        )
    summary["passed"] = all(e["passed"] for e in summary["executions"])
    write_summary(summary)
    if not summary["passed"]:
        raise SystemExit(
            "Mixed full-layer validation failed; inspect per-parameter checks"
        )
    (DEST / "complete.txt").write_text("complete\n")
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
