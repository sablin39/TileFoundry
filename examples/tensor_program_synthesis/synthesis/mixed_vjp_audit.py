"""Check all analytic VJPs at matching BF16 forward arithmetic; retain other errors."""

import argparse
import gc
import json

import torch

from .full_layer import boundary_policy, full_layer_graph
from .graph import evaluate_graph
from .lower import execute, lower
from .mixed_layer_run import DEST
from .mixed_precision import (
    compare_mixed,
    master_reference_inputs,
    mixed_inputs,
    reference_mixed_layer,
)
from .multihead import configuration
from .planner import plan_from_saved


def boundary_errors(actual, expected):
    result = {}
    for name, x in actual.items():
        y = expected[name]
        xf, yf = x.float(), y.float()
        delta = xf - yf
        row = dict(
            max_abs=delta.abs().max().item(),
            relative_l2=(
                delta.double().norm() / yf.double().norm().clamp_min(1e-30)
            ).item(),
            changed_elements=int(torch.count_nonzero(x != y)),
            elements=x.numel(),
        )
        if x.dtype == y.dtype == torch.bfloat16:

            def ordered(t):
                bits = t.contiguous().view(torch.int16).int()
                return torch.where(bits < 0, 32768 - (bits & 32767), bits + 32768)

            distance = (ordered(x) - ordered(y)).abs()
            row["max_bf16_ulp"] = int(distance.max())
            row["more_than_one_ulp"] = int(torch.count_nonzero(distance > 1))
        result[name] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = configuration("Qwen3.5-397B-A17B")
    graph = full_layer_graph(config, args.length, precision="bf16_fp32")
    plan = plan_from_saved(graph, boundary_policy(graph))
    values = mixed_inputs(graph)
    refs = master_reference_inputs(values)
    names = graph.metadata["gradient_inputs"]
    torch.manual_seed(909)
    cot = torch.randn(1, args.length, config["hidden_size"]).bfloat16()
    print("Autograd with matching forward arithmetic", flush=True)
    y, stages = reference_mixed_layer(
        refs, config, core="candidate_autograd", return_stages=True
    )
    gradient = torch.autograd.grad(y, [refs[n] for n in names], cot)
    expected = tuple(x.detach() for x in (y, *gradient))
    stages = {name: x.detach() for name, x in stages.items()}
    del y, gradient, refs
    gc.collect()
    inputs = {i: values[graph.nodes[i].name] for i in graph.forward_inputs}
    inputs[graph.backward_inputs[0]] = cot
    print("Native HIR explicit backward", flush=True)
    program = lower(graph, plan, DEST / "generated/vjp_audit.py")
    out, grad = execute(program, inputs)
    checks = compare_mixed((*out, *grad), expected, ["output", *names])
    report = dict(
        reference="matching forward arithmetic, independently differentiated by PyTorch autograd; candidate backward remains supplied formulas",
        shape=dict(model="Qwen3.5-397B-A17B", batch=1, length=args.length),
        output_bitwise_equal=torch.equal(out[0], expected[0]),
        checks=checks,
        passed=all(c["passed"] for c in checks),
    )
    (DEST / "vjp_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    del program, out, grad, expected
    gc.collect()
    print("Independent forward BF16 boundary audit", flush=True)
    with torch.no_grad():
        _, official_stages = reference_mixed_layer(
            master_reference_inputs(values), config, return_stages=True
        )
        report["forward_boundaries_vs_independent_reference"] = boundary_errors(
            stages, official_stages
        )
        # Confirm the matching-forward reference really uses the candidate's
        # stored values, rather than merely yielding a close final output.
        mapping = {
            "qkv_projection": "layer.qkv_projection",
            "conv_pre": "layer.conv_pre_activation",
            "conv": "layer.conv_activation",
            "z": "layer.z_projection",
            "beta": "layer.beta",
            "log_decay": "layer.log_decay",
            "core_output": "layer.core_output",
            "rms_unit_storage": "layer.rms_unit_storage",
            "gated_norm_output": "layer.gated_norm_output",
        }
        ids = {
            name: next(n.id for n in graph.nodes if n.name == label)
            for name, label in mapping.items()
        }
        result, _ = evaluate_graph(
            graph, list(ids.values()), inputs, retain_intermediates=False
        )
        report["matching_forward_boundary_exact"] = {
            name: torch.equal(tensor.float(), stages[name].float())
            for name, tensor in zip(ids, result, strict=True)
        }
    report["passed"] &= all(report["matching_forward_boundary_exact"].values())
    (DEST / "vjp_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("Mixed VJP audit failed; original criteria retained")
    print("VJP audit complete", flush=True)


if __name__ == "__main__":
    main()
