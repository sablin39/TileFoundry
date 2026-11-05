"""Locate BF16 backward rounding differences without changing acceptance criteria."""

import json

import torch

from .full_layer import boundary_policy, full_layer_graph
from .lower import execute, lower
from .mixed_layer_run import DEST
from .mixed_precision import master_reference_inputs, mixed_inputs
from .multihead import configuration
from .native_vjp_oracle import evaluate_oracle
from .planner import plan_from_saved


def stats(x, y):
    a, b = x.float(), y.float()
    d = a - b
    return dict(
        dtype=str(x.dtype),
        exact=torch.equal(a, b),
        max_abs=d.abs().max().item(),
        relative_l2=(d.double().norm() / b.double().norm().clamp_min(1e-30)).item(),
        bf16_changed_elements=int(torch.count_nonzero(a.bfloat16() != b.bfloat16())),
        elements=a.numel(),
    )


def main():
    torch.set_num_threads(8)
    graph = full_layer_graph(
        configuration("Qwen3.5-397B-A17B"), 8192, precision="bf16_fp32"
    )
    values = mixed_inputs(graph)
    refs = master_reference_inputs(values)
    torch.manual_seed(909)
    cot = torch.randn(1, 8192, 4096).bfloat16()
    fwd_names = (
        "layer.gated_norm_output",
        "layer.core_output_fp32",
        "layer.qkv_projection",
    )
    bwd_names = (
        "layer.d_gated_norm_output",
        "layer.d_core_output_fp32",
        "layer.d_qkv_projection",
    )
    print("Native forward autograd checkpoint VJPs", flush=True)
    out = evaluate_oracle(graph, DEST / "generated/rounding_oracle.py", refs, fwd_names)
    names = graph.metadata["gradient_inputs"]
    expected = torch.autograd.grad(
        out[0], [refs[n] for n in names] + list(out[1:]), cot
    )
    expected = tuple(t.detach() for t in expected)
    del out, refs
    graph.backward_outputs += [
        next(n.id for n in graph.nodes if n.name == name) for name in bwd_names
    ]
    inputs = {i: values[graph.nodes[i].name] for i in graph.forward_inputs}
    inputs[graph.backward_inputs[0]] = cot
    print("Explicit backward checkpoints", flush=True)
    program = lower(
        graph,
        plan_from_saved(graph, boundary_policy(graph)),
        DEST / "generated/rounding_candidate.py",
    )
    _, actual = execute(program, inputs)
    comparisons = {
        name: stats(a.reshape(b.shape), b)
        for name, a, b in zip(bwd_names, actual[10:], expected[10:], strict=True)
    }
    print(json.dumps(comparisons), flush=True)
    x = (values["hidden_states"].float() * values["attention_mask"]).reshape(8192, 4096)
    a = actual[-1].reshape(8192, -1).float()
    b = expected[-1].reshape(8192, -1).float()
    # Recompute the two projection weight gradients using the two cotangents.
    # This tests whether the discrepancy is upstream of the dW contraction.
    pred_a = a.T @ x
    pred_b = b.T @ x
    result = dict(
        checkpoints=comparisons,
        candidate_qkv_dW_from_its_cotangent=stats(actual[1], pred_a),
        autograd_qkv_dW_from_its_cotangent=stats(expected[1], pred_b),
        qkv_dW_difference_explanation=stats(actual[1] - expected[1], pred_a - pred_b),
        scope="diagnostic attribution only; original strict failures are retained",
    )
    (DEST / "gradient_rounding_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
