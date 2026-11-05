"""Precision boundaries, FP32 parameter accumulation, and complete BF16 HIR VJPs."""

import pytest
import torch

from .full_layer import boundary_policy, full_layer_graph
from .graph import Graph, evaluate_graph
from .lower import check_program, deep_ir_visitors, execute, function, lower
from .mixed_precision import (
    compare_mixed,
    master_reference_inputs,
    mixed_inputs,
    reference_mixed_layer,
)
from .planner import plan_from_saved
from .test_full_layer import tiny_config


@pytest.mark.parametrize("length,group", [(1, 2), (7, 2), (9, 3)])
@pytest.mark.parametrize("save", [False, True])
def test_mixed_layer_all_parameter_gradients(tmp_path, length, group, save):
    config = tiny_config(group)
    graph = full_layer_graph(config, length, 2, 4, precision="bf16_fp32")
    tensors = mixed_inputs(graph)
    refs = master_reference_inputs(tensors)
    reference = reference_mixed_layer(refs, config, core="token")
    cot = torch.randn(reference.shape).bfloat16()
    names = graph.metadata["gradient_inputs"]
    grads = torch.autograd.grad(reference, [refs[n] for n in names], cot)
    plan = plan_from_saved(graph, boundary_policy(graph) if save else [])
    inputs = {i: tensors[graph.nodes[i].name] for i in graph.forward_inputs}
    inputs[graph.backward_inputs[0]] = cot
    forward, _ = evaluate_graph(graph, [*graph.forward_outputs, *plan.saved], inputs)
    binputs = {
        **inputs,
        **dict(zip([*graph.forward_outputs, *plan.saved], forward, strict=True)),
    }
    backward, _ = evaluate_graph(
        graph, graph.backward_outputs, binputs, recipes=plan.recipes
    )
    checks = compare_mixed(
        (forward[0], *backward), (reference, *grads), ["output", *names]
    )
    assert all(r["passed"] for r in checks), checks
    assert forward[0].dtype == backward[0].dtype == torch.bfloat16
    assert all(g.dtype == torch.float32 for g in backward[1:])
    assert plan.residual_bytes == sum(t.numel() * t.element_size() for t in forward[1:])
    program = lower(graph, plan, tmp_path / f"mixed_{length}_{save}.py")
    with deep_ir_visitors():
        for phase in ("forward", "backward"):
            check_program(program.module, function(program.module, phase))
    out, grad = execute(program, inputs)
    for x, y in zip((*out, *grad), (forward[0], *backward), strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    if length > 2:
        assert torch.count_nonzero(grad[0][0, -2:]) == 0


def test_bf16_matmul_retains_fp32_accumulator(tmp_path):
    graph = Graph("mixed_accumulator")
    graph.metadata["strict_dtypes"] = True
    a = graph.input("a", (1, 4), "bf16")
    b = graph.input("b", (4, 1), "bf16")
    value = graph.op("matmul", a, b, out_dtype="f32")
    graph.begin_backward([value])
    graph.finish([value])
    plan = plan_from_saved(graph, [])
    inputs = {
        a.id: torch.tensor([[1, 1, 1, 1]], dtype=torch.bfloat16),
        b.id: torch.tensor([[1], [0.001], [0.001], [0.001]], dtype=torch.bfloat16),
    }
    expected = inputs[a.id].float() @ inputs[b.id].float()
    program = lower(graph, plan, tmp_path / "accumulator.py")
    out, _ = execute(program, inputs)
    assert torch.equal(out[0], expected)
    assert not torch.equal(expected, expected.bfloat16().float())
    assert "out_dtype='f32'" in program.path.read_text()


def test_candidate_regions_preserve_full_layer(tmp_path):
    from .fusion_boundaries import (
        boundary_report,
        check_partitioned,
        execute_partitioned,
        lower_partitioned,
    )

    graph = full_layer_graph(tiny_config(), 9, 2, 4, precision="bf16_fp32")
    tensors = mixed_inputs(graph)
    plan = plan_from_saved(graph, boundary_policy(graph))
    inputs = {i: tensors[graph.nodes[i].name] for i in graph.forward_inputs}
    inputs[graph.backward_inputs[0]] = torch.randn(2, 9, 12).bfloat16()
    whole = lower(graph, plan, tmp_path / "whole.py")
    split = lower_partitioned(graph, plan, tmp_path / "regions.py")
    check_partitioned(split)
    actual = execute_partitioned(split, inputs)
    expected = execute(whole, inputs)
    for a, b in zip(actual, expected, strict=True):
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    report = boundary_report(graph, plan)
    assert report["inter_chunk_states"]
    assert any(r["label"] == "fwd.chunk_prepare" for r in report["phases"]["forward"])
    assert any(r["label"] == "bwd.state_adjoint" for r in report["phases"]["backward"])


def test_native_forward_vjp_oracle(tmp_path):
    from .native_vjp_oracle import evaluate_oracle

    graph = full_layer_graph(tiny_config(), 9, 2, 4, precision="bf16_fp32")
    tensors = mixed_inputs(graph)
    refs = master_reference_inputs(tensors)
    cot = torch.randn(2, 9, 12).bfloat16()
    names = graph.metadata["gradient_inputs"]
    out = evaluate_oracle(graph, tmp_path / "oracle.py", refs)
    grad = torch.autograd.grad(out[0], [refs[n] for n in names], cot)
    inputs = {i: tensors[graph.nodes[i].name] for i in graph.forward_inputs}
    inputs[graph.backward_inputs[0]] = cot
    program = lower(
        graph, plan_from_saved(graph, boundary_policy(graph)), tmp_path / "candidate.py"
    )
    y, g = execute(program, inputs)
    checks = compare_mixed((*y, *g), (*out, *grad), ["output", *names])
    assert all(c["passed"] for c in checks), checks


def test_bf16_graph_enforces_dtypes_without_optional_metadata():
    graph = Graph("implicit_mixed_contract")
    x = graph.input("x", (1, 4), "bf16")
    total = x.cast("f32").sum(1)
    values = {x.id: torch.tensor([[1, 0.001, 0.001, 0.001]], dtype=torch.bfloat16)}
    actual, _ = evaluate_graph(graph, [total.id], values)
    assert actual[0].dtype == torch.float32
    assert torch.equal(actual[0], values[x.id].float().sum(1, keepdim=True))
