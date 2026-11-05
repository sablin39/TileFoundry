"""Grouped-head VJPs, Q/K reduction, broadcasting and actual batched HIR."""

import pytest
import torch

from .graph import evaluate_graph
from .lower import lower
from .multihead import Core, candidate, make_graph, pack, recurrent64, sample_inputs
from .planner import plan_from_saved


@pytest.mark.parametrize("group", [2, 3, 4])
@pytest.mark.parametrize("variant", ["sequential", "scan", "qwen_prepared"])
def test_grouped_heads_and_normalization_all_gradients(group, variant):
    graph = make_graph(17, 2, 2, 2 * group, 8, 6, 4, variant)
    saved = [
        n.id
        for n in graph.nodes
        if n.name.endswith(".inverse") and n.id in graph.candidates
    ]
    plan = plan_from_saved(graph, saved)
    args, cot = sample_inputs(
        2,
        17,
        2,
        2 * group,
        8,
        6,
        dtype=torch.float64,
        case="weak_decay",
        initial_zero=False,
    )
    ref = recurrent64(*args)
    ref_grad = torch.autograd.grad(ref, args, grad_outputs=cot)
    out = candidate(graph, plan, *args)
    grad = torch.autograd.grad(out, args, grad_outputs=cot)
    for x, y in zip((*out, *grad), (*ref, *ref_grad), strict=True):
        torch.testing.assert_close(x, y, atol=2e-9, rtol=2e-9)
    kk = next(n for n in graph.nodes if n.name == "c0.KK")
    inverse = next(n for n in graph.nodes if n.name == "c0.inverse")
    assert kk.shape == (2, 2, 1, 4, 4)
    assert inverse.shape == (2, 2, group, 4, 4)


@pytest.mark.parametrize("variant", ["sequential", "scan", "qwen_prepared"])
def test_actual_grouped_hir(tmp_path, variant):
    graph = make_graph(17, 2, 2, 4, 8, 6, 4, variant)
    plan = plan_from_saved(
        graph, [i for i in graph.candidates if graph.nodes[i].name.endswith(".inverse")]
    )
    program = lower(graph, plan, tmp_path / f"grouped_{variant}.py")
    args, cot = sample_inputs(2, 17, 2, 4, 8, 6, case="weak_decay", initial_zero=False)
    expected = candidate(graph, plan, *args)
    expected_grad = torch.autograd.grad(expected, args, grad_outputs=cot)
    actual = candidate(graph, plan, *args, program=program)
    actual_grad = torch.autograd.grad(actual, args, grad_outputs=cot)
    for x, y in zip((*actual, *actual_grad), (*expected, *expected_grad), strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_lifetime_releasing_evaluator_preserves_values():
    graph = make_graph(17, 2, 2, 4, 8, 6, 4)
    args, _ = sample_inputs(2, 17, 2, 4, 8, 6)
    inputs = dict(zip(graph.forward_inputs, pack(args), strict=True))
    a, all_values = evaluate_graph(graph, graph.forward_outputs, inputs)
    b, retained = evaluate_graph(
        graph, graph.forward_outputs, inputs, retain_intermediates=False
    )
    assert len(retained) < len(all_values)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def test_core_retains_only_inputs_and_selected_residuals():
    graph = make_graph(17, 2, 2, 4, 8, 6, 4, "qwen_prepared")
    inverse = [i for i in graph.candidates if graph.nodes[i].name.endswith(".inverse")]
    plan = plan_from_saved(graph, inverse)
    args, _ = sample_inputs(2, 17, 2, 4, 8, 6)
    output, _ = Core.apply(graph, plan, None, *pack(args))
    retained = output.grad_fn.saved_tensors
    assert len(retained) == 6 + len(inverse)
    assert (
        sum(x.numel() * x.element_size() for x in retained[6:]) == plan.residual_bytes
    )
