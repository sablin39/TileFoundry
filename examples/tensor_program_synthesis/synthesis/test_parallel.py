"""Long-graph traversal, affine composition, and independent VJP validation."""

import pytest
import torch

from .graph import Graph, evaluate_graph
from .lower import execute, lower
from .parallel import scan_gated_delta_rule, plan_dependencies
from .planner import plan_from_saved, solve
from .problems import chunk_gated_delta_rule, mlp
from .validation import assert_results, oracle, sample


@pytest.mark.parametrize("shape", [(1, 1, 2, 3), (17, 4, 3, 2), (32, 4, 4, 3)])
@pytest.mark.parametrize(
    "case",
    [
        "random",
        "zero_decay",
        "strong_decay",
        "zero_beta",
        "unit_beta",
        "zero_keys",
        "zero_initial",
        "state_only",
        "output_only",
    ],
)
def test_scan_all_gradients_against_independent_recurrence(shape, case):
    graph = scan_gated_delta_rule(*shape)
    args, cotangents = sample(graph, seed=31, case=case, dtype=torch.float64)
    expected = oracle(graph, args, cotangents)
    inputs = dict(
        zip(
            graph.forward_inputs + graph.backward_inputs,
            (*args, *cotangents),
            strict=True,
        )
    )
    with torch.no_grad():
        actual_f, _ = evaluate_graph(graph, graph.forward_outputs, inputs)
        actual_b, _ = evaluate_graph(graph, graph.backward_outputs, inputs)
    assert_results(
        (*actual_f, *actual_b), (*expected[0], *expected[1]), dtype=torch.float64
    )


def test_scan_generated_hir_has_reduced_dependency_depth(tmp_path):
    graph = scan_gated_delta_rule(512, 64, 16, 12)
    plan = plan_from_saved(graph, [])
    program = lower(graph, plan, tmp_path / "scan512.py")
    args, cotangents = sample(graph, seed=17)
    expected = oracle(graph, args, cotangents)
    inputs = dict(
        zip(
            graph.forward_inputs + graph.backward_inputs,
            (*args, *cotangents),
            strict=True,
        )
    )
    with torch.no_grad():
        actual = execute(program, inputs)
    assert_results((*actual[0], *actual[1]), (*expected[0], *expected[1]))
    sequential = chunk_gated_delta_rule(512, 64, 16, 12)
    seq_report = plan_dependencies(sequential, plan_from_saved(sequential, []))
    scan_report = plan_dependencies(graph, plan)
    assert (
        scan_report["forward"]["dependent_matmul_depth"]
        < seq_report["forward"]["dependent_matmul_depth"]
    )
    assert (
        scan_report["backward"]["dependent_matmul_depth"]
        < seq_report["backward"]["dependent_matmul_depth"]
    )


def test_fixed_policy_and_complete_solver_hint_agree():
    graph = mlp()
    h = next(n.id for n in graph.nodes if n.name == "H")
    hint = plan_from_saved(graph, [h], budget=1024)
    plan = solve(graph, 1024, hint_plan=hint)
    assert plan.status == "OPTIMAL"
    assert plan.work <= hint.work
    fixed = solve(graph, 1024, fixed_saved=[h])
    # MLP has alternatives: fixed default recipes can cost more than optimized ones.
    assert fixed.work <= hint.work


def test_iterative_demand_walk_handles_deep_graph():
    graph = Graph("deep")
    x = graph.input("X", (1, 1))
    y = x
    for _ in range(3000):
        y = y + 1
    graph.begin_backward([y])
    graph.finish([y * 2])
    actual, _ = evaluate_graph(graph, graph.backward_outputs, {x.id: torch.zeros(1, 1)})
    assert actual[0].item() == 6000
    assert len(graph.ancestors(graph.backward_outputs)) > 3000
