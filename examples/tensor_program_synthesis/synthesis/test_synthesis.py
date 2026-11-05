"""Correctness, search optimality, generated interfaces, and recurrence checks."""

from itertools import combinations, product

import pytest
import torch

from .graph import Recipe, evaluate_graph
from .lower import execute, lower
from .planner import solve
from .problems import chunk_gated_delta_rule, mlp, relu_sum
from .validation import assert_results, oracle, sample


def enumerated_optimum(graph, budget):
    """Independent exhaustive evaluator for a small graph's residual subsets."""
    candidates = graph.candidates
    original = graph.ancestors(graph.forward_outputs)
    alternatives = [n.id for n in graph.nodes if len(n.recipes) > 1]
    best = None
    for size in range(len(candidates) + 1):
        for saved_tuple in combinations(candidates, size):
            saved = set(saved_tuple)
            memory = sum(graph.nodes[i].nbytes for i in saved)
            if memory > budget:
                continue
            extra = graph.ancestors(saved) - original - set(graph.forward_inputs)
            forward_work = sum(graph.work(i) for i in extra)
            for indices in product(
                *(range(len(graph.nodes[i].recipes)) for i in alternatives)
            ):
                choices = dict(zip(alternatives, indices))
                visited = set()

                def cost(i):
                    if i in visited or i in saved or i in graph.external:
                        return 0
                    visited.add(i)
                    recipe = graph.nodes[i].recipes[choices.get(i, 0)]
                    return graph.work(i, recipe) + sum(cost(j) for j in recipe.args)

                backward_work = sum(cost(i) for i in graph.backward_outputs)
                value = (forward_work + backward_work, memory)
                best = min(best, value) if best is not None else value
    return best


@pytest.mark.parametrize("budget", [0, 32, 128, 256, 512, 1024, 1280, 2048, 4096])
@pytest.mark.parametrize("factory", [mlp, relu_sum])
def test_solver_matches_exhaustive_enumeration(factory, budget):
    graph = factory()
    plan = solve(graph, budget)
    assert plan.status == "OPTIMAL"
    assert (plan.work, plan.residual_bytes) == enumerated_optimum(graph, budget)


def test_algebraic_recipe_changes_selected_residual():
    graph = mlp()
    with_rule = solve(graph, 1024)
    assert [graph.nodes[i].name for i in with_rule.saved] == ["H"]
    assert any(
        graph.nodes[i].recipes[r].rule == "relu_positive_mask"
        for i, r in with_rule.recipes.items()
    )
    for node in graph.nodes:
        node.recipes[:] = node.recipes[:1]
    without_rule = solve(graph, 1024)
    assert [graph.nodes[i].name for i in without_rule.saved] == ["Z"]
    assert with_rule.work < without_rule.work


def test_synthesizes_boolean_residual_in_forward(tmp_path):
    graph = relu_sum()
    plan = solve(graph, 256)
    assert plan.residual_bytes == 256
    assert len(plan.saved) == 1
    mask = graph.nodes[plan.saved[0]]
    assert mask.dtype == "bool" and mask.recipe.op == "gt"
    assert mask.id not in graph.ancestors(graph.forward_outputs)
    program = lower(graph, plan, tmp_path / "compact_mask.py")
    args, cotangents = sample(graph, seed=41)
    expected_f, expected_b = oracle(graph, args, cotangents)
    inputs = dict(
        zip(
            graph.forward_inputs + graph.backward_inputs,
            (*args, *cotangents),
            strict=True,
        )
    )
    with torch.no_grad():
        actual_f, actual_b = execute(program, inputs)
    assert_results((*actual_f, *actual_b), (*expected_f, *expected_b))


def test_shared_recomputation_is_charged_once():
    graph = mlp()
    plan = solve(graph, 0)
    assert plan.saved == ()
    assert (plan.work, 0) == enumerated_optimum(graph, 0)
    z = next(n for n in graph.nodes if n.name == "Z")
    assert z.id in plan.recipes


def test_rejects_invalid_budget_and_mutable_inputs():
    graph = mlp()
    with pytest.raises(ValueError, match="nonnegative"):
        solve(graph, -1)
    graph.metadata["immutable_inputs"] = False
    with pytest.raises(ValueError, match="immutable"):
        solve(graph, 0)


def test_rejects_recipe_cycles():
    graph = mlp()
    z = next(n for n in graph.nodes if n.name == "Z")
    h = next(n for n in graph.nodes if n.name == "H")
    z.recipes.append(Recipe("neg", (h.id,)))
    with pytest.raises(ValueError, match="acyclic"):
        graph.validate()


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 2, 3),
        (5, 2, 3, 2),
        (5, 3, 3, 2),
        (10, 4, 3, 2),
        (8, 8, 4, 3),
        (17, 8, 8, 6),
    ],
)
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
def test_chunk_formulas_and_all_gradients_match_recurrence(shape, case):
    graph = chunk_gated_delta_rule(*shape)
    args, cotangents = sample(graph, seed=7, case=case, dtype=torch.float64)
    expected_f, expected_b = oracle(graph, args, cotangents)
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
        (*actual_f, *actual_b), (*expected_f, *expected_b), dtype=torch.float64
    )


def test_chunk_vjp_directional_finite_difference():
    graph = chunk_gated_delta_rule(7, 3, 3, 4)
    args, cotangents = sample(graph, seed=13, dtype=torch.float64)
    generator = torch.Generator().manual_seed(19)
    directions = [
        torch.randn(x.shape, dtype=x.dtype, generator=generator) for x in args
    ]
    inputs = dict(
        zip(
            graph.forward_inputs + graph.backward_inputs,
            (*args, *cotangents),
            strict=True,
        )
    )
    gradients, _ = evaluate_graph(graph, graph.backward_outputs, inputs)
    analytical = sum(
        (gradient * tangent).sum()
        for gradient, tangent in zip(gradients, directions, strict=True)
    )
    epsilon = 1e-6

    def objective(sign):
        perturbed = [
            x + sign * epsilon * tangent
            for x, tangent in zip(args, directions, strict=True)
        ]
        values, _ = evaluate_graph(
            graph,
            graph.forward_outputs,
            dict(zip(graph.forward_inputs, perturbed, strict=True)),
        )
        return sum((y * dy).sum() for y, dy in zip(values, cotangents, strict=True))

    numerical = (objective(1) - objective(-1)) / (2 * epsilon)
    torch.testing.assert_close(analytical, numerical, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("budget", [0, 128, 512, 2048])
def test_selected_chunk_plan_reconstructs_only_declared_residuals(tmp_path, budget):
    graph = chunk_gated_delta_rule()
    plan = solve(graph, budget)
    assert plan.status == "OPTIMAL" and plan.residual_bytes <= budget
    program = lower(graph, plan, tmp_path / f"gdr_{budget}.py")
    args, cotangents = sample(graph, seed=23)
    expected_f, expected_b = oracle(graph, args, cotangents)
    inputs = dict(
        zip(
            graph.forward_inputs + graph.backward_inputs,
            (*args, *cotangents),
            strict=True,
        )
    )
    with torch.no_grad():
        actual_f, actual_b = execute(program, inputs)
    assert_results((*actual_f, *actual_b), (*expected_f, *expected_b))
    allowed = graph.external | set(plan.saved)
    assert set(program.backward.inputs) <= allowed
    if budget == 512:
        assert any(graph.nodes[i].name.startswith("S_after_chunk_") for i in plan.saved)


def test_large_chunk_generated_hir_matches_recurrence(tmp_path):
    graph = chunk_gated_delta_rule(65, 64, 16, 12)
    plan = solve(graph, 0)
    program = lower(graph, plan, tmp_path / "chunk64_tail.py")
    for case in ("random", "strong_decay", "zero_decay"):
        args, cotangents = sample(graph, seed=29, case=case)
        expected_f, expected_b = oracle(graph, args, cotangents)
        inputs = dict(
            zip(
                graph.forward_inputs + graph.backward_inputs,
                (*args, *cotangents),
                strict=True,
            )
        )
        with torch.no_grad():
            actual_f, actual_b = execute(program, inputs)
        assert_results((*actual_f, *actual_b), (*expected_f, *expected_b))
