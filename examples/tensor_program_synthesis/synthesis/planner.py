"""Bounded exact residual/recipe selection with shared recomputation accounting."""

from dataclasses import dataclass
from math import floor

from ortools.sat.python import cp_model


@dataclass
class Plan:
    budget: int
    saved: tuple[int, ...]
    recipes: dict[int, int]
    residual_bytes: int
    backward_work: int
    extra_forward_work: int
    status: str
    lower_bound_work: int
    wall_seconds: float

    @property
    def work(self):
        return self.backward_work + self.extra_forward_work

    def dump(self, graph):
        original_forward = graph.ancestors(graph.forward_outputs)
        return {
            "residual_budget_bytes": self.budget,
            "residual_bytes": self.residual_bytes,
            "backward_work_units": self.backward_work,
            "extra_forward_work_units": self.extra_forward_work,
            "objective_work_units": self.work,
            "solver_status": self.status,
            "optimal_within_recipe_model": self.status == "OPTIMAL",
            "lower_bound_work_units": self.lower_bound_work,
            "solver_seconds": self.wall_seconds,
            "saved": [
                {
                    "id": i,
                    "name": graph.nodes[i].name,
                    "shape": graph.nodes[i].shape,
                    "bytes": graph.nodes[i].nbytes,
                    "synthesized_in_forward": i not in original_forward,
                }
                for i in self.saved
            ],
            "backward_recipes": {
                str(i): {
                    "name": graph.nodes[i].name,
                    "recipe": choice,
                    "rule": graph.nodes[i].recipes[choice].rule,
                }
                for i, choice in self.recipes.items()
            },
        }


def plan_from_saved(graph, saved, *, budget=None):
    """Construct a legal default-recipe policy, without an optimality claim."""
    saved = tuple(sorted(set(saved)))
    if not set(saved) <= set(graph.candidates):
        raise ValueError("Requested saved values are not eligible residuals")
    nbytes = sum(graph.nodes[i].nbytes for i in saved)
    budget = nbytes if budget is None else budget
    if budget < nbytes:
        raise ValueError("Policy exceeds its residual budget")
    external = graph.external
    order = graph.schedule(graph.backward_outputs, available=external | set(saved))
    original = graph.ancestors(graph.forward_outputs) | set(graph.forward_inputs)
    extra = graph.ancestors(saved) - original
    return Plan(
        budget,
        saved,
        {i: 0 for i in order},
        nbytes,
        sum(graph.work(i) for i in order),
        sum(graph.work(i) for i in extra),
        "FIXED_POLICY",
        0,
        0.0,
    )


def solve(graph, budget, *, timeout=15, fixed_saved=None, hint_plan=None):
    """Minimize elementary work, then residual bytes, under a residual budget.

    Each logical value has one backward source: external, saved, or one recipe.
    Shared recipes execute once. This does not constrain transient backward peak
    memory or optimize a multi-execution checkpoint schedule within backward.
    """
    if budget < 0:
        raise ValueError("Residual budget must be nonnegative")
    if not graph.metadata.get("immutable_inputs", True):
        raise ValueError("Recomputation requires immutable, versioned inputs")
    graph.validate()
    needed = graph.ancestors(graph.backward_outputs, alternatives=True)
    candidates = set(graph.candidates)
    external = graph.external
    if fixed_saved is not None and not set(fixed_saved) <= candidates:
        raise ValueError("Requested saved values are not eligible residuals")
    model = cp_model.CpModel()
    available = {i: model.new_bool_var(f"available_{i}") for i in needed}
    save = {i: model.new_bool_var(f"save_{i}") for i in candidates}
    choices = {}
    for i in sorted(needed):
        if i in external:
            model.add(available[i] == 1)
            continue
        choices[i] = [
            model.new_bool_var(f"recipe_{i}_{r}")
            for r in range(len(graph.nodes[i].recipes))
        ]
        model.add(sum(choices[i]) + save.get(i, 0) == available[i])
        for selected, recipe in zip(choices[i], graph.nodes[i].recipes, strict=True):
            for dependency in recipe.args:
                model.add(selected <= available[dependency])
    for i in graph.backward_outputs:
        model.add(available[i] == 1)
    if fixed_saved is not None:
        for i, variable in save.items():
            model.add(variable == int(i in fixed_saved))
    residual_bytes = sum(save[i] * graph.nodes[i].nbytes for i in candidates)
    model.add(residual_bytes <= budget)

    original = graph.ancestors(graph.forward_outputs) | set(graph.forward_inputs)
    extra_nodes = graph.forward_available - original
    extra = {
        i: model.new_bool_var(f"extra_forward_{i}")
        for i in extra_nodes
        if graph.nodes[i].recipe.op not in {"scalar", "constant", "input"}
    }
    for i, variable in extra.items():
        if i in save:
            model.add(variable >= save[i])
        for dependency in graph.nodes[i].recipe.args:
            if dependency in extra:
                model.add(variable <= extra[dependency])
    backward_work = sum(
        selected * graph.work(i, recipe)
        for i, variables in choices.items()
        for selected, recipe in zip(variables, graph.nodes[i].recipes, strict=True)
    )
    extra_work = sum(variable * graph.work(i) for i, variable in extra.items())
    lexicographic_factor = sum(graph.nodes[i].nbytes for i in candidates) + 1
    model.minimize((backward_work + extra_work) * lexicographic_factor + residual_bytes)
    if hint_plan is not None:
        if hint_plan.residual_bytes > budget:
            raise ValueError("Hint exceeds budget")
        saved_hint = set(hint_plan.saved)
        active_hint = set(hint_plan.recipes) | saved_hint | external
        for i, variable in available.items():
            model.add_hint(variable, int(i in active_hint))
        for i, variable in save.items():
            model.add_hint(variable, int(i in saved_hint))
        for i, variables in choices.items():
            for j, variable in enumerate(variables):
                model.add_hint(variable, int(hint_plan.recipes.get(i) == j))
        extra_hint = graph.ancestors(saved_hint) - original
        for i, variable in extra.items():
            model.add_hint(variable, int(i in extra_hint))
        model.add(
            (backward_work + extra_work) * lexicographic_factor + residual_bytes
            <= hint_plan.work * lexicographic_factor + hint_plan.residual_bytes
        )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Residual search: {solver.status_name(status)}")
    selected_saved = tuple(sorted(i for i in candidates if solver.value(save[i])))
    selected_recipes = {
        i: next(j for j, var in enumerate(variables) if solver.value(var))
        for i, variables in choices.items()
        if any(solver.value(var) for var in variables)
    }
    return Plan(
        budget=budget,
        saved=selected_saved,
        recipes=selected_recipes,
        residual_bytes=solver.value(residual_bytes),
        backward_work=solver.value(backward_work),
        extra_forward_work=solver.value(extra_work),
        status=solver.status_name(status),
        lower_bound_work=(
            solver.value(backward_work) + solver.value(extra_work)
            if status == cp_model.OPTIMAL
            else floor(solver.best_objective_bound / lexicographic_factor)
        ),
        wall_seconds=solver.wall_time,
    )


def pareto(plans):
    """Nondominated unique results among the budgets actually solved."""
    unique = {(p.residual_bytes, p.work): p for p in plans}
    return [
        p
        for key, p in sorted(unique.items())
        if not any(
            q.residual_bytes <= p.residual_bytes
            and q.work <= p.work
            and (q.residual_bytes < p.residual_bytes or q.work < p.work)
            for q in unique.values()
        )
    ]
