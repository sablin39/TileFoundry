"""Generate, solve, lower, validate, and analyze residual plans from formulas."""

import argparse
import json
from pathlib import Path

import torch

from .lower import execute, lower, static_reports
from .planner import pareto, solve
from .problems import chunk_gated_delta_rule, mlp, relu_sum
from .validation import assert_results, oracle, sample

ROOT = Path(__file__).resolve().parents[1]


def run_problem(graph, budgets, directory, *, analyze_plans=True, timeout=15):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "joint_graph.json").write_text(
        json.dumps(graph.dump(), indent=2) + "\n"
    )
    plans = []
    for budget in budgets:
        plan = solve(graph, budget, timeout=timeout)
        plans.append(plan)
        print(
            f"{graph.name}: budget={budget}, saved={plan.residual_bytes}, work={plan.work}, {plan.status}",
            flush=True,
        )
    frontier = pareto(plans)
    results = []
    for index, plan in enumerate(frontier):
        stem = f"plan_{index}_bytes_{plan.residual_bytes}"
        generated_path = ROOT / "generated" / directory.name / f"{stem}.py"
        program = lower(graph, plan, generated_path)
        destination = directory / stem
        destination.mkdir(exist_ok=True)
        print(f"  validating {stem}", flush=True)
        checks = []
        cases = (
            ["random", "relu_zero"]
            if graph.name != "chunk_gated_delta_rule"
            else [
                "random",
                "zero_decay",
                "strong_decay",
                "zero_beta",
                "unit_beta",
                "zero_keys",
                "zero_initial",
                "state_only",
                "output_only",
            ]
        )
        for seed, case in [(0, case) for case in cases] + [
            (1, "random"),
            (2, "random"),
        ]:
            args, cotangents = sample(graph, seed, case)
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
                forward_errors = assert_results(actual_f, expected_f)
                backward_errors = assert_results(actual_b, expected_b)
            checks.append(
                {
                    "seed": seed,
                    "case": case,
                    "forward_max_abs_errors": forward_errors,
                    "backward_max_abs_errors": backward_errors,
                    "passed": True,
                }
            )
        row = plan.dump(graph)
        row.update(
            {
                "generated_hir_source": str(generated_path.relative_to(ROOT)),
                "numerical_checks": checks,
                "numerical_tolerance": {"rtol": 2e-5, "atol": 2e-5, "dtype": "float32"},
                "phase_inputs": {
                    "forward": program.forward.inputs,
                    "backward": program.backward.inputs,
                },
            }
        )
        if analyze_plans:
            print(f"  analyzing {stem}", flush=True)
            row["tilefoundry"] = static_reports(program, destination)
        (destination / "plan.json").write_text(json.dumps(row, indent=2) + "\n")
        results.append(row)
    summary = {
        "problem": graph.name,
        "problem_metadata": graph.metadata,
        "tilefoundry_commit": (ROOT / "TILEFOUNDRY_COMMIT").read_text().strip(),
        "node_count": len(graph.nodes),
        "residual_candidate_count": len(graph.candidates),
        "budgets_solved": [plan.dump(graph) for plan in plans],
        "pareto_plans_among_solved_budgets": results,
        "contract": {
            "objective": "elementary-operation proxy: total backward work plus extra forward work",
            "exp_compare_cast_weight": 1,
            "budget": "retained residual bytes only; excludes immutable external inputs, constants, and ordinary outputs",
            "transient_memory": "reported by TileFoundry separately; not a solver constraint",
            "physical_placement": "whole-tensor gmem candidates; no tile/fusion search",
            "graph_scope": "fixed-size pure tensor DAG; chunk loops unrolled at graph construction",
            "recomputation": "at most one evaluation per logical value within backward",
            "correctness": "numerical oracle checks, no formal proof",
            "target": "nvidia.h200_sxm static model, CPU HIR evaluation",
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "mlp", "mask", "gdr"), default="all")
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=3)
    parser.add_argument("--value-dim", type=int, default=2)
    parser.add_argument(
        "--budgets",
        help="Comma-separated residual budgets in bytes; applies to each selected problem",
    )
    parser.add_argument("--solver-timeout", type=float, default=15)
    parser.add_argument(
        "--run-name", help="A directory label for retaining a separate experiment run"
    )
    parser.add_argument("--skip-static-analysis", action="store_true")
    args = parser.parse_args()
    if args.run_name and (
        not args.run_name.replace("-", "").replace("_", "").isalnum()
    ):
        parser.error(
            "--run-name must contain only letters, digits, hyphens, and underscores"
        )
    torch.set_num_threads(1)
    problems = []
    if args.case in {"all", "mlp"}:
        problems.append((mlp(), [0, 256, 1024, 1280, 2048]))
    if args.case in {"all", "mask"}:
        problems.append((relu_sum(), [0, 256, 1024]))
    if args.case in {"all", "gdr"}:
        problems.append(
            (
                chunk_gated_delta_rule(
                    args.length, args.chunk_size, args.key_dim, args.value_dim
                ),
                [0, 128, 256, 512, 1024, 2048],
            )
        )
    all_results = []
    for graph, defaults in problems:
        budgets = (
            sorted(set(map(int, args.budgets.split(",")))) if args.budgets else defaults
        )
        label = f"{graph.name}_{args.run_name}" if args.run_name else graph.name
        all_results.append(
            run_problem(
                graph,
                budgets,
                ROOT / "results" / "synthesis" / label,
                analyze_plans=not args.skip_static_analysis,
                timeout=args.solver_timeout,
            )
        )
    root = ROOT / "results" / "synthesis"
    summary_name = f"run_summary_{args.run_name or args.case}.json"
    (root / summary_name).write_text(json.dumps(all_results, indent=2) + "\n")
    print(
        f"Validated {sum(len(x['pareto_plans_among_solved_budgets']) for x in all_results)} generated plans. Reports: {root}"
    )


if __name__ == "__main__":
    main()
