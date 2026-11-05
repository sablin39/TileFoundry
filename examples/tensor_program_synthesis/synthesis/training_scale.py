"""Larger multi-chunk residual policies, bounded search, and dependency spans."""

import argparse
import json
import time
from pathlib import Path

import torch

from .lower import execute, lower
from .parallel import plan_dependencies, scan_gated_delta_rule
from .planner import plan_from_saved, solve
from .problems import chunk_gated_delta_rule
from .validation import assert_results, oracle, sample

ROOT = Path(__file__).resolve().parents[1]


def policies(graph):
    candidates = set(graph.candidates)
    names = {n.id: n.name for n in graph.nodes if n.id in candidates}
    inverse = {i for i, name in names.items() if name.endswith(".inverse")}
    states = {
        i
        for i, name in names.items()
        if name.startswith(("S_after_chunk_", "S_before_chunk_"))
    }
    basis = {
        i for i, name in names.items() if name.endswith((".W", ".U_base", ".A", ".Z"))
    }
    u = {i for i, name in names.items() if name.endswith(".U")}
    result = {
        "recompute": plan_from_saved(graph, []),
        "all_inverses": plan_from_saved(graph, inverse),
        "all_states": plan_from_saved(graph, states),
        "inverses_and_states": plan_from_saved(graph, inverse | states),
        "inverses_states_U": plan_from_saved(graph, inverse | states | u),
    }
    if basis:
        result["affine_basis"] = plan_from_saved(graph, basis)
        result["inverses_and_basis"] = plan_from_saved(graph, inverse | basis)
    return result


def compact_plan(graph, plan):
    row = plan.dump(graph)
    row.pop("backward_recipes")
    row["dependencies"] = plan_dependencies(graph, plan)
    row["total_training_work_units"] = (
        row["dependencies"]["forward"]["work_units"]
        + row["dependencies"]["backward"]["work_units"]
    )
    row["work_bound_gap_fraction"] = (
        1 - plan.lower_bound_work / plan.work
        if plan.status in {"OPTIMAL", "FEASIBLE"} and plan.work
        else None
    )
    return row


def run_shape(
    length,
    chunk_size,
    key_dim,
    value_dim,
    directory,
    timeout,
    *,
    search=True,
    resume=False,
):
    result = {
        "length": length,
        "chunk_size": chunk_size,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "chunks": (length + chunk_size - 1) // chunk_size,
        "replicas_executed": 1,
        "variants": {},
    }
    if resume and (directory / "summary.json").exists():
        result = json.loads((directory / "summary.json").read_text())
    reference_cache = {}
    for variant, factory in (
        ("sequential", chunk_gated_delta_rule),
        ("affine_scan", scan_gated_delta_rule),
    ):
        previous = result["variants"].get(variant)
        if (
            resume
            and previous
            and all(
                p.get("checks") or "validation_same_as" in p
                for p in previous["policies"].values()
            )
        ):
            print(f"Reuse completed {length} {variant}", flush=True)
            continue
        graph = factory(length, chunk_size, key_dim, value_dim)
        variant_path = directory / variant
        variant_path.mkdir(parents=True, exist_ok=True)
        candidates = policies(graph)
        row = {
            "node_count": len(graph.nodes),
            "candidate_count": len(graph.candidates),
            "metadata": graph.metadata,
            "policies": {},
            "search": [],
        }
        result["variants"][variant] = row
        for label, plan in candidates.items():
            row["policies"][label] = compact_plan(graph, plan)
        if search:
            for budget in (
                candidates["all_inverses"].residual_bytes,
                candidates["inverses_and_states"].residual_bytes,
            ):
                hint = min(
                    (p for p in candidates.values() if p.residual_bytes <= budget),
                    key=lambda p: (p.work, p.residual_bytes),
                )
                started = time.perf_counter()
                try:
                    plan = solve(graph, budget, timeout=timeout, hint_plan=hint)
                    label = f"searched_{budget}"
                    candidates[label] = plan
                    row["policies"][label] = compact_plan(graph, plan)
                    row["search"].append(
                        {
                            "label": label,
                            "budget": budget,
                            "elapsed_seconds": time.perf_counter() - started,
                            "hint_work": hint.work,
                            "status": plan.status,
                        }
                    )
                except RuntimeError as error:
                    row["search"].append(
                        {
                            "budget": budget,
                            "error": str(error),
                            "elapsed_seconds": time.perf_counter() - started,
                            "fallback_policy_work": hint.work,
                        }
                    )
                print(
                    json.dumps(
                        {
                            "length": length,
                            "variant": variant,
                            "search": row["search"][-1],
                        }
                    ),
                    flush=True,
                )
                (directory / "summary.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
        # Lower and execute all distinct materialization sets. Shared references
        # are evaluated once per case, independently of the selected policy.
        references = []
        for case in ("random", "weak_decay", "zero_decay", "strong_decay"):
            if case not in reference_cache:
                args, cotangents = sample(graph, seed=43, case=case)
                f, b = oracle(graph, args, cotangents)
                reference_cache[case] = (
                    tuple(x.detach() for x in (*args, *cotangents)),
                    tuple(x.detach() for x in (*f, *b)),
                )
            input_values, expected = reference_cache[case]
            inputs = dict(
                zip(
                    graph.forward_inputs + graph.backward_inputs,
                    input_values,
                    strict=True,
                )
            )
            references.append((case, inputs, expected))
        seen = {}
        for label, plan in candidates.items():
            item = row["policies"][label]
            signature = (plan.saved, tuple(sorted(plan.recipes.items())))
            if signature in seen:
                item["validation_same_as"] = seen[signature]
                continue
            seen[signature] = label
            destination = variant_path / label
            destination.mkdir(exist_ok=True)
            generated = destination / "program.py"
            (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"Lower/validate T={length} {variant} {label}", flush=True)
            program = lower(graph, plan, generated)
            (destination / "plan.json").write_text(
                json.dumps(plan.dump(graph), indent=2) + "\n"
            )
            item["generated_source"] = str(generated.relative_to(ROOT))
            item["checks"] = []
            for case, inputs, expected in references:
                started = time.perf_counter()
                with torch.no_grad():
                    f, b = execute(program, inputs)
                    actual = (*f, *b)
                    errors = [
                        (x - y).abs().max().item()
                        for x, y in zip(actual, expected, strict=True)
                    ]
                    try:
                        assert_results(actual, expected)
                        passed, failure = True, None
                    except AssertionError as error:
                        passed, failure = False, str(error)
                item["checks"].append(
                    {
                        "case": case,
                        "seed": 43,
                        "dtype": "float32",
                        "atol": 2e-5,
                        "rtol": 2e-5,
                        "passed": passed,
                        "max_abs_errors": errors,
                        "failure": failure,
                        "cpu_validation_seconds": time.perf_counter() - started,
                    }
                )
                if not passed:
                    print(
                        f"Numerical mismatch: T={length} {variant} {label} {case}: {failure}",
                        flush=True,
                    )
            del program
            (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "length": length,
                    "variant": variant,
                    "checks": sum(
                        len(x.get("checks", [])) for x in row["policies"].values()
                    ),
                    "failures": sum(
                        not c["passed"]
                        for x in row["policies"].values()
                        for c in x.get("checks", [])
                    ),
                }
            ),
            flush=True,
        )
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="512,2048,8192")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--solver-timeout", type=float, default=15)
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse fully validated variants"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    destination = ROOT / "results" / "training_scale"
    destination.mkdir(parents=True, exist_ok=True)
    runs = []
    for length in map(int, args.lengths.split(",")):
        directory = (
            destination
            / f"T{length}_C{args.chunk_size}_K{args.key_dim}_V{args.value_dim}"
        )
        directory.mkdir(exist_ok=True)
        runs.append(
            run_shape(
                length,
                args.chunk_size,
                args.key_dim,
                args.value_dim,
                directory,
                args.solver_timeout,
                search=not args.no_search,
                resume=args.resume,
            )
        )
        (destination / f"summary_K{args.key_dim}_V{args.value_dim}.json").write_text(
            json.dumps(runs, indent=2) + "\n"
        )
    checks = [
        c
        for r in runs
        for v in r["variants"].values()
        for p in v["policies"].values()
        for c in p.get("checks", [])
    ]
    print(
        json.dumps({"checks": len(checks), "passed": sum(c["passed"] for c in checks)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
