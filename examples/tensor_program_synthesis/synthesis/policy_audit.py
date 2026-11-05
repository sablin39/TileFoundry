"""Check exact FP32 invariance of residual choices within each supplied DAG."""

import json
from pathlib import Path

import torch

from .graph import evaluate_graph
from .parallel import scan_gated_delta_rule
from .problems import chunk_gated_delta_rule
from .validation import sample

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    directory = ROOT / "results" / "training_scale"
    results = []
    for path in sorted(directory.glob("T*/summary.json")):
        summary = json.loads(path.read_text())
        shape = tuple(
            summary[key] for key in ("length", "chunk_size", "key_dim", "value_dim")
        )
        for variant, rows in summary["variants"].items():
            factory = (
                chunk_gated_delta_rule
                if variant == "sequential"
                else scan_gated_delta_rule
            )
            graph = factory(*shape)
            for case in ("random", "weak_decay", "zero_decay", "strong_decay"):
                args, cotangents = sample(graph, seed=43, case=case)
                inputs = dict(
                    zip(
                        graph.forward_inputs + graph.backward_inputs,
                        (x.detach() for x in (*args, *cotangents)),
                        strict=True,
                    )
                )
                with torch.no_grad():
                    baseline_f, _ = evaluate_graph(graph, graph.forward_outputs, inputs)
                    baseline_b, _ = evaluate_graph(
                        graph, graph.backward_outputs, inputs
                    )
                    for label, row in rows["policies"].items():
                        if "validation_same_as" in row or not row.get("checks"):
                            continue
                        plan = json.loads(
                            (path.parent / variant / label / "plan.json").read_text()
                        )
                        saved = [v["id"] for v in plan["saved"]]
                        recipes = {
                            int(k): v["recipe"]
                            for k, v in plan["backward_recipes"].items()
                        }
                        # Only declared residuals and ordinary outputs cross the boundary.
                        roots = [*graph.forward_outputs, *saved]
                        f_values, _ = evaluate_graph(graph, roots, inputs)
                        boundary = dict(zip(roots, f_values, strict=True))
                        actual_b, _ = evaluate_graph(
                            graph,
                            graph.backward_outputs,
                            inputs,
                            recipes=recipes,
                            saved=boundary,
                        )
                        actual = (*f_values[: len(baseline_f)], *actual_b)
                        expected = (*baseline_f, *baseline_b)
                        equal = all(
                            torch.equal(x, y)
                            for x, y in zip(actual, expected, strict=True)
                        )
                        results.append(
                            {
                                "shape": shape,
                                "variant": variant,
                                "policy": label,
                                "case": case,
                                "bitwise_equal_to_no_residual_formula": equal,
                                "max_abs_difference": max(
                                    (x - y).abs().max().item()
                                    for x, y in zip(actual, expected, strict=True)
                                ),
                                "scope": "tensor DAG evaluator; separate from HIR-vs-recurrence validation",
                            }
                        )
                print(f"Policy invariance {shape} {variant} {case}", flush=True)
                (directory / "policy_audit.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
    print(
        json.dumps(
            {
                "checks": len(results),
                "bitwise_equal": sum(
                    r["bitwise_equal_to_no_residual_formula"] for r in results
                ),
            }
        )
    )
    if not all(r["bitwise_equal_to_no_residual_formula"] for r in results):
        raise SystemExit("Residual policy changed formula results")


if __name__ == "__main__":
    main()
