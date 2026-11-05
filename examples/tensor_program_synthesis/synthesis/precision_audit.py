"""Separate formula accuracy from FP32 recurrence and reassociation errors."""

import json
from pathlib import Path

import torch

from .graph import evaluate_graph
from .parallel import scan_gated_delta_rule
from .problems import chunk_gated_delta_rule
from .validation import assert_results, oracle, sample

ROOT = Path(__file__).resolve().parents[1]


def metrics(actual, reference):
    error = actual.double() - reference
    norm = reference.norm().item()
    rms = reference.square().mean().sqrt().item()
    return {
        "max_abs": error.abs().max().item(),
        "relative_l2": error.norm().item() / max(norm, 1e-30),
        "max_abs_over_reference_rms": error.abs().max().item() / max(rms, 1e-30),
        "reference_rms": rms,
    }


def main():
    torch.set_num_threads(1)
    results = []
    destination = ROOT / "results" / "training_scale" / "precision_audit.json"
    names = ["O", "S_final", "dQ", "dK", "dV", "dg", "dbeta", "dS0"]
    for shape in (
        (512, 64, 64, 64),
        (2048, 64, 64, 64),
        (8192, 64, 64, 64),
        (2051, 64, 128, 64),
    ):
        graphs = [
            (name, factory(*shape))
            for name, factory in (
                ("sequential", chunk_gated_delta_rule),
                ("affine_scan", scan_gated_delta_rule),
            )
        ]
        for case in ("random", "weak_decay", "zero_decay", "strong_decay"):
            base = graphs[0][1]
            args32, cot32 = sample(base, seed=43, case=case)
            f32, b32 = oracle(base, args32, cot32)
            reference32 = tuple(x.detach() for x in (*f32, *b32))
            args64 = [x.detach().double().requires_grad_() for x in args32]
            cot64 = [x.double() for x in cot32]
            f64, b64 = oracle(base, args64, cot64)
            reference64 = tuple(x.detach() for x in (*f64, *b64))
            for variant, graph in graphs:
                outcomes = {}
                for dtype, args, cot in (
                    ("float32", args32, cot32),
                    ("float64", args64, cot64),
                ):
                    inputs = dict(
                        zip(
                            graph.forward_inputs + graph.backward_inputs,
                            (x.detach() for x in (*args, *cot)),
                            strict=True,
                        )
                    )
                    with torch.no_grad():
                        f, _ = evaluate_graph(graph, graph.forward_outputs, inputs)
                        b, _ = evaluate_graph(graph, graph.backward_outputs, inputs)
                    actual = (*f, *b)
                    target_dtype = (
                        torch.float64 if dtype == "float64" else torch.float32
                    )
                    try:
                        assert_results(
                            actual,
                            tuple(x.to(target_dtype) for x in reference64),
                            dtype=target_dtype,
                        )
                        passed, failure = True, None
                    except AssertionError as error:
                        passed, failure = False, str(error)
                    outcomes[dtype] = {
                        "passed_original_tolerance_against_fp64_reference": passed,
                        "failure": failure,
                        "metrics": {
                            name: metrics(x, y)
                            for name, x, y in zip(
                                names, actual, reference64, strict=True
                            )
                        },
                    }
                row = {
                    "shape": shape,
                    "case": case,
                    "seed": 43,
                    "variant": variant,
                    "evaluation": "tensor formula evaluator; FP64 operands are exact casts of the same FP32 inputs",
                    "formula": outcomes,
                    "fp32_recurrence_vs_fp64": {
                        name: metrics(x, y)
                        for name, x, y in zip(
                            names, reference32, reference64, strict=True
                        )
                    },
                }
                results.append(row)
                print(
                    json.dumps(
                        {
                            "shape": shape,
                            "case": case,
                            "variant": variant,
                            "float64_passed": outcomes["float64"][
                                "passed_original_tolerance_against_fp64_reference"
                            ],
                            "fp32_worst_relative_l2": max(
                                m["relative_l2"]
                                for m in outcomes["float32"]["metrics"].values()
                            ),
                            "fp32_worst_max_abs": max(
                                m["max_abs"]
                                for m in outcomes["float32"]["metrics"].values()
                            ),
                        }
                    ),
                    flush=True,
                )
                destination.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
