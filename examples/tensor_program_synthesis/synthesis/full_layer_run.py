"""Incomplete full-layer GDN example: FP32 reference validation."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .full_layer import boundary_policy, full_layer_graph, load_official_layer
from .graph import evaluate_graph
from .lower import check_program, deep_ir_visitors, execute, function, lower
from .multihead import configuration
from .planner import plan_from_saved

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/full_gdn"


def compare(actual, expected, names):
    rows = []
    for name, x, y in zip(names, actual, expected, strict=True):
        try:
            torch.testing.assert_close(x, y, atol=3e-5, rtol=3e-5)
            strict_passed = True
        except AssertionError:
            strict_passed = False
        try:
            torch.testing.assert_close(x, y, atol=1e-4, rtol=3e-5)
            elementwise_passed = True
        except AssertionError:
            elementwise_passed = False
        delta = x.double() - y.double()
        relative_l2 = (delta.norm() / y.double().norm().clamp_min(1e-30)).item()
        rows.append(
            dict(
                name=name,
                passed=elementwise_passed and relative_l2 <= 2e-6,
                strict_3e_5_passed=strict_passed,
                max_abs=delta.abs().max().item(),
                relative_l2=relative_l2,
            )
        )
    return rows


def error_metrics(actual, reference):
    delta = actual.double() - reference.double()
    return dict(
        max_abs=delta.abs().max().item(),
        relative_l2=(delta.norm() / reference.double().norm().clamp_min(1e-30)).item(),
    )


def validate_case(model, batch, length, policies):
    config = configuration(model)
    torch.manual_seed(707)
    block = load_official_layer()(SimpleNamespace(**config), layer_idx=0).float()
    # Synthetic parameters; no model checkpoint weights are downloaded.
    with torch.no_grad():
        block.A_log.fill_(-2)
        block.dt_bias.fill_(-3)
        block.norm.weight.copy_(1 + 0.1 * torch.randn_like(block.norm.weight))
    hidden = torch.randn(batch, length, config["hidden_size"], requires_grad=True)
    mask = torch.ones(batch, length)
    mask[0, -2:] = 0
    cot = torch.randn_like(hidden)
    graph = full_layer_graph(config, length, batch)
    named = {
        "hidden_states": hidden,
        "attention_mask": mask.unsqueeze(-1),
        **dict(block.named_parameters()),
    }
    assert set(named) == {graph.nodes[i].name for i in graph.forward_inputs}
    target_names = graph.metadata["gradient_inputs"]
    print(f"Reference {model} B={batch} T={length}", flush=True)
    expected_output = block(hidden, attention_mask=mask)
    expected_grads = torch.autograd.grad(
        expected_output, [named[n] for n in target_names], cot
    )
    expected = tuple(t.detach() for t in (expected_output, *expected_grads))
    del expected_output, expected_grads
    inputs = {i: named[graph.nodes[i].name].detach() for i in graph.forward_inputs}
    inputs[graph.backward_inputs[0]] = cot
    case = dict(
        model=model,
        batch=batch,
        length=length,
        dtype="float32",
        acceptance={
            "atol": 1e-4,
            "rtol": 3e-5,
            "relative_l2_max": 2e-6,
            "strict_check_retained": "atol=rtol=3e-5",
            "rationale": "full-layer projection gradients reduce many token contributions; both FP32 implementations are audited against FP64 at the long shape",
        },
        config=graph.metadata,
        weights="synthetic random weights with weak decay",
        reference="Complete GDN module and fallback functions from pinned Transformers fd3cf0140de03f7619ac95b5cfed05b711ff9e22; dispatch decorators removed, bodies unchanged",
        backend="entire generated forward and backward in TileFoundry HIR CPU evaluator",
        policies=[],
    )
    audit64 = None
    if length >= 1024:
        print("FP64 audit of the supplied whole-layer formulas", flush=True)
        audit64, _ = evaluate_graph(
            graph,
            [*graph.forward_outputs, *graph.backward_outputs],
            {i: value.double() for i, value in inputs.items()},
            retain_intermediates=False,
        )
        case["precision_audit"] = {
            "reference": "same supplied DAG evaluated in FP64; an arithmetic audit, not an independent full-size derivation",
            "official_fp32_vs_dag_fp64": {
                name: error_metrics(actual, precise)
                for name, actual, precise in zip(
                    ["output", *target_names], expected, audit64, strict=True
                )
            },
        }
    graph_path = DEST / f"{model}_B{batch}_T{length}.graph.json"
    graph_path.write_text(json.dumps(graph.dump(), indent=2) + "\n")
    for policy in policies:
        saved = boundary_policy(graph) if policy == "selected_boundaries" else []
        plan = plan_from_saved(graph, saved)
        label = f"{model}_B{batch}_T{length}_{policy}"
        print(f"HIR {label}", flush=True)
        program = lower(graph, plan, DEST / "generated" / f"{label}.py")
        with deep_ir_visitors():
            for phase in ("forward", "backward"):
                check_program(program.module, function(program.module, phase))
        actual_output, actual_grads = execute(program, inputs)
        checks = compare(
            (*actual_output, *actual_grads), expected, ["output", *target_names]
        )
        entry = dict(
            name=policy,
            typechecked=True,
            passed=all(r["passed"] for r in checks),
            checks=checks,
            saved=plan.dump(graph)["saved"],
            residual_bytes=plan.residual_bytes,
            forward_interface=[graph.nodes[i].name for i in program.forward.outputs],
            backward_interface=[graph.nodes[i].name for i in program.backward.inputs],
            generated_source=str(program.path.relative_to(ROOT)),
        )
        if audit64 is not None:
            entry["hir_fp32_vs_dag_fp64"] = {
                name: error_metrics(actual, precise)
                for name, actual, precise in zip(
                    ["output", *target_names],
                    (*actual_output, *actual_grads),
                    audit64,
                    strict=True,
                )
            }
            entry["fp64_audit_passed"] = all(
                metrics["relative_l2"] <= 2e-6
                for measurements in (
                    entry["hir_fp32_vs_dag_fp64"],
                    case["precision_audit"]["official_fp32_vs_dag_fp64"],
                )
                for metrics in measurements.values()
            )
            entry["passed"] = entry["passed"] and entry["fp64_audit_passed"]
        case["policies"].append(entry)
        print(
            json.dumps(
                {
                    "case": label,
                    "passed": entry["passed"],
                    "failed": [r for r in checks if not r["passed"]],
                    "max_abs": max(r["max_abs"] for r in checks),
                }
            ),
            flush=True,
        )
        del actual_output, actual_grads, program
        gc.collect()
    case["passed"] = all(p["passed"] for p in case["policies"])
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "complete.txt").unlink(missing_ok=True)
    summary = dict(
        scope="complete fixed-shape full-sequence GDN training module; no optimization",
        candidate="explicit formulas in native TileFoundry HIR, no opaque PyTorch callbacks",
        reference_source=json.loads(
            (ROOT / "references/qwen35/sources.json").read_text()
        ),
        runs=[],
    )
    specs = [
        ("Qwen3.5-35B-A3B", 2, 129, ("recompute", "selected_boundaries")),
        ("Qwen3.5-397B-A17B", 1, 1025, ("recompute", "selected_boundaries")),
    ]
    for spec in specs:
        summary["runs"].append(validate_case(*spec))
        (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        gc.collect()
    summary["passed"] = all(r["passed"] for r in summary["runs"])
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise SystemExit("Whole-layer verification failed; see recorded tensor checks")
    (DEST / "complete.txt").write_text("complete\n")


if __name__ == "__main__":
    main()
