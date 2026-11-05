"""Verify grouped Qwen3.5 cores against pinned official code and a full block."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .lower import lower
from .multihead import (
    candidate,
    configuration,
    load_official,
    make_graph,
    official,
    recurrent64,
    sample_inputs,
)
from .planner import solve
from .training_scale import policies, compact_plan

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/qwen35"
NAMES = ("O", "S_final", "dQ_shared", "dK_shared", "dV", "dg", "dbeta", "dS0")


def compare(actual, expected, *, dtype):
    tolerance = (
        2e-9 if dtype == torch.float64 else (5e-3 if dtype == torch.bfloat16 else 2e-5)
    )
    rows = {}
    for name, x, y in zip(NAMES, actual, expected, strict=True):
        delta = x.double() - y.double()
        try:
            torch.testing.assert_close(x, y, rtol=tolerance, atol=tolerance)
            passed = True
        except AssertionError:
            passed = False
        rows[name] = {
            "passed": passed,
            "max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / y.double().norm().clamp_min(1e-30)).item(),
            "atol": tolerance,
            "rtol": tolerance,
        }
    return {"passed": all(x["passed"] for x in rows.values()), "tensors": rows}


def reference_check(spec):
    model, batch, length, dtype_name, case = spec
    config = configuration(model)
    hk, hv = config["linear_num_key_heads"], config["linear_num_value_heads"]
    kd, vd = config["linear_key_head_dim"], config["linear_value_head_dim"]
    dtype = getattr(torch, dtype_name)
    args, cot = sample_inputs(
        batch,
        length,
        hk,
        hv,
        kd,
        vd,
        dtype=dtype,
        case=case,
        initial_zero=dtype != torch.float64,
    )
    if dtype != torch.float64:
        cot = (
            cot[0],
            torch.zeros_like(cot[1]),
        )  # ordinary full-sequence training boundary
    started = time.perf_counter()
    ref_fn = (
        recurrent64
        if dtype == torch.float64
        else lambda *x: official(load_official()["torch_chunk_gated_delta_rule"], *x)
    )
    ref = ref_fn(*args)
    grad = torch.autograd.grad(ref, args, grad_outputs=cot)
    expected = tuple(x.detach() for x in (*ref, *grad))
    del ref, grad
    row = {
        "model": model,
        "batch": batch,
        "length": length,
        "key_heads": hk,
        "value_heads": hv,
        "key_dim": kd,
        "value_dim": vd,
        "dtype": dtype_name,
        "case": case,
        "reference": "independent FP64 normalized token recurrence"
        if dtype == torch.float64
        else "pinned official Qwen3.5 chunk fallback with autograd",
        "initial_state": "nonzero" if dtype == torch.float64 else "zero",
        "final_state_cotangent": "nonzero" if dtype == torch.float64 else "zero",
        "reference_seconds_not_benchmark": time.perf_counter() - started,
        "variants": {},
    }
    print(f"Reference ready: {spec}", flush=True)
    for variant in ("qwen_prepared", "scan"):
        graph = make_graph(length, batch, hk, hv, kd, vd, variant=variant)
        all_policies = policies(graph)
        row["variants"][variant] = {
            "policies": {k: compact_plan(graph, p) for k, p in all_policies.items()},
            "checks": [],
        }
        for label in ("recompute", "all_inverses"):
            plan = all_policies[label]
            # Validate actual batched HIR at full head count, including one long case.
            use_hir = (
                dtype == torch.float32
                and label == "all_inverses"
                and (
                    length == 129
                    or (
                        model.endswith("397B-A17B")
                        and length == 8192
                        and variant == "qwen_prepared"
                    )
                )
            )
            program = None
            if use_hir:
                stem = f"{model}_B{batch}_T{length}_{variant}_{case}_{label}"
                print(f"Lowering batched HIR: {stem}", flush=True)
                program = lower(graph, plan, DEST / "generated" / f"{stem}.py")
            out = candidate(graph, plan, *args, program=program)
            grads = torch.autograd.grad(out, args, grad_outputs=cot)
            check = compare((*out, *grads), expected, dtype=dtype)
            check.update(
                {
                    "policy": label,
                    "backend": "TileFoundry HIR" if use_hir else "tensor DAG",
                    "saved_bytes_fp32": plan.residual_bytes,
                }
            )
            row["variants"][variant]["checks"].append(check)
            print(
                json.dumps(
                    {
                        "shape": spec,
                        "variant": variant,
                        "policy": label,
                        "backend": check["backend"],
                        "passed": check["passed"],
                        "max_abs": max(m["max_abs"] for m in check["tensors"].values()),
                    }
                ),
                flush=True,
            )
            del out, grads, program
    return row


def full_block_check(model="Qwen3.5-397B-A17B"):
    """Same official block/weights, replacing only its recurrent core callback."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    import transformers

    torch.manual_seed(7)
    config = configuration(model)
    block = Qwen3_5GatedDeltaNet(SimpleNamespace(**config), layer_idx=0).float()
    # Synthetic weak-decay parameters keep the test sensitive across chunks.
    with torch.no_grad():
        block.A_log.fill_(-2)
        block.dt_bias.fill_(-3)
    batch, length = 1, 129
    hidden = torch.randn(batch, length, config["hidden_size"], requires_grad=True)
    cot = torch.randn_like(hidden)
    params = tuple(block.parameters())
    implementation = load_official()["torch_chunk_gated_delta_rule"]
    block.chunk_gated_delta_rule = implementation
    ref = block(hidden)
    ref_grad = torch.autograd.grad(ref, (hidden, *params), grad_outputs=cot)
    expected_out, expected_grad = ref.detach(), tuple(x.detach() for x in ref_grad)
    del ref, ref_grad
    graph = make_graph(
        length,
        batch,
        config["linear_num_key_heads"],
        config["linear_num_value_heads"],
        config["linear_key_head_dim"],
        config["linear_value_head_dim"],
        variant="qwen_prepared",
    )
    inv = policies(graph)["all_inverses"]
    calls = []

    def kernel(
        q, k, v, g, beta, initial_state=None, output_final_state=False, **kwargs
    ):
        calls.append(tuple(q.shape))
        group = config["linear_num_value_heads"] // config["linear_num_key_heads"]
        # The official block has explicitly repeated Q/K; fold that broadcast.
        q, k = q[:, :, ::group], k[:, :, ::group]
        if initial_state is None:
            initial_state = torch.zeros(
                batch,
                config["linear_num_value_heads"],
                config["linear_key_head_dim"],
                config["linear_value_head_dim"],
            )
        out, state = candidate(graph, inv, q, k, v, g, beta, initial_state)
        return out, state if output_final_state else None

    block.chunk_gated_delta_rule = kernel
    actual = block(hidden)
    gradients = torch.autograd.grad(actual, (hidden, *params), grad_outputs=cot)
    names = [
        "block_output",
        "hidden_gradient",
        *(n for n, _ in block.named_parameters()),
    ]
    rows = []
    for name, x, y in zip(
        names, (actual, *gradients), (expected_out, *expected_grad), strict=True
    ):
        try:
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)
            passed = True
        except AssertionError:
            passed = False
        delta = x.double() - y.double()
        rows.append(
            {
                "name": name,
                "passed": passed,
                "max_abs": delta.abs().max().item(),
                "relative_l2": (
                    delta.norm() / y.double().norm().clamp_min(1e-30)
                ).item(),
            }
        )
    result = {
        "model_shape": model,
        "batch": batch,
        "length": length,
        "block_implementation_version": transformers.__version__,
        "weights": "random synthetic block weights; weak decay set for sensitivity; no checkpoint weights loaded",
        "official_reference_core": "pinned current official fallback",
        "candidate_kernel_calls": calls,
        "checks": rows,
        "passed": all(x["passed"] for x in rows),
    }
    print("Full block", json.dumps(result), flush=True)
    return result


def search_groups():
    results = []
    for group in (2, 4):
        graph = make_graph(512, 1, 1, group, variant="qwen_prepared")
        candidates = policies(graph)
        rows = []
        for name in ("all_inverses", "inverses_and_states"):
            hint = candidates[name]
            plan = solve(graph, hint.residual_bytes, timeout=15, hint_plan=hint)
            rows.append(compact_plan(graph, plan))
        results.append(
            {
                "length": 512,
                "key_heads": 1,
                "value_heads": group,
                "scope": "uniform grouped-tensor policy; not an optimum over all layers or individual heads",
                "search": rows,
            }
        )
        print(
            "Group search",
            group,
            [(r["solver_status"], r["residual_bytes"]) for r in rows],
            flush=True,
        )
    return results


def memory_projection():
    rows = []
    for model in ("Qwen3.5-35B-A3B", "Qwen3.5-27B", "Qwen3.5-397B-A17B"):
        config = configuration(model)
        graph = make_graph(
            8192,
            1,
            config["linear_num_key_heads"],
            config["linear_num_value_heads"],
            variant="qwen_prepared",
        )
        candidates = policies(graph)
        layers = config["layer_types"].count("linear_attention")
        row = {
            "model": model,
            "length": 8192,
            "batch": 1,
            "gdn_layers": layers,
            "key_heads": config["linear_num_key_heads"],
            "value_heads": config["linear_num_value_heads"],
            "state_bytes_current_only": config["linear_num_value_heads"]
            * 128
            * 128
            * 4,
            "assumption": "FP32 residuals, no layer activation checkpointing, all GDN-layer caches coexist; excludes inputs, parameters, ordinary outputs and transient workspace",
            "policies": {},
        }
        for name in (
            "all_inverses",
            "all_states",
            "inverses_and_states",
            "inverses_states_U",
        ):
            size = candidates[name].residual_bytes
            row["policies"][name] = {
                "bytes_per_layer": size,
                "bytes_all_gdn_layers_without_checkpointing": size * layers,
            }
        rows.append(row)
    (DEST / "memory_projection.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed experiment rows after interruption",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    DEST.mkdir(exist_ok=True, parents=True)
    (DEST / "complete.txt").unlink(missing_ok=True)
    summary_path = DEST / "summary.json"
    summary = (
        json.loads(summary_path.read_text())
        if args.resume and summary_path.exists()
        else {"runs": []}
    )
    specs = [
        ("Qwen3.5-35B-A3B", 2, 129, "float32", "weak_decay"),
        ("Qwen3.5-397B-A17B", 2, 129, "float32", "weak_decay"),
        ("Qwen3.5-397B-A17B", 1, 129, "float64", "weak_decay"),
        ("Qwen3.5-35B-A3B", 1, 8192, "float32", "weak_decay"),
        ("Qwen3.5-397B-A17B", 1, 8192, "float32", "weak_decay"),
        ("Qwen3.5-397B-A17B", 2, 2048, "bfloat16", "gated"),
        ("Qwen3.5-27B", 1, 2051, "float32", "weak_decay"),
    ]
    keys = ("model", "batch", "length", "dtype", "case")
    for spec in specs:
        if any(tuple(r[k] for k in keys) == spec for r in summary["runs"]):
            continue
        summary["runs"].append(reference_check(spec))
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if "full_block" not in summary:
        summary["full_block"] = full_block_check()
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if "group_search" not in summary:
        summary["group_search"] = search_groups()
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    checks = [
        c for r in summary["runs"] for v in r["variants"].values() for c in v["checks"]
    ]
    print(
        json.dumps(
            {
                "core_checks": len(checks),
                "passed": sum(c["passed"] for c in checks),
                "full_block_passed": summary["full_block"]["passed"],
            }
        ),
        flush=True,
    )
    memory_projection()
    if not all(c["passed"] for c in checks) or not summary["full_block"]["passed"]:
        raise SystemExit("Qwen3.5 validation failed; see the recorded tensor checks")
    (DEST / "complete.txt").write_text("complete\n")


if __name__ == "__main__":
    main()
