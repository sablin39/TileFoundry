"""Check the static-analysis integration and compare authored residual plans.

Cross-invocation residual bytes are accounted for by this harness. TileFoundry
analyzes each invocation separately. CPU evaluation checks the supplied formulas;
it does not generate those formulas or run a GPU kernel.
"""

import json
from dataclasses import replace
from pathlib import Path

import torch

from models import BufferReuse, Lifetimes, ResidualMLP
from tilefoundry.analysis import ComputeCostMetadata, MemoryMetadata, RegionMemoryMetadata
from tilefoundry.analysis import analyze, check_program
from tilefoundry.analysis.allocation import AllocationSummaryMetadata
from tilefoundry.analysis.liveness import LivenessMetadata
from tilefoundry.analysis.memory import MemoryOptions
from tilefoundry.evaluator import evaluate
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core.expr import SSAValue
from tilefoundry.ir.core.kinds import UnaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.visitor import ExprCloner, collect_exprs


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FAMILIES = ("compute-cost", "memory", "roofline", "performance")


def function(module, name):
    return next(fn for fn in module.functions if fn.name == name)


def analyze_one(module, name, label=None):
    """Write both a full JSON report and annotated HIR for one invocation."""
    result = analyze(
        *check_program(module, function(module, name)),
        analysis=FAMILIES,
        options=MemoryOptions(timeout_seconds=10, workers=1, random_seed=0),
    )
    rendered = render_analysis(result)
    prefix = RESULTS / (label or name)
    prefix.with_suffix(".json").write_text(json.dumps(rendered.data, indent=2) + "\n")
    prefix.with_suffix(".hir.txt").write_text(rendered.annotated + "\n")
    memory = get_metadata(result.function, RegionMemoryMetadata)
    if memory.errors:
        raise AssertionError(f"{name}: {memory.errors}")
    assert get_metadata(result.function, LivenessMetadata) is not None
    assert get_metadata(result.function, AllocationSummaryMetadata) is not None
    return result


def summary(result):
    """Keep traffic, operation counts, peaks, and solver status distinct."""
    memory = get_metadata(result.function, RegionMemoryMetadata)
    compute = get_metadata(result.function, ComputeCostMetadata)
    return {
        "flops": sum(spread.total for _, spread in compute.flops.kinds),
        "flops_by_kind": {kind: spread.total for kind, spread in compute.flops.kinds},
        "traffic_bytes": {
            kind: {"read": spread.total.read, "write": spread.total.write}
            for kind, spread in memory.traffic.storage.kinds
        },
        "peak_bytes": {peak.memory_level: peak.peak_bytes for peak in memory.peaks},
        "solver_status": memory.solver_status,
        "advisories": list(memory.advisories),
    }


class EliminateDoubleNegation(ExprCloner):
    """A small concrete HIR rewrite, valid for the finite inputs used here."""

    def visit_Call(self, call, ctx):
        rewritten = self.default_visit(call, ctx)
        if isinstance(rewritten.target, Unary) and rewritten.target.kind is UnaryKind.NEG:
            inner = rewritten.args[0]
            if (
                isinstance(inner, Call)
                and isinstance(inner.target, Unary)
                and inner.target.kind is UnaryKind.NEG
            ):
                return inner.args[0]
        return rewritten


def lifetime_probes():
    """Exercise shared SSA uses, loop windows, and an analyzed HIR rewrite."""
    shared = analyze_one(Lifetimes, "shared")
    live = get_metadata(shared.function, LivenessMetadata)
    repeated = [
        life for life in live.intervals
        if isinstance(life.value.expr, Call)
        and len({use.at for use in live.uses if use.value == life.value and not use.synthetic}) > 1
    ]
    assert repeated, "Expected an intermediate with multiple distinct consumer events"
    for life in repeated:
        uses = [use.at for use in live.uses if use.value == life.value]
        assert life.last_used_at >= max(uses)

    carried = analyze_one(Lifetimes, "carried")
    loop_live = get_metadata(carried.function, LivenessMetadata)
    assert loop_live.loop_windows, "Loop-carried state must create a liveness window"

    original = function(Lifetimes, "shared")
    body = EliminateDoubleNegation().visit(original.body)
    assert body is not original.body
    rewritten = replace(original, name="shared_rewritten", body=body, metadata=())
    rewritten_module = replace(
        Lifetimes, name="RewrittenLifetimes", functions=(rewritten,), entry=rewritten.name
    )
    rewritten_result = analyze_one(rewritten_module, rewritten.name)
    before, after = summary(shared), summary(rewritten_result)
    assert after["flops"] < before["flops"]
    x = torch.linspace(-2.0, 2.0, 64).reshape(8, 8)
    torch.testing.assert_close(evaluate(original, x), 2 * x.square())
    torch.testing.assert_close(evaluate(rewritten, x), evaluate(original, x))
    torch.testing.assert_close(evaluate(function(Lifetimes, "carried"), x), 4 * x)
    return {
        "shared": before,
        "shared_rewritten": after,
        "carried": summary(carried),
        "shared_value_liveness_checked": True,
        "loop_windows": len(loop_live.loop_windows),
        "rewrite_equivalence_checked": True,
    }


def reuse_probes():
    """Verify actual buffer addresses distinguish dead and live arguments."""
    outputs = {}
    for name in ("reuse", "keep_live"):
        result = analyze_one(BufferReuse, name)
        call = next(
            expr for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, Function)
        )
        actual = get_metadata(call.args[0], MemoryMetadata)
        returned = get_metadata(call, MemoryMetadata)
        assert actual.offsets and returned.offsets
        reused = actual.offsets == returned.offsets
        assert reused == (name == "reuse")
        allocation = get_metadata(result.function, AllocationSummaryMetadata)
        assert (allocation.root(SSAValue(call)) is allocation.root(SSAValue(call.args[0]))) == reused
        outputs[name] = {
            **summary(result),
            "actual_offsets": list(actual.offsets),
            "result_offsets": list(returned.offsets),
            "reused": reused,
        }
    return outputs


def residual_probes():
    """Compare three phase contracts and check them against a CPU oracle."""
    candidates = {}
    for strategy in ("save_zh", "save_h", "recompute"):
        fwd = analyze_one(ResidualMLP, f"forward_{strategy}")
        bwd = analyze_one(ResidualMLP, f"backward_{strategy}")
        candidates[strategy] = {"forward": summary(fwd), "backward": summary(bwd)}

    checks = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        x = torch.randn(8, 16, requires_grad=True)
        w = torch.randn(16, 32, requires_grad=True)
        v = torch.randn(32, 8, requires_grad=True)
        dy = torch.randn(8, 8)
        if seed == 2:
            with torch.no_grad():
                w.zero_()
        reference = torch.relu(x @ w) @ v
        expected = torch.autograd.grad(reference, (x, w, v), grad_outputs=dy)
        with torch.no_grad():
            for strategy, candidate in candidates.items():
                result = evaluate(function(ResidualMLP, f"forward_{strategy}"), x, w, v)
                if strategy == "recompute":
                    y, residuals = result, ()
                else:
                    y, *residuals = result
                torch.testing.assert_close(y, reference, rtol=1e-5, atol=1e-5)
                gradients = evaluate(
                    function(ResidualMLP, f"backward_{strategy}"), x, w, v, dy, *residuals
                )
                for actual, wanted in zip(gradients, expected, strict=True):
                    torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-5)
                candidate["cross_phase_residual_bytes"] = sum(
                    t.numel() * t.element_size() for t in residuals
                )
                checks.append({"seed": seed, "strategy": strategy, "passed": True})

    assert candidates["save_zh"]["cross_phase_residual_bytes"] == 2048
    assert candidates["save_h"]["cross_phase_residual_bytes"] == 1024
    assert candidates["recompute"]["cross_phase_residual_bytes"] == 0
    baseline = candidates["save_h"]["backward"]["flops"]
    for candidate in candidates.values():
        candidate["extra_backward_flops_vs_save_h"] = candidate["backward"]["flops"] - baseline
    assert candidates["recompute"]["extra_backward_flops_vs_save_h"] > 0
    return candidates, checks


def main():
    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    lifetime = lifetime_probes()
    reuse = reuse_probes()
    candidates, checks = residual_probes()
    output = {
        "tilefoundry_commit": (ROOT / "TILEFOUNDRY_COMMIT").read_text().strip(),
        "model_target": "nvidia.h200_sxm",
        "execution": "CPU HIR interpreter; static GPU analysis; no GPU kernel execution",
        "contract": {
            "externally_available_at_backward": ["X", "W", "V", "dY"],
            "inputs_immutable_between_phases": True,
            "forward_output_Y_excluded_from_extra_residual_bytes": True,
            "relu_derivative_at_zero": 0,
            "dtype": "f32",
            "residual_accounting": "harness, outside per-invocation TileFoundry analysis",
            "candidate_generation": "three explicitly authored candidates, no automatic synthesis",
        },
        "lifetime_probes": lifetime,
        "buffer_reuse": reuse,
        "residual_candidates": candidates,
        "numerical_checks": checks,
    }
    (RESULTS / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print("All integration probes passed.")
    print("strategy      residual bytes    backward FLOPs    extra backward FLOPs")
    for strategy, candidate in candidates.items():
        print(
            f"{strategy:12}  {candidate['cross_phase_residual_bytes']:14d}"
            f"  {candidate['backward']['flops']:16d}"
            f"  {candidate['extra_backward_flops_vs_save_h']:20d}"
        )
    print(f"Reports: {RESULTS}")


if __name__ == "__main__":
    main()
