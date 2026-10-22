"""A program is priced, and its wave stated, by the mesh that runs it.

The identity's ``wave`` is the most positions the root's meshes run against the
units the machine holds. The roofline and the timeline read their rates for that
many units: a share of the HBM peak, but by Little's law no more than each CTA's
bytes in flight carry over the global-memory latency.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tilefoundry.analysis import (
    PerformanceMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
    analyze,
)
from tilefoundry.analysis.check import check_program
from tilefoundry.dsl import *
from tilefoundry.inspection.analysis_report import render_analysis, render_text
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 132),)
_ROW = 8192
_TMA_GEMM = (
    Path(__file__).parents[1] / "fixtures" / "schedule" / "hir" / "gemm_8192x17408x5120_tma_store.py"
)


def _per_cta(inflight: int) -> int:
    return inflight * 1_000_000_000 // _TARGET.device.gmem_latency_ns


_SYNC_CTA = _per_cta(_TARGET.architecture.max_threads_per_cta * 16)


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class UnderFilled:
    """Forty of the 132 declared lanes run the whole program."""

    @func
    def f(x: Tensor[(40, _ROW), "bf16"]):
        with Mesh(("cta",), layout=(40,), names=("lane",)) as cta:
            held = tf.reshard(x, (40 @ cta.lane, _ROW), "rmem")
            return tf.reshard(held, (40, _ROW), "gmem")


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class Meshless:
    """No mesh at all: one position runs on a 132-lane level."""

    @func
    def f(x: Tensor[(_ROW,), "bf16"]):
        return tf.reshard(x, (_ROW,), "gmem")


def _analyzed(owner):
    return analyze(*check_program(owner, owner.entry_function()), analysis=("roofline", "performance"))


def test_forty_lanes_state_their_wave_and_share_the_hbm_peak_among_forty() -> None:
    result = _analyzed(UnderFilled)
    rendering = render_analysis(result)
    assert rendering.data["wave"] == {"active": 40, "physical": 132}
    assert "wave=40/132" in render_text(rendering)
    load = next(expr for expr in collect_exprs(result.function.body) if isinstance(expr, Call))
    timeline = get_metadata(load, PerformanceMetadata).timeline
    rate = min(_SYNC_CTA, _TARGET.device.hbm_bandwidth_bytes_per_second // 40)
    assert timeline.end_ns - timeline.start_ns == -(-(_ROW * 2 * 1_000_000_000) // rate)


def test_one_position_is_bound_by_its_bytes_in_flight() -> None:
    result = _analyzed(Meshless)
    assert render_analysis(result).data["wave"] == {"active": 1, "physical": 132}
    footprint = get_metadata(result.function, RegionMemoryMetadata).footprint.bytes_at("gmem")[1]
    roofline = get_metadata(result.function, RooflineMetadata)
    assert roofline.memory_ns == -(-(footprint * 1_000_000_000) // _SYNC_CTA)


def test_an_asynchronous_copy_keeps_every_stage_in_flight(tmp_path) -> None:
    """One CTA's TMA load of a 16 KiB tile runs faster the more stages its ring holds."""
    durations = []
    for stages in (3, 4):
        source = tmp_path / f"gemm_{stages}.py"
        source.write_text(
            _TMA_GEMM.read_text().replace("STAGES = 3", f"STAGES = {stages}", 1),
            encoding="utf-8",
        )
        spec = importlib.util.spec_from_file_location(source.stem, source)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        owner = loaded.GEMM_8192X17408X5120_TMA_STORE
        result = analyze(*check_program(owner, owner.entry_function()), analysis=("performance",))
        load = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, ScheduleOp)
        )
        timeline = get_metadata(load, PerformanceMetadata).timeline
        durations.append(timeline.end_ns - timeline.start_ns)
        assert durations[-1] == -(-(16384 * 1_000_000_000) // _per_cta(stages * 16384))
    assert durations[1] < durations[0]
