"""A program is priced, and its wave stated, by the mesh that runs it.

The identity's ``wave`` is the most positions the root's meshes run against the
units the machine holds. The roofline and the timeline read their rates for that
many units: a share of the HBM peak, but never more than the active SMs' L1 data
paths carry.
"""

from __future__ import annotations

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
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 132),)
_ROW = 8192
_L1_PATH = (
    _TARGET.architecture.shared_memory_bandwidth_bytes_per_clock_per_sm * _TARGET.device.sm_clock_hz
)


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
    rate = min(_L1_PATH, _TARGET.device.hbm_bandwidth_bytes_per_second // 40)
    assert timeline.end_ns - timeline.start_ns == -(-(_ROW * 2 * 1_000_000_000) // rate)


def test_one_position_is_bound_by_one_l1_path() -> None:
    result = _analyzed(Meshless)
    assert render_analysis(result).data["wave"] == {"active": 1, "physical": 132}
    footprint = get_metadata(result.function, RegionMemoryMetadata).footprint.bytes_at("gmem")[1]
    roofline = get_metadata(result.function, RooflineMetadata)
    assert roofline.memory_ns == -(-(footprint * 1_000_000_000) // _L1_PATH)
