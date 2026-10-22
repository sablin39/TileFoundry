"""A window every position writes alike moves its bytes once.

Traffic counts what each position asks for: ``@total`` is one position's share
times the executing positions, the amount the L2 side sees. HBM sees the unique
bytes, so the timeline prices one position's share of the occurrence's
footprint: a window every position writes alike costs the window over the
positions, and a window of each position's own costs that window.
"""

from __future__ import annotations

from tilefoundry.analysis import MemoryMetadata, PerformanceMetadata, RegionMemoryMetadata, analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget, PerformanceServiceFacts

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 4),)
_POSITIONS = 4
_WINDOW = 1 * 8 * 16 * 2
_BUFFER = 4 * _WINDOW


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class SharedWindow:
    """A zeros and a loop-placed insert_slice every position performs alike."""

    @func
    def f(y: Tensor[(1, 8, 16), "bf16"]) -> Tensor[(4, 8, 16), "bf16"]:
        with Mesh(("cta",), layout=(4,), names=("tile",)) as _cta:
            out = tf.zeros(Tensor[(4, 8, 16), "bf16"])
            for t in tf.tile(4, 1):
                out = tf.insert_slice(out, y, (t, 0, 0))
            return out


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class PlacedWindow:
    """Offsets naming the mesh coordinate give each position a window of its own."""

    @func
    def f(y: Tensor[(1, 8, 16), "bf16"]) -> Tensor[(4, 8, 16), "bf16"]:
        with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
            out = tf.zeros(Tensor[(4, 8, 16), "bf16"])
            return tf.insert_slice(out, y, (cta.tile, 0, 0))


def _analyzed(owner):
    return analyze(*check_program(owner, owner.entry_function()), analysis=("memory", "performance"))


def _insert(owner) -> Call:
    (call,) = (
        expr
        for expr in collect_exprs(_analyzed(owner).function.body)
        if isinstance(expr, Call) and isinstance(expr.target, InsertSlice)
    )
    return call


def test_total_is_every_position_s_share() -> None:
    for owner in (SharedWindow, PlacedWindow):
        spread = get_metadata(_insert(owner), MemoryMetadata).traffic.storage.of("gmem")
        assert (spread.at(0).read, spread.at(0).write) == (_WINDOW, _WINDOW)
        assert (spread.total.read, spread.total.write) == (
            _POSITIONS * _WINDOW,
            _POSITIONS * _WINDOW,
        )


def test_the_timeline_prices_one_position_s_share_of_the_footprint() -> None:
    rate = _TARGET.get_facts(PerformanceServiceFacts, ("cta", _POSITIONS)).bandwidth("gmem")
    durations = {}
    for owner, window in ((SharedWindow, 2 * _WINDOW), (PlacedWindow, _WINDOW + _BUFFER)):
        call = _insert(owner)
        assert get_metadata(call, MemoryMetadata).footprint.bytes_at("gmem")[0] == window
        timeline = get_metadata(call, PerformanceMetadata).timeline
        durations[owner] = timeline.end_ns - timeline.start_ns
        assert durations[owner] == -(-(-(-window // _POSITIONS) * 1_000_000_000) // rate)
    assert durations[SharedWindow] < durations[PlacedWindow]


def test_footprint_still_sees_the_whole_window_per_position() -> None:
    """Footprint keeps whole-window residency per position."""
    memory = get_metadata(_analyzed(SharedWindow).function, RegionMemoryMetadata)
    assert memory is not None
    pairs = sorted(
        (held.of("gmem").logical, held.of("gmem").total)
        for _buffer, _ownership, held in memory.footprint.buffers
    )
    assert pairs == [(_WINDOW, _BUFFER), (_WINDOW, _BUFFER)]
