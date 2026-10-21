"""Windows every position writes alike are billed once at a shared level.

An insert_slice whose offsets never name a mesh coordinate, and a zeros of a
mesh-replicated buffer, perform the same write at every executing position.
At a level the target owns there is one store behind the positions, so the
window is billed once -- the per-unit share is the window divided among the
positions -- instead of each position paying for the whole of it. Offsets
that do name a mesh coordinate, and levels a topology unit owns, keep
per-position billing: those bytes physically are each position's own.
"""

from __future__ import annotations

from tilefoundry.analysis import MemoryMetadata, RegionMemoryMetadata, analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.zeros import Zeros
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget

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


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class RangedWindow:
    """A loop whose bounds name the mesh coordinate ranges over per-position windows."""

    @func
    def f(y: Tensor[(1, 8, 16), "bf16"]) -> Tensor[(4, 8, 16), "bf16"]:
        with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
            out = tf.zeros(Tensor[(4, 8, 16), "bf16"])
            for t in range(cta.tile % 2, 4, 2):
                out = tf.insert_slice(out, y, (t, 0, 0))
            return out


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class PrivateFill:
    """A level a topology unit owns is still filled by every position for itself."""

    @func
    def f() -> Tensor[(8, 16), "f32"]:
        with Mesh(("cta",), layout=(4,), names=("tile",)) as _cta:
            local = tf.zeros(Tensor[(8, 16), "f32", "smem"])
            return tf.reshard(local, (8, 16), "gmem")


def _analyzed(owner):
    return analyze(*check_program(owner, owner.entry_function()), analysis=("memory",))


def _op_spread(owner, op_type, level="gmem"):
    result = _analyzed(owner)
    matches = [
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, op_type)
    ]
    assert len(matches) == 1
    moved = get_metadata(matches[0], MemoryMetadata)
    assert moved is not None
    spread = moved.traffic.storage.of(level)
    assert spread is not None
    return moved, spread


def _function_spread(owner, level="gmem"):
    memory = get_metadata(_analyzed(owner).function, RegionMemoryMetadata)
    assert memory is not None
    spread = memory.traffic.storage.of(level)
    assert spread is not None
    return spread


def test_shared_window_is_billed_once_not_once_per_position() -> None:
    moved, spread = _op_spread(SharedWindow, InsertSlice)
    assert (spread.logical.read, spread.logical.write) == (_WINDOW, _WINDOW)
    assert (spread.total.read, spread.total.write) == (_WINDOW, _WINDOW)
    assert (spread.at(0).read, spread.at(0).write) == (
        _WINDOW // _POSITIONS,
        _WINDOW // _POSITIONS,
    )
    rmem = moved.traffic.storage.of("rmem")
    assert rmem is not None
    assert rmem.total.read == _POSITIONS * rmem.at(0).read


def test_shared_zeros_is_billed_once_not_once_per_position() -> None:
    _moved, spread = _op_spread(SharedWindow, Zeros)
    assert spread.logical.write == _BUFFER
    assert spread.total.write == _BUFFER
    assert spread.at(0).write == _BUFFER // _POSITIONS


def test_mesh_placed_window_keeps_per_position_billing() -> None:
    _moved, spread = _op_spread(PlacedWindow, InsertSlice)
    assert spread.at(0).write == _WINDOW
    assert spread.total.write == _POSITIONS * _WINDOW


def test_mesh_bounded_loop_keeps_per_position_billing() -> None:
    _moved, spread = _op_spread(RangedWindow, InsertSlice)
    assert spread.at(0).write == _WINDOW
    assert spread.total.write == _POSITIONS * _WINDOW


def test_private_level_fill_keeps_per_position_billing() -> None:
    _moved, spread = _op_spread(PrivateFill, Zeros, level="smem")
    assert spread.at(0).write == 8 * 16 * 4
    assert spread.total.write == _POSITIONS * 8 * 16 * 4


def test_function_total_matches_the_logical_sweep() -> None:
    spread = _function_spread(SharedWindow)
    assert (spread.total.read, spread.total.write) == (_BUFFER, 2 * _BUFFER)
    assert spread.total == spread.logical


def test_footprint_still_sees_the_whole_window_per_position() -> None:
    """Footprint keeps whole-window residency per position; only pricing changed."""
    memory = get_metadata(_analyzed(SharedWindow).function, RegionMemoryMetadata)
    assert memory is not None
    pairs = sorted(
        (held.of("gmem").logical, held.of("gmem").total)
        for _buffer, _ownership, held in memory.footprint.buffers
    )
    assert pairs == [(_WINDOW, _BUFFER), (_WINDOW, _BUFFER)]
