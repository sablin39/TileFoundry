"""The report identity states how many of the declared lanes a program fills.

``wave`` is machine context: positions one wave holds against the units the
level declares. A program whose meshes activate fewer positions than the level
declares leaves the rest idle, and nothing in the wave figures says so. The
identity's ``lanes`` is the program-side reading: the most positions any of the
selected Function's meshes activates at the level, against the declared units.
It is stated only when the two differ -- a full fill is the silence case, and a
machine-truncated wave is the wave field's to state, not the lane fill's.
"""

from __future__ import annotations

from tilefoundry.analysis import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.analysis.report import report_data
from tilefoundry.dsl import *
from tilefoundry.inspection.analysis_report import render_analysis, render_text
from tilefoundry.target import CudaTarget

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 132),)


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class UnderFilled:
    """Forty of the 132 declared lanes run the whole program."""

    @func
    def f(x: Tensor[(40, 8), "bf16"]):
        with Mesh(("cta",), layout=(40,), names=("lane",)) as cta:
            held = tf.reshard(x, (40 @ cta.lane, 8), "rmem")
            return tf.reshard(held, (40, 8), "gmem")


@module(entry="f", target=_TARGET, topologies=(Topology("cta", 4),))
class FullFill:
    """Every declared lane runs, so there is no fill to state."""

    @func
    def f(x: Tensor[(4, 8), "bf16"]):
        with Mesh(("cta",), layout=(4,), names=("lane",)) as cta:
            held = tf.reshard(x, (4 @ cta.lane, 8), "rmem")
            return tf.reshard(held, (4, 8), "gmem")


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class Meshless:
    """No mesh at all: one position runs on a 132-lane level."""

    @func
    def f(x: Tensor[(8,), "bf16"]):
        return tf.reshard(x, (8,), "gmem")


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class SiblingFill:
    """The widest sibling region states the fill."""

    @func
    def f(x: Tensor[(96, 8), "bf16"]):
        with Mesh(("cta",), layout=(40,), names=("lane",)) as cta:
            _head = tf.reshard(x[0:40, :], (40 @ cta.lane, 8), "rmem")
        with Mesh(("cta",), layout=(96,), names=("lane",)) as cta:
            tail = tf.reshard(x, (96 @ cta.lane, 8), "rmem")
            return tf.reshard(tail, (96, 8), "gmem")


def _report(owner) -> dict:
    result = analyze(*check_program(owner, owner.entry_function()), analysis=("memory",))
    return report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        topology_level=result.topology_level,
        executed=result.executed,
        metadata_types=result.metadata_types,
    )


def _identity(owner) -> str:
    result = analyze(*check_program(owner, owner.entry_function()), analysis=("memory",))
    return render_text(render_analysis(result))


def test_under_fill_is_stated_against_the_declared_lanes() -> None:
    data = _report(UnderFilled)
    assert data["wave"] == {"counted": 132, "declared": 132, "occupied": 40}
    assert "wave=132/132 lanes=40/132" in _identity(UnderFilled)


def test_full_fill_states_no_lanes() -> None:
    data = _report(FullFill)
    assert data["wave"] == {"counted": 4, "declared": 4, "occupied": 4}
    identity = _identity(FullFill)
    assert "wave=4/4" in identity
    assert "lanes=" not in identity


def test_a_meshless_program_occupies_one_lane() -> None:
    data = _report(Meshless)
    assert data["wave"] == {"counted": 132, "declared": 132, "occupied": 1}
    assert "wave=132/132 lanes=1/132" in _identity(Meshless)


def test_the_widest_region_states_the_fill() -> None:
    data = _report(SiblingFill)
    assert data["wave"] == {"counted": 132, "declared": 132, "occupied": 96}
    assert "wave=132/132 lanes=96/132" in _identity(SiblingFill)
