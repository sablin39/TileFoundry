"""Lay out an explicitly staged tile pipeline on serial resources."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class TileStage:
    """One operation per tile, with a resource and a duration in caller-selected units."""

    name: str
    resource: str
    duration: float


@dataclass(frozen=True)
class TileInterval:
    """The half-open execution interval of one stage on one tile."""

    stage: TileStage
    tile: int
    start: float
    end: float


def tile_pipeline(
    phases: tuple[tuple[TileStage, ...], ...],
    *,
    tiles: int,
    slots: int,
    policy: Literal["bulk", "tile_ready"],
) -> tuple[TileInterval, ...]:
    """Schedule in authored order, fencing the end of each phase.

    Bulk execution materializes each stage. Tile-ready execution issues each tile
    through the phase and reuses an edge's staging slot after its consumer finishes.
    A resource serves its issued operations in FIFO order across all stages.
    """
    if isinstance(tiles, bool) or not isinstance(tiles, int) or tiles < 0:
        raise ValueError("tiles must be a nonnegative integer")
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
        raise ValueError("slots must be a positive integer")
    if policy not in ("bulk", "tile_ready"):
        raise ValueError(f"unknown tile pipeline policy: {policy!r}")
    spans = []
    free = {}
    fence = 0.0
    for phase in phases:
        for stage in phase:
            if not stage.name or not stage.resource:
                raise ValueError("each tile stage must name its operation and resource")
            if not isfinite(stage.duration) or stage.duration <= 0:
                raise ValueError(f"{stage.name}: duration must be positive and finite")
        ends = [[fence] * tiles for _ in phase]
        if policy == "bulk":
            order = ((s, t) for s in range(len(phase)) for t in range(tiles))
        else:
            order = ((s, t) for t in range(tiles) for s in range(len(phase)))
        for s, t in order:
            stage = phase[s]
            start = max(fence, free.get(stage.resource, 0.0))
            if s:
                start = max(start, ends[s - 1][-1 if policy == "bulk" else t])
            if policy == "tile_ready" and s + 1 < len(phase) and t >= slots:
                start = max(start, ends[s + 1][t - slots])
            end = start + stage.duration
            spans.append(TileInterval(stage, t, start, end))
            ends[s][t] = free[stage.resource] = end
        fence = max(free.values(), default=fence)
    return tuple(spans)
