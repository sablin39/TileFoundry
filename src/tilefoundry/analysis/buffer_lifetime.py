"""Check physical storage reuse on an explicitly supplied execution timeline."""

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class BufferAccess:
    """An operation's half-open access interval in a caller-selected time unit."""

    operation: str
    start: float
    end: float


@dataclass(frozen=True)
class BufferLifetime:
    """One produced value and all its consumers in a physical byte range.

    Allocation names identify disjoint storage, including its owning device.
    Distinct writes use distinct value names, even when reusing an allocation.
    An initially available input can have a zero-duration producer.
    """

    value: str
    allocation: str
    offset: int
    nbytes: int
    producer: BufferAccess
    consumers: tuple[BufferAccess, ...] = ()

    @property
    def end(self) -> float:
        """Return the end of the last access, including the producer."""
        return max((self.producer.end, *(use.end for use in self.consumers)))


@dataclass(frozen=True)
class BufferHazard:
    """A readiness or overlapping-storage violation with its responsible operations."""

    kind: str
    allocation: str
    values: tuple[str, ...]
    operations: tuple[str, ...]


def check_buffer_lifetimes(lifetimes: tuple[BufferLifetime, ...]) -> tuple[BufferHazard, ...]:
    """Check declared producer readiness and live physical byte-range collisions.

    Consumer completion includes any asynchronous peer read or transfer. Times
    must share an ordered clock domain. This checks the supplied trace; it does
    not infer accesses from HIR or establish device memory visibility.
    A new producer may overwrite only after the old value's last consumer.
    """
    names = set()
    hazards = []
    for value in lifetimes:
        if not value.value or value.value in names or not value.allocation:
            raise ValueError("values must be uniquely named and identify an allocation")
        names.add(value.value)
        if (
            type(value.offset) is not int
            or type(value.nbytes) is not int
            or value.offset < 0
            or value.nbytes <= 0
        ):
            raise ValueError("buffer ranges require a nonnegative offset and positive byte count")
        for use in (value.producer, *value.consumers):
            if (
                not use.operation
                or not isinstance(use.start, Real)
                or not isinstance(use.end, Real)
                or not isfinite(use.start)
                or not isfinite(use.end)
                or not 0 <= use.start <= use.end
            ):
                raise ValueError("access intervals must be named, finite and nonnegative")
        for use in value.consumers:
            if use.start < value.producer.end:
                hazards.append(
                    BufferHazard(
                        "read_before_ready",
                        value.allocation,
                        (value.value,),
                        (value.producer.operation, use.operation),
                    )
                )
    ordered = sorted(lifetimes, key=lambda value: (value.producer.start, value.value))
    for old, new in combinations(ordered, 2):
        if old.allocation != new.allocation:
            continue
        byte_overlap = max(old.offset, new.offset) < min(
            old.offset + old.nbytes, new.offset + new.nbytes
        )
        if byte_overlap and new.producer.start < old.end:
            blockers = tuple(
                use.operation
                for use in (old.producer, *old.consumers)
                if use.end > new.producer.start
            )
            hazards.append(
                BufferHazard(
                    "overwrite_live_value",
                    old.allocation,
                    (old.value, new.value),
                    (*blockers, new.producer.operation),
                )
            )
    return tuple(hazards)
