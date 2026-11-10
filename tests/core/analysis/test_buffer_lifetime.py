from dataclasses import replace

import pytest

from tilefoundry.analysis import BufferAccess, BufferLifetime, check_buffer_lifetimes


def test_deferred_reader_prevents_alias_reuse_until_its_completion():
    hidden = BufferLifetime(
        "hidden",
        "gpu0.scratch",
        0,
        512,
        BufferAccess("forward", 0, 1),
        (BufferAccess("weight_gradient", 8, 10),),
    )
    grad = BufferLifetime("gradient", "gpu0.scratch", 256, 1024, BufferAccess("derivative", 4, 5))
    (hazard,) = check_buffer_lifetimes((hidden, grad))
    assert hazard.kind == "overwrite_live_value"
    assert hazard.operations == ("weight_gradient", "derivative")
    assert not check_buffer_lifetimes((hidden, replace(grad, offset=512)))
    assert not check_buffer_lifetimes((hidden, replace(grad, allocation="gpu1.scratch")))
    assert not check_buffer_lifetimes(
        (hidden, replace(grad, producer=BufferAccess("reuse", 10, 11)))
    )


def test_readiness_and_remote_completion_are_different_constraints():
    packet = BufferLifetime(
        "packet0",
        "source.slot",
        0,
        64,
        BufferAccess("produce", 0, 3),
        (BufferAccess("peer_read", 2, 6),),
    )
    reuse = BufferLifetime("packet1", "source.slot", 0, 64, BufferAccess("produce_next", 4, 5))
    assert [h.kind for h in check_buffer_lifetimes((packet, reuse))] == [
        "read_before_ready",
        "overwrite_live_value",
    ]
    packet = replace(packet, consumers=(BufferAccess("peer_read", 3, 6),))
    assert len(check_buffer_lifetimes((packet, reuse))) == 1
    assert not check_buffer_lifetimes((packet, replace(reuse, producer=BufferAccess("next", 6, 7))))


def test_multiple_readers_extend_liveness_and_input_order_does_not_matter():
    value = BufferLifetime(
        "input",
        "slot",
        0,
        16,
        BufferAccess("available", 0, 0),
        (BufferAccess("input_gradient", 1, 2), BufferAccess("weight_gradient", 5, 7)),
    )
    reuse = BufferLifetime("next", "slot", 0, 16, BufferAccess("next_forward", 3, 4))
    assert value.end == 7
    assert check_buffer_lifetimes((value, reuse)) == check_buffer_lifetimes((reuse, value))
    assert check_buffer_lifetimes((value, reuse))[0].operations == (
        "weight_gradient",
        "next_forward",
    )


def test_malformed_trace_is_not_reported_as_safe():
    value = BufferLifetime("input", "slot", 0, 16, BufferAccess("available", 0, 0))
    for bad in (
        replace(value, nbytes=0),
        replace(value, offset=-1),
        replace(value, producer=BufferAccess("bad", 0, float("nan"))),
    ):
        with pytest.raises(ValueError):
            check_buffer_lifetimes((bad,))
    with pytest.raises(ValueError, match="uniquely"):
        check_buffer_lifetimes((value, value))
