"""Compute-action projection of DeepSeek's eight-step DualPipe schedule.

Adapted from deepseek-ai/DualPipe at 030ce4325f4ebeb437da4ebc6d00a70469dd58ae,
dualpipe/dualpipe.py, DualPipe.step.
The original MIT license is reproduced in LICENSE.DualPipe.
Communication and module execution are modeled by the example's caller.
"""

from collections import deque


def actions(ranks=4, microbatches=8):
    """Return per-rank groups; a multi-action group is an upstream overlap opportunity."""
    if ranks < 2 or ranks % 2 or microbatches % 2 or microbatches < 2 * ranks:
        raise ValueError("DualPipe requires even ranks and an even microbatch count >= 2 * ranks")
    result = []
    for rank in range(ranks):
        half_rank = min(rank, ranks - 1 - rank)
        middle = rank in (ranks // 2 - 1, ranks // 2)
        counts = {kind: [0, 0] for kind in ("fwd", "bwd_input")}
        groups, deferred = [], deque()

        def make(kind, phase):
            direction = phase ^ int(rank >= ranks // 2)
            batch = counts[kind][direction] + direction * (microbatches // 2)
            counts[kind][direction] += 1
            stage = rank if direction == 0 else ranks - 1 - rank
            return kind, direction, batch, stage

        def forward(phase):
            groups.append([make("fwd", phase)])

        def backward(phase, defer=False):
            action = make("bwd_input", phase)
            weight = ("bwd_weight", *action[1:])
            groups.append([action])
            if defer:
                deferred.append(weight)
            else:
                groups[-1].append(weight)

        def pair(fwd_phase, bwd_phase):
            forward(fwd_phase)
            backward(bwd_phase)
            groups[-2].extend(groups.pop())

        def weight():
            groups.append([deferred.popleft()])

        for _ in range((ranks // 2 - half_rank - 1) * 2):
            forward(0)
        for _ in range(half_rank + 1):
            forward(0)
            forward(1)
        for _ in range(ranks // 2 - half_rank - 1):
            backward(1, True)
            weight()
            forward(1)
        for i in range(microbatches // 2 - ranks + half_rank + 1):
            if i == 0 and middle:
                forward(0)
                backward(1)
            else:
                pair(0, 1)
            pair(1, 0)
        for _ in range(ranks // 2 - half_rank - 1):
            backward(1)
            pair(1, 0)
        defer = False
        for i in range(half_rank + 1):
            if i == (half_rank + 1) // 2 and half_rank % 2 == 1:
                defer = True
            backward(1, defer)
            if i == (half_rank + 1) // 2 and half_rank % 2 == 0:
                defer = True
            backward(0, defer)
        for _ in range(ranks // 2 - half_rank - 1):
            weight()
            backward(0, True)
        for _ in range(half_rank + 1):
            weight()
        assert not deferred
        assert all(count == microbatches // 2 for values in counts.values() for count in values)
        result.append(groups)
    return result
