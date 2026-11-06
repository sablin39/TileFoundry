"""Compose each CP × EP pair and check the FSDP gradient shard of the MLA KV projection."""

from itertools import product

import context_parallel as cp
import expert_parallel as ep
import torch
from reference import D, F
from workload import TOP_K, expert_ids


def token_order():
    # EP's fixture numbers source ranks round-robin. Re-index local rows without moving a token.
    return torch.arange(cp.S).reshape(cp.P, cp.L).T.reshape(-1)


def check():
    torch.manual_seed(235)
    order = token_order()
    ids = torch.tensor([expert_ids(t, "colocated") for t in range(cp.S)])

    def rand(*shape):
        return (torch.randn(shape) * 0.4).requires_grad_()

    w = rand(cp.C, cp.H * (cp.K + cp.V))
    wo = rand(cp.H * cp.V, D)
    wg, wu, wd = rand(32, D, F), rand(32, D, F), rand(32, F, D)
    # Four independent toy sequences on four data ranks. The loss averages the four contributions.
    batches = [
        (
            rand(cp.S, cp.H, cp.K + cp.R),
            rand(cp.S, cp.C),
            rand(cp.S, cp.R),
            rand(cp.S, TOP_K).softmax(-1),
            torch.randn(cp.S, D) / cp.P,
        )
        for _ in range(cp.P)
    ]
    expected, loss = [], 0
    for q, c, r, score, seed in batches:
        attention = cp.dense(q, c, r, w)
        y = ep.dense(attention.flatten(1)[order] @ wo, score, ids, wg, wu, wd)
        loss = loss + (y * seed).sum()
        expected.append(y)
    leaves = [x for q, c, r, score, _ in batches for x in (q, c, r, score)]
    *local, global_dw = torch.autograd.grad(loss, (*leaves, w))
    weights_shape = (cp.C, cp.P, w.shape[1] // cp.P)
    with torch.no_grad():
        shards = w.chunk(cp.P, dim=1)
        replicas = cp.move(shards, weights_shape, 1, "replica")
        for method, route in product((cp.ulysses, cp.ring), ("naive", "compact", "peer")):
            contributions = []
            errors = []

            def compare(value, ref):
                torch.testing.assert_close(value, ref, rtol=2e-5, atol=4e-6)
                errors.append((value - ref).abs().max().item())

            for rank, (q, c, r, score, seed) in enumerate(batches):
                attention, adjoint = method(q, c, r, replicas[rank].reshape_as(w))
                x = attention.flatten(1)[order] @ wo
                y, dx, ds, *_ = ep.run(route, x, score, ids, wg, wu, wd, seed)
                da = (dx @ wo.T)[order.argsort()].reshape_as(attention)
                dq, dc, dr, dw = adjoint(da)
                for value, ref in zip(
                    (dq, dc, dr, ds), local[rank * 4 : rank * 4 + 4], strict=True
                ):
                    compare(value, ref)
                compare(y, expected[rank])
                contributions.append(dw)
            local_dw = cp.move(contributions, weights_shape, "sum", 1)
            for value, ref in zip(local_dw, global_dw.chunk(cp.P, dim=1), strict=True):
                compare(value.reshape_as(ref), ref)
            print(
                f"{method.__name__:8s} × {route:8s}  output, input/score gradients, FSDP dWkv shards  max |error| {max(errors):.2g}"
            )


if __name__ == "__main__":
    torch.set_num_threads(1)
    check()
