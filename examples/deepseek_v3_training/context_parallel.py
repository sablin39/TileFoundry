"""Compare causal MLA training layouts on four logical CP ranks."""

import math
from functools import cache

import torch
from reference import LEVEL, LogicalRanks

from tilefoundry.analysis import check_program
from tilefoundry.core import ShardTensor
from tilefoundry.dsl import (
    Broadcast,
    Layout,
    Mesh,
    Partial,
    ShardLayout,
    Split,
    Tensor,
    func,
    module,
    tf,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.types import Placement
from tilefoundry.target import CpuTarget

P, S, H = 4, 12, 8
C, K, R, V = 3, 2, 2, 3  # normalized latent, content key, shared position key, value
L, G = S // P, H // P
F32 = "f32"
MESH = Mesh((LEVEL,), Layout((P,), (1,)), ("cp",))
RANKS = tuple(Placement.each(MESH))


@cache
def transfer(shape, src, dst):
    """Split axes are explicit rank factors of size P in the logical tensor."""

    def layout(axis):
        if axis == "sum":
            return ShardLayout(Layout(shape), (Partial("sum"),), MESH)
        if axis == "replica":
            return ShardLayout(Layout(shape), (Broadcast(),), MESH)
        local = list(shape)
        local[axis] = 1
        strides = tuple(0 if i == axis else math.prod(local[i + 1 :]) for i in range(len(shape)))
        return ShardLayout(Layout(shape, strides), (Split(axis),), MESH)

    before, after = layout(src), layout(dst)

    @module(entry="move", target=LogicalRanks(), topologies=(LEVEL,))
    class Exchange:
        @func
        def move(x: Tensor[shape, F32, before]):
            return tf.local(tf.reshard(x, layout=after, storage="gmem"))

    return check_program(Exchange, Exchange.functions[0])[1]


def move(rows, shape, src, dst):
    fn = transfer(shape, src, dst)
    local = tuple(1 if i == src else n for i, n in enumerate(shape))
    shards = tuple((p, rows[p.ids["rank"]].reshape(local).contiguous()) for p in RANKS)
    result = evaluate(fn, ShardTensor(fn.params[0].type, shards))
    if dst == "replica" and isinstance(result, torch.Tensor):
        return [result] * P
    return [result.to_local(p) for p in RANKS]


@cache
def blocks(heads, nq, nk):
    Q, Kshape, Vshape = (heads, nq, K + R), (heads, nk, K + R), (heads, nk, V)
    Y, Stats, Mask = (heads, nq, V), (heads, nq, 1), (nq, nk)
    scale = (K + R) ** -0.5

    @module(entry="update", target=CpuTarget())
    class Attention:
        @func
        def update(
            q: Tensor[Q, F32],
            k: Tensor[Kshape, F32],
            v: Tensor[Vshape, F32],
            mask: Tensor[Mask, F32],
            row_max: Tensor[Stats, F32],
            normalizer: Tensor[Stats, F32],
            numerator: Tensor[Y, F32],
        ):
            scores = tf.matmul(q, k, b_layout="NK") * scale + mask
            hi = tf.max(row_max, tf.reduce(scores, axes=(-1,), kind="max", keepdim=True))
            corr = tf.exp(row_max - hi)
            weights = tf.exp(scores - hi)
            return (
                hi,
                corr * normalizer + tf.reduce(weights, axes=(-1,), kind="sum", keepdim=True),
                corr * numerator + tf.matmul(weights, v),
            )

        @func
        def backward(
            q: Tensor[Q, F32],
            k: Tensor[Kshape, F32],
            v: Tensor[Vshape, F32],
            mask: Tensor[Mask, F32],
            row_max: Tensor[Stats, F32],
            normalizer: Tensor[Stats, F32],
            y: Tensor[Y, F32],
            dy: Tensor[Y, F32],
        ):
            p = tf.exp(tf.matmul(q, k, b_layout="NK") * scale + mask - row_max) / normalizer
            delta = tf.reduce(dy * y, axes=(-1,), kind="sum", keepdim=True)
            ds = p * (tf.matmul(dy, v, b_layout="NK") - delta) * scale
            return (
                tf.matmul(ds, k),
                tf.matmul(ds, q, a_layout="KM"),
                tf.matmul(p, dy, a_layout="KM"),
            )

    return tuple(check_program(Attention, f)[1] for f in Attention.functions)


@cache
def projections(rows):
    @module(entry="expand", target=CpuTarget())
    class KVProjection:
        @func
        def expand(c: Tensor[(rows, C), F32], w: Tensor[(C, H * (K + V)), F32]):
            return tf.matmul(c, w)

        @func
        def adjoint(
            c: Tensor[(rows, C), F32],
            w: Tensor[(C, H * (K + V)), F32],
            dkv: Tensor[(rows, H * (K + V)), F32],
        ):
            return tf.matmul(dkv, w, b_layout="NK"), tf.matmul(c, dkv, a_layout="KM")

    return tuple(check_program(KVProjection, f)[1] for f in KVProjection.functions)


def project(c, w):
    return evaluate(projections(len(c))[0], c, w).reshape(len(c), H, K + V)


def project_back(c, w, dk, dv):
    dkv = torch.cat([dk, dv], -1).reshape(len(c), H * (K + V))
    return evaluate(projections(len(c))[1], c, w, dkv)


def keys(kv, pos):
    k, v = kv.split((K, V), -1)
    return torch.cat([k, pos[:, None, :].expand(-1, k.shape[1], -1)], -1), v


def mask(q_start, k_start, nq, nk):
    future = torch.arange(nk)[None, :] + k_start > torch.arange(nq)[:, None] + q_start
    return torch.zeros(nq, nk).masked_fill(future, -torch.inf)


def empty(q):
    n, h, _ = q.shape
    return torch.full((h, n, 1), -torch.inf), torch.zeros(h, n, 1), torch.zeros(h, n, V)


def update(q, k, v, bias, state):
    fn = blocks(q.shape[1], q.shape[0], k.shape[0])[0]
    return evaluate(fn, q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), bias, *state)


def backward(q, k, v, bias, state, dy):
    row_max, normalizer, numerator = state
    fn = blocks(q.shape[1], q.shape[0], k.shape[0])[1]
    out = evaluate(
        fn,
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        bias,
        row_max,
        normalizer,
        numerator / normalizer,
        dy.transpose(0, 1),
    )
    return tuple(x.transpose(0, 1) for x in out)


def ulysses(q, c, r, w):
    def transpose(rows, src, dst):
        d = rows[0].shape[-1]
        shape = (P, L, P, G, d)
        out = move(rows, shape, src, dst)
        return [x.reshape((S, G, d) if dst == 2 else (L, H, d)) for x in out]

    cs = c.chunk(P)
    packed = [torch.cat([a, project(b, w)], -1) for a, b in zip(q.chunk(P), cs)]
    heads = transpose(packed, 0, 2)
    pos = move(r.chunk(P), (P, L, R), 0, "replica")
    bias = mask(0, 0, S, S)
    ys, states = [], []
    for packed, shared in zip(heads, pos):
        local_q, kv = packed.split((K + R, K + V), -1)
        key, val = keys(kv, shared.reshape(S, R))
        state = update(local_q, key, val, bias, empty(local_q))
        ys.append((state[2] / state[1]).transpose(0, 1))
        states.append((local_q, key, val, state))
    y = torch.cat(transpose(ys, 2, 0))

    def adjoint(dy):
        grads, dr_parts = [], []
        for (local_q, key, val, state), local_dy in zip(states, transpose(dy.chunk(P), 0, 2)):
            dq, dk, dv = backward(local_q, key, val, bias, state, local_dy)
            grads.append(torch.cat([dq, dk[:, :, :K], dv], -1))
            dr_parts.append(dk[:, :, K:].sum(1))
        dr = torch.cat(move(dr_parts, (P, L, R), "sum", 0)).reshape(S, R)
        dxq, dcs, dw = [], [], torch.zeros_like(w)
        for latent, packed in zip(cs, transpose(grads, 2, 0)):
            dq, dk, dv = packed.split((K + R, K, V), -1)
            dc, local_dw = project_back(latent, w, dk, dv)
            dxq.append(dq)
            dcs.append(dc)
            dw += local_dw
        return torch.cat(dxq), torch.cat(dcs), dr, dw

    return y, adjoint


def compact_gather(q, c, r, w):
    """Matched-payload control with gathered compact KV retained through backward."""
    shape = (P, L, C + R)
    replicas = move(torch.cat([c, r], -1).chunk(P), shape, 0, "replica")
    outputs, saved = [], []
    for rank, (local_q, replica) in enumerate(zip(q.chunk(P), replicas)):
        latent, pos = replica.reshape(S, C + R).split((C, R), -1)
        key, val = keys(project(latent, w), pos)
        bias = mask(rank * L, 0, L, S)
        state = update(local_q, key, val, bias, empty(local_q))
        outputs.append((state[2] / state[1]).transpose(0, 1))
        saved.append((local_q, latent, key, val, bias, state))

    def adjoint(dy):
        dqs, parts, dw = [], [], torch.zeros_like(w)
        for (local_q, latent, key, val, bias, state), local_dy in zip(saved, dy.chunk(P)):
            dq, dk, dv = backward(local_q, key, val, bias, state, local_dy)
            dc, local_dw = project_back(latent, w, dk[:, :, :K], dv)
            parts.append(torch.cat((dc, dk[:, :, K:].sum(1)), -1))
            dqs.append(dq)
            dw += local_dw
        owned = torch.cat(move(parts, shape, "sum", 0)).reshape(S, C + R)
        dc, dr = owned.split((C, R), -1)
        return torch.cat(dqs), dc, dr, dw

    return torch.cat(outputs), adjoint


def shift(rows):
    """One directed peer hop per rank on the closed ring."""
    return [rows[-1], *rows[:-1]]


def ring(q, c, r, w):
    qs = q.chunk(P)
    original = list(torch.cat([c, r], -1).chunk(P))
    states = [empty(a) for a in qs]
    blocks_in_flight = original
    for step in range(P):
        for rank, packet in enumerate(blocks_in_flight):
            owner = (rank - step) % P
            if owner <= rank:
                latent, pos = packet.split((C, R), -1)
                k, v = keys(project(latent, w), pos)
                states[rank] = update(qs[rank], k, v, mask(rank * L, owner * L, L, L), states[rank])
        if step + 1 < P:
            blocks_in_flight = shift(blocks_in_flight)
    y = torch.cat([(numerator / normalizer).transpose(0, 1) for _, normalizer, numerator in states])

    def adjoint(dy):
        dys = dy.chunk(P)
        # Replay KV. Its gradient accumulator travels alongside it, then takes one final hop home.
        blocks_in_flight = original
        accum = [torch.zeros_like(t) for t in original]
        dqs, dw = [torch.zeros_like(a) for a in qs], torch.zeros_like(w)
        for step in range(P):
            for rank, packet in enumerate(blocks_in_flight):
                owner = (rank - step) % P
                if owner <= rank:
                    latent, pos = packet.split((C, R), -1)
                    k, v = keys(project(latent, w), pos)
                    dq, dk, dv = backward(
                        qs[rank], k, v, mask(rank * L, owner * L, L, L), states[rank], dys[rank]
                    )
                    dc, local_dw = project_back(latent, w, dk[:, :, :K], dv)
                    accum[rank] += torch.cat([dc, dk[:, :, K:].sum(1)], -1)
                    dqs[rank] += dq
                    dw += local_dw
            if step + 1 < P:
                blocks_in_flight = shift(blocks_in_flight)
            accum = shift(accum)
        dc, dr = torch.cat(accum).split((C, R), -1)
        return torch.cat(dqs), dc, dr, dw

    return y, adjoint


def dense(q, c, r, w):
    n, heads, _ = q.shape
    kv = (c @ w).reshape(n, heads, K + V)
    content, val = kv.split((K, V), -1)
    key = torch.cat((content, r[:, None].expand(-1, heads, -1)), -1)
    scores = torch.einsum("thd,shd->hts", q, key) / math.sqrt(K + R)
    causal = torch.ones(n, n, dtype=torch.bool).triu(1)
    return torch.einsum("hts,shv->thv", scores.masked_fill(causal, -torch.inf).softmax(-1), val)


def check():
    torch.manual_seed(671)
    q, c, r, w, dy = [
        torch.randn(shape) for shape in ((S, H, K + R), (S, C), (S, R), (C, H * (K + V)), (S, H, V))
    ]
    for t in (q, c, r, w):
        t.requires_grad_()
    y = dense(q, c, r, w)
    expected = (y, *torch.autograd.grad(y, (q, c, r, w), dy))
    with torch.no_grad():
        for method in (ulysses, ring, compact_gather):
            y, adjoint = method(q, c, r, w)
            result = (y, *adjoint(dy))
            for got, want in zip(result, expected, strict=True):
                torch.testing.assert_close(got, want, atol=4e-6, rtol=2e-5)
            errors = [
                (got - want).abs().max().item() for got, want in zip(result, expected, strict=True)
            ]
            print(
                f"{method.__name__:8s}  max |error|  "
                + "  ".join(
                    f"{k}={e:.2g}"
                    for k, e in zip(
                        (
                            "attention output",
                            "query gradient",
                            "latent gradient",
                            "positional-key gradient",
                            "KV weight gradient",
                        ),
                        errors,
                    )
                )
            )


if __name__ == "__main__":
    torch.set_num_threads(1)
    check()
