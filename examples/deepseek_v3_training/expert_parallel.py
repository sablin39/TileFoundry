"""Check assignment exchange, compact dispatch and peer addressing on the same top-eight routes."""

from collections import defaultdict
from functools import cache

import torch
from reference import BOX, D, Expert, F, M, R, Route, hop
from workload import TOP_K, expert_ids

from tilefoundry.analysis import check_program
from tilefoundry.evaluator import evaluate


@cache
def functions():
    return {f.name: check_program(mod, f)[1] for mod in (Expert, Route) for f in mod.functions}


def exchange(rows, pairs, *, back=False):
    """Move ragged records in bounded HIR mailboxes, retaining their route identities."""
    bins = defaultdict(list)
    for i, pair in enumerate(pairs):
        bins[pair].append(i)
    out = torch.empty_like(rows)
    for start in range(0, max(map(len, bins.values())), R):
        slots = [
            ((*divmod(src, 2), *divmod(dst, 2), j), i)
            for (src, dst), idx in bins.items()
            for j, i in enumerate(idx[start : start + R])
        ]
        box = torch.zeros(BOX)
        for slot, i in slots:
            box[slot][: rows.shape[1]] = rows[i]
        if back:
            box = hop(functions()["nvlink_back"], box, (2, 3), (2, 1))
            box = hop(functions()["ib_back"], box, (2, 1), (0, 1))
        else:
            box = hop(functions()["ib_out"], box, (0, 1), (2, 1))
            box = hop(functions()["nvlink_out"], box, (2, 1), (2, 3))
        for slot, i in slots:
            out[i] = box[slot][: rows.shape[1]]
    return out


def expert_mlp(ids, wg, wu, wd, read_x, write_y):
    """Group assignments by expert. Tail rows are zero-padded for the fixed HIR tile."""

    def call(name, *args):
        return evaluate(functions()[name], *args)

    groups = defaultdict(list)
    for i, e in enumerate(ids):
        groups[e].append(i)
    saved = []
    for e, batch in groups.items():
        for offset in range(0, len(batch), M):
            idx = batch[offset : offset + M]
            x = torch.zeros(M, D)
            x[: len(idx)] = read_x(idx)
            g, u, h = call("w13", x, wg[e], wu[e])
            write_y(idx, call("w2", h, wd[e])[: len(idx)])
            saved.append((e, idx, x, g, u, h))

    def adjoint(read_dy, write_dx):
        dwg, dwu, dwd = (torch.zeros_like(z) for z in (wg, wu, wd))
        for e, idx, x, g, u, h in saved:
            dy = torch.zeros(M, D)
            dy[: len(idx)] = read_dy(idx)
            dh = call("dgrad_w2", dy, wd[e])
            dg, du = call("gate_backward", dh, g, u)
            write_dx(idx, call("dgrad_w13", dg, du, wg[e], wu[e])[: len(idx)])
            dwd[e] += call("wgrad_w2", h, dy)
            a, b = call("wgrad_w13", x, dg, du)
            dwg[e] += a
            dwu[e] += b
        return dwg, dwu, dwd

    return adjoint


def run(method, x, score, ids, wg, wu, wd, dy):
    # Assignment order is (source token, selected slot). Routing IDs stay fixed in backward.
    tokens = torch.arange(len(x)).repeat_interleave(TOP_K)
    experts = ids.flatten().tolist()
    destinations = [e // TOP_K for e in experts]
    scores = score.reshape(-1, 1)
    if method == "peer":
        # A source publishes each token once. Expert work addresses (rank, local row);
        # W2 writes the result directly into the original token's selected slot.
        published = [x[rank::4].clone() for rank in range(4)]
        addresses = [(t % 4, t // 4, slot) for t in range(len(x)) for slot in range(TOP_K)]
        output_slots = [x.new_empty(len(part), TOP_K, D) for part in published]
        grad_slots = [torch.empty_like(part) for part in output_slots]

        def read_x(indices):
            return torch.stack([published[r][row] for r, row, _ in (addresses[i] for i in indices)])

        def store(buffers, indices, values):
            for i, value in zip(indices, values, strict=True):
                r, row, slot = addresses[i]
                buffers[r][row, slot] = value

        adjoint = expert_mlp(
            experts, wg, wu, wd, read_x, lambda idx, values: store(output_slots, idx, values)
        )
        origin_out = torch.stack([output_slots[t % 4][t // 4] for t in range(len(x))])
        y = (origin_out * score[:, :, None]).sum(1)
        ds = (origin_out * dy[:, None, :]).sum(-1)
        weighted = [dy[r::4, None, :] * score[r::4, :, None] for r in range(4)]

        def read_dy(indices):
            return torch.stack(
                [weighted[r][row, slot] for r, row, slot in (addresses[i] for i in indices)]
            )

        dwg, dwu, dwd = adjoint(read_dy, lambda idx, values: store(grad_slots, idx, values))
        dx = torch.stack([grad_slots[t % 4][t // 4].sum(0) for t in range(len(x))])
        return y, dx, ds, dwg, dwu, dwd

    assignment_keys = list(zip(tokens.tolist(), destinations))
    if method == "compact":
        records = list(dict.fromkeys(assignment_keys))
        lookup = {key: i for i, key in enumerate(records)}
        packet = torch.tensor([lookup[key] for key in assignment_keys])
    else:
        records = assignment_keys
        packet = torch.arange(len(records))
    src = torch.tensor([t for t, _ in records])
    peers = [(t % 4, rank) for t, rank in records]
    inputs = exchange(x[src], peers)[packet]
    out = torch.empty_like(inputs)
    adjoint = expert_mlp(experts, wg, wu, wd, inputs.__getitem__, out.__setitem__)
    if method == "compact":
        all_pairs = [(t % 4, rank) for t, rank in assignment_keys]
        owner_scores = exchange(scores, all_pairs)
        # Scores travel with dispatch. Weight and fold co-located experts before the return.
        combined = x.new_zeros(len(records), D).index_add_(0, packet, out * owner_scores)
        y = torch.zeros_like(x).index_add_(0, src, exchange(combined, peers, back=True))
        owner_dy = exchange(dy[src], peers)[packet]
        dscore = (owner_dy * out).sum(-1, keepdim=True)
        ds = exchange(dscore, all_pairs, back=True).reshape_as(score)
        dout = owner_dy * owner_scores
    else:
        at_source = exchange(out, peers, back=True)
        y = torch.zeros_like(x).index_add_(0, src, at_source * scores)
        ds = (dy[src] * at_source).sum(-1).reshape_as(score)
        dout = exchange(dy[src] * scores, peers)
    dx = torch.empty_like(dout)
    dwg, dwu, dwd = adjoint(dout.__getitem__, dx.__setitem__)
    if method == "compact":
        dx = x.new_zeros(len(records), D).index_add_(0, packet, dx)
    dx = torch.zeros_like(x).index_add_(0, src, exchange(dx, peers, back=True))
    return y, dx, ds, dwg, dwu, dwd


def dense(x, score, ids, wg, wu, wd):
    gate = torch.einsum("td,tkdf->tkf", x, wg[ids])
    up = torch.einsum("td,tkdf->tkf", x, wu[ids])
    out = torch.einsum("tkf,tkfd->tkd", torch.nn.functional.silu(gate) * up, wd[ids])
    return (out * score[:, :, None]).sum(1)


def check():
    torch.manual_seed(47)

    def rand(*shape):
        return (torch.randn(shape) * 0.3).requires_grad_()

    x, score = rand(7, D), rand(7, TOP_K).softmax(-1)
    wg, wu, wd = rand(32, D, F), rand(32, D, F), rand(32, F, D)
    dy = torch.randn_like(x)
    for case in ("colocated", "spread", "empty"):
        ids = torch.tensor([expert_ids(t, case) for t in range(len(x))])
        y = dense(x, score, ids, wg, wu, wd)
        expected = (y, *torch.autograd.grad(y, (x, score, wg, wu, wd), dy))
        with torch.no_grad():
            pairs = [(t, e // TOP_K) for t, row in enumerate(ids.tolist()) for e in row]
            for method in ("naive", "compact", "peer"):
                result = run(method, x, score, ids, wg, wu, wd, dy)
                for got, want in zip(result, expected, strict=True):
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
                err = max((a - b).abs().max().item() for a, b in zip(result, expected, strict=True))
                print(
                    f"{case:9s} {method:8s}  {len(pairs)} assignments  {len(set(pairs))} token/owner pairs  max |error| {err:.2g}"
                )


if __name__ == "__main__":
    torch.set_num_threads(1)
    check()
