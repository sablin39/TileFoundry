"""Check a two-node, two-GPU-per-node route with the HIR evaluator and dense autograd."""

import math

import torch
from workload import PASSES, trace

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
    Topology,
    func,
    module,
    tf,
)
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.types import Placement
from tilefoundry.target import CpuTarget, TopologyFacts, TopologyLevelFacts

R, D, F, T = 2, 6, 4, 21
F32 = "f32"
M = 4 * R
BOX = (2, 2, 2, 2, R, D)  # source node / GPU, expert node / GPU, slot, channel
LEVEL = Topology("rank", 4)
MESH = Mesh((LEVEL,), Layout((2, 2), (2, 1)), ("node", "gpu"))
RANKS = tuple(Placement.each(MESH))


class LogicalRanks(CpuTarget):
    def get_facts(self, facts_type, query=None):
        level = TopologyLevelFacts("rank", 4, 4)
        if facts_type is TopologyFacts and query is None:
            return TopologyFacts((level,))
        if facts_type is TopologyLevelFacts and query == "rank":
            return level
        return super().get_facts(facts_type, query)


def placed(axes):
    local = [1 if i in axes else n for i, n in enumerate(BOX)]
    strides = tuple(0 if i in axes else math.prod(local[i + 1 :]) for i in range(len(BOX)))
    return ShardLayout(Layout(BOX, strides), tuple(Split(i) for i in axes), MESH)


ORIGIN, GATEWAY, OWNER = (placed(a) for a in ((0, 1), (2, 1), (2, 3)))
PART = ShardLayout(Layout((T, D)), (Partial("sum"), Partial("sum")), MESH)
FULL = ShardLayout(Layout((T, D)), (Broadcast(), Broadcast()), MESH)


@module(entry="ib_out", target=LogicalRanks(), topologies=(LEVEL,))
class Route:
    @func
    def ib_out(x: Tensor[BOX, F32, ORIGIN]):
        return tf.local(tf.reshard(x, layout=GATEWAY, storage="gmem"))

    @func
    def nvlink_out(x: Tensor[BOX, F32, GATEWAY]):
        return tf.local(tf.reshard(x, layout=OWNER, storage="gmem"))

    @func
    def nvlink_back(x: Tensor[BOX, F32, OWNER]):
        return tf.local(tf.reshard(x, layout=GATEWAY, storage="gmem"))

    @func
    def ib_back(x: Tensor[BOX, F32, GATEWAY]):
        return tf.local(tf.reshard(x, layout=ORIGIN, storage="gmem"))

    @func
    def resolve(x: Tensor[(T, D), F32, PART]):
        return tf.reshard(x, layout=FULL, storage="gmem")


@module(entry="w13", target=CpuTarget())
class Expert:
    @func
    def w13(x: Tensor[(M, D), F32], gate: Tensor[(D, F), F32], up: Tensor[(D, F), F32]):
        g = tf.matmul(x, gate)
        u = tf.matmul(x, up)
        return g, u, tf.silu(g) * u

    @func
    def w2(h: Tensor[(M, F), F32], wd: Tensor[(F, D), F32]):
        return tf.matmul(h, wd)

    @func
    def dgrad_w2(dy: Tensor[(M, D), F32], wd: Tensor[(F, D), F32]):
        return tf.matmul(dy, wd, b_layout="NK")

    @func
    def wgrad_w2(h: Tensor[(M, F), F32], dy: Tensor[(M, D), F32]):
        return tf.matmul(h, dy, a_layout="KM")

    @func
    def gate_backward(dh: Tensor[(M, F), F32], g: Tensor[(M, F), F32], u: Tensor[(M, F), F32]):
        s = tf.sigmoid(g)
        return dh * u * s * (1.0 + g * (1.0 - s)), dh * g * s

    @func
    def dgrad_w13(
        dg: Tensor[(M, F), F32],
        du: Tensor[(M, F), F32],
        wg: Tensor[(D, F), F32],
        wu: Tensor[(D, F), F32],
    ):
        return tf.matmul(dg, wg, b_layout="NK") + tf.matmul(du, wu, b_layout="NK")

    @func
    def wgrad_w13(x: Tensor[(M, D), F32], dg: Tensor[(M, F), F32], du: Tensor[(M, F), F32]):
        return tf.matmul(x, dg, a_layout="KM"), tf.matmul(x, du, a_layout="KM")


def window(rank, axes):
    cut = [slice(None)] * len(BOX)
    for axis, coord in zip(axes, divmod(rank, 2), strict=True):
        cut[axis] = slice(coord, coord + 1)
    return tuple(cut)


def hop(fn, box, src, dst):
    shards = tuple((p, box[window(p.ids["rank"], src)].contiguous()) for p in RANKS)
    out = evaluate(fn, ShardTensor(fn.params[0].type, shards))
    result = torch.empty_like(box)
    for p in RANKS:
        target = result[window(p.ids["rank"], dst)]
        target.copy_(out.to_local(p).reshape_as(target))
    return result


def check(policy, empty_node):
    torch.manual_seed(671)

    def rand(*shape):
        return (torch.randn(*shape) * 0.3).requires_grad_()

    x, wg, wu, wd = rand(T, D), rand(4, D, F), rand(4, D, F), rand(4, F, D)
    score = rand(T, 2).softmax(-1)
    # Two selected experts per token. Each source sends to the opposite node in the first case.
    ids = torch.tensor([[0, 1] if empty_node or t % 4 >= 2 else [2, 3] for t in range(T)])
    g = torch.einsum("td,tkdf->tkf", x, wg[ids])
    u = torch.einsum("td,tkdf->tkf", x, wu[ids])
    y_ref = (
        torch.einsum("tkf,tkfd->tkd", torch.nn.functional.silu(g) * u, wd[ids]) * score[:, :, None]
    ).sum(1)
    dy = torch.randn_like(y_ref)
    expected = torch.autograd.grad((y_ref * dy).sum(), (x, wg, wu, wd, score))
    x, wg, wu, wd, score = (v.detach() for v in (x, wg, wu, wd, score))
    functions = {
        fn.name: check_program(mod, fn)[1] for mod in (Route, Expert) for fn in mod.functions
    }

    def call(name, *args):
        return evaluate(functions[name], *args)

    counts = torch.zeros(4, 4, dtype=torch.long)
    routes = []
    for t, choices in enumerate(ids.tolist()):
        source = t % 4
        for k, expert in enumerate(choices):
            slot = int(counts[source, expert])
            counts[source, expert] += 1
            routes.append(
                (t, k, expert, slot // R, (*divmod(source, 2), *divmod(expert, 2), slot % R))
            )
    n = math.ceil(int(counts.max()) / R)
    state = [[{} for _ in range(4)] for _ in range(n)]
    boxes = [None] * n
    y, dx, ds = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(score)
    dwg, dwu, dwd = (torch.zeros_like(w) for w in (wg, wu, wd))
    partials = [torch.zeros_like(x) for _ in RANKS]
    for phases in PASSES.values():
        for event in sorted(trace(phases, n, 2, policy, {}), key=lambda span: span.end):
            t0, key = event.tile, event.stage.name
            entries = [r for r in routes if r[3] == t0]
            if key in ("x_ib", "dy_ib"):
                box = torch.zeros(BOX)
                for t, k, _, _, index in entries:
                    box[index] = x[t] if key == "x_ib" else dy[t] * score[t, k]
                boxes[t0] = hop(functions["ib_out"], box, (0, 1), (2, 1))
            elif key in ("x_nvl", "dy_nvl"):
                box = hop(functions["nvlink_out"], boxes[t0], (2, 1), (2, 3))
                for e, v in enumerate(state[t0]):
                    v["x" if key == "x_nvl" else "dy"] = box[:, :, e // 2, e % 2].reshape(M, D)
            elif key in ("y_nvl", "dx_nvl"):
                box = torch.zeros(BOX)
                for e, v in enumerate(state[t0]):
                    box[:, :, e // 2, e % 2] = v["y" if key == "y_nvl" else "dx"].reshape(
                        2, 2, R, D
                    )
                boxes[t0] = hop(functions["nvlink_back"], box, (2, 3), (2, 1))
            elif key in ("y_ib", "dx_ib"):
                box = hop(functions["ib_back"], boxes[t0], (2, 1), (0, 1))
                for t, k, e, _, index in entries:
                    if key == "y_ib":
                        y[t] += score[t, k] * box[index]
                        ds[t, k] = dy[t] @ box[index]
                    else:
                        dx[t] += box[index]
                        partials[e][t] += box[index]
            else:
                for e, v in enumerate(state[t0]):
                    if key == "gate_up":
                        v["g"], v["u"], v["h"] = call("w13", v["x"], wg[e], wu[e])
                    elif key == "down":
                        v["y"] = call("w2", v["h"], wd[e])
                    elif key == "down_input":
                        v["dh"] = call("dgrad_w2", v["dy"], wd[e])
                    elif key == "down_weight":
                        dwd[e] += call("wgrad_w2", v["h"], v["dy"])
                    elif key == "up_input":
                        v["dg"], v["du"] = call("gate_backward", v["dh"], v["g"], v["u"])
                        v["dx"] = call("dgrad_w13", v["dg"], v["du"], wg[e], wu[e])
                    elif key == "up_weight":
                        a, b = call("wgrad_w13", v["x"], v["dg"], v["du"])
                        dwg[e] += a
                        dwu[e] += b
                    else:
                        raise ValueError(key)
    part = ShardTensor(
        functions["resolve"].params[0].type, tuple((p, partials[p.ids["rank"]]) for p in RANKS)
    )
    got = (y, call("resolve", part), dx, dwg, dwu, dwd, ds)
    want = (y_ref, expected[0], *expected)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    return max((a - b).abs().max().item() for a, b in zip(got, want, strict=True))


if __name__ == "__main__":
    torch.set_num_threads(1)
    for empty in (False, True):
        for policy in ("bulk", "tile_ready"):
            print(f"{policy}, empty expert node={empty}: max error {check(policy, empty):.3g}")
