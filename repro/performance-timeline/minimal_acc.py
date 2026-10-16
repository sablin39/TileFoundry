from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

# I-2 companion: NO insert_slice, no big output tensor — per-lane work is a
# fixed serial chain (40 trips x load 2 tiles + one 64x64x64 matmul + add into
# a smem accumulator), one small final store. 40-lane and 132-lane variants do
# identical per-lane work; only the machine fill differs.
#   minimal_acc.py      LANES=40  of 132
#   minimal_acc_132.py  LANES=132 of 132

LANES = 40
TRIPS = 40
TILES = LANES * TRIPS

_TARGET = CudaTarget("nvidia.h200_sxm")


@module(entry="scan", target=_TARGET, topologies=(Topology("cta", 132),))
class Scan:
    @func
    def scan(
        a: Tensor[(1, TILES, 64, 64), "bf16"],
        b: Tensor[(1, TILES, 64, 64), "bf16"],
    ):
        with Mesh((Topology("cta", 132),), layout=(LANES,), names=("lane",)) as cta:
            acc = tf.zeros(type=Tensor[(1, LANES, 64, 64), "f32", (1, LANES @ cta.lane, 64, 64), "smem"])
            for t in tf.tile(TILES, LANES):
                ta = a[:, t, :, :]
                sa = tf.reshard(ta, layout=(1, LANES @ cta.lane, 64, 64), storage="smem")
                tb = b[:, t, :, :]
                sb = tf.reshard(tb, layout=(1, LANES @ cta.lane, 64, 64), storage="smem")
                m = tf.matmul(sa, sb, out_dtype="f32")
                ms = tf.reshard(m, layout=(1, LANES @ cta.lane, 64, 64), storage="smem")
                acc = tf.add(acc, ms)
            out = tf.cast(acc, dtype="bf16")
            r = tf.reshard(out, layout=(1, LANES, 64, 64), storage="gmem")
        return r
