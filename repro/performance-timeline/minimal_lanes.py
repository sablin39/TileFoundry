from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

# Minimal lane-count A/B for the --performance artifacts (ISSUES.md I-1/I-2).
# Per-lane work is IDENTICAL in both variants: TRIPS=40 sequential trips, each
# one 64x64x64 bf16 matmul + one tile store. Only the mesh fill differs:
#   minimal_lanes.py      LANES=40  of 132  (TILES=1600)
#   minimal_lanes_132.py  LANES=132 of 132  (TILES=5280)
# A timing model that accounts for occupancy must not give both the same
# predicted-ns; and the insert_slice/zeros billing shows up directly in the
# per-op intervals.

LANES = 40
TRIPS = 40
TILES = LANES * TRIPS

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPO = (Topology("cta", 132),)


@module(entry="stream", target=_TARGET, topologies=_TOPO)
class Stream:
    @func
    def stream(
        a: Tensor[(1, TILES, 64, 64), "bf16"],
        b: Tensor[(1, TILES, 64, 64), "bf16"],
    ):
        with Mesh((Topology("cta", 132),), layout=(LANES,), names=("lane",)) as cta:
            o_all = tf.zeros(type=Tensor[(1, TILES, 64, 64), "bf16"])
            for t in tf.tile(TILES, LANES):
                ta = a[:, t, :, :]
                sa = tf.reshard(ta, layout=(1, LANES @ cta.lane, 64, 64), storage="smem")
                tb = b[:, t, :, :]
                sb = tf.reshard(tb, layout=(1, LANES @ cta.lane, 64, 64), storage="smem")
                m = tf.matmul(sa, sb, out_dtype="f32")
                ms = tf.reshard(m, layout=(1, LANES @ cta.lane, 64, 64), storage="smem")
                mc = tf.cast(ms, dtype="bf16")
                mg = tf.reshard(mc, layout=(1, LANES, 64, 64), storage="gmem")
                o_new = tf.insert_slice(o_all, mg, (0, t, 0, 0))
                o_all = o_new
        return o_all
