# Repro: the `--performance` timeline misprices twice

Self-contained reproducers for the combined issue
"the performance timeline bills `insert_slice`/`tf.zeros` once per mesh
position at whole-window extent, and cannot distinguish 40 from 132 active
lanes". Affects `main` @ `aaa8054`. No GPU needed — `analyze` is a static
model.

## Files

| file | mesh fill | per-lane work | demonstrates |
|---|---|---|---|
| `minimal_lanes.py` | 40 of 132 lanes | 40 trips × (64³ matmul + tile store) | I-1: per-position billing |
| `minimal_dest2x.py` | 40 of 132 lanes | same, but destination tensor 2× (25 MB) | I-1 discriminator: insert billing is window×positions, NOT destination-sized |
| `minimal_lanes_132.py` | 132 of 132 lanes | same per-lane work (3.3× the tiles) | I-1: prediction scales with positions for equal per-lane work |
| `minimal_acc.py` | 40 of 132 lanes | 40 serial accumulator steps | I-2: lane-fill blindness |
| `minimal_acc_132.py` | 132 of 132 lanes | same per-lane work | I-2: identical prediction for 3.3× the work |

Per-lane work is identical across each pair; only the mesh fill (or the
destination size, for the discriminator) differs.

## Run

```bash
PYTHONPATH=src python -m tilefoundry.cli analyze \
  --compute-cost --memory --roofline --performance \
  repro/performance-timeline/minimal_lanes.py:Stream out.txt
# likewise for :Stream in minimal_dest2x.py / minimal_lanes_132.py
# and :Scan in minimal_acc.py / minimal_acc_132.py
```

## Observed at `aaa8054` (target `nvidia.h200_sxm`)

### I-1 — insert_slice / tf.zeros billed per mesh position at whole-window extent

`minimal_lanes.py` (40 lanes, 12.5 MB destination):

```text
# roofline ideal-ns=13654 bound-by=memory
# performance root=Stream::stream predicted-ns=1117769 waves=1
o_all = tf.zeros(...)                w12.50MB@logical, w12.50MB@cta, w500.00MB@total   performance=[0,360449)
v9  = tf.insert_slice(o_all, v8, …)  r320.00KB/w320.00KB@logical, r320.00KB/w320.00KB@cta,
                                     r12.50MB/w12.50MB@total            performance=[…,18933t+379382)*40
```

Each position's shard of the update `v8` is 8 KB (the matching `reshard`
load of the same tensor bills `r8.00KB@cta`), yet the insert bills the whole
320 KB window to **every** position (`@cta`), and `@total` multiplies by the
40 executing positions — 12.5 MB for a 320 KB disjoint write. `tf.zeros` of
the mesh-replicated destination bills the whole 12.5 MB buffer per position
(500 MB total), 360 µs on the timeline for what is one 12.5 MB memset.

`minimal_dest2x.py` — same program, destination doubled to 25 MB, window
and lanes unchanged:

```text
o_all = tf.zeros(...)                w1000.00MB@total   performance=[0,720897)   # doubles with the buffer
v9  = tf.insert_slice(o_all, v8, …)  r12.50MB/w12.50MB@total                     # UNCHANGED
# performance predicted-ns=1478217   # +360,448 ns, exactly the zeros doubling
```

The insert billing does not track the destination at all — it is
window × positions. The zeros billing is buffer × positions. (#194/#195
stopped replicated movement from being counted per mesh position for
copies; these two write paths still multiply.)

Mechanism anchors: `_occurrence_shares` scales the per-unit share by
`positions(unit)` (`src/tilefoundry/analysis/memory.py:327,331`); the
per-unit share for `insert_slice` comes from the whole update window
(`src/tilefoundry/ir/hir/tensor/insert_slice.py:56`,
`_insert_slice_access`), not the position's shard; `--performance` turns
the per-unit traffic into time at `hbm_rate ÷ max_physical_units`, so
`predicted-ns` inherits the multiply — 82× the roofline floor here, and
27× against hardware on a real streaming kernel (DPLR `ho` apply: 3.60 ms
predicted vs 133 µs measured at 79% of DRAM peak, ncu).

### I-2 — lane-fill blindness

`minimal_acc.py` (40 lanes) vs `minimal_acc_132.py` (132 lanes), identical
per-lane serial chains:

```text
# analysis target=nvidia.h200_sxm module=Scan function=scan topology=cta wave=132/132
# roofline ideal-ns=5530 bound-by=memory      (40 lanes)
# performance root=Scan::scan predicted-ns=31628 waves=1

# analysis target=nvidia.h200_sxm module=Scan function=scan topology=cta wave=132/132
# roofline ideal-ns=18248 bound-by=memory     (132 lanes, 3.3× the work)
# performance root=Scan::scan predicted-ns=31628 waves=1
```

Identical predicted-ns; the header reads `wave=132/132` in both. Per-CTA
service rates are peak/N regardless of how many mesh lanes the layout
activates: `wave_of` returns `(min(declared, max), declared)` from the
topology extent (`src/tilefoundry/analysis/footprint.py:1317`) and
`waves = ceil(topology_extent / max_physical_units)`
(`src/tilefoundry/analysis/performance.py:169`) — the layout's active
extent is never consulted.

## Why it matters (measurement cross-check)

In an occupancy-gated regime this names the wrong winner. For the DPLR
`ho` segment at 40/132 lanes (flash-linear-attention RWKV-7 path), the
model predicted fused 4.17 ms < split 11.05 ms; on an sm_90 part the
measured order is the opposite (fused 2017 µs > split 1522 µs; ncu shows
3–6% occupancy, DRAM idle). Spec §1.2.4 disclaims occupancy guarantees, but
the report gives the reader no signal — nothing prints "40 of 132 lanes
active" or discounts the verdict.
