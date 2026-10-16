# Repro: the `--performance` timeline misprices twice

Self-contained reproducers for the combined issue
"the performance timeline prices `insert_slice`/`tf.zeros` as whole-tensor
copies, and cannot distinguish 40 from 132 active lanes".
Affects `main` @ `aaa8054`. No GPU needed — `analyze` is a static model.

## Files

| file | mesh fill | per-lane work | demonstrates |
|---|---|---|---|
| `minimal_lanes.py` | 40 of 132 lanes | 40 trips × (64³ matmul + tile store) | I-1: insert_slice/zeros billing |
| `minimal_lanes_132.py` | 132 of 132 lanes | same per-lane work (3.3× the tiles) | I-1: predicted time tracks destination size, not work |
| `minimal_acc.py` | 40 of 132 lanes | 40 serial accumulator steps | I-2: lane-fill blindness |
| `minimal_acc_132.py` | 132 of 132 lanes | same per-lane work | I-2: identical prediction for 3.3× the work |

Per-lane work is identical across each pair; only the mesh fill differs.

## Run

```bash
PYTHONPATH=src python -m tilefoundry.cli analyze \
  --compute-cost --memory --roofline --performance \
  repro/performance-timeline/minimal_lanes.py:Stream out.txt
# likewise for :Stream in minimal_lanes_132.py and :Scan in minimal_acc*.py
```

## Observed at `aaa8054` (target `nvidia.h200_sxm`)

I-1 — `minimal_lanes.py`:

```text
# roofline ideal-ns=13654 bound-by=memory
# performance root=Stream::stream predicted-ns=1117769 waves=1
o_all = tf.zeros(...)      performance=[0,360449)          # 360 µs to zero 12.50 MB
v9 = tf.insert_slice(...)  gmem:r320KB/w320KB@logical, r12.50MB/w12.50MB@total
```

- `tf.zeros` of a 12.50 MB buffer is billed `w500.00MB@total` (40× logical,
  one multiply per mesh position) and priced at 360 µs; a 12.5 MB memset is
  ~2.6 µs at this target's 4.8 TB/s.
- Each `insert_slice` of a 320 KB slice is billed as a read+write of the
  whole 12.50 MB destination.
- Net: `predicted-ns=1117769` vs `roofline ideal-ns=13654` — an 82×
  inflation, almost entirely this artifact. `minimal_lanes_132.py` (3.3×
  the tiles, same per-lane work) predicts 3,604,839 ns: predicted time
  scales with the *destination tensor size*, not the program's work.

I-2 — `minimal_acc.py` vs `minimal_acc_132.py`:

```text
# analysis target=nvidia.h200_sxm module=Scan function=scan topology=cta wave=132/132
# roofline ideal-ns=5530 bound-by=memory      (40 lanes)
# performance root=Scan::scan predicted-ns=31628 waves=1

# analysis target=nvidia.h200_sxm module=Scan function=scan topology=cta wave=132/132
# roofline ideal-ns=18248 bound-by=memory     (132 lanes, 3.3× the work)
# performance root=Scan::scan predicted-ns=31628 waves=1
```

Identical predicted-ns for 40 vs 132 active lanes; the header reads
`wave=132/132` in both. Per-CTA service rates are peak/N regardless of how
many mesh lanes the layout activates.

## Why it matters (measurement cross-check)

In an occupancy-gated regime this names the wrong winner. For the DPLR
`ho` segment at 40/132 lanes (flash-linear-attention RWKV-7 path), the
model predicted fused 4.17 ms < split 11.05 ms; on an sm_90 part the
measured order is the opposite (fused 2017 µs > split 1522 µs; ncu shows
3–6% occupancy, DRAM idle). Spec §1.2.4 disclaims occupancy guarantees, but
the report gives the reader no signal — nothing prints "40 of 132 lanes
active" or discounts the verdict.
