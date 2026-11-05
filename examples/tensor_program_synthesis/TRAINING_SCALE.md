# Multi-chunk gated delta rule experiment

The prototype now compares the original chunk recurrence with a supplied affine-scan formulation, using the same residual planner and TileFoundry lowering path.

The tested shapes are `T=512, 2048, 8192`, chunk size 64, `K=V=64`: 8, 32, and 128 full chunks. A second case uses `T=2051, K=128, V=64`, including a three-token final chunk. These are individual sequence/head programs. Batch and head parallelism is additional; batched GPU kernels were not executed.

## What the longer sequences reveal

The original graph already has independent chunk preparation: gate prefixes, Gram matrices, causal attention matrices, and triangular inverses do not depend on the incoming recurrent state. At `T=8192, K=V=64`, 70.6% of its forward work proxy is independent of the initial state. A scheduler can prepare those chunks concurrently.

State propagation remains a chain. Saving forward states does not remove the backward adjoint chain. The alternative formulation exposes both directions as associative affine scans, followed by independent local output/gradient calculations.

Here, depth means the largest number of dependent whole-matrix matmuls in the DAG. It is neither CUDA launch count nor measured latency. The table uses zero saved residuals for both formulations.

| Tokens | Chunks | Original forward depth | Scan forward depth | Original backward depth | Scan backward depth |
|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 30 | 16 | 31 | 18 |
| 2048 | 32 | 102 | 20 | 103 | 22 |
| 8192 | 128 | 390 | 24 | 391 | 26 |

At 8192 tokens, the scan increases forward work from 1.155 to 1.617 billion elementary-operation units (40.0%) and total forward/backward work from 3.136 to 4.447 billion (41.8%). With `K=128, V=64, T=2051`, the scan costs 2.37 times the original total work. Its extra dense `K x K` matrix compositions are therefore a consequential tradeoff.

## Affine formulation

For a chunk, use the existing inverse `J`, row decay `d`, end decay `e`, suffix-weighted keys `K_s`, scaled queries `Q_s`, and causal attention matrix `M`. All products with beta and d below broadcast over rows.

```text
U_base = J @ (beta * V)
W      = J @ (beta * d * K)
U      = U_base - W @ S_in
A      = e * I - K_s.T @ W
B      = K_s.T @ U_base
Z      = d * Q_s - M @ W

S_out  = A @ S_in + B
O      = Z @ S_in + M @ U_base
dS_in  = A.T @ dS_out + Z.T @ dO
```

Compose two affine maps in temporal order with `(A2 @ A1, A2 @ B1 + B2)`. A balanced upsweep/downsweep constructs forward state prefixes. A reverse scan over `(A.T, Z.T @ dO)` constructs adjoint prefixes. Once these are available, the existing supplied local VJP computes all input and gate gradients independently per chunk. Non-power-of-two chunk counts and short chunks are supported.

This algebraic identity and scan algorithm were supplied explicitly. The residual solver did not invent them. It searches materialization/reconstruction choices separately within each supplied graph.

## Residual choices at 8192 tokens

Bytes count only additional retained residuals per sequence/head. Immutable inputs, ordinary outputs, constants, and transient allocations are excluded. Work is the elementary-operation objective, not GPU time.

| Formulation / policy | Residual MiB | Backward work (billions) | Backward matmul depth |
|---|---:|---:|---:|
| Original, recompute | 0 | 1.981 | 391 |
| Original, all inverses | 2.000 | 1.304 | 387 |
| Original, all internal states | 1.984 | 1.913 | 391 |
| Original, inverses + states | 3.984 | 1.237 | 387 |
| Scan, recompute | 0 | 2.830 | 26 |
| Scan, all inverses | 2.000 | 2.154 | 20 |
| Scan, inverses + states | 3.984 | 1.767 | 20 |

Caching inverses cuts the original backward work by 34.1%. Saving states alone reduces work but leaves its longest backward dependency chain unchanged. The scan exposes more concurrency at additional arithmetic cost. This motivates comparing work, residual memory, and dependency span together.

The search now accepts complete feasible policy hints, and caches its external-node set during model construction. At 8 and 32 chunks, the two tested budgets for the original graph were proved optimal within this recipe model. At 128 chunks they remained feasible, with work-bound gaps about 0.28% and 0.26% after 15-second searches. Scan results retain their actual solver statuses and bounds; they are not claimed optimal. Fixed policies are explicitly labeled `FIXED_POLICY`.

The triangular inverse is still the prototype's finite matrix polynomial. A tuned triangular solve can change both arithmetic costs and residual rankings.

## Numerical and compiler validation

- 113 regression tests pass, including affine composition, all six input gradients, short tails, generated HIR, solver hints, and deep DAG traversal.
- All 32 large-shape FP64 formula/oracle checks pass with `atol=rtol=2e-9`. The oracle is an independent token recurrence differentiated with PyTorch autograd. The candidate backward is supplied analytically.
- All 208 residual-policy comparisons are bitwise equal to their no-residual DAG results.
- Generated FP32 HIR completed 208 cases: 104 pass and 104 fail the original strict tolerance. It is exercised with random, weak-decay, zero-decay, and strong-decay inputs. Weak/zero-decay cases exceed the old `atol=rtol=2e-5` comparison against an FP32 recurrence. These failures are preserved in the JSON reports, rather than hidden by changing the tolerance.
- A separate precision audit uses FP64 casts of exactly the same FP32 input values. Across random/weak/zero-decay cases, maximum per-output relative L2 error is below `8.3e-7`; the largest absolute error is `1.23e-4`. Relative error alone is misleading for near-zero strong-decay gradients, so those cases retain absolute metrics too.
- A separate policy audit compares each materialization plan with its own no-residual DAG using exact FP32 equality. This checks residual handling separately from algebraic reassociation and oracle rounding; it is not an additional HIR or GPU benchmark.

The default Python recursion limit caused TileFoundry type inference to fail on the 8192-token sequential graph. A scoped limit of 20000 around loading/evaluation lets the prototype handle that graph without modifying upstream source. The wrapper is for this single-threaded experiment, not a general compiler fix.

TileFoundry legality/compute analysis succeeded for the complete 64- and 512-token programs. All four static analysis families also succeeded for the single-chunk stages. Monolithic compute analysis at 2048 tokens was stopped after several minutes; full-sequence memory optimization is not claimed. Structured chunk/stage IR is preferable to indefinitely growing flat graphs.

All results here use CPU numerical evaluation and a static H200 target model. There are no native GPU timing, mixed-precision training, or kernel throughput results. The actual server GPUs are L20D. The experiment establishes numerical equivalence in the stated tests and additional DAG parallelism, not a measured GPU speedup or formal proof.

## Reproduction and files

```bash
cd examples/tensor_program_synthesis
./run_training_scale.sh

# Individual experiments
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m synthesis.training_scale --lengths 512,2048,8192
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m synthesis.training_scale --lengths 2051 --key-dim 128 --value-dim 64 --no-search
OMP_NUM_THREADS=1 python -m synthesis.precision_audit
OMP_NUM_THREADS=1 python -m synthesis.policy_audit
OMP_NUM_THREADS=1 python -m synthesis.static_scale
```

`synthesis/parallel.py` contains the affine scan and dependency analysis. `synthesis/training_scale.py` runs policies and bounded searches. The `results/training_scale/T*/summary.json` files are the authoritative per-shape results, with generated `program.py` and complete `plan.json` files beside them. `precision_audit.json`, `policy_audit.json`, `static/summary.json`, `infrastructure_notes.json`, and `tests.xml` retain the separate validation results and limitations.

The residual objective still allows at most one computation of each logical value during backward. It does not optimize transient peak memory, repeated checkpoint replay, tiling, fusion, kernel partitioning, or hardware scheduling. Source and environment versions remain pinned by `TILEFOUNDRY_COMMIT` and `requirements.lock`.
