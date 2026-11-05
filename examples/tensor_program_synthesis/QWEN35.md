# What the Qwen3.5 multi-head experiment means

Verified on 2026-11-05 using the CPU HIR evaluator.

The tensor-survival approach extends to real Qwen3.5 GDN head geometry. The strongest implication is that residual selection needs a training-memory budget that accounts for heads and layers. A shorter state-dependency chain alone is insufficient to choose the best implementation.

## Verified model geometry and implementation

These dimensions come from the actual checkpoint configurations, rather than Transformers class defaults:

| Checkpoint | Q/K heads | Value heads / recurrent states | Key and value width | GDN layers |
|---|---:|---:|---:|---:|
| Qwen3.5-35B-A3B | 16 | 32 | 128 | 30 |
| Qwen3.5-27B | 16 | 48 | 128 | 48 |
| Qwen3.5-397B-A17B | 16 | 64 | 128 | 45 |

The official configs and current Transformers implementation are archived under `references/qwen35/`, with immutable revisions, URLs, and SHA256 hashes in `sources.json`. The official pure-PyTorch chunk function is executed from that snapshot with its kernel-dispatch decorator removed. Its function body is unchanged. Q/K normalization, query scaling, head repetition, gate/beta conventions, and dtype behavior are included in the reference comparison.

Actual batched tensors were evaluated, including shared-head gradient reductions. This is more than multiplying single-head operation counts. `Graph` and the TileFoundry emitter now support batched matmul, transpose, slicing, concatenation, and reductions. Q/K use a singleton group dimension, while values, gates, and states use the full value-head group dimension.

The Qwen-style baseline now precomputes transformed values `U_base` and transformed keys `W` for every chunk before propagating states sequentially. This follows the preparation structure of the official implementation. Its state adjoint reuses `W.T` as well. The earlier unfactored toy recurrence was a less appropriate performance baseline for Qwen.

## What changes in the interpretation

**Multiple heads already provide independent work.** Each value head has its own memory state. A batch of sequences gives additional independent state streams. The sequence recurrence within one stream is not a serialization of the whole layer. Whether a tree scan helps depends on how efficiently the existing head and batch work uses the device, and on its memory traffic. We have not measured that device behavior.

**Qwen already performs substantial work across chunks independently.** Its reference implementation prepares the local solve and transformed keys/values before the state loop. The full affine scan adds dense 128-by-128 state-map compositions to reduce dependencies in the remaining loop. In this prototype it requires more than twice the modeled arithmetic of the Qwen-style prepared recurrence at the tested 8k-token shapes. That does not establish its GPU speed, but it is a clear reason to retain both candidates rather than choose the scan solely from dependency depth.

**Sharing keys does not mean sharing histories.** Four value heads in the 397B configuration can use the same Q/K head while having different values, update strengths, and decay gates. Their states and gate-dependent triangular inverses therefore generally differ. The key Gram matrix and query-key products can be shared within that group. The backward must sum the value-head contributions into each shared Q/K gradient. The comparisons check this directly.

**Training checkpoints can be far larger than the live recurrent state.** A single current state for all 64 value heads is only 4 MiB per layer in FP32. Keeping an internal state at every chunk boundary for an 8k-token sequence is a different allocation: 508 MiB per layer. Small per-head retention decisions accumulate across the model.

For batch 1, sequence length 8192, chunk size 64, with FP32 caches and all GDN-layer caches retained until backward:

| Additional cache policy | 35B-A3B, all 30 GDN layers | 397B-A17B, all 45 GDN layers |
|---|---:|---:|
| Keep per-chunk inverse matrices | 1.875 GiB | 5.625 GiB |
| Keep all internal chunk-boundary states | 7.441 GiB | 22.324 GiB |

These are alternative residual-cache costs, not measurements of Qwen's production memory use or total training peak. They exclude weights, gradients, optimizer state, immutable inputs, ordinary outputs, and temporary workspace. Layer activation checkpointing, tensor/context parallel placement, and different retained dtypes change the totals. Microbatch size scales them too.

The practical implication is to compare smaller solve summaries, shared Q/K products, selected states, and recomputation under a realistic memory limit. Saving every state is not automatically the useful default. The group-level searches selected all inverses at the smaller tested budgets and mixed shared products, activations, and states at larger budgets. Only the smaller-budget results were proved optimal within this bounded cost model; the larger searches ended feasible, with roughly 19% work-bound gaps.

## Verification and scope

All 28 core comparisons passed. Each comparison checks both outputs and all six input gradients, including gradients for shared Q/K. Cases include:

- Batch 2, full 32- and 64-value-head geometry, 129 tokens with a short final chunk, both formulations, and generated batched TileFoundry HIR.
- Batch 1, 8192 tokens, all 32 or 64 value heads, weak decay, both formulations, with and without retained inverses. The 64-head prepared/inverse program was also lowered and executed as TileFoundry HIR at this full length.
- Batch 2, 2048 tokens, all 64 value heads, BF16 inputs/outputs and FP32 core intermediates/state.
- The 27B checkpoint's 48 value heads at 2051 tokens, including a short final chunk.
- A full 64-head FP64 check against an independently written normalized token recurrence, with a nonzero initial state and final-state cotangent.

FP32 core comparisons use `atol=rtol=2e-5`; FP64 uses `2e-9`; the BF16 boundary comparison uses `5e-3`. The numerical results and per-tensor error metrics are retained in `results/qwen35/summary.json`.

The candidate was additionally inserted into an actual Transformers 5.14.1 `Qwen3_5GatedDeltaNet` block configured with the 397B model's GDN dimensions. That block used synthetic random weights and weak decay for sensitivity across chunks. Against the same block using the pinned current official reference core, its output, hidden-state gradient, and every parameter gradient passed the FP32 check. This covers the surrounding projections, convolution, gates, gated normalization, and output projection. It is a single-block correctness check; checkpoint weights were not loaded.

The test suite has 127 passing regression tests. The adapter retains only inputs, selected residuals, and any ordinary outputs actually needed by backward; it avoids pinning an unused FP32 core output behind the BF16 output cast.

No end-to-end pretrained-model training, fused FLA CUDA execution, or GPU speedup is claimed. The prototype still uses a polynomial triangular inverse instead of the official row-elimination or production fused triangular kernels. Consequently its arithmetic objective cannot establish which tensors a tuned CUDA implementation should save. The solver also selects uniform whole batched tensors within a supplied graph; it does not yet jointly allocate memory across all model layers or choose a different policy for every head.

## Reproduce

```bash
cd examples/tensor_program_synthesis
./run_qwen35.sh
# If a connection or process was interrupted:
./run_qwen35.sh --resume
```

Source: `synthesis/multihead.py`, `synthesis/qwen35_run.py`, the prepared recurrence in `synthesis/parallel.py`, and the batched extensions in `synthesis/graph.py` / `synthesis/lower.py`.

Evidence: `results/qwen35/summary.json`, `memory_projection.json`, `tests.xml`, and `generated/`. The environment and TileFoundry commit remain pinned by `requirements.lock` and `TILEFOUNDRY_COMMIT`.
