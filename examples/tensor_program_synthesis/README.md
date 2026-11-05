# Tensor-program synthesis tools and an incomplete GDN example

This experimental suite expresses supplied forward and backward tensor formulas,
plans which values survive between them, lowers the selected plan to TileFoundry
HIR, and checks the result against independent references. The tooling lives in
this example package; it is not a stable TileFoundry public API.

**The full Qwen3.5 GDN layer is an incomplete example.** At 8,192 tokens the
whole and partitioned HIR programs execute and agree bitwise, but strict
parameter-gradient comparisons still fail. The failures remain visible and the
runner exits nonzero. GPU lowering and execution have not been validated.

## Current status

The full-layer graph covers input projections, causal depthwise convolution,
SiLU, Q/K normalization, the grouped-head chunked delta rule, gated RMSNorm,
output projection, and explicit backward formulas for all nine parameter
tensors. The 8k case uses Qwen3.5-397B-A17B dimensions with synthetic weights,
BF16 compute operands and activation boundaries, FP32 contractions/reductions,
and an FP32 chunk core and recurrent state.

| Check | Recorded result |
|---|---|
| Whole-program and region-program native legality | Pass |
| Region partition versus whole output and all gradients | Bitwise equal |
| Independent full-layer reference | All normwise checks pass; QKV, z and output projection weight gradients fail the strict elementwise condition |
| Autograd of an exactly matching native forward | Eight parameter tensors pass; QKV still fails the strict elementwise condition |
| QKV gradient diagnostic | Each weight-gradient contraction is exact for its own incoming gradient; upstream rounding explains their difference |
| Small regression suite | 149 tests pass |

[STATUS.json](STATUS.json) records the incomplete status. The
[8k report](BF16_8K.md) explains the precision contract, numerical failures and
candidate fusion boundaries. [Recorded evidence](evidence/README.md) preserves
the original per-tensor verdicts. The [FP32 report](FULL_GDN.md) documents earlier
smaller full-layer checks; those passes do not resolve the 8k BF16 failures.

## Run

Use Python 3.12 or newer and an editable checkout, following the repository's
[development guide](../../docs/develop.md). From the repository root:

```bash
python -m pip install -e '.[test]' -e targets/nvidia -e targets/apple
cd examples/tensor_program_synthesis
OMP_NUM_THREADS=1 python -m pytest synthesis -q
```

The full-size study is deliberately separate from the small regression suite.
It evaluates large tensors on CPU and needs substantial host memory:

```bash
bash run_bf16_8k.sh
```

This command currently exits nonzero after writing the failed strict checks to
`results/full_gdn_bf16_8k/summary.json`. Run the diagnostics separately:

```bash
OMP_NUM_THREADS=8 python -m synthesis.mixed_vjp_audit
OMP_NUM_THREADS=8 python -m synthesis.native_vjp_oracle
OMP_NUM_THREADS=8 python -m synthesis.gradient_rounding_audit
```

The VJP comparisons also exit nonzero while strict equivalence fails. These
failures are not skipped or marked as expected passes. All shell runners use
the active `python`, or the interpreter selected by `PYTHON`. Generated programs,
graphs, reports and logs are written locally and ignored by Git.

`requirements.lock` records the original CPU package versions;
`TILEFOUNDRY_COMMIT` records the upstream base used for the experiments, before
the call-checker fix in this branch. Pinned official configuration and reference
sources, their hashes, and the reference source license are in `references/`.
No model checkpoint download is needed.

## Tooling

| Module | Responsibility |
|---|---|
| `synthesis/graph.py` | Typed joint tensor DAG, casts and residual byte accounting |
| `synthesis/planner.py` | Save/recompute constraints, explicit policies and bounded residual planning |
| `synthesis/lower.py` | Native HIR emission and execution with declared forward/backward interfaces |
| `synthesis/parallel.py` | Supplied recurrence and affine-scan alternatives |
| `synthesis/fusion_boundaries.py` | Typed region interfaces, dependency extraction, acyclicity checks and callable HIR partitions |
| `synthesis/full_layer.py` | Incomplete full GDN forward and supplied backward formulas |
| `synthesis/mixed_precision.py` | Independent references and fixed numerical criteria |
| `synthesis/native_vjp_oracle.py` | Forward-only native HIR differentiated independently with autograd |
| `synthesis/gradient_rounding_audit.py` | Attribution of intermediate rounding and QKV gradient differences |

The 8k runner uses one fixed residual policy and supplied stage annotations. It
does not search for an optimal fusion plan. Extracted dependencies identify
where chunks or backward branches may run independently; they do not prove
that a region fits one GPU kernel. Logical tensor sizes are not physical memory
traffic or total peak storage.

Earlier studies remain runnable through `run_synthesis.sh`,
`run_training_scale.sh`, `run_qwen35.sh` and `run_full_gdn.sh`. Their reports are
[SYNTHESIS.md](SYNTHESIS.md), [TRAINING_SCALE.md](TRAINING_SCALE.md),
[QWEN35.md](QWEN35.md) and [FULL_GDN.md](FULL_GDN.md). Some earlier studies contain
bounded residual searches; the full-layer 8k study does not invoke them.

## Remaining full-layer work

- Resolve or explicitly specify the remaining mixed-precision numerical
  equivalence, preserving BF16 rounding boundaries and FP32 accumulation.
- Validate an actual GPU lowering, including tiling, resource limits and final
  parameter-gradient reductions.
- Connect the extracted interfaces to future scheduling and fusion exploration.

The branch also fixes native call checking to label call occurrences by identity
instead of recursively hashing shared expression DAGs. Its regression is in
`tests/nvidia/cuda/analysis/test_shared_call_check.py` at the repository root.
