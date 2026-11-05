# Incomplete example: complete Qwen3.5 GDN layer in TileFoundry HIR

**Status: incomplete.** The 8k BF16 example retains strict parameter-gradient
mismatches and has not been lowered to executable GPU kernels. See [README.md](README.md)
for the current status and [BF16_8K.md](BF16_8K.md) for the numerical evidence.

This extension expresses the complete `Qwen3_5GatedDeltaNet` full-sequence
training module, with its forward output and explicit gradients for hidden
states and all nine parameter tensors. The earlier experiment replaced only
the recurrent core inside a PyTorch block; this candidate lowers the surrounding
operations and their gradients into TileFoundry too.


## Verified results

Verified on 2026-11-05 (Asia/Singapore). Both native HIR programs pass
`check_program` and execute through TileFoundry's CPU evaluator for each case:

| Official checkpoint geometry | Hidden width | Q/K heads | Value heads | Batch | Tokens | Chunk size |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B | 2,048 | 16 | 32 | 2 | 129 | 64 |
| Qwen3.5-397B-A17B | 4,096 | 16 | 64 | 1 | 1,025 | 64 |

Both use key/value width 128 and convolution width four. The longer case has
16 complete chunks plus a final one-token chunk. Both fixed retention policies
pass for both geometries: the output, hidden-state gradient, and all nine
parameter gradients match the complete pinned official layer within the
numerical criteria below. These are synthetic weights with weak decay so the
tests remain sensitive to cross-chunk state propagation.

All 139 prototype regression tests and 272 affected upstream tests pass.
This establishes an executable whole-layer representation and explicit
forward/backward contracts. It does not establish an optimal implementation
or GPU throughput.

## Representation

The authored graph includes input padding-mask multiplication; QKV, z, beta,
and decay projections; four-tap causal depthwise convolution and SiLU; Q/K L2
normalization; sigmoid beta and `-exp(A_log) * softplus(a + dt_bias)` decay;
grouped-head chunked gated delta rule; per-head gated RMS normalization; and
the output projection.

Every stage has supplied analytical backward formulas. This includes convolution
weight gradients, shared-Q/K head reductions, normalization gradients, and all
projection weight gradients. The candidate does not invoke PyTorch autograd or
call an opaque recurrent operator. The TileFoundry interpreter itself uses
PyTorch to execute native HIR operations on the CPU.

No new mathematical operation was needed in TileFoundry. The prototype frontend
gained mappings for reshape, general permutation and slicing, zero tensors,
SiLU, sigmoid, softplus, inverse square root, and selection. The short convolution
is expressed as shifted slices, broadcast products, and additions; this is a
legal formulation, not a claim that it is the best convolution implementation.

`synthesis/full_layer.py` composes the existing grouped chunk graph into one
joint graph. `synthesis/full_layer_run.py` emits separate forward and backward
programs and checks their phase interfaces. Forward receives hidden states,
the immutable mask, and parameters; backward additionally receives the output
cotangent and the explicitly chosen residuals. Backward returns gradients for
hidden states and all nine parameter tensors. Immutable parameters and inputs
remain available across phases.

## What this adds for future planning

Residual choices now span the whole layer. The planner can see, for example,
that regenerating a core input can require an input projection, convolution,
activation, and normalization. It can also see that the same projected hidden
states feed several branches, and that parameter gradients consume values that
a core-only backward would not need.

This gives subsequent search a meaningful dependency graph for choosing among
retaining projected inputs, convolution intermediates, normalized Q/K, gate
values, chunk quantities, core outputs, and normalization intermediates. The
extension also supplies an executable correctness oracle for each emitted
forward/backward interface.

The demonstration uses two fixed policies: recompute intermediate values, or
retain a selected set of outer-layer and core intermediates. It does not search
for an optimal policy. The existing planner still selects whole tensors from
supplied formulas and recipes. It does not discover model equations, derive
arbitrary gradients, choose arbitrary fusion/tiling, or provide a general
multi-execution checkpoint scheduler.

Residual byte counts refer to selected values across the forward/backward
boundary. They exclude immutable inputs, parameters, gradients, ordinary
outputs, and transient workspace. They are not total training memory.

## Native checker fix

Whole-layer validation exposed a scalability bug in the pinned TileFoundry
checker: `_CheckCalls` used expression objects as dictionary keys for diagnostic
labels. Their structural hashes recursively revisit shared DAG subexpressions.
The large backward program stalled in this bookkeeping even though native type
inference and numerical evaluation could handle the graph.

This branch changes that label map to use node identity in
`src/tilefoundry/analysis/check.py`. It changes no operator semantics. The
focused regression is in `tests/nvidia/cuda/analysis/test_shared_call_check.py`
at the repository root. `TILEFOUNDRY_COMMIT` records the upstream experiment base.

## Reference fidelity

The whole reference layer is executed from the pinned official Transformers
source at commit `fd3cf0140de03f7619ac95b5cfed05b711ff9e22`, including its convolution,
gated normalization, masking, and chunk fallback. Kernel-dispatch decorators are
removed and annotations postponed; function and class bodies remain unchanged.
Checkpoint configuration revisions and source hashes are in
`references/qwen35/sources.json`.

An initial comparison combined the installed Transformers 5.14.1 outer block
with the newer pinned core. That installed outer block ignores padding masks
when batch size is one; the pinned source applies them. This produced a mismatch
for the masked batch-one example. The diagnostic result is preserved in
`results/full_gdn/installed_reference_mask_mismatch.json`. The final runner uses
the complete pinned reference consistently, and regression tests cover masking
at both batch sizes one and two.

Official sources:

- [Qwen3.5 GDN implementation](https://github.com/huggingface/transformers/blob/fd3cf0140de03f7619ac95b5cfed05b711ff9e22/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [35B-A3B configuration](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)
- [397B-A17B configuration](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/resolve/8472618112abcbd45acbcdc58436aff4233c23f7/config.json)

## Scope

This proves a fixed-shape, FP32, full-sequence GDN training module with zero
initial convolution and recurrent states, synthetic weights, padding masks,
and partial final chunks. Small FP64 tests independently audit the equations
against a token recurrence and standard PyTorch convolution/autograd.

The GDN mixer is the scope, rather than the outer transformer decoder block
with its residual connections and MLP/MoE. Full-layer BF16 rounding, packed
variable-length sequences, mutable decode-cache paths, pretrained-model
training, native CUDA code generation, kernel fusion, and GPU performance have
not been verified by this extension. The previous core-only 8k-token and BF16
checks remain separate evidence, not claims about this complete-layer run.

## Numerical criteria

Full-size FP32 comparisons require elementwise `atol=1e-4, rtol=3e-5` and
relative L2 error at most `2e-6` for every output/gradient tensor. The long case
also requires both the official FP32 implementation and candidate FP32 HIR to
stay within `2e-6` relative L2 error of the supplied whole-layer DAG evaluated
in FP64. That FP64 run audits arithmetic; independent derivation is checked by
the small token-recurrence/conv1d/autograd tests at `atol=rtol=2e-9`.

The original stricter FP32 `atol=rtol=3e-5` checks are retained per tensor. At
1,025 tokens, the output and hidden-state gradient satisfy that check, while
the QKV, beta-projection, and output-projection weight gradients have some
entries outside it. Their maximum pairwise absolute differences are about
`8.4e-5`, `6.1e-5`, and `4.6e-5`, respectively, with relative L2 errors around
`5e-7`. Against FP64, both implementations show similar rounding: for the QKV
weight gradient, relative L2 error is about `9.3e-7` for each. This supports a
rounding explanation for the long reductions, not bitwise equivalence.
The original strict run is preserved in `strict_rounding_diagnostic.json` and
the full precision audit in `precision_audit.json` and the final summary.

## Reproduce

After installing this checkout as described in [README.md](README.md):

```bash
cd examples/tensor_program_synthesis
bash run_full_gdn.sh
```

The checker fix is committed in this branch. Model checkpoint weights are not
downloaded. Run the affected native regression from the repository root with
`python -m pytest tests/nvidia/cuda/analysis/test_shared_call_check.py -q`.

Source: `synthesis/full_layer.py`, `synthesis/full_layer_run.py`, the new tests,
and the frontend additions in `synthesis/graph.py` and `synthesis/lower.py`.
Evidence: `results/full_gdn/summary.json`, `tests.xml`, `generated/`, graph JSON,
and the logs. Native GPU execution is not part of this runbook.
