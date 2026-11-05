# Recorded 8k evidence

These JSON files preserve the completed 8k CPU HIR run before publication into
the example directory. Numerical formulas and acceptance thresholds are retained.
Publication changes the directory layout, runner interpreter selection and
documentation, and adds an explicit incomplete status.

| File | Evidence |
|---|---|
| [summary.json](bf16_8k/summary.json) | Whole and partitioned execution, all nine parameter gradients, and the independent formula reference |
| [native_vjp_oracle.json](bf16_8k/native_vjp_oracle.json) | Independent differentiation of an exactly matching native forward |
| [vjp_audit.json](bf16_8k/vjp_audit.json) | Forward-stage and matching-arithmetic diagnostics |
| [gradient_rounding_audit.json](bf16_8k/gradient_rounding_audit.json) | Intermediate cotangents and exact attribution of the QKV weight-gradient difference |

The `passed: false` verdicts are intentional records of unresolved strict
numerical mismatches. They are not interrupted runs or successful validation.
Relative L2 checks pass, while some elementwise comparisons fail.

Paths inside the snapshots refer to generated output locations relative to the
example directory. Rerunning the commands in the main README recreates those
outputs under `results/`; it does not overwrite these recorded snapshots.
Large generated graph and region-interface JSON files are regenerated rather
than checked in. The 149-test regression suite and the unresolved full-size
checks are separate validation scopes.
