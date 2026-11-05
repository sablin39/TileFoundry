#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
mkdir -p logs results/training_scale
"${PYTHON:-python}" -m pytest synthesis -q --junitxml=results/training_scale/tests.xml
"${PYTHON:-python}" -u -m synthesis.training_scale "$@" | tee logs/training-scale.log
"${PYTHON:-python}" -u -m synthesis.training_scale --lengths 2051 --key-dim 128 --value-dim 64 --no-search | tee logs/training-scale-tail.log
"${PYTHON:-python}" -u -m synthesis.precision_audit | tee logs/training-precision.log
"${PYTHON:-python}" -u -m synthesis.policy_audit | tee logs/training-policies.log
"${PYTHON:-python}" -u -m synthesis.static_scale | tee logs/training-static.log
"${PYTHON:-python}" -u -m synthesis.training_report
"${PYTHON:-python}" - <<'PY'
import json
from pathlib import Path
root = Path('results/training_scale')
summaries = [json.loads(p.read_text()) for p in sorted(root.glob('T*/summary.json'))]
checks = [c for r in summaries for v in r['variants'].values()
          for p in v['policies'].values() for c in p.get('checks', [])]
audit = json.loads((root / 'precision_audit.json').read_text())
print('FP32 HIR cases:', len(checks), 'strict tolerance passes:', sum(c['passed'] for c in checks))
print('FP64 formula cases:', len(audit), 'passes:', sum(x['formula']['float64']['passed_original_tolerance_against_fp64_reference'] for x in audit))
print('Weak/zero-decay FP32 mismatches remain recorded; this is an experiment, not GPU performance validation.')
if not all(x['formula']['float64']['passed_original_tolerance_against_fp64_reference'] for x in audit):
    raise SystemExit('FP64 formula validation failed')
PY
