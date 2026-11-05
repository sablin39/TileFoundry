#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=''
mkdir -p logs results/full_gdn
OMP_NUM_THREADS=1 "${PYTHON:-python}" -m pytest synthesis -q --junitxml=results/full_gdn/tests.xml
OMP_NUM_THREADS=8 "${PYTHON:-python}" -u -m synthesis.full_layer_run "$@" | tee logs/full_gdn.log
