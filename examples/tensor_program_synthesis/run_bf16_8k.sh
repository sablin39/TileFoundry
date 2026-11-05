#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=''
mkdir -p logs results/full_gdn_bf16_8k
OMP_NUM_THREADS=1 "${PYTHON:-python}" -m pytest synthesis -q --junitxml=results/full_gdn_bf16_8k/tests.xml
OMP_NUM_THREADS=8 "${PYTHON:-python}" -u -m synthesis.mixed_layer_run "$@" | tee logs/full_gdn_bf16_8k.log
