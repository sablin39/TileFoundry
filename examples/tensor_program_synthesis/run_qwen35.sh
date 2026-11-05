#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=''
mkdir -p logs results/qwen35
OMP_NUM_THREADS=1 "${PYTHON:-python}" -m pytest synthesis -q --junitxml=results/qwen35/tests.xml
OMP_NUM_THREADS=8 "${PYTHON:-python}" -u -m synthesis.qwen35_run "$@" | tee logs/qwen35.log
