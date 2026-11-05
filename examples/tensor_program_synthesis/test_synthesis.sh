#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$task_root"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
mkdir -p results/synthesis
exec "${PYTHON:-python}" -m pytest synthesis/test_synthesis.py -q \
    --junitxml=results/synthesis/tests.xml "$@"
