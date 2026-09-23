#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RECIPE="${1:-progress-db-count-v1}"
export TAU2_RUN_NAME="${RUN_STAMP}-${RECIPE}-train"
export TAU2_SERIAL_CLEANUP=1
experiment="${PROJECT_ROOT}/output/experiments/tau2-async-db-count-four-domain"
run_dir="${experiment}/arms/async/${TAU2_RUN_NAME}"
mkdir -p "${run_dir}"
cd "${PROJECT_ROOT}"
python3 -m pytest tests/test_tau2_continuous.py -k four_domain_monitor -q
env -u LOAD_DIR -u CKPT_STEP bash scripts/run_tau2_async_four_domain.sh "${RECIPE}" train \
  2>&1 | tee "${run_dir}/run.log"
ray stop --force
eval_dir="${experiment}/eval/${RECIPE}-iter555"
mkdir -p "${eval_dir}"
bash scripts/eval_tau2_async_four_domain.sh "${RECIPE}" 555 \
  2>&1 | tee "${eval_dir}/run_${RUN_STAMP}.log"
