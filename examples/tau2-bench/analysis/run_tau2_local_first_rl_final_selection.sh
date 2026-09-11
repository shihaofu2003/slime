#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-local-first-dependency-safe-k2-fieldreward"

exec python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/check_tau2_rl_promotion.py" select \
  --comparison "${EXPERIMENT_DIR}/ITER199_COMPARISON.json" \
  --eligibility "new_rl100=${EXPERIMENT_DIR}/ITER99_PROMOTION.json" \
  --eligibility "new_rl200=${EXPERIMENT_DIR}/ITER199_ELIGIBILITY.json" \
  --iteration new_rl100=99 \
  --iteration new_rl200=199 \
  --output "${EXPERIMENT_DIR}/FINAL_SELECTION.json"
