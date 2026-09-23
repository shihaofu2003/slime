#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
RUN_STAMP="${RUN_STAMP:-20260915_092700}"
EXPERIMENT_DIR=output/experiments/tau2-async-db-count-four-domain
for recipe in vanilla-grpo progress-db-count-v1; do
  python3 scripts/plot_tau2_four_domain_rewards.py \
    --run-dir "${EXPERIMENT_DIR}/arms/async/${RUN_STAMP}-${recipe}-train" \
    --output-dir "${EXPERIMENT_DIR}/plots/${RUN_STAMP}/${recipe}" \
    --label "${recipe}"
done
