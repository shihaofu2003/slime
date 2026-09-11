#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-local-first-dependency-safe-k2-fieldreward"
CHECKPOINT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_local_first_dependency_safe_k2_fieldreward_lr2e6_20260803"
ITERATION="${1:-99}"
TRAINING_LOG="${2:-${TRAINING_LOG:-}}"

if [[ $# -gt 2 || ( "${ITERATION}" != "99" && "${ITERATION}" != "199" ) ]]; then
  echo "Usage: $0 [99|199] <training-run.log>" >&2
  exit 2
fi
if [[ -z "${TRAINING_LOG}" || ! -f "${TRAINING_LOG}" ]]; then
  echo "[ERROR] A completed training run log is required as argument 2 or TRAINING_LOG" >&2
  exit 1
fi

bash "${PROJECT_ROOT}/examples/tau2-bench/analysis/run_tau2_local_first_rl_comparison.sh" "${ITERATION}"

if [[ "${ITERATION}" == "99" ]]; then
  CANDIDATE="new_rl100"
  COMPARISON="${EXPERIMENT_DIR}/ITER99_COMPARISON.json"
  TRAJECTORIES="${EXPERIMENT_DIR}/trajectories/iter0000-0099.jsonl"
  OUTPUT="${EXPERIMENT_DIR}/ITER99_PROMOTION.json"
  EXTRA_ARGS=(--gate-mode promote100 --prior old_rl100)
else
  CANDIDATE="new_rl200"
  COMPARISON="${EXPERIMENT_DIR}/ITER199_COMPARISON.json"
  TRAJECTORIES="${EXPERIMENT_DIR}/trajectories/iter0100-0199.jsonl"
  OUTPUT="${EXPERIMENT_DIR}/ITER199_ELIGIBILITY.json"
  EXTRA_ARGS=(--gate-mode eligibility)
fi

exec python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/check_tau2_rl_promotion.py" gate \
  --comparison "${COMPARISON}" \
  --candidate "${CANDIDATE}" \
  --baseline new_sft \
  "${EXTRA_ARGS[@]}" \
  --training-log "${TRAINING_LOG}" \
  --trajectory-dump "${TRAJECTORIES}" \
  --trajectory-window 384 \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --expected-latest "${ITERATION}" \
  --output "${OUTPUT}"
