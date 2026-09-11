#!/usr/bin/env bash

set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
CHECKPOINT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_local_first_dependency_safe_k2_fieldreward_lr2e6_20260803"
ITERATION="${1:-${ITERATION:-99}}"

if [[ $# -gt 1 || ( "${ITERATION}" != "99" && "${ITERATION}" != "199" ) ]]; then
  echo "Usage: $0 [99|199]" >&2
  exit 2
fi

printf -v ITER_PADDED '%07d' "${ITERATION}"
export MODEL_PATH="${CHECKPOINT_ROOT}/iter_${ITER_PADDED}_hf"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Missing converted checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi

export MODEL_NAME="Qwen3-4B-tau2-agent-rl-local-first-dependency-safe-lr2e6-iter${ITER_PADDED}"
export AGENT_PROTOCOL_PROFILE="dependency-safe-multi"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_dependency_safe_multi_user_stop_parser_${RUN_STAMP}"
export SUMMARY_OUTPUT="${OFFICIAL_DIR}/outputs/${MODEL_NAME}/dependency_safe_multi_user_stop_parser_pass4_summary.json"

exec bash "${OFFICIAL_DIR}/models/run_full_tau2_agent_rl_stability_user_stop_parser.sh"
