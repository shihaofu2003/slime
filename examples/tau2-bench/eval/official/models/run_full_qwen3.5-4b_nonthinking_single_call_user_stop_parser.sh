#!/usr/bin/env bash

set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-qwen35-nonthinking-eval"
SEED="${1:-${SEED:-300}}"

if [[ $# -gt 1 || ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [seed]" >&2
  exit 2
fi

export MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B-nonthinking-single-call}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"
export NUM_TASKS=""
export NUM_TRIALS=4
export SEED
export TASK_SPLIT="${TASK_SPLIT:-test}"
export AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.6}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-8192}"
export MAX_STEPS="${MAX_STEPS:-200}"
export AGENT_PROTOCOL_PROFILE="current-single"
export AGENT_LLM_ARGS_JSON='{"enable_thinking":false}'
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${EXPERIMENT_DIR}/eval/seed${SEED}_summary.json}"

exec bash "${OFFICIAL_DIR}/run_eval.sh"
