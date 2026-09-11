#!/usr/bin/env bash

set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
OFFICIAL_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/official"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-tau2-grpo-v1"
export MODEL_NAME="Qwen3-4B-tau2-grpo-v1"
export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft/iter_0006637_hf"
export USER_MODEL="Qwen3-4B-tau2-user-sft-iter0006637"
export NUM_TASKS=""
export NUM_TRIALS="4"
export TASK_SPLIT="${TASK_SPLIT:-test}"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_user_sft_${RUN_STAMP}"
export SUMMARY_OUTPUT="${OFFICIAL_DIR}/outputs/${MODEL_NAME}/user_sft_pass4_summary.json"

exec bash "${OFFICIAL_DIR}/run_eval.sh"
