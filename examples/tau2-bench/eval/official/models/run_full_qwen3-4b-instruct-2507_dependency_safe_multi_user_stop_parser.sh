#!/usr/bin/env bash

set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
OFFICIAL_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/official"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
export MODEL_NAME="Qwen3-4B-Instruct-2507"
export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf"
export USER_MODEL="Qwen3-4B-tau2-user-sft-stop-iter0006899"
export NUM_TASKS=""
export NUM_TRIALS="4"
export TASK_SPLIT="${TASK_SPLIT:-test}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-8192}"
export AGENT_PROTOCOL_PROFILE="dependency-safe-multi"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_dependency_safe_multi_user_stop_parser_${RUN_STAMP}"
export SUMMARY_OUTPUT="${OFFICIAL_DIR}/outputs/${MODEL_NAME}/dependency_safe_multi_user_stop_parser_pass4_summary.json"

exec bash "${OFFICIAL_DIR}/run_eval.sh"
