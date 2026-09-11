#!/usr/bin/env bash

set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
OFFICIAL_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/official"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714/iter_0002413_hf"
export MODEL_NAME="Qwen3-4B-tau2-agent-sft-multitool-iter0002413"
export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf"
export USER_MODEL="Qwen3-4B-tau2-user-sft-stop-iter0006899"
export NUM_TASKS=""
export NUM_TRIALS="4"
export TASK_SPLIT="${TASK_SPLIT:-test}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-8192}"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_user_stop_${RUN_STAMP}"
export SUMMARY_OUTPUT="${OFFICIAL_DIR}/outputs/${MODEL_NAME}/user_stop_pass4_summary.json"

exec bash "${OFFICIAL_DIR}/run_eval.sh"
