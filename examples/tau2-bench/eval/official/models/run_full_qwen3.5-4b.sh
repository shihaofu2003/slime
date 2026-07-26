#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
OFFICIAL_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/official"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B"
export MODEL_NAME="Qwen3.5-4B"
export NUM_TASKS=""
export NUM_TRIALS="4"
export TASK_SPLIT="${TASK_SPLIT:-test}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-8192}"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${OFFICIAL_DIR}/outputs/${MODEL_NAME}/pass4_summary.json"

exec bash "${OFFICIAL_DIR}/run_eval.sh"
