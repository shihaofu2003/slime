#!/usr/bin/env bash

set -euo pipefail

export AGENT_TOOL_CALL_PARSER="qwen3_coder"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
OFFICIAL_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/official"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B"
export MODEL_NAME="Qwen3.5-4B"
export NUM_TASKS="${NUM_TASKS:-1}"
export NUM_TRIALS="${NUM_TRIALS:-1}"
export TASK_SPLIT="${TASK_SPLIT:-test}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-8192}"
export SAVE_PREFIX="tau2_official_smoke_${MODEL_NAME}"
export SUMMARY_OUTPUT="${OFFICIAL_DIR}/outputs/${MODEL_NAME}/smoke_summary.json"

exec bash "${OFFICIAL_DIR}/run_eval.sh"
