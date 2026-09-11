#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
export MODEL_NAME="Qwen3-4B-Instruct-2507"
export VITA_AGENT_ENABLE_THINKING=false
export VITA_AGENT_TOOL_CALL_PARSER=qwen25
export VITA_RESULT_DIR="${VITA_RESULT_DIR:-${PROJECT_ROOT}/output/experiments/vitabench-qwen3-4b-instruct-2507-full-eval/artifacts}"

exec bash "${PROJECT_ROOT}/examples/vita-bench/run_qwen3_5_4b_full_eval_local_roles.sh" "$@"
