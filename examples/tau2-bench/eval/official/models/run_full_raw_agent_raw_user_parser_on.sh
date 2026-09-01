#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-raw-agent-raw-user-parser-on"

MODEL_KIND="${1:-}"
SEED="${2:-300}"
USER_KIND="${3:-raw}"

if [[ $# -gt 3 || ! "${MODEL_KIND}" =~ ^(qwen3|qwen35-thinking|qwen35-nonthinking)$ || ! "${SEED}" =~ ^[0-9]+$ || ! "${USER_KIND}" =~ ^(raw|v1)$ ]]; then
  echo "Usage: $0 <qwen3|qwen35-thinking|qwen35-nonthinking> [seed] [raw|v1]" >&2
  exit 2
fi

export MODEL_PATH=""
export MODEL_NAME=""
export AGENT_MAX_TOKENS="1200"
export AGENT_PROTOCOL_PROFILE=""
export AGENT_LLM_ARGS_JSON=""

case "${MODEL_KIND}" in
  qwen3)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
    export MODEL_NAME="Qwen3-4B-raw-agent-raw-user-parser-on"
    ;;
  qwen35-thinking)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B"
    export MODEL_NAME="Qwen3.5-4B-thinking-raw-agent-raw-user-parser-on"
    export AGENT_MAX_TOKENS="8192"
    ;;
  qwen35-nonthinking)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B"
    export MODEL_NAME="Qwen3.5-4B-nonthinking-raw-agent-raw-user-parser-on"
    export AGENT_MAX_TOKENS="8192"
    export AGENT_PROTOCOL_PROFILE="current-single"
    export AGENT_LLM_ARGS_JSON='{"enable_thinking":false}'
    ;;
esac

# The parser converts Qwen <tool_call> output into structured User tool calls;
# it is explicit for both User checkpoints rather than inherited.
if [[ "${USER_KIND}" == "raw" ]]; then
  export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
  export USER_MODEL="Qwen3-4B-Instruct-2507-raw-user"
  USER_OUTPUT_DIR="${MODEL_KIND}"
else
  export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf"
  export USER_MODEL="Qwen3-4B-tau2-user-sft-stop-iter0006899"
  USER_OUTPUT_DIR="${MODEL_KIND}-v1-user"
fi
export USER_SGLANG_EXTRA_ARGS="--tool-call-parser qwen"
export USER_TEMPERATURE="0.0"
export USER_MAX_TOKENS="512"

export DOMAINS="airline,retail,telecom"
export TASK_SPLIT="test"
export NUM_TASKS=""
export NUM_TRIALS="4"
export SEED
export MAX_STEPS="200"
export MAX_ERRORS="10"
export MAX_CONCURRENCY="1"
export AGENT_TEMPERATURE="0.6"
export AGENT_TOP_P="1.0"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_${USER_KIND}-user_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${USER_OUTPUT_DIR}/seed${SEED}_summary.json"
export NAMESPACE_PROBE_OUTPUT=""

exec bash "${OFFICIAL_DIR}/run_eval.sh"
