#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(thinking|nonthinking)$ ]]; then
  echo "Usage: $0 <thinking|nonthinking>" >&2
  exit 2
fi

MODE="$1"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
FOUR_DOMAIN_WRAPPER="${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B"
export MODEL_NAME="Qwen3.5-4B-${MODE}-qwen36-user-async"
export AGENT_EVAL_MODE="official-native"
export AGENT_TOOL_CALL_PARSER="qwen3_coder"
export AGENT_MAX_TOKENS="8192"
export RETRIEVAL_CONFIG="${RETRIEVAL_CONFIG:-bm25}"
export EVAL_LABEL="${EVAL_LABEL:-four-domain-full-${RETRIEVAL_CONFIG}-${MODE}}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-eval-qwen35-official-native-four-domain}"

case "${MODE}" in
  thinking)
    export AGENT_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":true}}'
    ;;
  nonthinking)
    export AGENT_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
    ;;
esac

exec bash "${FOUR_DOMAIN_WRAPPER}" full
