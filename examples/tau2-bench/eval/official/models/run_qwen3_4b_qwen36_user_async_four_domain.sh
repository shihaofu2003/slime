#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(smoke|full)$ ]]; then
  echo "Usage: $0 <smoke|full>" >&2
  exit 2
fi

MODE="$1"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
BASE_WRAPPER="${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh"

export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-eval-qwen36-user-four-domain}"
export DOMAINS="airline,retail,telecom,banking_knowledge"
export DOMAIN_CONCURRENCY="airline:1,retail:2,telecom:2,banking_knowledge:4"
export GLOBAL_CONCURRENCY="9"
export BORROW_COMPLETED_DOMAIN_SLOTS="1"
export RETRIEVAL_CONFIG="${RETRIEVAL_CONFIG:-bm25}"

case "${MODE}" in
  smoke)
    export NUM_TASKS="${NUM_TASKS:-3}"
    export NUM_TRIALS="${NUM_TRIALS:-1}"
    export EVAL_LABEL="${EVAL_LABEL:-four-domain-smoke-${RETRIEVAL_CONFIG}}"
    ;;
  full)
    export NUM_TASKS="${NUM_TASKS-}"
    export NUM_TRIALS="${NUM_TRIALS:-4}"
    export EVAL_LABEL="${EVAL_LABEL:-four-domain-full-${RETRIEVAL_CONFIG}}"
    ;;
esac

exec bash "${BASE_WRAPPER}"
