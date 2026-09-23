#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"
SEED="${2:-300}"
if [[ "${MODE}" != "sft4505" && "${MODE}" != "opd" ]]; then
  echo "Usage: $0 <sft4505|opd> [seed]" >&2
  exit 2
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] seed must be an integer: ${SEED}" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tau2-opd-airline-retail-pilot}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
USER_ENDPOINT_FILE="${USER_ENDPOINT_FILE:-${EXPERIMENT_DIR}/user_endpoint.env}"

if [[ -s "${USER_ENDPOINT_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${USER_ENDPOINT_FILE}"
fi
: "${TAU2_USER_API_BASE:?User endpoint is missing; set TAU2_USER_API_BASE or USER_ENDPOINT_FILE}"
: "${TAU2_USER_API_KEY:=EMPTY}"

SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920}"
if [[ "${MODE}" == "sft4505" ]]; then
  MODEL_PATH="${MODEL_PATH:-${SFT_CKPT_ROOT}/iter_0004505_hf}"
  MODEL_NAME="${MODEL_NAME:-Qwen3-4B-tau2-opd-sft4505}"
  EVAL_LABEL="sft4505"
else
  TAU2_OPD_RUN_TAG="${TAU2_OPD_RUN_TAG-current-bounded-lr2e6}"
  ITERATION="${TAU2_OPD_EVAL_ITERATION:-59}"
  printf -v padded '%07d' "${ITERATION}"
  RUN_TAG="${TAU2_OPD_RUN_TAG:+-${TAU2_OPD_RUN_TAG}}"
  MODEL_PATH="${MODEL_PATH:-${EXPERIMENT_DIR}/pilot${RUN_TAG}/checkpoints/iter_${padded}_hf}"
  MODEL_NAME="${MODEL_NAME:-Qwen3-4B-tau2-opd${RUN_TAG}-iter${ITERATION}}"
  EVAL_LABEL="opd${RUN_TAG}-iter${ITERATION}"
fi
[[ -f "${MODEL_PATH}/config.json" ]] || {
  echo "[ERROR] Hugging Face checkpoint is not ready: ${MODEL_PATH}" >&2
  exit 1
}

# The external User service is the independent four-GPU job. This evaluation
# job only allocates Agent GPUs; four TP=1 replicas are a useful default for a
# four-GPU worker, and callers may override the group list for a larger job.
export EXPERIMENT_DIR MODEL_PATH MODEL_NAME EVAL_LABEL
export AGENT_SERVED_MODEL_NAME="${MODEL_NAME}"
export AGENT_EVAL_MODE=official-native
export AGENT_TOOL_CALL_PARSER=qwen
export AGENT_REPLICA_CUDA_GROUPS="${AGENT_REPLICA_CUDA_GROUPS:-0;1;2;3}"
export AGENT_WORKER_PORT_BASE="${AGENT_WORKER_PORT_BASE:-31000}"
export AGENT_ROUTER_POLICY=cache_aware
export USER_SGLANG=0
export TAU2_USER_API_BASE TAU2_USER_API_KEY
export USER_MODEL="${TAU2_USER_MODEL:-Qwen3.6-27B-tau2-user-nonthinking}"
export USER_TEMPERATURE=0.0
export USER_MAX_TOKENS=512
export USER_TOP_P=1.0
export USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'

export DOMAINS="${DOMAINS:-airline,retail}"
export TASK_SPLIT=test
export NUM_TASKS=""
export NUM_TRIALS=4
export SEED="${SEED}"
export MAX_STEPS=200
export MAX_ERRORS=10
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
export DOMAIN_CONCURRENCY="${DOMAIN_CONCURRENCY:-airline:2,retail:2}"
export GLOBAL_CONCURRENCY="${GLOBAL_CONCURRENCY:-4}"
export PARALLEL_DOMAINS=1
export BORROW_COMPLETED_DOMAIN_SLOTS=1
export AGENT_TEMPERATURE=0.6
export AGENT_TOP_P=1.0
export AGENT_MAX_TOKENS=1200
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="${SAVE_PREFIX:-tau2_opd_${EVAL_LABEL}_${MODEL_NAME}_seed${SEED}_${RUN_STAMP}}"
export SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed${SEED}_${RUN_STAMP}_summary.json}"

echo "[tau2-opd-eval] mode=${MODE} model=${MODEL_PATH} user=${TAU2_USER_API_BASE} domains=${DOMAINS} seed=${SEED} trials=${NUM_TRIALS}"
exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
