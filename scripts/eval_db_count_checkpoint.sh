#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 <zero-based-checkpoint-iteration>" >&2
  exit 2
fi

CHECKPOINT_NUMBER=$((10#$1))
CHECKPOINT_ITERATION="${CHECKPOINT_NUMBER}"
STEP_COUNT=$((CHECKPOINT_NUMBER + 1))
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${DB_COUNT_PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime-credit-assignment}"
RUN_DIR="${DB_COUNT_RUN_DIR:?Set DB_COUNT_RUN_DIR to the training run directory}"
CHECKPOINT_ROOT="${RUN_DIR}/checkpoints"
printf -v PADDED '%07d' "${CHECKPOINT_NUMBER}"
ITER_DIR="${CHECKPOINT_ROOT}/iter_${PADDED}"
HF_DIR="${CHECKPOINT_ROOT}/iter_${PADDED}_hf"
EVAL_ROOT="${DB_COUNT_EVAL_ROOT:-${RUN_DIR}/eval/iter_${PADDED}}"

if [[ ! -f "${ITER_DIR}/metadata.json" || ! -f "${ITER_DIR}/.metadata" ]]; then
  echo "[ERROR] checkpoint iter_${PADDED} is incomplete: ${ITER_DIR}" >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}"
ORIGIN_HF_DIR="${DB_COUNT_ORIGIN_HF_DIR:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903/final_hf}"

if [[ ! -f "${HF_DIR}/config.json" ]]; then
  echo "[CONVERT] checkpoint_iter=${CHECKPOINT_ITERATION} step=${STEP_COUNT}"
  ITER_DIR="${ITER_DIR}" OUTPUT_DIR="${HF_DIR}" ORIGIN_HF_DIR="${ORIGIN_HF_DIR}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh"
else
  echo "[CONVERT] existing ${HF_DIR}"
fi

SUMMARY="${EVAL_ROOT}/summary.json"
if [[ -f "${SUMMARY}" ]]; then
  echo "[EVAL] existing ${SUMMARY}"
  exit 0
fi

export PROJECT_ROOT
export EXPERIMENT_DIR="${EVAL_ROOT}"
export MODEL_PATH="${HF_DIR}"
export MODEL_NAME="tau2-db-count-step${STEP_COUNT}"
export EVAL_LABEL="step${STEP_COUNT}-iter${CHECKPOINT_ITERATION}-four-domain-bm25"
export DOMAINS="airline,retail,telecom,banking_knowledge"
export DOMAIN_CONCURRENCY="airline:1,retail:2,telecom:2,banking_knowledge:4"
export GLOBAL_CONCURRENCY=9
export BORROW_COMPLETED_DOMAIN_SLOTS=1
export RETRIEVAL_CONFIG=bm25
export TASK_SPLIT=test
export NUM_TASKS="${NUM_TASKS-}"
export NUM_TRIALS="${NUM_TRIALS:-4}"
export SEED="${SEED:-300}"
export SUMMARY_OUTPUT="${SUMMARY}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh" full
