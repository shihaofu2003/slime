#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^(sft|vanilla-grpo|progress-db-count-v1)$ ]]; then
  echo "Usage: $0 <sft|vanilla-grpo|progress-db-count-v1> [iteration]" >&2
  exit 2
fi

LABEL="$1"
ITERATION="${2:-555}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_AGENT_ROOT="$(dirname "${PROJECT_ROOT}")"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-async-db-count-four-domain"
RUN_NAME="${TAU2_RUN_NAME:-}"

if [[ "${LABEL}" == sft ]]; then
  ITERATION=4673
  MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909/iter_0004673_hf"
else
  printf -v padded '%07d' "${ITERATION}"
  if [[ -z "${RUN_NAME}" ]]; then
    shopt -s nullglob
    candidates=("${EXPERIMENT_DIR}"/arms/async/*-"${LABEL}"-train/checkpoints/iter_"${padded}")
    if [[ ${#candidates[@]} -ne 1 ]]; then
      echo "Expected one ${LABEL} training run at iter${ITERATION}; set TAU2_RUN_NAME explicitly (found ${#candidates[@]})." >&2
      exit 2
    fi
    CHECKPOINT_ROOT="$(dirname "${candidates[0]}")"
  else
    CHECKPOINT_ROOT="${EXPERIMENT_DIR}/arms/async/${RUN_NAME}/checkpoints"
  fi
  MODEL_PATH="${CHECKPOINT_ROOT}/iter_${padded}_hf"
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    ITER_DIR="${CHECKPOINT_ROOT}/iter_${padded}" OUTPUT_DIR="${MODEL_PATH}" \
      ORIGIN_HF_DIR="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909/iter_0004673_hf" \
      bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh"
  fi
fi

export PROJECT_ROOT SERVICE_AGENT_ROOT EXPERIMENT_DIR MODEL_PATH
export MODEL_NAME="Qwen3-4B-${LABEL}-iter${ITERATION}"
export EVAL_LABEL="${LABEL}-iter${ITERATION}"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/trajectories/seed300_${RUN_STAMP}"
export DOMAINS="airline,retail,telecom,banking_knowledge"
export DOMAIN_CONCURRENCY="airline:1,retail:2,telecom:2,banking_knowledge:4"
export GLOBAL_CONCURRENCY=9
export BORROW_COMPLETED_DOMAIN_SLOTS=1
export RETRIEVAL_CONFIG=bm25
export NUM_TASKS=""
export AGENT_MAX_TOKENS=1200
export NUM_TRIALS=4
export SEED=300

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh" full
