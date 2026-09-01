#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809}"
LATEST_FILE="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"

if [[ ! -f "${LATEST_FILE}" ]]; then
  echo "[ERROR] missing latest iteration file: ${LATEST_FILE}" >&2
  exit 1
fi
LATEST_ITERATION="$(tr -d '[:space:]' < "${LATEST_FILE}")"
if [[ ! "${LATEST_ITERATION}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] invalid latest iteration: ${LATEST_ITERATION}" >&2
  exit 1
fi

printf -v ITER_NAME 'iter_%07d' "${LATEST_ITERATION}"
export ITER_DIR="${ITER_DIR:-${CHECKPOINT_ROOT}/${ITER_NAME}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/final_hf}"
export ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"

exec bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
