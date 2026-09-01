#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_local_first_relaxed_20260803}"
LATEST_FILE="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"

[[ -f "${LATEST_FILE}" ]] || {
  echo "[ERROR] Missing latest iteration file: ${LATEST_FILE}" >&2
  exit 1
}
LATEST_ITERATION="$(tr -d '[:space:]' < "${LATEST_FILE}")"
[[ "${LATEST_ITERATION}" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] Invalid latest iteration: ${LATEST_ITERATION}" >&2
  exit 1
}
printf -v ITER_NAME 'iter_%07d' "${LATEST_ITERATION}"
export ITER_DIR="${ITER_DIR:-${CHECKPOINT_ROOT}/${ITER_NAME}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/final_hf}"

exec bash "${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
