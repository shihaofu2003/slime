#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_20260809}"
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809}"
ITERATION="${1:-99}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! "${ITERATION}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [iteration] [--force|--dry-run-only]" >&2
  exit 2
fi

LATEST_FILE="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
if [[ ! -f "${LATEST_FILE}" ]]; then
  echo "[ERROR] missing latest checkpoint marker: ${LATEST_FILE}" >&2
  exit 1
fi
LATEST="$(tr -d '[:space:]' < "${LATEST_FILE}")"
if [[ ! "${LATEST}" =~ ^[0-9]+$ || "${LATEST}" -lt "${ITERATION}" ]]; then
  echo "[ERROR] iteration ${ITERATION} is incomplete; latest=${LATEST}" >&2
  exit 1
fi
if [[ ! -d "${SFT_CKPT_ROOT}/final_hf" ]]; then
  echo "[ERROR] missing SFT origin: ${SFT_CKPT_ROOT}/final_hf" >&2
  exit 1
fi

printf -v ITER_PADDED '%07d' "${ITERATION}"
export ITER_DIR="${CHECKPOINT_ROOT}/iter_${ITER_PADDED}"
export OUTPUT_DIR="${CHECKPOINT_ROOT}/iter_${ITER_PADDED}_hf"
export ORIGIN_HF_DIR="${SFT_CKPT_ROOT}/final_hf"

exec bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
