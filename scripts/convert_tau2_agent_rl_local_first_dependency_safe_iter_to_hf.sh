#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_local_first_dependency_safe_k2_fieldreward_lr2e6_20260803"
SFT_CKPT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_local_first_relaxed_20260803"
ITERATION="${ITERATION:-99}"

if [[ $# -gt 0 && ( "$1" == "99" || "$1" == "199" ) ]]; then
  ITERATION="$1"
  shift
fi
if [[ "${ITERATION}" != "99" && "${ITERATION}" != "199" ]]; then
  echo "Usage: $0 [99|199] [--force|--dry-run-only]" >&2
  exit 2
fi

printf -v ITER_PADDED '%07d' "${ITERATION}"
LATEST_FILE="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
if [[ ! -f "${LATEST_FILE}" ]]; then
  echo "[ERROR] Missing checkpoint marker: ${LATEST_FILE}" >&2
  exit 1
fi
LATEST="$(tr -d '[:space:]' < "${LATEST_FILE}")"
if [[ ! "${LATEST}" =~ ^[0-9]+$ || "${LATEST}" -lt "${ITERATION}" ]]; then
  echo "[ERROR] iteration ${ITERATION} is not complete; latest=${LATEST}" >&2
  exit 1
fi

export CHECKPOINT_ROOT
export ITER_DIR="${CHECKPOINT_ROOT}/iter_${ITER_PADDED}"
export OUTPUT_DIR="${CHECKPOINT_ROOT}/iter_${ITER_PADDED}_hf"
export ORIGIN_HF_DIR="${SFT_CKPT_ROOT}/final_hf"

exec bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
