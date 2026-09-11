#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
ARM="${TAU2_BOUNDARY_SFT_ARM:-contract-only}"
if [[ $# -gt 0 && "$1" =~ ^(contract-only|contract-boundary)$ ]]; then
  ARM="$1"
  shift
fi
case "${ARM}" in
  contract-only) SUFFIX="contract_only" ;;
  contract-boundary) SUFFIX="contract_boundary" ;;
  *)
    echo "Usage: $0 [contract-only|contract-boundary] [--force|--dry-run-only]" >&2
    exit 2
    ;;
esac

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_${SUFFIX}_20260804}"
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

exec bash "${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
