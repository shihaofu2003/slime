#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint-dir> [--force|--dry-run-only]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_BASE="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_turn_filtered_20260908"

case "${CHECKPOINT}" in
  iter_0000399|iter_0000799|iter_0001199|iter_0001599|iter_0001855|iter_0001999|iter_0002399|iter_0002799|iter_0003199|iter_0003599)
    HF_BASENAME="${CHECKPOINT}_hf"
    ;;
  iter_0003711)
    HF_BASENAME=final_hf
    ;;
  *)
    echo "[ERROR] unsupported checkpoint: ${CHECKPOINT}" >&2
    exit 2
    ;;
esac

export PROJECT_ROOT
export ITER_DIR="${RUN_ROOT}/${CHECKPOINT}"
export OUTPUT_DIR="${RUN_ROOT}/${HF_BASENAME}"
export ORIGIN_HF_DIR="${MODEL_ROOT}/Qwen3-4B-Instruct-2507"

exec bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
