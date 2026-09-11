#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <smoke|full> <checkpoint-dir> [--force|--dry-run-only]" >&2
  exit 2
fi

MODE="$1"
CHECKPOINT="$2"
shift 2

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_BASE="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"

case "${MODE}" in
  smoke)
    if [[ "${CHECKPOINT}" != "iter_0000001" ]]; then
      echo "[ERROR] smoke conversion requires iter_0000001" >&2
      exit 2
    fi
    RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_longest32_smoke_v2_20260903"
    HF_BASENAME=final_hf
    ;;
  full)
    case "${CHECKPOINT}" in
      iter_0000399|iter_0000799|iter_0001199|iter_0001599|iter_0001897|iter_0001999|iter_0002399|iter_0002799|iter_0003199|iter_0003599)
        HF_BASENAME="${CHECKPOINT}_hf"
        ;;
      iter_0003795)
        HF_BASENAME=final_hf
        ;;
      *)
        echo "[ERROR] unsupported full checkpoint: ${CHECKPOINT}" >&2
        exit 2
        ;;
    esac
    RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903"
    ;;
  *)
    echo "[ERROR] mode must be smoke or full" >&2
    exit 2
    ;;
esac

export PROJECT_ROOT
export ITER_DIR="${RUN_ROOT}/${CHECKPOINT}"
export OUTPUT_DIR="${RUN_ROOT}/${HF_BASENAME}"
export ORIGIN_HF_DIR="${MODEL_ROOT}/Qwen3-4B-Instruct-2507"

exec bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
