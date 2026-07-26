#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-}"

# submit.sh copies this script into output/jobs/... before execution, so deriving
# the repo root from BASH_SOURCE would point at the log directory. Prefer the
# injected PROJECT_ROOT and only fall back to a relative path for direct runs.
if [[ -z "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3-4B}"
SAVE_DIR="${SAVE_DIR:-/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3-4B_torch_dist}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run-only)
      DRY_RUN_ONLY="1"
      shift 1
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${PROJECT_ROOT}" || ! -f "${PROJECT_ROOT}/tools/convert_hf_to_torch_dist.py" ]]; then
  echo "[ERROR] Invalid PROJECT_ROOT: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "[ERROR] Hugging Face checkpoint directory not found: ${HF_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"

# Load Qwen3-4B Megatron model arguments used by the official slime examples.
source "${PROJECT_ROOT}/scripts/models/qwen3-4B.sh"

cd "${PROJECT_ROOT}"

echo "[INFO] Project root : ${PROJECT_ROOT}"
echo "[INFO] HF checkpoint: ${HF_CHECKPOINT}"
echo "[INFO] Save dir     : ${SAVE_DIR}"
echo "[INFO] NPROC        : ${NPROC_PER_NODE}"
echo "[INFO] Python       : ${PYTHON_BIN}"
echo "[INFO] Megatron-LM  : ${MEGATRON_LM_PATH}"

export PYTHONPATH="${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

CONVERT_CMD=(
  "${PYTHON_BIN}"
  tools/convert_hf_to_torch_dist.py
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --save "${SAVE_DIR}"
)

if [[ "${NPROC_PER_NODE}" != "1" ]]; then
  CONVERT_CMD=(
    torchrun
    --nproc_per_node "${NPROC_PER_NODE}"
    tools/convert_hf_to_torch_dist.py
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "${HF_CHECKPOINT}"
    --save "${SAVE_DIR}"
  )
fi

printf '[INFO] Command      : '
printf '%q ' "${CONVERT_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
  echo "[INFO] DRY_RUN_ONLY=1, skipping actual conversion."
  exit 0
fi

"${CONVERT_CMD[@]}"
