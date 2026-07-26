#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-}"

# submit.sh copies this script into output/jobs/... before execution, so prefer
# the injected PROJECT_ROOT and only fall back to a relative path for direct runs.
if [[ -z "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_sft_areal_strict_no_thinking_epoch2_20260713}"
ITER_DIR="${ITER_DIR:-${CHECKPOINT_ROOT}/iter_0000881}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/iter_0000881_hf}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
FORCE="${FORCE:-0}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
VOCAB_SIZE="${VOCAB_SIZE:-151936}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE="1"
      shift 1
      ;;
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

if [[ ! -d "${PROJECT_ROOT}" || ! -f "${PROJECT_ROOT}/tools/convert_torch_dist_to_hf.py" ]]; then
  echo "[ERROR] Invalid PROJECT_ROOT: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${ITER_DIR}" || ! -f "${ITER_DIR}/common.pt" || ! -f "${ITER_DIR}/.metadata" ]]; then
  echo "[ERROR] Invalid Megatron checkpoint iter dir: ${ITER_DIR}" >&2
  exit 1
fi

if [[ ! -d "${ORIGIN_HF_DIR}" || ! -f "${ORIGIN_HF_DIR}/config.json" ]]; then
  echo "[ERROR] Invalid origin Hugging Face directory: ${ORIGIN_HF_DIR}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${MEGATRON_LM_PATH}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONVERT_CMD=(
  "${PYTHON_BIN}"
  tools/convert_torch_dist_to_hf.py
  --input-dir "${ITER_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --origin-hf-dir "${ORIGIN_HF_DIR}"
  --add-missing-from-origin-hf
  --vocab-size "${VOCAB_SIZE}"
)

if [[ "${FORCE}" == "1" ]]; then
  CONVERT_CMD+=(--force)
fi

echo "[INFO] Project root : ${PROJECT_ROOT}"
echo "[INFO] Input dir    : ${ITER_DIR}"
echo "[INFO] Output dir   : ${OUTPUT_DIR}"
echo "[INFO] Origin HF    : ${ORIGIN_HF_DIR}"
echo "[INFO] Python       : ${PYTHON_BIN}"
echo "[INFO] Megatron-LM  : ${MEGATRON_LM_PATH}"
echo "[INFO] Vocab size   : ${VOCAB_SIZE}"
printf '[INFO] Command      : '
printf '%q ' "${CONVERT_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
  echo "[INFO] DRY_RUN_ONLY=1, skipping actual conversion."
  exit 0
fi

"${CONVERT_CMD[@]}"
