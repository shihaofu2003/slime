#!/usr/bin/env bash
set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"

if [[ -z "${CHECKPOINT_ROOT:-}" ]]; then
  echo "[ERROR] CHECKPOINT_ROOT is required" >&2
  exit 1
fi

export ITER_DIR="${CHECKPOINT_ROOT}/iter_0000099"
export OUTPUT_DIR="${CHECKPOINT_ROOT}/iter_0000099_hf"

exec bash "${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
