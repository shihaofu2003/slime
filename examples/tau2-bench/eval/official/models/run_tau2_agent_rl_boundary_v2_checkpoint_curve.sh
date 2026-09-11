#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(9|39|69|89)$ ]]; then
  echo "Usage: $0 9|39|69|89" >&2
  exit 2
fi

ITERATION="$1"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CURVE_HF_ROOT="${TAU2_RL_CURVE_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_checkpoint_curve_hf_20260805}"
printf -v ITER_PADDED '%07d' "${ITERATION}"
HF_DIR="${CURVE_HF_ROOT}/iter_${ITER_PADDED}_hf"

if [[ ! -f "${HF_DIR}/config.json" ]]; then
  CONVERSION_ARGS=(curve "${ITERATION}")
  if [[ -d "${HF_DIR}" ]]; then
    CONVERSION_ARGS+=(--force)
  fi
  bash "${PROJECT_ROOT}/scripts/convert_tau2_agent_rl_boundary_v2_iter_to_hf.sh" \
    "${CONVERSION_ARGS[@]}"
else
  echo "[INFO] reusing completed curve conversion: ${HF_DIR}"
fi

exec bash \
  "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_tau2_agent_rl_boundary_v2_user_stop_parser.sh" \
  curve "${ITERATION}" 300 no-probe
