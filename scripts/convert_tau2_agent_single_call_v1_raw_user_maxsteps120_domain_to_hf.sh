#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
ARM="${1:-}"
ITERATION="${2:-}"

if [[ $# -lt 2 || $# -gt 3 || ! "${ARM}" =~ ^(mixed|airline|retail|telecom)$ || ! "${ITERATION}" =~ ^(109|119|129)$ ]]; then
  echo "Usage: $0 <mixed|airline|retail|telecom> <109|119|129> [--force|--dry-run-only]" >&2
  exit 2
fi

export CHECKPOINT_ROOT="${TAU2_RL_SOURCE_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_from_iter99_domain_${ARM}_20260816}"
CONVERT_ARGS=("${ITERATION}")
if [[ $# -eq 3 ]]; then
  CONVERT_ARGS+=("$3")
fi
exec bash "${PROJECT_ROOT}/scripts/convert_tau2_agent_rl_single_call_v1_iter_to_hf.sh" "${CONVERT_ARGS[@]}"
