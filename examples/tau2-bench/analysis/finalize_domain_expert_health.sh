#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <airline|retail|telecom|mixed> <109|119|129> TRAINING_LOG [RESUME_LOG ...]" >&2
  exit 2
fi

TARGET="$1"
ITERATION="$2"
shift 2
if [[ ! "${TARGET}" =~ ^(airline|retail|telecom|mixed)$ || ! "${ITERATION}" =~ ^(109|119|129)$ ]]; then
  echo "[ERROR] target must be airline|retail|telecom|mixed and iteration must be 109|119|129" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts"
LONG100_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
LONG200_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long200"
SOURCE_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
SOURCE_HEALTH="${TAU2_RL_LONG100_ITER99_GATE:-${LONG100_EXPERIMENT}/ITER99_HEALTH_GATE.json}"

if [[ "${TARGET}" == "mixed" ]]; then
  VARIANT="mixed-control"
  CHECKPOINT_ROOT="${TAU2_RL_LONG200_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805}"
  LINEAGE="${CHECKPOINT_ROOT}/LONG200_LINEAGE.json"
  OUTPUT="${HEALTH_OUTPUT:-${EXPERIMENT_DIR}/ITER${ITERATION}_PREFIX_HEALTH_GATE.json}"
  DOMAIN_ARGS=()
  mapfile -t TRAJECTORIES < <(
    find "${LONG200_EXPERIMENT}/trajectories" -maxdepth 1 -type f \
      -name 'long200-final_iter0100-0199_*.jsonl' -print | sort
    find "${LONG200_EXPERIMENT}/trajectories" -maxdepth 1 -type f \
      -name 'long200-final-kl-waiver_iter0100-0199_*.jsonl' -print | sort
  )
else
  VARIANT="domain-expert"
  case "${TARGET}" in
    airline) CHECKPOINT_ROOT="${TAU2_RL_AIRLINE_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_airline_expert_20260806}" ;;
    retail) CHECKPOINT_ROOT="${TAU2_RL_RETAIL_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_retail_expert_20260806}" ;;
    telecom) CHECKPOINT_ROOT="${TAU2_RL_TELECOM_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_telecom_expert_20260806}" ;;
  esac
  LINEAGE="${CHECKPOINT_ROOT}/DOMAIN_EXPERT_LINEAGE.json"
  UPPER_TARGET="${TARGET^^}"
  OUTPUT="${HEALTH_OUTPUT:-${EXPERIMENT_DIR}/${UPPER_TARGET}_ITER${ITERATION}_HEALTH_GATE.json}"
  DOMAIN_ARGS=(--domain "${TARGET}")
  mapfile -t TRAJECTORIES < <(
    find "${EXPERIMENT_DIR}/trajectories" -maxdepth 1 -type f \
      -name "domain-expert_${TARGET}_iter0100-0129_*.jsonl" -print | sort
  )
fi

if [[ ${#TRAJECTORIES[@]} -eq 0 ]]; then
  echo "[ERROR] no trajectory artifacts found for ${TARGET}" >&2
  exit 1
fi
for path in "$@" "${SOURCE_HEALTH}" "${LINEAGE}" "${TRAJECTORIES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing health artifact: ${path}" >&2
    exit 1
  fi
done

ARGS=(
  --variant "${VARIANT}"
  --iteration "${ITERATION}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --source-checkpoint-root "${SOURCE_ROOT}"
  --source-health "${SOURCE_HEALTH}"
  --lineage "${LINEAGE}"
  --output "${OUTPUT}"
  "${DOMAIN_ARGS[@]}"
)
for path in "$@"; do
  ARGS+=(--training-log "${path}")
done
for path in "${TRAJECTORIES[@]}"; do
  ARGS+=(--trajectories "${path}")
done

exec python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/check_domain_expert_health.py" "${ARGS[@]}"
