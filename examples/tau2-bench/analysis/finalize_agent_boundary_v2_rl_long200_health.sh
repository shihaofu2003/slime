#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 partial|final|partial-kl-waiver|final-kl-waiver TRAINING_LOG [RESUME_LOG ...]" >&2
  exit 2
fi

MODE="$1"
shift
WAIVE_KL=0
case "${MODE}" in
  partial|final)
    HEALTH_MODE="${MODE}"
    ;;
  partial-kl-waiver)
    HEALTH_MODE="partial"
    WAIVE_KL=1
    ;;
  final-kl-waiver)
    HEALTH_MODE="final"
    WAIVE_KL=1
    ;;
  *)
    echo "[ERROR] mode must be partial, final, partial-kl-waiver, or final-kl-waiver" >&2
    exit 2
    ;;
esac

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long200"
LONG100_EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
SFT_GATE="${TAU2_SFT_PROMOTION_GATE:-${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2/SFT_PROMOTION.json}"
SOURCE_HEALTH="${TAU2_RL_LONG100_ITER99_GATE:-${LONG100_EXPERIMENT_DIR}/ITER99_HEALTH_GATE.json}"
SOURCE_CKPT_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
DESTINATION_CKPT_ROOT="${TAU2_RL_LONG200_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805}"
LINEAGE="${DESTINATION_CKPT_ROOT}/LONG200_LINEAGE.json"
ANALYSIS_DIR="${PROJECT_ROOT}/examples/tau2-bench/analysis"

SFT_CKPT_ROOT="$(python3 - "${SFT_GATE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
gate = json.loads(path.read_text(encoding="utf-8"))
if (
    gate.get("status") != "pass"
    or gate.get("gate_version") != "turn-aware-rl-v3"
    or gate.get("selected") != "contract-boundary"
):
    raise SystemExit(f"[ERROR] invalid Contract + boundary SFT promotion gate: {path}")
print(str(Path(gate["selected_checkpoint_root"]).resolve()))
PY
)"

mapfile -t TRAJECTORIES < <(
  find "${EXPERIMENT_DIR}/trajectories" -maxdepth 1 -type f \
    -name 'long200-final_iter0100-0199_*.jsonl' -print | sort
  find "${EXPERIMENT_DIR}/trajectories" -maxdepth 1 -type f \
    -name 'long200-final-kl-waiver_iter0100-0199_*.jsonl' -print | sort
)
if [[ ${#TRAJECTORIES[@]} -eq 0 ]]; then
  echo "[ERROR] no long200 trajectory artifacts found" >&2
  exit 1
fi

REQUIRED_PATHS=("$@" "${SOURCE_HEALTH}" "${LINEAGE}" "${TRAJECTORIES[@]}")
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing required long200 health artifact: ${path}" >&2
    exit 1
  fi
done

if [[ "${HEALTH_MODE}" == "final" && "${WAIVE_KL}" == "1" ]]; then
  OUTPUT="${HEALTH_OUTPUT:-${EXPERIMENT_DIR}/ITER199_KL_WAIVER_HEALTH_GATE.json}"
elif [[ "${HEALTH_MODE}" == "final" ]]; then
  OUTPUT="${HEALTH_OUTPUT:-${EXPERIMENT_DIR}/ITER199_HEALTH_GATE.json}"
elif [[ "${WAIVE_KL}" == "1" ]]; then
  OUTPUT="${HEALTH_OUTPUT:-${EXPERIMENT_DIR}/health/KL_WAIVER_PARTIAL_HEALTH_$(date +%Y%m%d_%H%M%S).json}"
else
  OUTPUT="${HEALTH_OUTPUT:-${EXPERIMENT_DIR}/health/PARTIAL_HEALTH_$(date +%Y%m%d_%H%M%S).json}"
fi

HEALTH_ARGS=(
  --stage iter199
  --mode "${HEALTH_MODE}"
  --checkpoint-root "${DESTINATION_CKPT_ROOT}"
  --expected-checkpoint-root "${DESTINATION_CKPT_ROOT}"
  --source-checkpoint-root "${SOURCE_CKPT_ROOT}"
  --source-health "${SOURCE_HEALTH}"
  --lineage "${LINEAGE}"
  --sft-checkpoint-root "${SFT_CKPT_ROOT}"
  --output "${OUTPUT}"
)
if [[ "${WAIVE_KL}" == "1" ]]; then
  HEALTH_ARGS+=(--waive-kl)
fi
for log_path in "$@"; do
  HEALTH_ARGS+=(--training-log "${log_path}")
done
for trajectory_path in "${TRAJECTORIES[@]}"; do
  HEALTH_ARGS+=(--trajectories "${trajectory_path}")
done

exec python3 "${ANALYSIS_DIR}/check_agent_boundary_rl_health.py" "${HEALTH_ARGS[@]}"
