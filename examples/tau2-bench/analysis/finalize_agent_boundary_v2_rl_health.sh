#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 iter9|iter19|iter99 TRAINING_LOG [RESUME_LOG ...]" >&2
  exit 2
fi

STAGE="$1"
shift
case "${STAGE}" in
  iter9)
    TRAJECTORY_STAGE="long100-a_iter0000-0009"
    PREREQUISITE=""
    ;;
  iter19)
    TRAJECTORY_STAGE="long100-b_iter0010-0019"
    PREREQUISITE="ITER9_HEALTH_GATE.json"
    ;;
  iter99)
    TRAJECTORY_STAGE="long100-final_iter0020-0099"
    PREREQUISITE="ITER19_HEALTH_GATE.json"
    ;;
  *)
    echo "[ERROR] stage must be iter9, iter19, or iter99" >&2
    exit 2
    ;;
esac

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
SFT_GATE="${TAU2_SFT_PROMOTION_GATE:-${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2/SFT_PROMOTION.json}"
LONG100_CKPT_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
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

TRAJECTORIES="${EXPERIMENT_DIR}/trajectories/${TRAJECTORY_STAGE}.jsonl"
REQUIRED_PATHS=("$@" "${TRAJECTORIES}")
if [[ -n "${PREREQUISITE}" ]]; then
  REQUIRED_PATHS+=("${EXPERIMENT_DIR}/${PREREQUISITE}")
fi
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing required health artifact: ${path}" >&2
    exit 1
  fi
done

HEALTH_ARGS=(
  --stage "${STAGE}"
  --checkpoint-root "${LONG100_CKPT_ROOT}"
  --expected-checkpoint-root "${LONG100_CKPT_ROOT}"
  --sft-checkpoint-root "${SFT_CKPT_ROOT}"
  --trajectories "${TRAJECTORIES}"
  --output "${EXPERIMENT_DIR}/$(printf 'ITER%s_HEALTH_GATE.json' "${STAGE#iter}")"
)
for log_path in "$@"; do
  HEALTH_ARGS+=(--training-log "${log_path}")
done
if [[ -n "${PREREQUISITE}" ]]; then
  HEALTH_ARGS+=(--prerequisite-health "${EXPERIMENT_DIR}/${PREREQUISITE}")
fi

exec python3 "${ANALYSIS_DIR}/check_agent_boundary_rl_health.py" "${HEALTH_ARGS[@]}"
