#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 SMOKE_RUN_LOG" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2"
SMOKE_CKPT_ROOT="${SMOKE_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_turn_credit_smoke_r3_20260804}"

exec python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/check_agent_boundary_rl_smoke.py" \
  --checkpoint-root "${SMOKE_CKPT_ROOT}" \
  --training-log "$1" \
  --trajectories "${EXPERIMENT_DIR}/trajectories/smoke_r3_iter0000-0001.jsonl" \
  --window 96 \
  --output "${EXPERIMENT_DIR}/SMOKE_GATE.json"
