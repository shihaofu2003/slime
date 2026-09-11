#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2"
export PYTHONPATH="${PROJECT_ROOT}:${SERVICE_AGENT_ROOT}/tau2-bench/src:${PYTHONPATH:-}"

exec python3 "${PROJECT_ROOT}/examples/tau2-bench/sft/build_agent_boundary_v2.py" build \
  --arm both \
  --ordinary-reviewed "${EXPERIMENT_DIR}/ordinary_telecom_reviewed.jsonl" \
  --handoff-candidates "${EXPERIMENT_DIR}/natural_handoff_candidates.jsonl" \
  "$@"
