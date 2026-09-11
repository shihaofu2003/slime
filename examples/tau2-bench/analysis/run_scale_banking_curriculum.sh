#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OUT="${BANKING_SCALE_OUT:-${PROJECT_ROOT}/output/experiments/tau2-banking-task-curriculum-scale-v2}"
PYTHON_BIN="${BANKING_CURRICULUM_PYTHON:-$(command -v python3)}"

export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tests/test_tau2_banking_curriculum.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/scale_banking_curriculum.py" \
  build --output-dir "${OUT}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/scale_banking_curriculum.py" \
  validate --environment --output-dir "${OUT}"
