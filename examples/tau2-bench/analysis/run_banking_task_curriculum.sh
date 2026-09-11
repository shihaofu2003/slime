#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OUT="${BANKING_CURRICULUM_OUT:-${PROJECT_ROOT}/output/experiments/tau2-banking-task-curriculum}"
PYTHON_BIN="${BANKING_CURRICULUM_PYTHON:-$(command -v python3)}"
COMMAND="${1:-validate}"

if [[ "${COMMAND}" != "validate" && "${COMMAND}" != "validate-expanded" ]]; then
  echo "Expected validate or validate-expanded, got: ${COMMAND}" >&2
  exit 2
fi

export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tests/test_tau2_banking_curriculum.py"
"${PYTHON_BIN}" -m unittest discover \
  -s "${SERVICE_AGENT_ROOT}/tau2-bench/tests" \
  -p "test_evaluator_communicate.py"
if [[ "${COMMAND}" == "validate-expanded" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_task_curriculum.py" \
    validate \
    --environment \
    --output-dir "${OUT}"
fi
"${PYTHON_BIN}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_task_curriculum.py" \
  "${COMMAND}" \
  --environment \
  --output-dir "${OUT}"
