#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
BANKING_DATA="${BANKING_DATA:-${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge}"
OUT="${BANKING_SYNTHETIC_OUT:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"
CANDIDATES="${OUT}/candidates"
MATRIX="${OUT}/template_matrix"

export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PROJECT_ROOT}/tests/test_tau2_banking_synthetic.py"

python3 "${PIPELINE}/isolation.py" build-allowlist \
  --benchmark-tasks-dir "${BANKING_DATA}/tasks" \
  --documents-dir "${BANKING_DATA}/documents" \
  --output "${OUT}/allowed_documents.json"

python3 "${PIPELINE}/generate.py" \
  --phase pilot \
  --documents-dir "${BANKING_DATA}/documents" \
  --allowlist "${OUT}/allowed_documents.json" \
  --output-dir "${CANDIDATES}"

python3 "${PIPELINE}/validate.py" \
  --tasks "${CANDIDATES}/pilot_tasks.json" \
  --specs "${CANDIDATES}/pilot_scenario_specs.jsonl" \
  --contracts "${CANDIDATES}/pilot_contracts.jsonl" \
  --allowlist "${OUT}/allowed_documents.json" \
  --documents-dir "${BANKING_DATA}/documents" \
  --output "${OUT}/pilot_reference_validation.json" \
  --failures "${OUT}/pilot_reference_failures.json" \
  --environment

python3 "${PIPELINE}/isolation.py" audit \
  --benchmark-tasks-dir "${BANKING_DATA}/tasks" \
  --tasks "${CANDIDATES}/pilot_tasks.json" \
  --specs "${CANDIDATES}/pilot_scenario_specs.jsonl" \
  --contracts "${CANDIDATES}/pilot_contracts.jsonl" \
  --allowlist "${OUT}/allowed_documents.json" \
  --output "${OUT}/pilot_isolation_audit.json" \
  --failures "${OUT}/pilot_isolation_failures.json"

python3 "${PIPELINE}/generate.py" \
  --phase matrix \
  --documents-dir "${BANKING_DATA}/documents" \
  --allowlist "${OUT}/allowed_documents.json" \
  --output-dir "${MATRIX}"

python3 "${PIPELINE}/validate.py" \
  --tasks "${MATRIX}/matrix_tasks.json" \
  --specs "${MATRIX}/matrix_scenario_specs.jsonl" \
  --contracts "${MATRIX}/matrix_contracts.jsonl" \
  --allowlist "${OUT}/allowed_documents.json" \
  --documents-dir "${BANKING_DATA}/documents" \
  --output "${MATRIX}/reference_validation.json" \
  --failures "${MATRIX}/reference_failures.json" \
  --environment

python3 "${PIPELINE}/isolation.py" audit \
  --benchmark-tasks-dir "${BANKING_DATA}/tasks" \
  --tasks "${MATRIX}/matrix_tasks.json" \
  --specs "${MATRIX}/matrix_scenario_specs.jsonl" \
  --contracts "${MATRIX}/matrix_contracts.jsonl" \
  --allowlist "${OUT}/allowed_documents.json" \
  --output "${MATRIX}/isolation_audit.json" \
  --failures "${MATRIX}/isolation_failures.json"

echo "BANKING_SYNTHETIC_PILOT_BUILD_OK out=${OUT}"
