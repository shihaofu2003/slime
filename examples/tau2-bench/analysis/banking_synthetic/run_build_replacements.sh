#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
BANKING_DATA="${BANKING_DATA:-${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"

REQUESTS="${1:?replacement request JSON is required}"
START="${2:?start is required}"
LIMIT="${3:?limit is required}"
LABEL="${4:?batch label is required}"
SERIAL_BASE="${5:-400000}"
OUT="${EXPERIMENT_DIR}/scale/replacements/${LABEL}"

export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PIPELINE}/resample.py" \
  --requests "${REQUESTS}" \
  --documents-dir "${BANKING_DATA}/documents" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --serial-base "${SERIAL_BASE}" \
  --start "${START}" \
  --limit "${LIMIT}" \
  --output-dir "${OUT}" \
  --prefix replacement

python3 "${PIPELINE}/validate.py" \
  --tasks "${OUT}/replacement_tasks.json" \
  --specs "${OUT}/replacement_scenario_specs.jsonl" \
  --contracts "${OUT}/replacement_contracts.jsonl" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --documents-dir "${BANKING_DATA}/documents" \
  --output "${OUT}/reference_validation.json" \
  --failures "${OUT}/reference_failures.json" \
  --workers 32 \
  --environment

python3 "${PIPELINE}/isolation.py" audit \
  --benchmark-tasks-dir "${BANKING_DATA}/tasks" \
  --tasks "${OUT}/replacement_tasks.json" \
  --specs "${OUT}/replacement_scenario_specs.jsonl" \
  --contracts "${OUT}/replacement_contracts.jsonl" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --output "${OUT}/isolation_audit.json" \
  --failures "${OUT}/isolation_failures.json"

echo "BANKING_SYNTHETIC_REPLACEMENT_BATCH_OK start=${START} limit=${LIMIT} out=${OUT}"
