#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
BANKING_DATA="${BANKING_DATA:-${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"

PHASE="${1:?phase is required}"
START="${2:?start is required}"
LIMIT="${3:?limit is required}"
LABEL="${4:?batch label is required}"
OUT="${EXPERIMENT_DIR}/scale/candidates/${LABEL}"

if [[ "${PHASE}" != "train" && "${PHASE}" != "dev" && "${PHASE}" != "challenge" ]]; then
  echo "Expected train, dev, or challenge, got: ${PHASE}" >&2
  exit 2
fi

export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PIPELINE}/generate.py" \
  --phase "${PHASE}" \
  --start "${START}" \
  --limit "${LIMIT}" \
  --documents-dir "${BANKING_DATA}/documents" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --output-dir "${OUT}"

python3 "${PIPELINE}/validate.py" \
  --tasks "${OUT}/${PHASE}_tasks.json" \
  --specs "${OUT}/${PHASE}_scenario_specs.jsonl" \
  --contracts "${OUT}/${PHASE}_contracts.jsonl" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --documents-dir "${BANKING_DATA}/documents" \
  --output "${OUT}/reference_validation.json" \
  --failures "${OUT}/reference_failures.json" \
  --workers 32 \
  --environment

python3 "${PIPELINE}/isolation.py" audit \
  --benchmark-tasks-dir "${BANKING_DATA}/tasks" \
  --tasks "${OUT}/${PHASE}_tasks.json" \
  --specs "${OUT}/${PHASE}_scenario_specs.jsonl" \
  --contracts "${OUT}/${PHASE}_contracts.jsonl" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --output "${OUT}/isolation_audit.json" \
  --failures "${OUT}/isolation_failures.json"

echo "BANKING_SYNTHETIC_BUILD_BATCH_OK phase=${PHASE} start=${START} limit=${LIMIT} out=${OUT}"
