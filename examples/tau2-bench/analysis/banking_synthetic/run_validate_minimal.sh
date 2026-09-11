#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
BANKING_DATA="${BANKING_DATA:-${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"
FINAL_DIR="${FINAL_DIR:-${EXPERIMENT_DIR}/final-minimal}"
COMBINED="${FINAL_DIR}/combined"

export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PIPELINE}/validate.py" \
  --tasks "${COMBINED}/all_tasks.json" \
  --specs "${COMBINED}/all_scenario_specs.jsonl" \
  --contracts "${COMBINED}/all_contracts.jsonl" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --documents-dir "${BANKING_DATA}/documents" \
  --output "${FINAL_DIR}/reference_validation.json" \
  --failures "${FINAL_DIR}/reference_failures.json" \
  --workers 32 \
  --environment

python3 "${PIPELINE}/isolation.py" audit \
  --benchmark-tasks-dir "${BANKING_DATA}/tasks" \
  --tasks "${COMBINED}/all_tasks.json" \
  --specs "${COMBINED}/all_scenario_specs.jsonl" \
  --contracts "${COMBINED}/all_contracts.jsonl" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --output "${FINAL_DIR}/isolation_audit.json" \
  --failures "${FINAL_DIR}/isolation_failures.json"

echo "BANKING_SYNTHETIC_MINIMAL_VALIDATED final_dir=${FINAL_DIR}"
