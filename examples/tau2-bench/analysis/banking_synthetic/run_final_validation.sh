#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
BANKING_DATA="${BANKING_DATA:-${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"

FINAL_DIR="${1:-${EXPERIMENT_DIR}/final}"
POOL_MANIFEST="${2:-${FINAL_DIR}/pool_manifest.json}"
MODEL_MANIFESTS="${3:?comma-separated runtime model manifests are required}"
COMBINED="${FINAL_DIR}/combined"

export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PIPELINE}/merge.py" \
  --tasks "${FINAL_DIR}/train_tasks.json" \
  --specs "${FINAL_DIR}/train_scenario_specs.jsonl" \
  --contracts "${FINAL_DIR}/train_contracts.jsonl" \
  --tasks "${FINAL_DIR}/dev_tasks.json" \
  --specs "${FINAL_DIR}/dev_scenario_specs.jsonl" \
  --contracts "${FINAL_DIR}/dev_contracts.jsonl" \
  --tasks "${FINAL_DIR}/challenge_tasks.json" \
  --specs "${FINAL_DIR}/challenge_scenario_specs.jsonl" \
  --contracts "${FINAL_DIR}/challenge_contracts.jsonl" \
  --output-dir "${COMBINED}" \
  --prefix all

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

FINALIZE_ARGS=(
  --train-tasks "${FINAL_DIR}/train_tasks.json"
  --train-specs "${FINAL_DIR}/train_scenario_specs.jsonl"
  --train-contracts "${FINAL_DIR}/train_contracts.jsonl"
  --dev-tasks "${FINAL_DIR}/dev_tasks.json"
  --dev-specs "${FINAL_DIR}/dev_scenario_specs.jsonl"
  --dev-contracts "${FINAL_DIR}/dev_contracts.jsonl"
  --challenge-tasks "${FINAL_DIR}/challenge_tasks.json"
  --challenge-specs "${FINAL_DIR}/challenge_scenario_specs.jsonl"
  --challenge-contracts "${FINAL_DIR}/challenge_contracts.jsonl"
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json"
  --pool-manifest "${POOL_MANIFEST}"
  --successful-trajectories "${FINAL_DIR}/successful_trajectories.jsonl"
  --reference-report "${FINAL_DIR}/reference_validation.json"
  --isolation-report "${FINAL_DIR}/isolation_audit.json"
  --output "${FINAL_DIR}/acceptance.json"
  --failures "${FINAL_DIR}/acceptance_failures.json"
)
IFS=',' read -r -a MANIFEST_PATHS <<< "${MODEL_MANIFESTS}"
for manifest in "${MANIFEST_PATHS[@]}"; do
  FINALIZE_ARGS+=(--model-manifest "${manifest}")
done
python3 "${PIPELINE}/finalize.py" "${FINALIZE_ARGS[@]}"

echo "BANKING_SYNTHETIC_FINAL_ACCEPTED final_dir=${FINAL_DIR}"
