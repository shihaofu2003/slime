#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"
EVAL_SET="${EXPERIMENT_DIR}/final-minimal/eval-set"

VARIANT="${1:?retrieval variant is required}"
shift
if [[ "$#" -eq 0 ]]; then
  echo "At least one evaluation seed is required" >&2
  exit 2
fi

export SYNTHETIC_TASKS="${EVAL_SET}/eval_tasks.json"
export SYNTHETIC_CONTRACTS="${EVAL_SET}/eval_contracts.jsonl"

for seed in "$@"; do
  bash "${PIPELINE}/run_qwen38_eval.sh" scale "${VARIANT}" 0 1 "${seed}"
done
