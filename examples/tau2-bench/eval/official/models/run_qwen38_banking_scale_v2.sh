#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
SCALE_DIR="${PROJECT_ROOT}/output/experiments/tau2-banking-task-curriculum-scale-v2"

export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-scale-v2-qwen38-eval}"
export CURRICULUM_TASKS="${SCALE_DIR}/scaled_tasks.json"
export CURRICULUM_CONTRACTS="${SCALE_DIR}/scaled_contracts.jsonl"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen38_banking_curriculum.sh" "$@"
