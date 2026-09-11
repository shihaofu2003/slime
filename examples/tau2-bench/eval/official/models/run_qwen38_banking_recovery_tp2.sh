#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"

export QWEN38_AGENT_TP="2"
export AGENT_REPLICA_CUDA_GROUPS="0,1"
export USER_REPLICA_CUDA_GROUPS="2"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen38_banking_curriculum.sh" "$@"
