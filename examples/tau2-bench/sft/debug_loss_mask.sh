#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

python3 examples/tau2-bench/sft/debug_loss_mask.py "$@"
