#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

exec python3 "${PROJECT_ROOT}/examples/tau2-bench/sft/build_raw_all_sft.py" "$@"
