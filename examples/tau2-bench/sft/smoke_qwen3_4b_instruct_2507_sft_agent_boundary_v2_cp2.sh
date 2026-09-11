#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
export TAU2_BOUNDARY_SFT_ARM="${1:-${TAU2_BOUNDARY_SFT_ARM:-contract-only}}"
export TAU2_BOUNDARY_SFT_SMOKE=1
export CONTEXT_PARALLEL_SIZE=2

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft_agent_boundary_v2.sh"
