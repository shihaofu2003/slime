#!/usr/bin/env bash
#
# Eight-card tau2-bench GRPO with the ORIGINAL (non-SFT) Qwen3-4B-Instruct-2507
# as the user simulator — the no-user-SFT baseline for the A/B/C comparison.
# Expect weaker user behaviour (no STOP/tool-call training); the run is
# otherwise identical to real_8g.

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"

export TAU2_RL_RUN_ID="${TAU2_RL_RUN_ID:-real8g_user_base_$(date +%Y%m%d_%H%M%S)}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-Instruct-2507-base}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh"
