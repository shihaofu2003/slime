#!/usr/bin/env bash
#
# Eight-card tau2-bench GRPO with the v2 trained user simulator
# (STOP-trained, aligned system prompt, drops content_and_tool mixed targets).
# Same hyperparameters as real_8g; only the user model differs.

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"

export TAU2_RL_RUN_ID="${TAU2_RL_RUN_ID:-real8g_user_v2_$(date +%Y%m%d_%H%M%S)}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop_v2/iter_0006311_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-v2-iter0006311}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh"
