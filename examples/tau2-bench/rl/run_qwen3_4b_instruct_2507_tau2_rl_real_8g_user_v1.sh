#!/usr/bin/env bash
#
# Eight-card tau2-bench GRPO with the v1 trained user simulator
# (STOP-trained, terminates + runs telecom tools). Same hyperparameters as
# real_8g; this wrapper pins the user model explicitly for the A/B/C comparison.

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"

export TAU2_RL_RUN_ID="${TAU2_RL_RUN_ID:-real8g_user_v1_$(date +%Y%m%d_%H%M%S)}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh"
