#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
BASE_RUNNER="${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_single_call_v1.sh"

export EXPERIMENT_NAME="tau2-agent-single-call-v1-raw-user-rl200"
export RL_CKPT_ROOT="${RL_CKPT_ROOT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_20260812}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-Instruct-2507}"
export WANDB_GROUP="${WANDB_GROUP:-tau2-agent-single-call-v1-raw-user-rl200}"

exec bash "${BASE_RUNNER}" stage-long200
