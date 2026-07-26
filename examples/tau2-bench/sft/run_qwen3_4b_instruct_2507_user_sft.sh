#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
DATASET_DIR="${DATASET_DIR:-${SERVICE_AGENT_ROOT}/datasets/tau2-bench-user-sft}"

export DATASET_DIR
export SFT_DATA_PATH="${SFT_DATA_PATH:-${DATASET_DIR}/mixed_user_sft_no_thinking.jsonl}"
export SAVE_DIR="${SAVE_DIR:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft}"
export WANDB_GROUP="${WANDB_GROUP:-qwen3-4B-tau2-user-sft}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-tau2-user-sft}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
