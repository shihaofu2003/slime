#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"

export SFT_DATA_PATH="${SFT_DATA_PATH:-${SERVICE_AGENT_ROOT}/datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking.jsonl}"
export SAVE_DIR="${SAVE_DIR:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_sft_areal_strict_no_thinking_epoch2_20260713}"
export NUM_EPOCH="${NUM_EPOCH:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-serviceagent-tau2-sft}"
export WANDB_GROUP="${WANDB_GROUP:-areal-strict-no-thinking-epoch2-20260713}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
