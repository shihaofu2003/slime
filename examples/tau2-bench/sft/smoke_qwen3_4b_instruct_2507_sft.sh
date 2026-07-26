#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
DATASET_DIR="${DATASET_DIR:-${SERVICE_AGENT_ROOT}/datasets/tau2-bench-sft}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-8}"

export SFT_DATA_PATH="${SFT_DATA_PATH:-${DATASET_DIR}/areal_tau2_sft_strict_no_thinking.jsonl@[0:${SMOKE_SAMPLES}]}"
export SAVE_DIR="${SAVE_DIR:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_sft_smoke}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
export NUM_EPOCH="${NUM_EPOCH:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
export USE_WANDB="${USE_WANDB:-0}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-tau2-sft-smoke}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
