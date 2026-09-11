#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"

export SFT_DATA_SLICE="${SFT_DATA_SLICE:-@[0:32]}"
export NUM_EPOCH="${NUM_EPOCH:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_local_first_relaxed_smoke_20260803}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft_agent_local_first_relaxed.sh"
