#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
DATASET_DIR="${DATASET_DIR:-${SERVICE_AGENT_ROOT}/datasets/tau2-bench-user-sft-v2}"

# v2 user-SFT data: aligned system prompt (no USER_TOOL_FORMAT preamble,
# telecom uses simulation_guidelines_tools.md), cleaned scenario (reason_for_call
# only, no Task id/Difficulty), and content_and_tool mixed targets dropped.
export DATASET_DIR
export SFT_DATA_PATH="${SFT_DATA_PATH:-${DATASET_DIR}/mixed_user_sft_no_thinking.jsonl}"
export SAVE_DIR="${SAVE_DIR:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop_v2}"
export WANDB_GROUP="${WANDB_GROUP:-qwen3-4B-tau2-user-sft-stop-v2}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-tau2-user-sft-stop-v2}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
