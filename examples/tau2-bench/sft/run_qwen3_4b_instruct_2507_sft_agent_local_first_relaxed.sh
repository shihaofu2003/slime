#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
FILTER_DIR="${TAU2_LOCAL_FILTER_OUT:-${PROJECT_ROOT}/output/experiments/tau2-sft-local-first-relaxed}"

BUILT_SFT_DATA_PATH="${BUILT_SFT_DATA_PATH:-${FILTER_DIR}/sft_train_budget_matched_19318.jsonl}"
SFT_DATA_SLICE="${SFT_DATA_SLICE:-}"
export SFT_DATA_PATH="${SFT_DATA_PATH:-${BUILT_SFT_DATA_PATH}${SFT_DATA_SLICE}}"
FILTER_SUMMARY="${FILTER_SUMMARY:-${FILTER_DIR}/filter_summary.json}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist}"
export LOAD_DIR="${LOAD_DIR:-${TORCH_DIST_DIR}}"
export SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_local_first_relaxed_20260803}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export NUM_EPOCH="${NUM_EPOCH:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export WANDB_GROUP="${WANDB_GROUP:-tau2-agent-sft-local-first-relaxed-20260803}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-tau2-agent-sft-local-first-relaxed}"

python3 "${PROJECT_ROOT}/examples/tau2-bench/sft/validate_local_first_sft.py" \
  --data "${BUILT_SFT_DATA_PATH}" \
  --summary "${FILTER_SUMMARY}" \
  --tokenizer "${HF_CHECKPOINT}" \
  --expected-rows 19318 \
  --max-total-tokens 8192

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
