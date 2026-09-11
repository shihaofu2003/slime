#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
BANKING_EXPERT_SFT_DIR="${BANKING_EXPERT_SFT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-expert-sft/data}"

MODE="${1:-full}"
case "${MODE}" in
  smoke)
    SFT_DATA_PATH="${SFT_DATA_PATH:-${BANKING_EXPERT_SFT_DIR}/banking_expert_sft_smoke32.jsonl}"
    SAVE_SUFFIX="smoke"
    export NUM_EPOCH="${NUM_EPOCH:-1}"
    export SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
    ;;
  full)
    SFT_DATA_PATH="${SFT_DATA_PATH:-${BANKING_EXPERT_SFT_DIR}/banking_expert_sft.jsonl}"
    SAVE_SUFFIX="full"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

export SFT_DATA_PATH
export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist}"
export LOAD_DIR="${LOAD_DIR:-${TORCH_DIST_DIR}}"
export SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_banking_expert_${SAVE_SUFFIX}_20260908}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export NUM_EPOCH="${NUM_EPOCH:-2}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export TRAIN_SEED="${TRAIN_SEED:-1234}"
export LOSS_MASK_TYPE="qwen3_full"
export TOOL_KEY="tools"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export WANDB_GROUP="${WANDB_GROUP:-tau2-sft-banking-expert}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-tau2-sft-banking-expert-${SAVE_SUFFIX}}"

if [[ ! -s "${SFT_DATA_PATH}" ]]; then
  echo "[banking-expert-sft] missing data: ${SFT_DATA_PATH}" >&2
  echo "Run build_banking_expert.sh ${MODE} first." >&2
  exit 1
fi

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
