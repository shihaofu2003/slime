#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"

SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-${SERVICE_AGENT_ROOT}/datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking.jsonl}"
FILTERED_DATA_DIR="${FILTERED_DATA_DIR:-${PROJECT_ROOT}/output/datasets/tau2-bench-sft}"
FILTERED_DATA_PATH="${FILTERED_DATA_PATH:-${FILTERED_DATA_DIR}/areal_tau2_sft_strict_no_thinking_max8192.jsonl}"
FILTER_STATS_PATH="${FILTER_STATS_PATH:-${FILTERED_DATA_DIR}/areal_tau2_sft_strict_no_thinking_max8192.stats.json}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/output/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${FILTERED_DATA_DIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"

python3 "${PROJECT_ROOT}/examples/tau2-bench/sft/filter_sft_by_tokens.py" \
  --input "${SOURCE_DATA_PATH}" \
  --output "${FILTERED_DATA_PATH}" \
  --stats-output "${FILTER_STATS_PATH}" \
  --tokenizer "${MODEL_ROOT}/Qwen3-4B-Instruct-2507" \
  --loss-mask-type qwen3 \
  --max-total-tokens "${MAX_TOTAL_TOKENS}"

export DATASET_DIR="${FILTERED_DATA_DIR}"
export SFT_DATA_PATH="${FILTERED_DATA_PATH}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist}"
export LOAD_DIR="${LOAD_DIR:-${TORCH_DIST_DIR}}"
export SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
export WANDB_GROUP="${WANDB_GROUP:-tau2-agent-sft-multitool-max8192-20260714}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-tau2-agent-sft-multitool-max8192}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
