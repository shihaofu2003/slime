#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(smoke|full)$ ]]; then
  echo "Usage: $0 <smoke|full>" >&2
  exit 2
fi

MODE="$1"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-raw-all-max16384"

case "${MODE}" in
  smoke)
    SFT_DATA_PATH="${EXPERIMENT_DIR}/data/raw_all_max16384_longest32.jsonl"
    SAVE_DIR="${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_longest32_smoke_20260903"
    NUM_EPOCH=1
    SAVE_INTERVAL=2
    EXPECTED_ROWS=32
    ;;
  full)
    SFT_DATA_PATH="${EXPERIMENT_DIR}/data/raw_all_max16384.jsonl"
    SAVE_DIR="${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903"
    NUM_EPOCH=2
    SAVE_INTERVAL=4068
    EXPECTED_ROWS=32548
    ;;
esac

if [[ ! -f "${SFT_DATA_PATH}" ]]; then
  echo "[tau2-raw-all-sft] missing data: ${SFT_DATA_PATH}" >&2
  exit 1
fi
ACTUAL_ROWS="$(wc -l < "${SFT_DATA_PATH}" | tr -d '[:space:]')"
if [[ "${ACTUAL_ROWS}" != "${EXPECTED_ROWS}" ]]; then
  echo "[tau2-raw-all-sft] expected ${EXPECTED_ROWS} rows, found ${ACTUAL_ROWS}" >&2
  exit 1
fi

export SFT_DATA_PATH
export SAVE_DIR
export HF_CHECKPOINT="${MODEL_ROOT}/Qwen3-4B-Instruct-2507"
export TORCH_DIST_DIR="${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist"
export LOAD_DIR="${TORCH_DIST_DIR}"
export NUM_GPUS=8
export ROLLOUT_BATCH_SIZE=16
export GLOBAL_BATCH_SIZE=16
export NUM_EPOCH
export SAVE_INTERVAL
export MAX_TOKENS_PER_GPU=16384
export LR=1e-5
export MIN_LR=1e-6
export TRAIN_SEED=1234
export START_ROLLOUT_ID=0
export LOSS_MASK_TYPE=qwen3_full
export TOOL_KEY=tools
export LABEL_KEY=answer
export ROLLOUT_FUNCTION_PATH=slime.rollout.prompt_answer_sft_rollout.generate_rollout
export CONTEXT_PARALLEL_SIZE=1
export WANDB_GROUP=tau2-sft-raw-all-max16384
export WANDB_EXP_NAME="tau2-sft-raw-all-max16384-${MODE}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
