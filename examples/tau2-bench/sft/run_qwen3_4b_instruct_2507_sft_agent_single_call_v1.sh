#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-agent-single-call-v1"
MODE="${1:-full}"

if [[ $# -gt 1 || ! "${MODE}" =~ ^(smoke|full)$ ]]; then
  echo "Usage: $0 <smoke|full>" >&2
  exit 2
fi

case "${MODE}" in
  smoke)
    export SFT_DATA_PATH="${SFT_DATA_PATH:-${EXPERIMENT_DIR}/data/single_call_v1_longest32_smoke.jsonl}"
    export SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_longest32_smoke_20260809}"
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
    export NUM_EPOCH="${NUM_EPOCH:-1}"
    export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
    EXPECTED_ROWS=32
    ;;
  full)
    export SFT_DATA_PATH="${SFT_DATA_PATH:-${EXPERIMENT_DIR}/data/single_call_v1_25708.jsonl}"
    export SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809}"
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
    export NUM_EPOCH="${NUM_EPOCH:-2}"
    export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
    EXPECTED_ROWS=25708
    ;;
esac

python3 "${PROJECT_ROOT}/examples/tau2-bench/sft/build_agent_single_call_v1.py" validate \
  --input "${SFT_DATA_PATH}" \
  --expected-rows "${EXPECTED_ROWS}"

export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist}"
export LOAD_DIR="${LOAD_DIR:-${TORCH_DIST_DIR}}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export TRAIN_SEED="${TRAIN_SEED:-1234}"
export LOSS_MASK_TYPE="qwen3_full"
export TOOL_KEY="tools"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-tau2-agent-single-call-v1-sft}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
