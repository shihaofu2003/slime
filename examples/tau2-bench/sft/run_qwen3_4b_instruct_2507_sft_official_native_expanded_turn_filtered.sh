#!/usr/bin/env bash

set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "resume" ) ]]; then
  echo "Usage: $0 [resume]" >&2
  exit 2
fi

MODE="${1:-fresh}"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
SOURCE_EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-turn-quality-qwen38-xhigh-v1"

SFT_DATA_PATH="${SOURCE_EXPERIMENT_DIR}/data/agent_official_native_expanded_turn_filtered.jsonl"
SAVE_DIR="${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_turn_filtered_20260908"
EXPECTED_ROWS=29701

if [[ ! -f "${SFT_DATA_PATH}" ]]; then
  echo "[tau2-official-native-turn-filtered-sft] missing data: ${SFT_DATA_PATH}" >&2
  exit 1
fi

ACTUAL_ROWS="$(wc -l < "${SFT_DATA_PATH}" | tr -d '[:space:]')"
if [[ "${ACTUAL_ROWS}" != "${EXPECTED_ROWS}" ]]; then
  echo "[tau2-official-native-turn-filtered-sft] expected ${EXPECTED_ROWS} rows, found ${ACTUAL_ROWS}" >&2
  exit 1
fi

python3 "${PROJECT_ROOT}/examples/tau2-bench/sft/build_agent_official_native_expanded.py" validate \
  --input "${SFT_DATA_PATH}" \
  --tokenizer "${MODEL_ROOT}/Qwen3-4B-Instruct-2507" \
  --max-tokens 16384

export SFT_DATA_PATH
export SAVE_DIR
export HF_CHECKPOINT="${MODEL_ROOT}/Qwen3-4B-Instruct-2507"
export TORCH_DIST_DIR="${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist"
if [[ "${MODE}" == "resume" ]]; then
  if [[ ! -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "[tau2-official-native-turn-filtered-sft] no checkpoint to resume: ${SAVE_DIR}" >&2
    exit 1
  fi
  export LOAD_DIR="${SAVE_DIR}"
  unset START_ROLLOUT_ID
else
  export LOAD_DIR="${TORCH_DIST_DIR}"
  export START_ROLLOUT_ID=0
fi
export NUM_GPUS=8
export ROLLOUT_BATCH_SIZE=16
export GLOBAL_BATCH_SIZE=16
export NUM_EPOCH=2
export SAVE_INTERVAL=400
export MAX_TOKENS_PER_GPU=16384
export LR=1e-5
export MIN_LR=1e-6
export TRAIN_SEED=1234
export LOSS_MASK_TYPE=qwen3_full
export TOOL_KEY=tools
export CONTEXT_PARALLEL_SIZE=1
export WANDB_GROUP=tau2-sft-official-native-expanded-turn-filtered
export WANDB_EXP_NAME=tau2-sft-official-native-expanded-turn-filtered

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
