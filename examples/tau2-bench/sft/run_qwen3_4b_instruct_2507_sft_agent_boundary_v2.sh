#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
BOUNDARY_DIR="${TAU2_BOUNDARY_SFT_OUT:-${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2}"
ARM="${1:-${TAU2_BOUNDARY_SFT_ARM:-contract-only}}"
VALIDATE_ARGS=()

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [contract-only|contract-boundary]" >&2
  exit 2
fi

case "${ARM}" in
  contract-only)
    DATA_BASENAME="contract_only_19318.jsonl"
    SMOKE_BASENAME="contract_only_longest32_smoke.jsonl"
    SAVE_SUFFIX="contract_only"
    ;;
  contract-boundary)
    DATA_BASENAME="contract_boundary_19318.jsonl"
    SMOKE_BASENAME="contract_boundary_longest32_smoke.jsonl"
    SAVE_SUFFIX="contract_boundary"
    ;;
  *)
    echo "[tau2-boundary-sft] invalid TAU2_BOUNDARY_SFT_ARM=${ARM}" >&2
    exit 2
    ;;
esac

if [[ "${TAU2_BOUNDARY_SFT_SMOKE:-0}" != "0" ]]; then
  DATA_BASENAME="${SMOKE_BASENAME}"
  SAVE_SUFFIX="${SAVE_SUFFIX}_longest32_smoke"
  export NUM_EPOCH="${NUM_EPOCH:-1}"
  export SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
  VALIDATE_ARGS+=(--smoke)
fi

export SFT_DATA_PATH="${SFT_DATA_PATH:-${BOUNDARY_DIR}/${DATA_BASENAME}}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist}"
export LOAD_DIR="${LOAD_DIR:-${TORCH_DIST_DIR}}"
export SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_${SAVE_SUFFIX}_20260804}"
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
export WANDB_GROUP="${WANDB_GROUP:-tau2-sft-agent-user-boundary-v2}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-tau2-sft-boundary-v2-${SAVE_SUFFIX}}"

python3 "${PROJECT_ROOT}/examples/tau2-bench/sft/build_agent_boundary_v2.py" validate \
  --input "${SFT_DATA_PATH}" \
  --arm "${ARM}" \
  "${VALIDATE_ARGS[@]}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
