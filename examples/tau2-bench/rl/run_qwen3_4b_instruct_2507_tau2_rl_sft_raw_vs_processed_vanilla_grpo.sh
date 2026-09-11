#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^(raw-all|processed)$ || ! "$2" =~ ^(smoke|train100)$ ]]; then
  echo "Usage: $0 <raw-all|processed> <smoke|train100>" >&2
  exit 2
fi

ARM="$1"
MODE="$2"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_BASE="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
EXPERIMENT_NAME="tau2-rl-sft-raw-vs-processed-vanilla-grpo"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
ARTIFACT_DIR="${EXPERIMENT_DIR}/arms/${ARM}"

case "${ARM}" in
  raw-all)
    SFT_CKPT_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903"
    RL_CKPT_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_rl_raw_all_vanilla_grpo_qwen36_20260903"
    SMOKE_CKPT_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_rl_raw_all_vanilla_grpo_qwen36_smoke_20260903"
    ;;
  processed)
    SFT_CKPT_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903"
    RL_CKPT_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_rl_official_native_expanded_vanilla_grpo_qwen36_20260903"
    SMOKE_CKPT_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_rl_official_native_expanded_vanilla_grpo_qwen36_smoke_20260903"
    ;;
esac

if [[ ! -f "${SFT_CKPT_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  echo "[ERROR] missing SFT torch-dist checkpoint root: ${SFT_CKPT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SFT_CKPT_ROOT}/final_hf/config.json" ]]; then
  echo "[ERROR] missing SFT final_hf checkpoint: ${SFT_CKPT_ROOT}/final_hf" >&2
  exit 1
fi

if [[ "${MODE}" == "smoke" ]]; then
  ACTIVE_CKPT_ROOT="${SMOKE_CKPT_ROOT}"
  NUM_ROLLOUT_VALUE=1
  SAVE_INTERVAL_VALUE=1
else
  ACTIVE_CKPT_ROOT="${RL_CKPT_ROOT}"
  NUM_ROLLOUT_VALUE=100
  SAVE_INTERVAL_VALUE=10
fi

LATEST_FILE="${ACTIVE_CKPT_ROOT}/latest_checkpointed_iteration.txt"
if [[ -f "${LATEST_FILE}" ]]; then
  latest="$(tr -d '[:space:]' < "${LATEST_FILE}")"
  if [[ ! "${latest}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] invalid checkpoint iteration in ${LATEST_FILE}: ${latest}" >&2
    exit 1
  fi
  if [[ "${MODE}" == "smoke" || "${latest}" -ge 99 ]]; then
    echo "[ERROR] ${ARM} ${MODE} already completed at iteration ${latest}" >&2
    exit 1
  fi
elif [[ -d "${ACTIVE_CKPT_ROOT}" ]] && find "${ACTIVE_CKPT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  echo "[ERROR] checkpoint root is non-empty but has no resume marker: ${ACTIVE_CKPT_ROOT}" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}/data" "${ARTIFACT_DIR}/trajectories"

export SFT_CKPT_ROOT
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/final_hf"
export REF_LOAD="${SFT_CKPT_ROOT}"
export SAVE_DIR="${ACTIVE_CKPT_ROOT}"
export LOAD_DIR="${ACTIVE_CKPT_ROOT}"
export PREPARED_RL_DATA="${ARTIFACT_DIR}/data/areal_tau2_rl_train.jsonl"

export TAU2_AGENT_PROTOCOL_PROFILE="official-native"
export LOSS_MASK_TYPE="qwen3_full"
export TAU2_TURN_CREDIT_VERSION=""
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
export DATA_SOURCE_PATH="filters.DomainQuotaDataSource"
export TAU2_RL_DOMAIN_QUOTA="airline:1,retail:2,telecom:2"
export TAU2_RL_DOMAIN=""
export SKIP_PREFLIGHT=0

export TOTAL_GPUS=8
export RAY_GPUS=4
export AGENT_CUDA_VISIBLE_DEVICES="0,1,2,3"
export USER_CUDA_VISIBLE_DEVICES="4,5"
export USER_TP=2
export ROLLOUT_BATCH_SIZE=5
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=40
export NUM_ROLLOUT="${NUM_ROLLOUT_VALUE}"
export SAVE_INTERVAL="${SAVE_INTERVAL_VALUE}"
export USE_DYNAMIC_FILTER=1

export TRAIN_SEED=1234
export ROLLOUT_SEED=42
export LR=2e-6
export KL_LOSS_TYPE=k2
export KL_LOSS_COEF=0
export KL_COEF=0
export ENTROPY_COEF=0
export EPS_CLIP=0.2
export EPS_CLIP_HIGH=0.2

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0
export AGENT_MAX_TOKENS=1200
export TAU2_MAX_STEPS=200
export TAU2_MAX_ERRORS=10
export TAU2_RL_MAX_TRAIN_TOKENS=16384
export TAU2_RL_MAX_ROLLOUT_RETRIES=2

export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B"
export USER_MODEL="Qwen3.6-27B-tau2-user-nonthinking"
export USER_MEM_FRACTION=0.90
export USER_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length 65536 --language-only --max-running-requests 8 --reasoning-parser qwen3 --tool-call-parser qwen3_coder"
export TAU2_USER_TEMPERATURE=0.0
export TAU2_USER_TOP_P=1.0
export TAU2_USER_MAX_TOKENS=512
export TAU2_USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'

export USE_REWARD_SHAPING=0
export TAU2_RL_TRAJECTORY_DUMP_PATH="${ARTIFACT_DIR}/trajectories/${MODE}.jsonl"
export USER_SGLANG_LOG="${ARTIFACT_DIR}/user_sglang_${MODE}.log"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
export WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}-${ARM}-${MODE}}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
