#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ! "$1" =~ ^(vanilla-grpo|progress-db-count-v1)$ || ! "$2" =~ ^(smoke|train)$ || ( $# -eq 3 && ! "$3" =~ ^(sync|async)$ ) ]]; then
  echo "Usage: $0 <vanilla-grpo|progress-db-count-v1> <smoke|train> [sync|async]" >&2
  exit 2
fi

RECIPE="$1"
MODE="$2"
ARM="${3:-async}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_AGENT_ROOT="$(dirname "${PROJECT_ROOT}")"
EXPERIMENT_NAME="tau2-async-db-count-four-domain"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
DATA_PATH="${EXPERIMENT_DIR}/data/train.jsonl"
RUN_NAME="${TAU2_RUN_NAME:-$(date +%Y%m%d_%H%M%S)-${RECIPE}-${MODE}}"

if [[ ! -f "${DATA_PATH}" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl/dependencies/tau2/src:${PYTHONPATH:-}" python3 "${PROJECT_ROOT}/examples/tau2-bench/rl/prepare_four_domain_rl_data.py" --output "${DATA_PATH}"
fi

export TAU2_RL_RECIPE="${RECIPE}"
export PROJECT_ROOT
export EXPERIMENT_NAME
export EXPERIMENT_DIR
export PREPARED_RL_DATA="${DATA_PATH}"
export SKIP_PREPARE_RL_DATA=1
export TAU2_RUN_NAME="${RUN_NAME}"
export WANDB_GROUP="${EXPERIMENT_NAME}-${RUN_NAME}"
export SFT_CKPT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909"
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/iter_0004673_hf"
export REF_LOAD="${SFT_CKPT_ROOT}"
export REF_CKPT_STEP=4673
export TAU2_TURN_CREDIT_VERSION="$([[ "${RECIPE}" == progress-db-count-v1 ]] && echo progress-db-count-v1 || true)"
export TAU2_RAW_TOKENS=1
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
export TAU2_DROP_UNIFORM_OUTCOME_GROUPS="${TAU2_DROP_UNIFORM_OUTCOME_GROUPS:-1}"
export NUM_ROLLOUT="$([[ "${MODE}" == smoke ]] && echo "${SMOKE_UPDATES:-5}" || echo 556)"
export SAVE_INTERVAL=10
export ROLLOUT_BATCH_SIZE=5
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=40
export TAU2_MAX_STEPS=200
export TAU2_RL_MAX_TRAIN_TOKENS=16384
export TAU2_RL_MAX_ROLLOUT_RETRIES=2
export LR=2e-6
export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export TRAIN_SEED=1234
export ROLLOUT_SEED=42
export TAU2_AGENT_PROTOCOL_PROFILE=official-native
export LOSS_MASK_TYPE=qwen3_full
export TAU2_POOL_CAPACITY=20
export TAU2_MAX_POLICY_LAG=-1
export TRAIN_ENTRYPOINT="$([[ "${ARM}" == sync ]] && echo train.py || echo train_async.py)"
export TAU2_RL_DOMAIN=""

export TAU2_SAMPLING_MODE="${ARM}"
exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_tau2_areal_async.sh" "${ARM}" "$([[ "${MODE}" == smoke ]] && echo smoke || echo train)"
