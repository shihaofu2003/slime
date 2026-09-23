#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 || ! "$1" =~ ^(sync|async)$ || ! "$2" =~ ^(smoke|train20|train100|train200|train)$ ]]; then
  echo "Usage: $0 <sync|async> <smoke|train20|train100|train200|train>" >&2
  exit 2
fi
ARM="$1"
MODE="$2"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SERVICE_AGENT_ROOT="$(dirname "${PROJECT_ROOT}")"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tau2-areal-async-rl}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
ARTIFACT_DIR="${EXPERIMENT_DIR}/arms/${ARM}/${TAU2_RUN_NAME:-ready-${MODE}}"
export SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${SFT_CKPT_ROOT}/final_hf}"
export REF_LOAD="${REF_LOAD:-${SFT_CKPT_ROOT}}"
export SAVE_DIR="${ARTIFACT_DIR}/checkpoints"
export LOAD_DIR="${LOAD_DIR:-${SAVE_DIR}}"
export PREPARED_RL_DATA="${PREPARED_RL_DATA:-${SERVICE_AGENT_ROOT}/datasets/tau2/rl/areal_tasks_slime.jsonl}"
export SKIP_PREPARE_RL_DATA=1
export TAU2_SRC="${TAU2_SRC:-${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl/dependencies/tau2/src}"
export TAU2_VALIDATE_APN=1
export TAU2_DATA_DIR="${SERVICE_AGENT_ROOT}/tau2-bench/data"
export TAU2_DISJOINT_GPUS=1
export TAU2_RAW_TOKENS="${TAU2_RAW_TOKENS:-1}"
export TAU2_RL_RAISE_ERRORS=1
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train_async.py}"
export SGLANG_SERVER_CONCURRENCY=16
export TAU2_SAMPLING_MODE="${ARM}"
export TAU2_POOL_CAPACITY="${TAU2_POOL_CAPACITY:-10}"
export DATA_SOURCE_PATH="${DATA_SOURCE_PATH:-random_tasks.RandomTaskDataSource}"
export TAU2_RL_DOMAIN_QUOTA=""
if [[ "${TAU2_RAW_TOKENS}" == 0 ]]; then
  [[ "${ARM}" == sync && "${MODE}" == smoke ]] || { echo "Legacy token control is a sync smoke diagnostic" >&2; exit 2; }
  export TRAIN_ENTRYPOINT=train.py
  export DATA_SOURCE_PATH="filters.DomainQuotaDataSource"
  export TAU2_RL_DOMAIN_QUOTA="airline:1,retail:2,telecom:2"
fi
case "${MODE}" in
  smoke) NUM_ROLLOUT_VALUE="${NUM_ROLLOUT:-${SMOKE_UPDATES:-3}}"; SAVE_INTERVAL_VALUE=1 ;;
  train20) NUM_ROLLOUT_VALUE=20; SAVE_INTERVAL_VALUE=10 ;;
  train100) NUM_ROLLOUT_VALUE=100; SAVE_INTERVAL_VALUE=10 ;;
  train200) NUM_ROLLOUT_VALUE=200; SAVE_INTERVAL_VALUE=10 ;;
  train) NUM_ROLLOUT_VALUE="${NUM_ROLLOUT:-556}"; SAVE_INTERVAL_VALUE="${SAVE_INTERVAL:-10}" ;;
esac
mkdir -p "${ARTIFACT_DIR}/trajectories"
export TAU2_AGENT_PROTOCOL_PROFILE="official-native"
export LOSS_MASK_TYPE="qwen3_full"
export TAU2_TURN_CREDIT_VERSION="${TAU2_TURN_CREDIT_VERSION:-}"
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
export TAU2_RL_DOMAIN=""
export SKIP_PREFLIGHT=0
export LOGURU_LEVEL="${LOGURU_LEVEL:-WARNING}"

export TOTAL_GPUS=8
# Async training uses the dedicated User job; sync retains its local-User default.
export USER_SGLANG="${USER_SGLANG:-$([[ "${ARM}" == async ]] && echo 0 || echo 1)}"
if [[ "${USER_SGLANG}" == 0 ]]; then
  export TAU2_USER_API_BASE="${TAU2_USER_API_BASE:-http://10.119.96.116:30000/v1}"
  export TAU2_USER_API_KEY="${TAU2_USER_API_KEY:-EMPTY}"
  export RAY_GPUS=8
  export AGENT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  export USER_CUDA_VISIBLE_DEVICES=""
else
  export RAY_GPUS=6
  export AGENT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
  export USER_CUDA_VISIBLE_DEVICES="6,7"
fi
export ROLLOUT_NUM_GPUS_PER_ENGINE=2
export AGENT_ROUTER_POLICY="${AGENT_ROUTER_POLICY:-round_robin}"
export USER_TP=2
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-5}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export NUM_ROLLOUT="${NUM_ROLLOUT_VALUE}"
export SAVE_INTERVAL="${SAVE_INTERVAL_VALUE}"
export USE_DYNAMIC_FILTER=1

export TRAIN_SEED="${TRAIN_SEED:-1234}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export LR="${LR:-2e-6}"
export KL_LOSS_TYPE=k2
export KL_LOSS_COEF=0
export KL_COEF=0
export ENTROPY_COEF=0
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.2}"

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
export TAU2_USER_CONCURRENCY="${TAU2_USER_CONCURRENCY:-32}"
export USER_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length 65536 --language-only --max-running-requests ${TAU2_USER_CONCURRENCY} --reasoning-parser qwen3 --tool-call-parser qwen3_coder"
export TAU2_USER_TEMPERATURE=0.0
export TAU2_USER_TOP_P=1.0
export TAU2_USER_MAX_TOKENS=512
export TAU2_USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'

export USE_REWARD_SHAPING="$([[ -n "${TAU2_TURN_CREDIT_VERSION}" ]] && echo 1 || echo 0)"
export TAU2_RL_TRAJECTORY_DUMP_PATH="${ARTIFACT_DIR}/trajectories/${MODE}.jsonl"
export USER_SGLANG_LOG="${ARTIFACT_DIR}/user_sglang_${MODE}_$(date +%Y%m%d_%H%M%S).log"
export USE_WANDB="${USE_WANDB:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
export WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}-${ARM}-${MODE}}"

# Bash reads scripts incrementally; run a snapshot while worktree edits continue.
RUN_SCRIPT="${ARTIFACT_DIR}/run_training_$(date +%Y%m%d_%H%M%S).sh"
cp "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh" "${RUN_SCRIPT}"
exec bash "${RUN_SCRIPT}"
