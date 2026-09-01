#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tau2-agent-single-call-v1}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
TAU2_RL_ARM="${TAU2_RL_ARM:-}"
if [[ -n "${TAU2_RL_ARM}" ]]; then
  ARTIFACT_DIR="${EXPERIMENT_DIR}/arms/${TAU2_RL_ARM}"
  RUN_ID_PREFIX="single_call_v1_${TAU2_RL_ARM}"
  WANDB_GROUP_DEFAULT="tau2-agent-single-call-v1-rl-${TAU2_RL_ARM}"
else
  ARTIFACT_DIR="${EXPERIMENT_DIR}"
  RUN_ID_PREFIX="single_call_v1"
  WANDB_GROUP_DEFAULT="tau2-agent-single-call-v1-rl"
fi
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809}"
RL_CKPT_ROOT="${RL_CKPT_ROOT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_20260809}"
SMOKE_CKPT_ROOT="${SMOKE_CKPT_ROOT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_smoke_20260809}"
STAGE="${1:-stage-a}"

if [[ $# -gt 1 || ! "${STAGE}" =~ ^(smoke|stage-a|stage-b|stage-final|stage-final130|stage-long200|stage-iter99-to-129)$ ]]; then
  echo "Usage: $0 <smoke|stage-a|stage-b|stage-final|stage-final130|stage-long200|stage-iter99-to-129>" >&2
  exit 2
fi
if [[ ! -d "${SFT_CKPT_ROOT}/final_hf" ]]; then
  echo "[ERROR] missing single-call-v1 SFT conversion: ${SFT_CKPT_ROOT}/final_hf" >&2
  exit 1
fi
if [[ ! -f "${SFT_CKPT_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  echo "[ERROR] missing SFT torch-dist checkpoint root: ${SFT_CKPT_ROOT}" >&2
  exit 1
fi

if [[ "${STAGE}" == "smoke" ]]; then
  ACTIVE_CKPT_ROOT="${SMOKE_CKPT_ROOT}"
else
  ACTIVE_CKPT_ROOT="${RL_CKPT_ROOT}"
fi
HF_CHECKPOINT_VALUE="${SFT_CKPT_ROOT}/final_hf"
LOAD_DIR_VALUE="${ACTIVE_CKPT_ROOT}"
START_ROLLOUT_ID_VALUE=""
LATEST_FILE="${ACTIVE_CKPT_ROOT}/latest_checkpointed_iteration.txt"
latest=""
if [[ -f "${LATEST_FILE}" ]]; then
  latest="$(tr -d '[:space:]' < "${LATEST_FILE}")"
  if [[ ! "${latest}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] invalid latest checkpoint iteration: ${latest}" >&2
    exit 1
  fi
fi

case "${STAGE}" in
  smoke)
    if [[ -n "${latest}" ]]; then
      echo "[ERROR] smoke checkpoint root already contains iteration ${latest}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=1
    SAVE_INTERVAL_VALUE=1
    DOMAIN_QUOTA="telecom:3,airline:2,retail:1"
    OVERRIDE_SCHEDULER=0
    ;;
  stage-a)
    if [[ -n "${latest}" && "${latest}" -ge 9 ]]; then
      echo "[ERROR] stage-a is already complete at iteration ${latest}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=10
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:3,airline:2,retail:1"
    OVERRIDE_SCHEDULER=0
    ;;
  stage-b)
    if [[ -z "${latest}" || "${latest}" -lt 9 || "${latest}" -ge 19 ]]; then
      echo "[ERROR] stage-b requires latest iteration 9..18; latest=${latest:-missing}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=20
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:3,airline:1,retail:2"
    OVERRIDE_SCHEDULER=1
    ;;
  stage-final)
    if [[ -z "${latest}" || "${latest}" -lt 19 || "${latest}" -ge 99 ]]; then
      echo "[ERROR] stage-final requires latest iteration 19..98; latest=${latest:-missing}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=100
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:2,airline:2,retail:2"
    OVERRIDE_SCHEDULER=1
    ;;
  stage-final130)
    if [[ -z "${latest}" || "${latest}" -lt 19 || "${latest}" -ge 129 ]]; then
      echo "[ERROR] stage-final130 requires latest iteration 19..128; latest=${latest:-missing}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=130
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:2,airline:2,retail:2"
    OVERRIDE_SCHEDULER=1
    ;;
  stage-long200)
    if [[ -z "${latest}" || "${latest}" -lt 99 || "${latest}" -ge 199 ]]; then
      echo "[ERROR] stage-long200 requires latest iteration 99..198; latest=${latest:-missing}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=200
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:2,airline:2,retail:2"
    OVERRIDE_SCHEDULER=1
    ;;
  stage-iter99-to-129)
    if [[ -z "${TAU2_RL_INIT_HF_CHECKPOINT:-}" || ! -d "${TAU2_RL_INIT_HF_CHECKPOINT}" ]]; then
      echo "[ERROR] stage-iter99-to-129 requires TAU2_RL_INIT_HF_CHECKPOINT" >&2
      exit 1
    fi
    HF_CHECKPOINT_VALUE="${TAU2_RL_INIT_HF_CHECKPOINT}"
    if [[ -z "${latest}" ]]; then
      LOAD_DIR_VALUE="${TAU2_RL_INIT_HF_CHECKPOINT}"
      START_ROLLOUT_ID_VALUE=100
    elif [[ "${latest}" -lt 109 || "${latest}" -ge 129 ]]; then
      echo "[ERROR] stage-iter99-to-129 resume requires latest iteration 109..128; latest=${latest}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=130
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:2,airline:2,retail:2"
    OVERRIDE_SCHEDULER=0
    ;;
esac

if [[ -n "${TAU2_RL_FIXED_DOMAIN_QUOTA:-}" ]]; then
  DOMAIN_QUOTA="${TAU2_RL_FIXED_DOMAIN_QUOTA}"
fi

export HF_CHECKPOINT="${HF_CHECKPOINT_VALUE}"
export REF_LOAD="${SFT_CKPT_ROOT}"
export SAVE_DIR="${ACTIVE_CKPT_ROOT}"
export LOAD_DIR="${LOAD_DIR_VALUE}"
if [[ -n "${START_ROLLOUT_ID_VALUE}" ]]; then
  export START_ROLLOUT_ID="${START_ROLLOUT_ID_VALUE}"
else
  unset START_ROLLOUT_ID
fi
export PREPARED_RL_DATA="${PREPARED_RL_DATA:-${ARTIFACT_DIR}/data/areal_tau2_rl_train_all.jsonl}"
export TAU2_AGENT_PROTOCOL_PROFILE="${TAU2_AGENT_PROTOCOL_PROFILE:-strict-single-v1}"
if [[ "${TAU2_AGENT_PROTOCOL_PROFILE}" != "strict-single-v1" ]]; then
  echo "[ERROR] single-call-v1 RL requires TAU2_AGENT_PROTOCOL_PROFILE=strict-single-v1" >&2
  exit 1
fi
export LOSS_MASK_TYPE="qwen3_full"
export TAU2_TURN_CREDIT_VERSION="${TAU2_TURN_CREDIT_VERSION:-turn-credit-v1}"
export TAU2_TURN_CREDIT_REALLOCATION_WEIGHT="${TAU2_TURN_CREDIT_REALLOCATION_WEIGHT:-0.1}"
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS="${TAU2_REPLACE_ZERO_SIGNAL_GROUPS:-1}"
export DATA_SOURCE_PATH="filters.DomainQuotaDataSource"
export TAU2_RL_DOMAIN_QUOTA="${DOMAIN_QUOTA}"
export TAU2_RL_DOMAIN=""
export SKIP_PREFLIGHT=0

export TOTAL_GPUS=8
export RAY_GPUS=6
export AGENT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
export USER_CUDA_VISIBLE_DEVICES="7"
export ROLLOUT_BATCH_SIZE=6
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=48
export NUM_ROLLOUT="${NUM_ROLLOUT_VALUE}"
export SAVE_INTERVAL="${SAVE_INTERVAL_VALUE}"
export OVERRIDE_OPT_PARAM_SCHEDULER="${OVERRIDE_SCHEDULER}"
export USE_DYNAMIC_FILTER=1

export LR=2e-6
export KL_LOSS_TYPE=k2
export KL_LOSS_COEF=0.01
export KL_COEF=0
export ENTROPY_COEF=0
export EPS_CLIP=0.4
export EPS_CLIP_HIGH=0.4

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}"
export AGENT_MAX_TOKENS=1024
export TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-60}"
export TAU2_MAX_ERRORS=10
export TAU2_USER_TEMPERATURE=0.0
export TAU2_USER_MAX_TOKENS=512
export TAU2_RL_MAX_TRAIN_TOKENS=16384
export TAU2_RL_MAX_ROLLOUT_RETRIES=2

export USE_REWARD_SHAPING=1
export TAU2_REWARD_ALPHA=0.25
export TAU2_PARTIAL_TOOL_NAME_WEIGHT=0.25
export TAU2_PARTIAL_ARGUMENT_WEIGHT=0.35
export TAU2_PARTIAL_DB_WEIGHT=0.25
export TAU2_PARTIAL_ENV_ASSERTION_WEIGHT=0.10
export TAU2_PARTIAL_COMMUNICATE_WEIGHT=0.05
export TAU2_PENALTY_MALFORMED_JSON=0.15
export TAU2_PENALTY_NONEXISTENT_TOOL=0.15
export TAU2_PENALTY_WRONG_ARGUMENT_FIELD=0.10
export TAU2_PENALTY_TOOL_EXECUTION_ERROR=0.25
export TAU2_PENALTY_REPETITION=0.05
export TAU2_PENALTY_MAX_STEPS=0.20

export USER_MODEL_PATH="${USER_MODEL_PATH:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"
export TAU2_RL_ARM
export TAU2_RL_RUN_ID="${TAU2_RL_RUN_ID:-${RUN_ID_PREFIX}_${STAGE}}"
export TAU2_RL_TRAJECTORY_DUMP_PATH="${TAU2_RL_TRAJECTORY_DUMP_PATH:-${ARTIFACT_DIR}/trajectories/${STAGE}.jsonl}"
export USER_SGLANG_LOG="${USER_SGLANG_LOG:-${ARTIFACT_DIR}/user_sglang_${STAGE}.log}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
export WANDB_GROUP="${WANDB_GROUP:-${WANDB_GROUP_DEFAULT}}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
