#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?Usage: $0 <smoke|train40|train100|continue100>}"
PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the credit-assignment worktree}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to an independent timestamped experiment directory}"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
SFT_CKPT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903"
ARTIFACT_DIR="${RUN_DIR}/${MODE}"
EXPERIMENT_NAME=tau2-credit-assignment
ARM="${TAU2_RL_RECIPE:-progress-rtg-v1}"
case "${ARM}" in
  progress-rtg-v1) export TAU2_TURN_CREDIT_VERSION=progress-rtg-v1 USE_REWARD_SHAPING=1 ;;
  vanilla-grpo) export TAU2_TURN_CREDIT_VERSION="" USE_REWARD_SHAPING=0 ;;
  *) echo "Unknown recipe: ${ARM}" >&2; exit 2 ;;
esac
case "$MODE" in
  smoke) ACTIVE_CKPT_ROOT="${RUN_DIR}/smoke-checkpoints"; NUM_ROLLOUT_VALUE=1; SAVE_INTERVAL_VALUE=1 ;;
  train40) ACTIVE_CKPT_ROOT="${RUN_DIR}/checkpoints"; NUM_ROLLOUT_VALUE=40; SAVE_INTERVAL_VALUE=10 ;;
  train100) ACTIVE_CKPT_ROOT="${RUN_DIR}/checkpoints"; NUM_ROLLOUT_VALUE=100; SAVE_INTERVAL_VALUE=10 ;;
  continue100)
    ACTIVE_CKPT_ROOT="${RUN_DIR}/checkpoints"; NUM_ROLLOUT_VALUE=100; SAVE_INTERVAL_VALUE=10
    test -f "${ACTIVE_CKPT_ROOT}/latest_checkpointed_iteration.txt"
    export OVERRIDE_OPT_PARAM_SCHEDULER=1 ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
mkdir -p "${ARTIFACT_DIR}/trajectories"
export SFT_CKPT_ROOT
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/final_hf"
export REF_LOAD="${SFT_CKPT_ROOT}"
export SAVE_DIR="${ACTIVE_CKPT_ROOT}"
export LOAD_DIR="${ACTIVE_CKPT_ROOT}"
export PREPARED_RL_DATA="${SERVICE_AGENT_ROOT}/datasets/tau2/rl/areal_tasks_slime.jsonl"
export SKIP_DATA_PREPARATION=1

export TAU2_AGENT_PROTOCOL_PROFILE="official-native"
export LOSS_MASK_TYPE="qwen3_full"
export TAU2_PROGRESS_GAMMA=0.98
export TAU2_PROGRESS_WEIGHT=1.0
export TAU2_FORMAT_WEIGHT=1.0
export TAU2_PROGRESS_DIAGNOSTICS_PATH="${ARTIFACT_DIR}/credit.jsonl"
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
export TAU2_RL_DOMAIN_QUOTA="${TAU2_RL_DOMAIN_QUOTA-}"
if [[ -n "${TAU2_RL_DOMAIN_QUOTA}" ]]; then
  export DATA_SOURCE_PATH=filters.DomainQuotaDataSource
else
  export DATA_SOURCE_PATH=slime.rollout.data_source.RolloutDataSourceWithBuffer
fi
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

export TAU2_RL_TRAJECTORY_DUMP_PATH="${ARTIFACT_DIR}/trajectories/${MODE}.jsonl"
export USER_SGLANG_LOG="${ARTIFACT_DIR}/user_sglang_${MODE}.log"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
export WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}-${ARM}-${MODE}}"

# Run recipe-specific tests in the job image before the existing launcher preflight.
TAU2_PROGRESS_DIAGNOSTICS_PATH="" \
PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/examples/tau2-bench/rl:${PROJECT_ROOT}/examples/tau2-bench/shared:${SERVICE_AGENT_ROOT}/tau2-bench/src:${PYTHONPATH:-}" \
  python3 "${PROJECT_ROOT}/examples/tau2-bench/rl/test_progress.py"
exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
