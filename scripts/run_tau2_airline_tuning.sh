#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^(vanilla-grpo|progress-db-count-v1)$ ]]; then
  echo "Usage: $0 <vanilla-grpo|progress-db-count-v1> <learning-rate>" >&2
  exit 2
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_AGENT_ROOT="$(dirname "${PROJECT_ROOT}")"
export EXPERIMENT_NAME=tau2-airline-db-count-tuning
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
export TAU2_RL_RECIPE="$1"
export LR="$2"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export TAU2_RUN_NAME="${TAU2_RUN_NAME:-${RUN_STAMP}-${TAU2_RL_RECIPE}-lr${LR}}"
RUN_DIR="${EXPERIMENT_DIR}/arms/async/${TAU2_RUN_NAME}"
export PREPARED_RL_DATA="${PREPARED_RL_DATA:-${EXPERIMENT_DIR}/data/airline_train.jsonl}"
export SFT_CKPT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909"
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/iter_0004673_hf"
export REF_LOAD="${SFT_CKPT_ROOT}"
export REF_CKPT_STEP=4673
export TAU2_TURN_CREDIT_VERSION="$([[ "$1" == progress-db-count-v1 ]] && echo progress-db-count-v1 || true)"
export TAU2_PROGRESS_WEIGHT="${TAU2_PROGRESS_WEIGHT:-1.0}"
export TAU2_FORMAT_WEIGHT="${TAU2_FORMAT_WEIGHT:-1.0}"
export TAU2_PROGRESS_GAMMA="${TAU2_PROGRESS_GAMMA:-0.98}"
export TAU2_DROP_UNIFORM_OUTCOME_GROUPS=1
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
export TAU2_RAW_TOKENS=1
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
export TAU2_POOL_CAPACITY="${TAU2_POOL_CAPACITY:-32}"
export TAU2_MAX_PENDING_GROUPS="${TAU2_MAX_PENDING_GROUPS:-$((2 * TAU2_POOL_CAPACITY))}"
# Keep the sampling recipe validated by fast21168 iter19. Uniform-source
# control21429 remains an explicit override; it scored lower on Airline.
export DATA_SOURCE_PATH="${DATA_SOURCE_PATH:-random_tasks.ShuffledTaskDataSource}"
export TAU2_ENVIRONMENT_WORKERS="${TAU2_ENVIRONMENT_WORKERS:-4}"
# H100/1200-token Agent requests finish within20s in the long-run trace.
# Bound a lost HTTP response before it holds the weight-update barrier for600s.
export TAU2_AGENT_TIMEOUT="${TAU2_AGENT_TIMEOUT:-60}"
# Consume ready groups without waiting for unfinished old-policy groups.
export TAU2_MAX_POLICY_LAG="${TAU2_MAX_POLICY_LAG:--1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-30}"
export SAVE_INTERVAL=10
export TRAIN_SEED=1234
export ROLLOUT_SEED=42
export TAU2_SERIAL_CLEANUP=1
# Each job starts in a fresh container; preserve the reward watcher below.
export TAU2_RL_CLEANUP=0
export USE_WANDB=0
export WANDB_MODE=offline

mkdir -p "${RUN_DIR}" "${EXPERIMENT_DIR}/plots/${TAU2_RUN_NAME}"
# The original Megatron SFT shards may have been removed while its HF export remains.
# Recreate weights only; training still starts with a fresh optimizer.
if [[ ! -d "${SFT_CKPT_ROOT}/iter_0004673" ]]; then
  export REF_LOAD="${RUN_DIR}/sft_init_torch_dist"
  export REF_CKPT_STEP=1
  if [[ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
    (
      source "${PROJECT_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh"
      export PYTHONPATH="/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
      cd "${PROJECT_ROOT}"
      python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${HF_CHECKPOINT}" --save "${REF_LOAD}"
    )
  fi
fi
echo "Airline tuning: recipe=${TAU2_RL_RECIPE} lr=${LR} batch=${GLOBAL_BATCH_SIZE} updates=${NUM_ROLLOUT} pool=${TAU2_POOL_CAPACITY} max_lag=${TAU2_MAX_POLICY_LAG} progress_weight=${TAU2_PROGRESS_WEIGHT}"
python3 -u "${PROJECT_ROOT}/scripts/watch_tau2_four_domain_rewards.py" \
  --experiment-dir "${EXPERIMENT_DIR}" --run-name "${TAU2_RUN_NAME}" \
  --batch-size "${GLOBAL_BATCH_SIZE}" --target-updates "${NUM_ROLLOUT}" \
  --interval 300 --title 'Airline asynchronous' >"${RUN_DIR}/reward_watch.log" 2>&1 &
WATCH_PID=$!
trap 'kill "${WATCH_PID}" 2>/dev/null || true' EXIT

if bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_tau2_areal_async.sh" async train \
    2>&1 | tee -a "${RUN_DIR}/run.log"; then
  train_status=0
else
  train_status=$?
fi
kill "${WATCH_PID}" 2>/dev/null || true
wait "${WATCH_PID}" 2>/dev/null || true
python3 "${PROJECT_ROOT}/scripts/plot_tau2_four_domain_rewards.py" \
  --run-dir "${RUN_DIR}" --output-dir "${EXPERIMENT_DIR}/plots/${TAU2_RUN_NAME}" \
  --label "${TAU2_RL_RECIPE} LR=${LR}" --batch-size "${GLOBAL_BATCH_SIZE}" \
  --title 'Airline asynchronous' || echo 'Reward plotting failed; see reward_watch.log.' >&2
if [[ "${train_status}" -ne 0 ]]; then
  exit "${train_status}"
fi
ray stop --force

if [[ "${RUN_BASELINE_EVAL:-0}" == 1 ]]; then
  bash "${PROJECT_ROOT}/scripts/eval_tau2_airline_tuning.sh" sft
fi
bash "${PROJECT_ROOT}/scripts/eval_tau2_airline_tuning.sh" "${TAU2_RUN_NAME}" "$((NUM_ROLLOUT - 1))"
