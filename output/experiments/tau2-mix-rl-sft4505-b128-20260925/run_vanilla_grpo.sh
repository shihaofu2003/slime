#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
cd "${PROJECT_ROOT}"
export EXPERIMENT_NAME=tau2-mix-rl-sft4505-b128-20260925
experiment_dir="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
export TAU2_RUN_NAME="${TAU2_RUN_NAME:-20260925_vanilla-grpo-train}"
run_dir="${experiment_dir}/arms/async/${TAU2_RUN_NAME}"
plot_dir="${experiment_dir}/plots/${TAU2_RUN_NAME}"
user_endpoint="${experiment_dir}/user_endpoint.env"

export SFT_CKPT_ROOT="$(dirname "${PROJECT_ROOT}")/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920"
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/iter_0004505_hf"
export REF_LOAD="${SFT_CKPT_ROOT}" REF_CKPT_STEP=4505
unset LOAD_DIR CKPT_STEP START_ROLLOUT_ID

export LR=2e-6 TRAIN_SEED=1234 ROLLOUT_SEED=42
export TAU2_TURN_CREDIT_VERSION=""
export USE_REWARD_SHAPING=0
export TAU2_PROGRESS_WEIGHT=1.0 TAU2_FORMAT_WEIGHT=1.0 TAU2_PROGRESS_GAMMA=0.98
export TAU2_RAW_TOKENS=1 TAU2_OPD_PURE=0
export TAU2_DROP_UNIFORM_OUTCOME_GROUPS=1 TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
export ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=8 GLOBAL_BATCH_SIZE=128
export NUM_ROLLOUT="${NUM_ROLLOUT:-140}" SAVE_INTERVAL=10
export TAU2_POOL_CAPACITY=32 TAU2_MAX_PENDING_GROUPS=64 TAU2_MAX_POLICY_LAG=-1
export DATA_SOURCE_PATH=random_tasks.RandomTaskDataSource
export PREPARED_RL_DATA="${experiment_dir}/data/train.jsonl"
export TAU2_ENVIRONMENT_WORKERS=4
export TAU2_AGENT_CONCURRENCY=48 TAU2_STEP_CONCURRENCY=160
export TAU2_AGENT_TIMEOUT=60 TAU2_BANKING_AGENT_TIMEOUT=600
export USER_SGLANG=0 TAU2_USER_API_KEY=EMPTY
export TAU2_RL_CLEANUP=0 TAU2_SERIAL_CLEANUP=1 USE_WANDB=0 WANDB_MODE=offline

mkdir -p "${experiment_dir}/data" "${run_dir}" "${plot_dir}"
# This combined pool was checked against all four SFT4505 expert task files.
cp "${PROJECT_ROOT}/output/experiments/tau2-async-db-count-four-domain/data/train.jsonl" "${PREPARED_RL_DATA}"

user_ready=0
for ((attempt=0; attempt<240; attempt++)); do
  if [[ -f "${user_endpoint}" ]]; then
    set -a
    source "${user_endpoint}"
    set +a
    if curl -fsS --max-time 5 "${TAU2_USER_API_BASE}/models" >/dev/null; then
      user_ready=1
      echo "MIX_RL_USER_READY api=${TAU2_USER_API_BASE}"
      break
    fi
  fi
  if (( attempt % 10 == 0 )); then echo "Waiting for User service: ${user_endpoint}"; fi
  sleep 30
done
[[ "${user_ready}" == 1 ]] || { echo 'User service did not become reachable within two hours.' >&2; exit 1; }

echo "MIX_RL_START updates=${NUM_ROLLOUT} source=${DATA_SOURCE_PATH} data=${PREPARED_RL_DATA}"
python3 -u scripts/watch_tau2_four_domain_rewards.py \
  --experiment-dir "${experiment_dir}" --run-name "${TAU2_RUN_NAME}" \
  --batch-size 128 --target-updates "${NUM_ROLLOUT}" --interval 300 \
  --title 'Four-domain vanilla GRPO' >"${run_dir}/reward_watch.log" 2>&1 &
watch_pid=$!
trap 'kill "${watch_pid}" 2>/dev/null || true' EXIT

if bash examples/tau2-bench/rl/run_tau2_areal_async.sh async train \
    2>&1 | tee "${run_dir}/run.log"; then
  train_status=0
else
  train_status=$?
fi
kill "${watch_pid}" 2>/dev/null || true
wait "${watch_pid}" 2>/dev/null || true
trap - EXIT
ray stop --force >"${run_dir}/ray_stop.log" 2>&1
[[ "${train_status}" == 0 ]] || exit "${train_status}"
python3 scripts/plot_tau2_four_domain_rewards.py --run-dir "${run_dir}" \
  --output-dir "${plot_dir}" --batch-size 128 --label 'LR2e-6 batch128 binary task reward' \
  --title 'Four-domain vanilla GRPO' || echo 'Reward plotting failed.' >&2
