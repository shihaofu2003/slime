#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
cd "${PROJECT_ROOT}"
export EXPERIMENT_NAME=tau2-mix-rl-sft4505-reward-v2-20260927
experiment_dir="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
export TAU2_RUN_NAME="${TAU2_RUN_NAME:-20260927_reward-v2-train}"
run_dir="${experiment_dir}/arms/async/${TAU2_RUN_NAME}"
plot_dir="${experiment_dir}/plots/${TAU2_RUN_NAME}"
user_endpoint="${TAU2_USER_ENDPOINT_FILE:-${experiment_dir}/user_endpoint.env}"

export SFT_CKPT_ROOT="$(dirname "${PROJECT_ROOT}")/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920"
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/iter_0004505_hf"
export REF_LOAD="${SFT_CKPT_ROOT}" REF_CKPT_STEP=4505
unset LOAD_DIR CKPT_STEP START_ROLLOUT_ID

export LR=2e-6 TRAIN_SEED=1234 ROLLOUT_SEED=42
export TAU2_TURN_CREDIT_VERSION=progress-db-count-v1
export TAU2_PROGRESS_WEIGHT=1.0 TAU2_FORMAT_WEIGHT=1.0 TAU2_PROGRESS_GAMMA=0.98
export TAU2_RAW_TOKENS=1 TAU2_OPD_PURE=0
export TAU2_DROP_UNIFORM_OUTCOME_GROUPS=0 TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
export TRAIN_ENTRYPOINT=train_tau2_reward_v2.py
# The base launcher resolves this to the new run's credit.jsonl after its
# preflight, so test fixtures cannot append records to experiment diagnostics.
export TAU2_PROGRESS_DIAGNOSTICS_PATH=""
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

if [[ "${1:-}" == --dry-run ]]; then
  python3 - <<'PY'
import json
import os
keys = ["EXPERIMENT_NAME", "TAU2_RUN_NAME", "TRAIN_ENTRYPOINT", "HF_CHECKPOINT", "REF_LOAD", "REF_CKPT_STEP",
        "LR", "TRAIN_SEED", "ROLLOUT_SEED", "TAU2_TURN_CREDIT_VERSION", "TAU2_PROGRESS_WEIGHT",
        "TAU2_FORMAT_WEIGHT", "TAU2_PROGRESS_GAMMA", "TAU2_DROP_UNIFORM_OUTCOME_GROUPS",
        "ROLLOUT_BATCH_SIZE", "N_SAMPLES_PER_PROMPT", "GLOBAL_BATCH_SIZE", "NUM_ROLLOUT",
        "TAU2_POOL_CAPACITY", "TAU2_MAX_PENDING_GROUPS", "TAU2_MAX_POLICY_LAG",
        "TAU2_ENVIRONMENT_WORKERS", "DATA_SOURCE_PATH", "PREPARED_RL_DATA"]
print(json.dumps({key: os.environ[key] for key in keys}, indent=2))
PY
  exit 0
fi

# This CPU preflight is separate so legacy source and CI files stay untouched.
python3 examples/tau2-bench/rl/test_credit_signal_v1.py
python3 examples/tau2-bench/rl/test_reward_v2.py
mkdir -p "${experiment_dir}/data" "${run_dir}" "${plot_dir}"
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

echo "MIX_RL_START recipe=reward-v2-credit-signal-v1 updates=${NUM_ROLLOUT} source=${DATA_SOURCE_PATH}"
python3 -u scripts/plot_tau2_credit_signal_rewards.py \
  --run-dir "${run_dir}" --output-dir "${plot_dir}" --prompt-data "${PREPARED_RL_DATA}" \
  --watch 300 >>"${run_dir}/reward_watch.log" 2>&1 &
watch_pid=$!
trap 'kill "${watch_pid}" 2>/dev/null || true' EXIT

if bash examples/tau2-bench/rl/run_tau2_areal_async.sh async train \
    2>&1 | tee -a "${run_dir}/run.log"; then
  train_status=0
else
  train_status=$?
fi
kill "${watch_pid}" 2>/dev/null || true
wait "${watch_pid}" 2>/dev/null || true
trap - EXIT
ray stop --force >"${run_dir}/ray_stop.log" 2>&1
[[ "${train_status}" == 0 ]] || exit "${train_status}"
python3 scripts/plot_tau2_credit_signal_rewards.py \
  --run-dir "${run_dir}" --output-dir "${plot_dir}" --prompt-data "${PREPARED_RL_DATA}" \
  || echo 'Reward plotting failed.' >&2
