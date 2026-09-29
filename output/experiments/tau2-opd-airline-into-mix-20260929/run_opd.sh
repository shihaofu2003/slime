#!/usr/bin/env bash
# Pure on-policy distillation: write the Airline domain expert's capability
# into the progress-db-count-v1 mix RL student (iter129 init).
#
# Hard requirement honored here: trajectory-level advantage is removed -- the
# OPD path sets every sample reward to 0.0, so GRPO advantages are identically
# zero and the only learning signal is the per-token OPD reverse-KL term
# (student logp - teacher logp) added in apply_opd_kl_to_advantages.
#
# Distillation estimator choice: the validated per-token reverse-KL OPD
# (tinker-cookbook style, teacher logprob at sampled tokens via sglang
# input_token_logprobs) is used instead of a top-k forward-KL variant; it is
# the path exercised by tau2-opd-four-domain-20260923 and needs no new
# training-side plumbing. Group size stays K=1: with rewards pinned to zero
# there is no group baseline, so larger groups only reduce prompt diversity.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
SERVICE_AGENT_ROOT="$(dirname "${PROJECT_ROOT}")"
cd "${PROJECT_ROOT}"

export EXPERIMENT_NAME=tau2-opd-airline-into-mix-20260929
experiment_dir="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
export TAU2_RUN_NAME="${TAU2_RUN_NAME:-20260929_airline-into-mix-iter129-opd}"
run_dir="${experiment_dir}/arms/async/${TAU2_RUN_NAME}"
teacher_endpoint="${experiment_dir}/teacher_endpoints.env"
user_endpoint="${experiment_dir}/user_endpoint.env"

# Student init: progress-db-count-v1 mix RL iter129 (torch_dist), fresh optimizer.
mix_ckpt_root="${experiment_dir}/../tau2-mix-rl-sft4505-credit-signal-v1-20260927/arms/async/20260927_credit-signal-v1-train/checkpoints"
export SFT_CKPT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920"
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/iter_0004505_hf"
export REF_LOAD="${PROJECT_ROOT}/$(realpath --relative-to="${PROJECT_ROOT}" "${mix_ckpt_root}")"
export REF_CKPT_STEP=129
export SAVE_DIR="${run_dir}/checkpoints"
export LOAD_DIR="${SAVE_DIR}"
unset CKPT_STEP START_ROLLOUT_ID

# Airline-only tasks (1148 rows), already in slime prompt format.
export PREPARED_RL_DATA="${PROJECT_ROOT}/output/experiments/tau2-domain-experts-sft4505-b128/data/airline_train.jsonl"
export SKIP_PREPARE_RL_DATA=1
export DATA_SOURCE_PATH=random_tasks.ShuffledTaskDataSource
export TAU2_RL_DOMAIN="" TAU2_RL_DOMAIN_QUOTA=""
export TAU2_SRC="${TAU2_SRC:-${SERVICE_AGENT_ROOT}/tau2-bench/src}"
export TAU2_DATA_DIR="${SERVICE_AGENT_ROOT}/tau2-bench/data"

# Pure OPD: reward 0, no shaping, no turn credit, no dynamic filter.
export TAU2_OPD_PURE=1
export OPD_KL_COEF="${TAU2_OPD_KL_COEF:-1.0}"
export OPD_USE_BEHAVIOR_LOGPROBS=0
export OPD_POST_UPDATE_LOG_INTERVAL=10
export TAU2_OPD_TEACHER_TIMEOUT="${TAU2_OPD_TEACHER_TIMEOUT:-120}"
export TAU2_TURN_CREDIT_VERSION="" TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0 TAU2_DROP_UNIFORM_OUTCOME_GROUPS=0
export USE_REWARD_SHAPING=0
export KL_LOSS_TYPE=k2 KL_LOSS_COEF=0 KL_COEF=0 ENTROPY_COEF=0
export EPS_CLIP=0.2 EPS_CLIP_HIGH=0.2

# Training hyperparameters (match the selected bounded-LR OPD recipe).
export LR="${TAU2_OPD_LR:-2e-6}" TRAIN_SEED=1234 ROLLOUT_SEED=42
export ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=1 GLOBAL_BATCH_SIZE=16
# 1148 airline tasks / 16 per update ~= 72 updates per source pass; 80 updates
# (~1.1 passes) aligns with the 10-update save interval.
export NUM_ROLLOUT="${TAU2_OPD_NUM_UPDATES:-80}" SAVE_INTERVAL=10

# Async rollout, 8 student GPUs: Trainer2 + three TP2 Agent generators.
export TRAIN_ENTRYPOINT=train_async.py TOTAL_GPUS=8 RAY_GPUS=8
export AGENT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export TAU2_DISJOINT_GPUS=1 ROLLOUT_NUM_GPUS_PER_ENGINE=2
export TAU2_RAW_TOKENS=1 TAU2_RL_RAISE_ERRORS=1 TAU2_SAMPLING_MODE=async
export TAU2_POOL_CAPACITY=32 TAU2_MAX_PENDING_GROUPS=64 TAU2_MAX_BUFFERED_GROUPS=32 TAU2_MAX_POLICY_LAG=-1
export TAU2_ENVIRONMENT_WORKERS=4
export TAU2_AGENT_CONCURRENCY=48 TAU2_STEP_CONCURRENCY=160 TAU2_AGENT_TIMEOUT=600
export SGLANG_SERVER_CONCURRENCY=16

# Rollout sampling / limits (on-policy distillation distribution).
export AGENT_MAX_TOKENS=1200 ROLLOUT_TEMPERATURE=1.0 ROLLOUT_TOP_P=1.0
export TAU2_MAX_STEPS=200 TAU2_MAX_ERRORS=10
export TAU2_RL_MAX_TRAIN_TOKENS=16384 TAU2_RL_MAX_ROLLOUT_RETRIES=2
export TAU2_AGENT_PROTOCOL_PROFILE=official-native LOSS_MASK_TYPE=qwen3_full

# External persistent services (separate 4-GPU User job and 2-GPU expert job).
export USER_SGLANG=0 TAU2_USER_API_KEY=EMPTY
export TAU2_USER_MODEL="Qwen3.6-27B-tau2-user-nonthinking"
export TAU2_USER_TEMPERATURE=0.0 TAU2_USER_TOP_P=1.0 TAU2_USER_MAX_TOKENS=512
export TAU2_USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'

export SKIP_PREFLIGHT=0 USE_DYNAMIC_FILTER=0 TAU2_RL_CLEANUP=0 TAU2_SERIAL_CLEANUP=1
export USE_WANDB=0 WANDB_MODE=offline
export TAU2_RL_TRAJECTORY_DUMP_PATH="${run_dir}/trajectories.jsonl"

if [[ "${1:-}" == --dry-run ]]; then
  python3 - <<'PY'
import json
import os
keys = ["EXPERIMENT_NAME", "TAU2_RUN_NAME", "TRAIN_ENTRYPOINT", "HF_CHECKPOINT", "REF_LOAD",
        "REF_CKPT_STEP", "SAVE_DIR", "PREPARED_RL_DATA", "DATA_SOURCE_PATH", "TAU2_OPD_PURE",
        "OPD_KL_COEF", "LR", "TRAIN_SEED", "ROLLOUT_SEED", "ROLLOUT_BATCH_SIZE",
        "N_SAMPLES_PER_PROMPT", "GLOBAL_BATCH_SIZE", "NUM_ROLLOUT", "SAVE_INTERVAL",
        "TAU2_POOL_CAPACITY", "TAU2_MAX_PENDING_GROUPS", "TAU2_MAX_POLICY_LAG",
        "TAU2_ENVIRONMENT_WORKERS", "ROLLOUT_TEMPERATURE", "TAU2_MAX_STEPS"]
print(json.dumps({key: os.environ.get(key) for key in keys}, indent=2))
PY
  exit 0
fi

for required in "${HF_CHECKPOINT}" "${REF_LOAD}/latest_checkpointed_iteration.txt" \
                "${REF_LOAD}/iter_0000129" "${PREPARED_RL_DATA}"; do
  [[ -e "${required}" ]] || { echo "[ERROR] Missing OPD input: ${required}" >&2; exit 1; }
done
mkdir -p "${run_dir}/logs" "${SAVE_DIR}"

python3 -m pytest "${PROJECT_ROOT}/examples/tau2-bench/opd/test_tau2_opd.py" -q

# Wait for the persistent Airline expert, then publish its URL to Ray workers.
teacher_ready=0
for ((attempt=0; attempt<240; attempt++)); do
  if [[ -s "${teacher_endpoint}" ]]; then
    # shellcheck disable=SC1090
    source "${teacher_endpoint}"
    if [[ -n "${TAU2_OPD_AIRLINE_URL:-}" ]] \
      && curl -fsS --max-time 10 "${TAU2_OPD_AIRLINE_URL%/generate}/health" >/dev/null 2>&1; then
      teacher_ready=1
      echo "OPD_TEACHER_READY url=${TAU2_OPD_AIRLINE_URL}"
      break
    fi
  fi
  if (( attempt % 10 == 0 )); then echo "Waiting for Airline expert: ${teacher_endpoint}"; fi
  sleep 30
done
[[ "${teacher_ready}" == 1 ]] || { echo "[ERROR] Airline expert did not become ready: ${teacher_endpoint}" >&2; exit 1; }
export TAU2_OPD_AIRLINE_URL

# Wait for the persistent 4-GPU User pool.
user_ready=0
for ((attempt=0; attempt<240; attempt++)); do
  if [[ -s "${user_endpoint}" ]]; then
    # shellcheck disable=SC1090
    source "${user_endpoint}"
    if [[ -n "${TAU2_USER_API_BASE:-}" ]] \
      && curl -fsS --max-time 10 "${TAU2_USER_API_BASE}/models" >/dev/null 2>&1; then
      user_ready=1
      echo "OPD_USER_READY api=${TAU2_USER_API_BASE}"
      break
    fi
  fi
  if (( attempt % 10 == 0 )); then echo "Waiting for User service: ${user_endpoint}"; fi
  sleep 30
done
[[ "${user_ready}" == 1 ]] || { echo "[ERROR] User service did not become ready: ${user_endpoint}" >&2; exit 1; }
export TAU2_USER_API_BASE

echo "OPD_AIRLINE_INTO_MIX_START lr=${LR} init=${REF_LOAD}@${REF_CKPT_STEP} updates=${NUM_ROLLOUT} time=$(date -Is)"

# Bash reads scripts incrementally; run a snapshot while worktree edits continue.
RUNNER_SNAPSHOT="${run_dir}/run_runner_$(date +%Y%m%d_%H%M%S).sh"
cp "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh" "${RUNNER_SNAPSHOT}"

echo "[OPD] starting training with eight student GPUs (physical GPUs 0-7 in this job)"
if CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" bash "${RUNNER_SNAPSHOT}" 2>&1 | tee -a "${run_dir}/run.log"; then
  train_status=0
else
  train_status=$?
fi
ray stop --force >"${run_dir}/ray_stop.log" 2>&1 || true
[[ "${train_status}" == 0 ]] || exit "${train_status}"
echo "OPD_AIRLINE_INTO_MIX_TRAIN_COMPLETE time=$(date -Is)"
