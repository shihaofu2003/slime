#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
cd "${PROJECT_ROOT}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-tau2-domain-experts-fast-b128}"
experiment_dir="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-$(dirname "${PROJECT_ROOT}")/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${SFT_CKPT_ROOT}/iter_0004673_hf}"
# Existing weights-only conversion of the same SFT; each run has a fresh optimizer.
export REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/output/experiments/tau2-airline-db-count-tuning/arms/async/20260919-progress-db-count-v1-lr2e-6-b128-pw1-external-control/sft_init_torch_dist}"
export REF_CKPT_STEP="${REF_CKPT_STEP:-1}"
unset LOAD_DIR CKPT_STEP START_ROLLOUT_ID

export LR=2e-6 TRAIN_SEED=1234 ROLLOUT_SEED=42
export TAU2_TURN_CREDIT_VERSION=progress-db-count-v1
export TAU2_PROGRESS_WEIGHT=1.0 TAU2_FORMAT_WEIGHT=1.0 TAU2_PROGRESS_GAMMA=0.98
export TAU2_RAW_TOKENS=1
export TAU2_DROP_UNIFORM_OUTCOME_GROUPS=1 TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
export ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=8 GLOBAL_BATCH_SIZE=128
export SAVE_INTERVAL=10
export TAU2_POOL_CAPACITY=32 TAU2_MAX_PENDING_GROUPS=64 TAU2_MAX_POLICY_LAG=-1
export DATA_SOURCE_PATH=random_tasks.ShuffledTaskDataSource
export TAU2_ENVIRONMENT_WORKERS=4
export TAU2_AGENT_CONCURRENCY=48 TAU2_STEP_CONCURRENCY=160 TAU2_AGENT_TIMEOUT=60
export USER_SGLANG=0
export TAU2_USER_API_BASE="${TAU2_USER_API_BASE:-http://10.119.96.116:30000/v1}"
export TAU2_USER_API_KEY=EMPTY
export TAU2_RL_CLEANUP=0 TAU2_SERIAL_CLEANUP=1 USE_WANDB=0 WANDB_MODE=offline

watch_pid=""
trap 'if [[ -n "${watch_pid}" ]]; then kill "${watch_pid}" 2>/dev/null || true; fi' EXIT

# SMOKE_DOMAINS can omit domains whose two-update smoke already passed.
# All four domains must have passed before starting the full training phase.
for phase in smoke train; do
  for spec in ${DOMAIN_SPECS:-airline:60 retail:30 telecom:20 banking:30}; do
    domain="${spec%:*}"
    if [[ "${phase}" == smoke && " ${SMOKE_DOMAINS:-airline retail telecom banking} " != *" ${domain} "* ]]; then
      continue
    fi
    # Banking retrieval produces longer prompts; 60s cancels healthy requests.
    export TAU2_AGENT_TIMEOUT=60
    [[ "${domain}" != banking ]] || export TAU2_AGENT_TIMEOUT=600
    export NUM_ROLLOUT="${spec#*:}"
    [[ "${phase}" != smoke ]] || export NUM_ROLLOUT=2
    export PREPARED_RL_DATA="${experiment_dir}/data/${domain}_train.jsonl"
    export TAU2_RUN_NAME="${RUN_STAMP}-${domain}-${phase}"
    run_dir="${experiment_dir}/arms/async/${TAU2_RUN_NAME}"
    plot_dir="${experiment_dir}/plots/${TAU2_RUN_NAME}"
    mkdir -p "${run_dir}" "${plot_dir}"
    echo "EXPERT_START domain=${domain} phase=${phase} updates=${NUM_ROLLOUT} run=${TAU2_RUN_NAME}"
    if [[ "${phase}" == train ]]; then
      python3 -u scripts/watch_tau2_four_domain_rewards.py \
        --experiment-dir "${experiment_dir}" --run-name "${TAU2_RUN_NAME}" \
        --batch-size 128 --target-updates "${NUM_ROLLOUT}" --interval 300 \
        --title "${domain} expert" >"${run_dir}/reward_watch.log" 2>&1 &
      watch_pid=$!
    fi
    if bash examples/tau2-bench/rl/run_tau2_areal_async.sh async "${phase}" \
        2>&1 | tee "${run_dir}/run.log"; then
      train_status=0
    else
      train_status=$?
    fi
    if [[ -n "${watch_pid}" ]]; then
      kill "${watch_pid}" 2>/dev/null || true
      wait "${watch_pid}" 2>/dev/null || true
      watch_pid=""
    fi
    ray stop --force >"${run_dir}/ray_stop.log" 2>&1
    [[ "${train_status}" == 0 ]] || exit "${train_status}"

    python3 - "${run_dir}" "${domain}" "${phase}" "${NUM_ROLLOUT}" <<'PY'
import ast
import math
import re
import sys
from pathlib import Path

run, domain, phase, expected = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
log = (run / "run.log").read_text(errors="replace")
completed = {int(i) for i in re.findall(r"tau2_trainer batch=(\d+) start=[\d.]+ end=[\d.]+", log)}
metrics = {int(i): ast.literal_eval(values) for i, values in re.findall(r" - step (\d+): (\{[^\n]+\})", log)}
if completed != set(range(expected)) or not completed.issubset(metrics):
    raise RuntimeError(f"{domain} {phase}: expected {expected} completed updates with metrics, got {sorted(completed)}")
for i in completed:
    for name in ("train/loss", "train/grad_norm"):
        if not math.isfinite(metrics[i][name]):
            raise RuntimeError(f"{domain} {phase}: non-finite {name} at update {i}")
checkpoint = run / "checkpoints"
if int((checkpoint / "latest_checkpointed_iteration.txt").read_text()) != expected - 1:
    raise RuntimeError(f"{domain} {phase}: final checkpoint not saved")
message = f"{domain} {phase}: {expected} updates completed; finite loss/gradient; iter{expected - 1} saved."
print(f"EXPERT_PASS {message}")
readme = run.parents[2] / "README.md"
with readme.open("a") as stream:
    stream.write(f"\n{message} [Run log](arms/async/{run.name}/run.log).\n")
PY
    python3 scripts/plot_tau2_four_domain_rewards.py --run-dir "${run_dir}" \
      --output-dir "${plot_dir}" --batch-size 128 --label 'LR2e-6 batch128 progress1' \
      --title "${domain} expert ${phase}" || echo "Reward plotting failed for ${TAU2_RUN_NAME}." >&2
  done
  echo "EXPERT_PHASE_COMPLETE phase=${phase}"
done
