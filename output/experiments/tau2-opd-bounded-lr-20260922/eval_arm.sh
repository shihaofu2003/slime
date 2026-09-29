#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-opd-bounded-lr-20260922"
arm="${1:?Usage: eval_arm.sh <lr2e6|lr5e6>}"
case "${arm}" in lr2e6|lr5e6) ;; *) echo 'Expected lr2e6 or lr5e6' >&2; exit 2 ;; esac
export MODEL_PATH="${EXPERIMENT_DIR}/pilot-${arm}-seed1235-43/checkpoints/iter_0000059_hf"
export MODEL_NAME="Qwen3-4B-opd-bounded-${arm}-seed1235-43"
export USER_SGLANG=1 DOMAINS=airline,retail,telecom
export DOMAIN_CONCURRENCY=airline:2,retail:2,telecom:5 GLOBAL_CONCURRENCY=9 MAX_CONCURRENCY=3
export AGENT_REPLICA_CUDA_GROUPS='0;1' BORROW_COMPLETED_DOMAIN_SLOTS=1
export AGENT_EVAL_MODE=official-native AGENT_MAX_TOKENS=1200 NUM_TRIALS=4 NUM_TASKS=""
export AUTO_RESUME=1
mkdir -p "${EXPERIMENT_DIR}/eval/${arm}"
cd "${PROJECT_ROOT}"
for seed in 300 301; do
  export SEED="${seed}" EVAL_LABEL="${arm}"
  export RUN_STAMP="20260922_${arm}_train1235_rollout43_seed${seed}"
  export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${arm}/trajectories/seed${seed}_${RUN_STAMP}"
  export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${arm}/seed${seed}_summary.json"
  echo "OPD_LR_EVAL_START arm=${arm} seed=${seed} time=$(date -Is)"
  bash examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh \
    2>&1 | tee -a "${EXPERIMENT_DIR}/eval/${arm}/seed${seed}.log"
  echo "OPD_LR_EVAL_COMPLETE arm=${arm} seed=${seed} summary=${SUMMARY_OUTPUT} time=$(date -Is)"
  printf '\n%s seed%s official evaluation completed. [Summary](eval/%s/seed%s_summary.json), [run log](eval/%s/seed%s.log).\n' \
    "${arm}" "${seed}" "${arm}" "${seed}" "${arm}" "${seed}" >>"${EXPERIMENT_DIR}/README.md"
done
