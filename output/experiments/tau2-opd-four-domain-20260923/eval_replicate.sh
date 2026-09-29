#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: eval_replicate.sh <seed>}"

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-opd-four-domain-20260923/eval-replicates/seed${seed}"
export MODEL_PATH="${PROJECT_ROOT}/output/experiments/tau2-opd-four-domain-20260923/pilot-lr2e6-seed1235-43/checkpoints/iter_0000159_hf"
export MODEL_NAME=Qwen3-4B-opd-four-domain-lr2e6-seed1235-43
export USER_SGLANG=1 AGENT_REPLICA_CUDA_GROUPS='0;1'
export AGENT_EVAL_MODE=official-native AGENT_MAX_TOKENS=1200
export AGENT_TEMPERATURE=0.6 AGENT_TOP_P=1.0 MAX_STEPS=200 MAX_ERRORS=10
export NUM_TRIALS=4 NUM_TASKS="" AUTO_RESUME=1 RETRIEVAL_CONFIG=bm25
export EVAL_LABEL=four-domain-lr2e6-iter159
export SEED="${seed}"
export RUN_STAMP="20260923_four_domain_lr2e6_train1235_rollout43_seed${seed}"
export SAVE_PREFIX="${EXPERIMENT_DIR}/trajectories/seed${seed}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/seed${seed}_summary.json"

mkdir -p "${EXPERIMENT_DIR}"
cd "${PROJECT_ROOT}"
echo "OPD_FOUR_DOMAIN_REPLICATE_START seed=${seed} time=$(date -Is)"
bash examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh full \
  2>&1 | tee -a "${EXPERIMENT_DIR}/run.log"
echo "OPD_FOUR_DOMAIN_REPLICATE_COMPLETE seed=${seed} summary=${SUMMARY_OUTPUT} time=$(date -Is)"
