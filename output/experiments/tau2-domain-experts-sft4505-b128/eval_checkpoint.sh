#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2
export SERVICE_AGENT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-domain-experts-sft4505-b128"
DOMAIN="$1"
ITERATION="$2"
printf -v padded '%07d' "${ITERATION}"
CHECKPOINT_ROOT="${EXPERIMENT_DIR}/arms/async/20260920_sft4505-${DOMAIN}-train/checkpoints"
export ITER_DIR="${CHECKPOINT_ROOT}/iter_${padded}"
export OUTPUT_DIR="${CHECKPOINT_ROOT}/iter_${padded}_hf"
export ORIGIN_HF_DIR="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
if [[ ! -f "${OUTPUT_DIR}/model.safetensors.index.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh"
fi
export MODEL_PATH="${OUTPUT_DIR}"
export MODEL_NAME="Qwen3-4B-sft4505-${DOMAIN}-iter${ITERATION}"
export EVAL_LABEL="${DOMAIN}-iter${ITERATION}-four-domain"
export RUN_STAMP="20260921_sft4505_${DOMAIN}_iter${ITERATION}"
export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/trajectories/seed300_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed300_${RUN_STAMP}_summary.json"
export DOMAINS=airline,retail,telecom,banking_knowledge
export DOMAIN_CONCURRENCY=airline:1,retail:2,telecom:2,banking_knowledge:4
export GLOBAL_CONCURRENCY=9
export BORROW_COMPLETED_DOMAIN_SLOTS=1
export RETRIEVAL_CONFIG=bm25
export NUM_TASKS=""
export AGENT_MAX_TOKENS=1200
export NUM_TRIALS=4
export SEED=300
exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh" full
