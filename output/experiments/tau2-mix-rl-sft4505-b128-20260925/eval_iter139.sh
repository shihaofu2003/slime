#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-mix-rl-sft4505-b128-20260925"
checkpoint_root="${EXPERIMENT_DIR}/arms/async/20260925_mix-retry1-train/checkpoints"
# Latest completed checkpoint at submission: iter139, after 140 updates.
export ITER_DIR="${checkpoint_root}/iter_0000139"
export OUTPUT_DIR="${checkpoint_root}/iter_0000139_hf"
export ORIGIN_HF_DIR="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
if [[ ! -f "${OUTPUT_DIR}/model.safetensors.index.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh"
fi

export MODEL_PATH="${OUTPUT_DIR}"
export MODEL_NAME=Qwen3-4B-mix-rl-iter139
export EVAL_LABEL=mix-iter139-four-domain
export RUN_STAMP=20260925_mix_iter139_seed300
export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/trajectories/seed300_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed300_${RUN_STAMP}_summary.json"
export AGENT_EVAL_MODE=official-native
export AGENT_TOOL_CALL_PARSER=qwen
export TP=1
export AGENT_REPLICA_CUDA_GROUPS='0;1'
export USER_SGLANG=1
export USER_REPLICA_CUDA_GROUPS='2,3;4,5;6,7'
export DOMAINS=airline,retail,telecom,banking_knowledge
export DOMAIN_CONCURRENCY=airline:1,retail:2,telecom:2,banking_knowledge:4
export GLOBAL_CONCURRENCY=9 BORROW_COMPLETED_DOMAIN_SLOTS=1
export RETRIEVAL_CONFIG=bm25
export NUM_TASKS="" AGENT_MAX_TOKENS=1200 NUM_TRIALS=4 SEED=300

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh" full
