#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-telecom-expert-sft4505-goal-v4-20260928"

checkpoint_root="${EXPERIMENT_DIR}/arms/async/20260928_telecom-expert-goal-v4-shuffled20-train/checkpoints"
export ITER_DIR="${checkpoint_root}/iter_0000019"
export OUTPUT_DIR="${checkpoint_root}/iter_0000019_hf"
export CHECKPOINT_ROOT="${checkpoint_root}"
export ORIGIN_HF_DIR="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"

if [[ ! -f "${OUTPUT_DIR}/model.safetensors.index.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh"
fi

export MODEL_PATH="${OUTPUT_DIR}"
export MODEL_NAME="Qwen3-4B-tau2-telecom-goal-v4-shuffled20-expert-iter19"
export EVAL_LABEL="three-domain-telecom-expert-shuffled20-iter19"
export RUN_STAMP="20260928_three_domain_telecom_expert_shuffled20_iter19_seed300"
export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/trajectories/seed300_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed300_${RUN_STAMP}_summary.json"

export AGENT_EVAL_MODE="official-native"
export AGENT_TOOL_CALL_PARSER="qwen"
export TP=1
export AGENT_REPLICA_CUDA_GROUPS='0;1'
export USER_SGLANG=1
export USER_REPLICA_CUDA_GROUPS='2,3;4,5;6,7'
export DOMAINS="airline,retail,telecom"
export DOMAIN_CONCURRENCY="airline:2,retail:2,telecom:5"
export GLOBAL_CONCURRENCY=9
export BORROW_COMPLETED_DOMAIN_SLOTS=1
export PARALLEL_DOMAINS=1
export RETRIEVAL_CONFIG=bm25
export NUM_TASKS=""
export NUM_TRIALS=4
export SEED=300
export MAX_STEPS=200
export MAX_ERRORS=10
export AGENT_TEMPERATURE=0.6
export AGENT_TOP_P=1.0
export AGENT_MAX_TOKENS=1200

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
