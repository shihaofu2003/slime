#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-eval-qwen38-nonthinking-banking-20260925"
export MODEL_PATH=/mnt/afs/models/Qwen3.8-27B
export MODEL_NAME=Qwen3.8-27B-nonthinking
export AGENT_EVAL_MODE=official-native
export AGENT_TOOL_CALL_PARSER=qwen3_coder
export AGENT_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
export AGENT_MAX_TOKENS=1200
export TP=2
export AGENT_REPLICA_CUDA_GROUPS='0,1;2,3'
export USER_SGLANG=1
export USER_REPLICA_CUDA_GROUPS='4,5;6,7'
export DOMAINS=banking_knowledge
export DOMAIN_CONCURRENCY=banking_knowledge:4
export BORROW_COMPLETED_DOMAIN_SLOTS=0
export NUM_TASKS="" NUM_TRIALS=4 SEED=300
export RETRIEVAL_CONFIG=bm25
export EVAL_LABEL=banking-full-bm25-nonthinking
export RUN_STAMP=20260925_qwen38_nonthinking_seed300
export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/trajectories/seed300_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed300_${RUN_STAMP}_summary.json"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
