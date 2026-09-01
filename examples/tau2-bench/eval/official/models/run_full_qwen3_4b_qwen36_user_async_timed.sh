#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-eval-qwen36-user-async-timed"
EVAL_LABEL="${EVAL_LABEL:-full}"

# Qwen3-4B-Instruct-2507_torch_dist is the Megatron training-format copy of
# these same source weights. SGLang serves the original HF directory.
export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
export MODEL_NAME="Qwen3-4B-Instruct-2507-qwen36-user-async"
export TP="1"
export MEM_FRACTION="0.85"
export AGENT_REPLICA_CUDA_GROUPS="0;1"
export AGENT_WORKER_PORT_BASE="31000"
export AGENT_SGLANG_EXTRA_ARGS="--show-time-cost --enable-request-time-stats-logging"

export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B"
export USER_MODEL="Qwen3.6-27B-tau2-user-nonthinking"
export USER_TP="2"
export USER_MEM_FRACTION="0.90"
export USER_REPLICA_CUDA_GROUPS="2,3;4,5;6,7"
export USER_WORKER_PORT_BASE="32000"
export USER_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length 65536 --max-running-requests 2 --reasoning-parser qwen3 --tool-call-parser qwen --show-time-cost --enable-request-time-stats-logging"
export USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
export USER_TEMPERATURE="0.0"
export USER_MAX_TOKENS="512"

export ROUTER_POLICY="cache_aware"
export AGENT_ROUTER_POLICY="cache_aware"
export USER_ROUTER_POLICY="round_robin"
export PARALLEL_DOMAINS="1"
export DOMAIN_CONCURRENCY="airline:2,retail:2,telecom:5"
export GLOBAL_CONCURRENCY="9"
export BORROW_COMPLETED_DOMAIN_SLOTS="1"
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE="0"
export DOMAINS="airline,retail,telecom"
export TASK_SPLIT="test"
export NUM_TASKS="${NUM_TASKS-}"
export NUM_TRIALS="${NUM_TRIALS:-4}"
export SEED="${SEED:-300}"
export MAX_STEPS="200"
export MAX_ERRORS="10"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-3}"
export AGENT_TEMPERATURE="0.6"
export AGENT_TOP_P="1.0"
export AGENT_MAX_TOKENS="1200"

export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_${EVAL_LABEL}_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed${SEED}_${RUN_STAMP}_summary.json}"
export NAMESPACE_PROBE_OUTPUT=""

start_epoch="$(date +%s)"
echo "[tau2-qwen36-async] stage=wrapper_start epoch=${start_epoch} timestamp=$(date --iso-8601=seconds)"
echo "[tau2-qwen36-async] topology=agent_2xtp1_gpu0_1,user_3xtp2_gpu2_7,domains_parallel agent_router_policy=${AGENT_ROUTER_POLICY} user_router_policy=${USER_ROUTER_POLICY} initial_domain_concurrency=${DOMAIN_CONCURRENCY} global_concurrency=${GLOBAL_CONCURRENCY} borrow_completed_domain_slots=${BORROW_COMPLETED_DOMAIN_SLOTS} num_tasks=${NUM_TASKS:-all} trials=${NUM_TRIALS}"
bash "${OFFICIAL_DIR}/run_eval.sh"
end_epoch="$(date +%s)"
echo "[tau2-qwen36-async] stage=wrapper_end epoch=${end_epoch} elapsed_seconds=$((end_epoch - start_epoch)) timestamp=$(date --iso-8601=seconds)"
