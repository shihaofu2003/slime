#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-eval-qwen36-user-timing-tp2"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
export MODEL_NAME="Qwen3-4B-Instruct-2507-qwen36-user"
export AGENT_CUDA_VISIBLE_DEVICES="0"
export TP="1"
export MEM_FRACTION="0.85"
export AGENT_SGLANG_EXTRA_ARGS="--show-time-cost --enable-request-time-stats-logging"

export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B"
export USER_MODEL="Qwen3.6-27B-tau2-user-nonthinking"
export USER_CUDA_VISIBLE_DEVICES="1,2"
export USER_TP="2"
export USER_MEM_FRACTION="0.90"
export USER_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length 65536 --max-running-requests 2 --reasoning-parser qwen3 --tool-call-parser qwen --show-time-cost --enable-request-time-stats-logging"
export USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
export USER_TEMPERATURE="0.0"
export USER_MAX_TOKENS="512"

export DOMAINS="airline,retail,telecom"
export TASK_SPLIT="test"
export NUM_TASKS=""
export NUM_TRIALS="4"
export SEED="${SEED:-300}"
export MAX_STEPS="200"
export MAX_ERRORS="10"
export MAX_CONCURRENCY="1"
export AGENT_TEMPERATURE="0.6"
export AGENT_TOP_P="1.0"
export AGENT_MAX_TOKENS="1200"

export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/seed${SEED}_summary.json"
export NAMESPACE_PROBE_OUTPUT=""

start_epoch="$(date +%s)"
echo "[tau2-qwen36-timing] stage=wrapper_start epoch=${start_epoch} timestamp=$(date --iso-8601=seconds)"
echo "[tau2-qwen36-timing] topology=agent_tp1_gpu0,user_tp2_gpu1_2,reserved_gpu3 user_context_length=65536 max_concurrency=${MAX_CONCURRENCY}"
bash "${OFFICIAL_DIR}/run_eval.sh"
end_epoch="$(date +%s)"
echo "[tau2-qwen36-timing] stage=wrapper_end epoch=${end_epoch} elapsed_seconds=$((end_epoch - start_epoch)) timestamp=$(date --iso-8601=seconds)"
