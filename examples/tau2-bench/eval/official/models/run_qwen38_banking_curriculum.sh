#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-curriculum-qwen38-eval}"
CURRICULUM_DIR="${CURRICULUM_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-task-curriculum}"
CURRICULUM_TASKS="${CURRICULUM_TASKS:-${CURRICULUM_DIR}/expanded_tasks.json}"
CURRICULUM_CONTRACTS="${CURRICULUM_CONTRACTS:-${CURRICULUM_DIR}/expanded_contracts.jsonl}"

MODE="${1:-smoke}"
RETRIEVAL_VARIANT="${2:-bm25}"
SHARD_INDEX="${3:-0}"
NUM_SHARDS="${4:-1}"
SMOKE_SOURCE_LIMIT="${5:-${SOURCE_LIMIT:-3}}"
TARGET_TASK_IDS="${6:-}"
AGENT_TOKEN_LIMIT="${7:-${QWEN38_AGENT_MAX_TOKENS:-16384}}"
EVAL_SEED="${8:-${SEED:-300}}"
AGENT_CONTEXT_LENGTH="${9:-${QWEN38_AGENT_CONTEXT_LENGTH:-262144}}"
AGENT_MEM_FRACTION="${10:-${QWEN38_AGENT_MEM_FRACTION:-0.95}}"
USER_TOKEN_LIMIT="${11:-${QWEN38_USER_MAX_TOKENS:-1024}}"
AGENT_PRESERVE_THINKING="${12:-${QWEN38_AGENT_PRESERVE_THINKING:-true}}"
case "${MODE}" in
  smoke)
    SOURCE_LIMIT="${SMOKE_SOURCE_LIMIT}"
    AGENT_GROUPS="${AGENT_REPLICA_CUDA_GROUPS:-0}"
    USER_GROUPS="${USER_REPLICA_CUDA_GROUPS:-1}"
    EVAL_CONCURRENCY="${MAX_CONCURRENCY:-1}"
    ;;
  full)
    SOURCE_LIMIT=""
    AGENT_GROUPS="${AGENT_REPLICA_CUDA_GROUPS:-0;1;2;3;4;5}"
    USER_GROUPS="${USER_REPLICA_CUDA_GROUPS:-6;7}"
    EVAL_CONCURRENCY="${MAX_CONCURRENCY:-6}"
    ;;
  full4)
    SOURCE_LIMIT=""
    AGENT_GROUPS="${AGENT_REPLICA_CUDA_GROUPS:-0;1;2}"
    USER_GROUPS="${USER_REPLICA_CUDA_GROUPS:-3}"
    EVAL_CONCURRENCY="${MAX_CONCURRENCY:-3}"
    ;;
  *)
    echo "Expected smoke, full, or full4, got: ${MODE}" >&2
    exit 2
    ;;
esac
if [[ "${RETRIEVAL_VARIANT}" != "bm25" && "${RETRIEVAL_VARIANT}" != "golden_retrieval" ]]; then
  echo "Expected bm25 or golden_retrieval, got: ${RETRIEVAL_VARIANT}" >&2
  exit 2
fi

SEED="${EVAL_SEED}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
VARIANT_LABEL="${RETRIEVAL_VARIANT//_/-}"
EVAL_LABEL="${MODE}-${VARIANT_LABEL}-shard${SHARD_INDEX}of${NUM_SHARDS}"
RUN_DIR="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed${SEED}_${RUN_STAMP}"
export TAU2_DATA_DIR="${RUN_DIR}/tau2_data"
mkdir -p "${RUN_DIR}"

PREPARE_ARGS=(
  --tasks "${CURRICULUM_TASKS}"
  --contracts "${CURRICULUM_CONTRACTS}"
  --data-root "${TAU2_DATA_DIR}"
  --retrieval-variant "${RETRIEVAL_VARIANT}"
  --shard-index "${SHARD_INDEX}"
  --num-shards "${NUM_SHARDS}"
)
if [[ -n "${SOURCE_LIMIT}" ]]; then
  PREPARE_ARGS+=(--source-limit "${SOURCE_LIMIT}")
fi
if [[ -n "${TARGET_TASK_IDS}" && "${TARGET_TASK_IDS}" != "-" ]]; then
  IFS=',' read -r -a TARGET_TASK_ID_LIST <<< "${TARGET_TASK_IDS}"
  for task_id in "${TARGET_TASK_ID_LIST[@]}"; do
    PREPARE_ARGS+=(--task-id "${task_id}")
  done
fi
python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/prepare_banking_curriculum_eval.py" \
  "${PREPARE_ARGS[@]}" | tee "${RUN_DIR}/prepare.log"

export MODEL_PATH="${QWEN38_MODEL_PATH:-/mnt/afs/models/Qwen3.8-27B}"
export MODEL_NAME="Qwen3.8-27B-banking-agent-xhigh"
export AGENT_SERVED_MODEL_NAME="${MODEL_NAME}"
export AGENT_EVAL_MODE="official-native"
export TP="${QWEN38_AGENT_TP:-1}"
export MEM_FRACTION="${AGENT_MEM_FRACTION}"
export AGENT_REPLICA_CUDA_GROUPS="${AGENT_GROUPS}"
export AGENT_WORKER_PORT_BASE="31000"
export AGENT_TOOL_CALL_PARSER="qwen3_coder"
export AGENT_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length ${AGENT_CONTEXT_LENGTH} --max-running-requests 1 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --show-time-cost --enable-request-time-stats-logging"
export AGENT_EXTRA_BODY_JSON="{\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0,\"chat_template_kwargs\":{\"enable_thinking\":true,\"preserve_thinking\":${AGENT_PRESERVE_THINKING},\"reasoning_effort\":\"xhigh\"}}"
export AGENT_TEMPERATURE="1.0"
export AGENT_TOP_P="0.95"
export AGENT_MAX_TOKENS="${AGENT_TOKEN_LIMIT}"

export USER_MODEL_PATH="${MODEL_PATH}"
export USER_MODEL="Qwen3.8-27B-banking-user-nonthinking"
export USER_SGLANG="1"
export USER_TP="1"
export USER_MEM_FRACTION="0.90"
export USER_REPLICA_CUDA_GROUPS="${USER_GROUPS}"
export USER_WORKER_PORT_BASE="32000"
export USER_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length 32768 --max-running-requests 4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder"
export USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
export USER_TEMPERATURE="0.0"
export USER_TOP_P="1.0"
export USER_MAX_TOKENS="${USER_TOKEN_LIMIT}"

export AGENT_ROUTER_POLICY="round_robin"
export USER_ROUTER_POLICY="round_robin"
export DOMAINS="banking_knowledge"
export TASK_SPLIT="generated"
export NUM_TASKS=""
export NUM_TRIALS="1"
export MAX_STEPS="200"
export MAX_ERRORS="10"
export MAX_CONCURRENCY="${EVAL_CONCURRENCY}"
export DOMAIN_CONCURRENCY="banking_knowledge:${EVAL_CONCURRENCY}"
export GLOBAL_CONCURRENCY="${EVAL_CONCURRENCY}"
export PARALLEL_DOMAINS="0"
export BORROW_COMPLETED_DOMAIN_SLOTS="0"
export RETRIEVAL_CONFIG="${RETRIEVAL_VARIANT}"
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE="0"
export RUN_STAMP
export SAVE_PREFIX="tau2_qwen38_banking_curriculum_${EVAL_LABEL}_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${RUN_DIR}/official_summary.json"
export NAMESPACE_PROBE_OUTPUT=""

echo "[banking-curriculum-qwen38] mode=${MODE} retrieval=${RETRIEVAL_VARIANT} shard=${SHARD_INDEX}/${NUM_SHARDS} agent_groups=${AGENT_GROUPS} agent_tp=${TP} user_groups=${USER_GROUPS} concurrency=${EVAL_CONCURRENCY} agent_max_tokens=${AGENT_TOKEN_LIMIT} agent_context=${AGENT_CONTEXT_LENGTH} agent_mem_fraction=${AGENT_MEM_FRACTION} user_max_tokens=${USER_TOKEN_LIMIT} preserve_thinking=${AGENT_PRESERVE_THINKING}"
bash "${OFFICIAL_DIR}/run_eval.sh"

RESULTS_FILE="${TAU2_DATA_DIR}/simulations/${SAVE_PREFIX}_banking_knowledge_${TASK_SPLIT}_${NUM_TRIALS}trials/results.json"
EXPECTED_TASKS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' "${TAU2_DATA_DIR}/tau2/domains/banking_knowledge/curriculum_manifest.json")"
python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/summarize_banking_curriculum_eval.py" \
  --results "${RESULTS_FILE}" \
  --contracts "${CURRICULUM_CONTRACTS}" \
  --expected-task-count "${EXPECTED_TASKS}" \
  --output "${RUN_DIR}/curriculum_metrics.json" \
  --markdown-output "${RUN_DIR}/curriculum_metrics.md"
echo "BANKING_CURRICULUM_QWEN38_EVAL_OK run_dir=${RUN_DIR} tasks=${EXPECTED_TASKS}"
