#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-curriculum-qwen36-eval}"
CURRICULUM_DIR="${PROJECT_ROOT}/output/experiments/tau2-banking-task-curriculum"

MODE="${1:-smoke}"
RETRIEVAL_VARIANT="${2:-bm25}"
SHARD_INDEX="${3:-0}"
NUM_SHARDS="${4:-1}"
case "${MODE}" in
  smoke)
    SOURCE_LIMIT="${SOURCE_LIMIT:-3}"
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
  *)
    echo "Expected smoke or full, got: ${MODE}" >&2
    exit 2
    ;;
esac
if [[ "${RETRIEVAL_VARIANT}" != "bm25" && "${RETRIEVAL_VARIANT}" != "golden_retrieval" ]]; then
  echo "Expected bm25 or golden_retrieval, got: ${RETRIEVAL_VARIANT}" >&2
  exit 2
fi

SEED="${SEED:-300}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
VARIANT_LABEL="${RETRIEVAL_VARIANT//_/-}"
EVAL_LABEL="${MODE}-${VARIANT_LABEL}-shard${SHARD_INDEX}of${NUM_SHARDS}"
RUN_DIR="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed${SEED}_${RUN_STAMP}"
export TAU2_DATA_DIR="${RUN_DIR}/tau2_data"
mkdir -p "${RUN_DIR}"

PREPARE_ARGS=(
  --tasks "${CURRICULUM_DIR}/expanded_tasks.json"
  --contracts "${CURRICULUM_DIR}/expanded_contracts.jsonl"
  --data-root "${TAU2_DATA_DIR}"
  --retrieval-variant "${RETRIEVAL_VARIANT}"
  --shard-index "${SHARD_INDEX}"
  --num-shards "${NUM_SHARDS}"
)
if [[ -n "${SOURCE_LIMIT}" ]]; then
  PREPARE_ARGS+=(--source-limit "${SOURCE_LIMIT}")
fi
python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/prepare_banking_curriculum_eval.py" \
  "${PREPARE_ARGS[@]}" | tee "${RUN_DIR}/prepare.log"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B"
export MODEL_NAME="Qwen3.6-27B-banking-curriculum-thinking"
export AGENT_SERVED_MODEL_NAME="${MODEL_NAME}"
export AGENT_EVAL_MODE="official-native"
export TP="1"
export MEM_FRACTION="0.90"
export AGENT_REPLICA_CUDA_GROUPS="${AGENT_GROUPS}"
export AGENT_WORKER_PORT_BASE="31000"
export AGENT_TOOL_CALL_PARSER="qwen3_coder"
export AGENT_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length 65536 --language-only --max-running-requests 1 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --show-time-cost --enable-request-time-stats-logging"
export AGENT_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":true}}'
export AGENT_TEMPERATURE="0.0"
export AGENT_TOP_P="1.0"
export AGENT_MAX_TOKENS="4096"

export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B"
export USER_MODEL="Qwen3.6-27B-banking-user-nonthinking"
export USER_SGLANG="1"
export USER_TP="1"
export USER_MEM_FRACTION="0.90"
export USER_REPLICA_CUDA_GROUPS="${USER_GROUPS}"
export USER_WORKER_PORT_BASE="32000"
export USER_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length 32768 --language-only --max-running-requests 4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder"
export USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
export USER_TEMPERATURE="0.0"
export USER_TOP_P="1.0"
export USER_MAX_TOKENS="512"
export SGLANG_ENCODER_BOOTSTRAP_PORT_OFFSET="20000"

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
export SAVE_PREFIX="tau2_qwen36_banking_curriculum_${EVAL_LABEL}_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${RUN_DIR}/official_summary.json"
export NAMESPACE_PROBE_OUTPUT=""

echo "[banking-curriculum-qwen36] mode=${MODE} retrieval=${RETRIEVAL_VARIANT} shard=${SHARD_INDEX}/${NUM_SHARDS} agent_groups=${AGENT_GROUPS} user_groups=${USER_GROUPS} concurrency=${EVAL_CONCURRENCY}"
bash "${OFFICIAL_DIR}/run_eval.sh"

RESULTS_FILE="${TAU2_DATA_DIR}/simulations/${SAVE_PREFIX}_banking_knowledge_${TASK_SPLIT}_${NUM_TRIALS}trials/results.json"
EXPECTED_TASKS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' "${TAU2_DATA_DIR}/tau2/domains/banking_knowledge/curriculum_manifest.json")"
python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/summarize_banking_curriculum_eval.py" \
  --results "${RESULTS_FILE}" \
  --contracts "${CURRICULUM_DIR}/expanded_contracts.jsonl" \
  --expected-task-count "${EXPECTED_TASKS}" \
  --output "${RUN_DIR}/curriculum_metrics.json" \
  --markdown-output "${RUN_DIR}/curriculum_metrics.md"
echo "BANKING_CURRICULUM_QWEN36_EVAL_OK run_dir=${RUN_DIR} tasks=${EXPECTED_TASKS}"
