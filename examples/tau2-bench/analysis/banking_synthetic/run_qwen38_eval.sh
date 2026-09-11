#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
BANKING_DATA="${BANKING_DATA:-${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge}"
MODEL_PATH="/mnt/afs/models/Qwen3.8-27B"

MODE="${1:-smoke}"
RETRIEVAL_VARIANT="${2:-bm25}"
SHARD_INDEX="${3:-0}"
NUM_SHARDS="${4:-1}"
EVAL_SEED="${5:-300}"
TASK_IDS_FILE="${6:-}"
TASK_LIMIT="${7:-}"
AGENT_TOKEN_LIMIT="${QWEN38_AGENT_MAX_TOKENS:-16384}"

if [[ "${RETRIEVAL_VARIANT}" != "bm25" && "${RETRIEVAL_VARIANT}" != "golden_retrieval" ]]; then
  echo "Expected bm25 or golden_retrieval, got: ${RETRIEVAL_VARIANT}" >&2
  exit 2
fi

case "${MODE}" in
  smoke)
    TASKS="${SYNTHETIC_TASKS:-${EXPERIMENT_DIR}/smoke/tasks_naturalized.json}"
    CONTRACTS="${SYNTHETIC_CONTRACTS:-${EXPERIMENT_DIR}/smoke/source/contracts.jsonl}"
    ATTEMPT_KIND="first"
    ;;
  pilot)
    TASKS="${SYNTHETIC_TASKS:-${EXPERIMENT_DIR}/pilot/tasks_naturalized.json}"
    CONTRACTS="${SYNTHETIC_CONTRACTS:-${EXPERIMENT_DIR}/candidates/pilot_contracts.jsonl}"
    ATTEMPT_KIND="first"
    ;;
  scale)
    TASKS="${SYNTHETIC_TASKS:?SYNTHETIC_TASKS is required for scale mode}"
    CONTRACTS="${SYNTHETIC_CONTRACTS:?SYNTHETIC_CONTRACTS is required for scale mode}"
    ATTEMPT_KIND="first"
    ;;
  recovery)
    TASKS="${SYNTHETIC_TASKS:-${EXPERIMENT_DIR}/pilot/tasks_naturalized.json}"
    CONTRACTS="${SYNTHETIC_CONTRACTS:-${EXPERIMENT_DIR}/candidates/pilot_contracts.jsonl}"
    ATTEMPT_KIND="recovery"
    AGENT_TOKEN_LIMIT="32768"
    if [[ -z "${TASK_IDS_FILE}" ]]; then
      echo "Recovery mode requires a task-id JSON file" >&2
      exit 2
    fi
    ;;
  smoke_recovery)
    TASKS="${SYNTHETIC_TASKS:-${EXPERIMENT_DIR}/smoke/tasks_naturalized.json}"
    CONTRACTS="${SYNTHETIC_CONTRACTS:-${EXPERIMENT_DIR}/smoke/source/contracts.jsonl}"
    ATTEMPT_KIND="recovery"
    AGENT_TOKEN_LIMIT="32768"
    if [[ -z "${TASK_IDS_FILE}" ]]; then
      echo "Smoke recovery mode requires a task-id JSON file" >&2
      exit 2
    fi
    ;;
  *)
    echo "Expected smoke, pilot, scale, recovery, or smoke_recovery, got: ${MODE}" >&2
    exit 2
    ;;
esac

for path in "${TASKS}" "${CONTRACTS}" "${EXPERIMENT_DIR}/allowed_documents.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required input is missing: ${path}" >&2
    exit 1
  fi
done

RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
VARIANT_LABEL="${RETRIEVAL_VARIANT//_/-}"
EVAL_LABEL="${MODE}-${VARIANT_LABEL}-shard${SHARD_INDEX}of${NUM_SHARDS}"
RUN_DIR="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed${EVAL_SEED}_${RUN_STAMP}"
export TAU2_DATA_DIR="${RUN_DIR}/tau2_data"
mkdir -p "${RUN_DIR}"

PREPARE_ARGS=(
  --tasks "${TASKS}"
  --contracts "${CONTRACTS}"
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json"
  --banking-data "${BANKING_DATA}"
  --data-root "${TAU2_DATA_DIR}"
  --retrieval-variant "${RETRIEVAL_VARIANT}"
  --shard-index "${SHARD_INDEX}"
  --num-shards "${NUM_SHARDS}"
)
if [[ -n "${TASK_IDS_FILE}" && "${TASK_IDS_FILE}" != "-" ]]; then
  PREPARE_ARGS+=(--task-ids-file "${TASK_IDS_FILE}")
fi
if [[ -n "${TASK_LIMIT}" ]]; then
  PREPARE_ARGS+=(--limit "${TASK_LIMIT}")
fi
python3 "${PIPELINE}/prepare_eval.py" "${PREPARE_ARGS[@]}" | tee "${RUN_DIR}/prepare.log"

export MODEL_PATH
export MODEL_NAME="Qwen3.8-27B-banking-synthetic-agent-xhigh"
export AGENT_SERVED_MODEL_NAME="${MODEL_NAME}"
export AGENT_EVAL_MODE="official-native"
export TP="1"
export MEM_FRACTION="0.95"
export AGENT_REPLICA_CUDA_GROUPS="0;1;2"
export AGENT_WORKER_PORT_BASE="31000"
export AGENT_TOOL_CALL_PARSER="qwen3_coder"
export AGENT_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length 262144 --max-running-requests 1 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --show-time-cost --enable-request-time-stats-logging"
export AGENT_EXTRA_BODY_JSON='{"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}}'
export AGENT_TEMPERATURE="1.0"
export AGENT_TOP_P="0.95"
export AGENT_MAX_TOKENS="${AGENT_TOKEN_LIMIT}"

export USER_MODEL_PATH="${MODEL_PATH}"
export USER_MODEL="Qwen3.8-27B-banking-synthetic-user-nonthinking"
export USER_SGLANG="1"
export USER_TP="1"
export USER_MEM_FRACTION="0.90"
export USER_REPLICA_CUDA_GROUPS="3"
export USER_WORKER_PORT_BASE="32000"
export USER_SGLANG_EXTRA_ARGS="--dtype bfloat16 --context-length 32768 --max-running-requests 4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder"
export USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
export USER_TEMPERATURE="0.0"
export USER_TOP_P="1.0"
export USER_MAX_TOKENS="1024"

if [[ "${MODEL_PATH}" != "/mnt/afs/models/Qwen3.8-27B" || "${USER_MODEL_PATH}" != "${MODEL_PATH}" ]]; then
  echo "Agent and User must both run /mnt/afs/models/Qwen3.8-27B" >&2
  exit 1
fi

MODEL_MANIFEST="${RUN_DIR}/runtime_models.json"
printf '{\n  "agent_model_path": "%s",\n  "user_model_path": "%s",\n  "agent_served_model": "%s",\n  "user_served_model": "%s",\n  "attempt_kind": "%s",\n  "agent_max_tokens": %s\n}\n' \
  "${MODEL_PATH}" "${USER_MODEL_PATH}" "${AGENT_SERVED_MODEL_NAME}" "${USER_MODEL}" \
  "${ATTEMPT_KIND}" "${AGENT_MAX_TOKENS}" >"${MODEL_MANIFEST}"

export AGENT_ROUTER_POLICY="round_robin"
export USER_ROUTER_POLICY="round_robin"
export DOMAINS="banking_knowledge"
export TASK_SPLIT="generated"
export NUM_TASKS=""
export NUM_TRIALS="1"
export MAX_STEPS="60"
export MAX_ERRORS="10"
export MAX_RETRIES="0"
export MAX_CONCURRENCY="3"
export DOMAIN_CONCURRENCY="banking_knowledge:3"
export GLOBAL_CONCURRENCY="3"
export PARALLEL_DOMAINS="0"
export BORROW_COMPLETED_DOMAIN_SLOTS="0"
export RETRIEVAL_CONFIG="${RETRIEVAL_VARIANT}"
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE="0"
export SEED="${EVAL_SEED}"
export RUN_STAMP
export SAVE_PREFIX="tau2_qwen38_banking_synthetic_${EVAL_LABEL}_seed${EVAL_SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${RUN_DIR}/official_summary.json"
export NAMESPACE_PROBE_OUTPUT=""

echo "[banking-synthetic-qwen38] mode=${MODE} retrieval=${RETRIEVAL_VARIANT} shard=${SHARD_INDEX}/${NUM_SHARDS} seed=${EVAL_SEED} agent_max_tokens=${AGENT_MAX_TOKENS}"
set +e
bash "${OFFICIAL_DIR}/run_eval.sh"
EVAL_STATUS=$?
set -e

RESULTS_FILE="${TAU2_DATA_DIR}/simulations/${SAVE_PREFIX}_banking_knowledge_${TASK_SPLIT}_${NUM_TRIALS}trials/results.json"
if [[ ! -f "${RESULTS_FILE}" ]]; then
  echo "Evaluation produced no results file: ${RESULTS_FILE}" >&2
  if [[ "${EVAL_STATUS}" -eq 0 ]]; then
    exit 1
  fi
  exit "${EVAL_STATUS}"
fi
python3 "${PIPELINE}/summarize.py" \
  --results "${RESULTS_FILE}" \
  --contracts "${CONTRACTS}" \
  --model-manifest "${MODEL_MANIFEST}" \
  --output "${RUN_DIR}/metrics.json" \
  --markdown-output "${RUN_DIR}/metrics.md"

EXPECTED_TASKS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' "${TAU2_DATA_DIR}/tau2/domains/banking_knowledge/synthetic_manifest.json")"
if [[ "${EVAL_STATUS}" -ne 0 ]]; then
  echo "BANKING_SYNTHETIC_QWEN38_EVAL_FAILED run_dir=${RUN_DIR} tasks=${EXPECTED_TASKS} exit=${EVAL_STATUS}" >&2
  exit "${EVAL_STATUS}"
fi
echo "BANKING_SYNTHETIC_QWEN38_EVAL_OK run_dir=${RUN_DIR} tasks=${EXPECTED_TASKS}"
