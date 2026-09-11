#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
DOCUMENTS_DIR="${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge/documents"
MODEL_PATH="/mnt/afs/models/Qwen3.8-27B"
SERVED_MODEL="Qwen3.8-27B-banking-node-teacher"
PORT="${TEACHER_PORT:-30000}"
MODE="${1:-smoke}"
TASK_IDS_FILE="${2:-}"
AUTONOMOUS_RESULTS="${3:-}"
SHARD_INDEX="${4:-0}"
NUM_SHARDS="${5:-1}"

if [[ -z "${AUTONOMOUS_RESULTS}" || "${AUTONOMOUS_RESULTS}" == "-" ]]; then
  echo "Node-guided teaching requires at least one autonomous results file" >&2
  exit 2
fi

export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  smoke)
    TASKS="${SYNTHETIC_TASKS:-${EXPERIMENT_DIR}/smoke/tasks_naturalized.json}"
    SPECS="${SYNTHETIC_SPECS:-${EXPERIMENT_DIR}/smoke/source/scenario_specs.jsonl}"
    CONTRACTS="${SYNTHETIC_CONTRACTS:-${EXPERIMENT_DIR}/smoke/source/contracts.jsonl}"
    ;;
  pilot)
    TASKS="${SYNTHETIC_TASKS:-${EXPERIMENT_DIR}/pilot/tasks_naturalized.json}"
    SPECS="${SYNTHETIC_SPECS:-${EXPERIMENT_DIR}/candidates/pilot_scenario_specs.jsonl}"
    CONTRACTS="${SYNTHETIC_CONTRACTS:-${EXPERIMENT_DIR}/candidates/pilot_contracts.jsonl}"
    ;;
  scale)
    TASKS="${SYNTHETIC_TASKS:?SYNTHETIC_TASKS is required for scale mode}"
    SPECS="${SYNTHETIC_SPECS:?SYNTHETIC_SPECS is required for scale mode}"
    CONTRACTS="${SYNTHETIC_CONTRACTS:?SYNTHETIC_CONTRACTS is required for scale mode}"
    ;;
  *)
    echo "Expected smoke, pilot, or scale, got: ${MODE}" >&2
    exit 2
    ;;
esac

RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
RUN_DIR="${EXPERIMENT_DIR}/teacher/${MODE}/shard${SHARD_INDEX}of${NUM_SHARDS}/${RUN_STAMP}"
mkdir -p "${RUN_DIR}"
SERVER_LOG="${RUN_DIR}/teacher_sglang.log"

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tp 1 \
  --dtype bfloat16 \
  --context-length 262144 \
  --max-running-requests 1 \
  --mem-fraction-static 0.95 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  for _ in $(seq 1 15); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  kill -KILL "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 100 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=3)" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "${ready}" -ne 1 ]]; then
  tail -n 100 "${SERVER_LOG}" >&2 || true
  exit 1
fi

TEACHER_ARGS=(
  --tasks "${TASKS}"
  --specs "${SPECS}"
  --contracts "${CONTRACTS}"
  --documents-dir "${DOCUMENTS_DIR}"
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json"
  --endpoint "http://127.0.0.1:${PORT}/v1"
  --model "${SERVED_MODEL}"
  --model-path "${MODEL_PATH}"
  --max-attempts 4
  --shard-index "${SHARD_INDEX}"
  --num-shards "${NUM_SHARDS}"
  --output "${RUN_DIR}/successful_trajectories.jsonl"
  --manifest "${RUN_DIR}/teacher_manifest.json"
  --failures "${RUN_DIR}/teacher_failures.json"
)
if [[ -n "${TASK_IDS_FILE}" && "${TASK_IDS_FILE}" != "-" ]]; then
  TEACHER_ARGS+=(--task-ids-file "${TASK_IDS_FILE}")
fi
IFS=',' read -r -a RESULT_FILES <<< "${AUTONOMOUS_RESULTS}"
for result_file in "${RESULT_FILES[@]}"; do
  TEACHER_ARGS+=(--autonomous-results "${result_file}")
done

python3 "${PIPELINE}/teacher.py" "${TEACHER_ARGS[@]}"
echo "BANKING_SYNTHETIC_QWEN38_TEACHER_OK run_dir=${RUN_DIR} shard=${SHARD_INDEX}/${NUM_SHARDS}"
