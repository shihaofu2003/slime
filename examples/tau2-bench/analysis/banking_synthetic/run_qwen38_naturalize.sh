#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
CANDIDATE_DIR="${EXPERIMENT_DIR}/candidates"
MODEL_PATH="/mnt/afs/models/Qwen3.8-27B"
SERVED_MODEL="Qwen3.8-27B-banking-naturalizer"
PORT="${NATURALIZE_PORT:-30000}"
MODE="${1:-smoke}"

export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  smoke)
    OUTPUT_DIR="${SYNTHETIC_SMOKE_DIR:-${EXPERIMENT_DIR}/smoke}"
    SOURCE_DIR="${OUTPUT_DIR}/source"
    mkdir -p "${SOURCE_DIR}" "${OUTPUT_DIR}"
    python3 "${PIPELINE}/select_tasks.py" \
      --tasks "${CANDIDATE_DIR}/pilot_tasks.json" \
      --specs "${CANDIDATE_DIR}/pilot_scenario_specs.jsonl" \
      --contracts "${CANDIDATE_DIR}/pilot_contracts.jsonl" \
      --per-category 2 \
      --output-dir "${SOURCE_DIR}"
    INPUT_TASKS="${SOURCE_DIR}/tasks.json"
    SPECS="${SOURCE_DIR}/scenario_specs.jsonl"
    CONTRACTS="${SOURCE_DIR}/contracts.jsonl"
    OUTPUT_TASKS="${OUTPUT_DIR}/tasks_naturalized.json"
    ;;
  pilot)
    OUTPUT_DIR="${EXPERIMENT_DIR}/pilot"
    mkdir -p "${OUTPUT_DIR}"
    INPUT_TASKS="${CANDIDATE_DIR}/pilot_tasks.json"
    SPECS="${CANDIDATE_DIR}/pilot_scenario_specs.jsonl"
    CONTRACTS="${CANDIDATE_DIR}/pilot_contracts.jsonl"
    OUTPUT_TASKS="${OUTPUT_DIR}/tasks_naturalized.json"
    ;;
  scale)
    INPUT_TASKS="${SYNTHETIC_TASKS:?SYNTHETIC_TASKS is required for scale mode}"
    SPECS="${SYNTHETIC_SPECS:?SYNTHETIC_SPECS is required for scale mode}"
    CONTRACTS="${SYNTHETIC_CONTRACTS:?SYNTHETIC_CONTRACTS is required for scale mode}"
    OUTPUT_DIR="${SYNTHETIC_NATURALIZE_DIR:?SYNTHETIC_NATURALIZE_DIR is required for scale mode}"
    mkdir -p "${OUTPUT_DIR}"
    OUTPUT_TASKS="${OUTPUT_DIR}/tasks_naturalized.json"
    ;;
  *)
    echo "Expected smoke, pilot, or scale, got: ${MODE}" >&2
    exit 2
    ;;
esac

SERVER_LOG="${OUTPUT_DIR}/naturalizer_sglang.log"
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tp 1 \
  --dtype bfloat16 \
  --context-length 32768 \
  --max-running-requests 8 \
  --mem-fraction-static 0.90 \
  --reasoning-parser qwen3 \
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

python3 "${PIPELINE}/naturalize.py" \
  --tasks "${INPUT_TASKS}" \
  --output "${OUTPUT_TASKS}" \
  --report "${OUTPUT_DIR}/naturalization_report.json" \
  --endpoint "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED_MODEL}" \
  --model-path "${MODEL_PATH}" \
  --concurrency 8

python3 "${PIPELINE}/validate.py" \
  --tasks "${OUTPUT_TASKS}" \
  --specs "${SPECS}" \
  --contracts "${CONTRACTS}" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --documents-dir "${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge/documents" \
  --output "${OUTPUT_DIR}/naturalized_static_validation.json" \
  --failures "${OUTPUT_DIR}/naturalized_static_failures.json"

python3 "${PIPELINE}/isolation.py" audit \
  --benchmark-tasks-dir "${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge/tasks" \
  --tasks "${OUTPUT_TASKS}" \
  --specs "${SPECS}" \
  --contracts "${CONTRACTS}" \
  --allowlist "${EXPERIMENT_DIR}/allowed_documents.json" \
  --output "${OUTPUT_DIR}/naturalized_isolation_audit.json" \
  --failures "${OUTPUT_DIR}/naturalized_isolation_failures.json"

echo "BANKING_SYNTHETIC_NATURALIZE_OK mode=${MODE} output=${OUTPUT_TASKS}"
