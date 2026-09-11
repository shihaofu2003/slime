#!/usr/bin/env bash

set -euo pipefail

STAGE="${ATOMIC_GAP_STAGE:-prepare}"
case "${STAGE}" in
  prepare|judge|replay-qwen3|replay-qwen35|summarize) ;;
  *)
    echo "[tau2-atomic-gap] ERROR: ATOMIC_GAP_STAGE must be prepare, judge, replay-qwen3, replay-qwen35, or summarize" >&2
    exit 2
    ;;
esac

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
ANALYSIS_PY="${PROJECT_ROOT}/examples/tau2-bench/analysis/atomic_gap_analysis.py"
TEST_PY="${PROJECT_ROOT}/tests/test_tau2_atomic_gap_analysis.py"
OUT="${ATOMIC_GAP_OUT:-${PROJECT_ROOT}/output/experiments/tau2-qwen3-qwen35-atomic-gap-analysis}"
VITABENCH_PY="/tmp/serviceagent-vitabench-venv/bin/python"
if [[ -n "${ATOMIC_GAP_PYTHON:-}" ]]; then
  PYTHON_BIN="${ATOMIC_GAP_PYTHON}"
elif [[ -x "${VITABENCH_PY}" ]]; then
  PYTHON_BIN="${VITABENCH_PY}"
else
  PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

JUDGE_MODEL_PATH="${ATOMIC_GAP_JUDGE_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
QWEN3_MODEL_PATH="${ATOMIC_GAP_QWEN3_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
QWEN35_MODEL_PATH="${ATOMIC_GAP_QWEN35_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B}"
HOST="${ATOMIC_GAP_HOST:-127.0.0.1}"
PORT="${ATOMIC_GAP_PORT:-30100}"
SERVER_PID=""
SERVER_LOG=""

log() { echo "[tau2-atomic-gap] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

cleanup() {
  trap - EXIT TERM INT HUP
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

wait_for_server() {
  local ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 160 "${SERVER_LOG}" >&2 || true
      die "SGLang exited before becoming ready"
    fi
    if "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('http://${HOST}:${PORT}/health', timeout=3).read()" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  [[ "${ready}" == "1" ]] || {
    tail -n 160 "${SERVER_LOG}" >&2 || true
    die "SGLang health check timed out"
  }
}

start_server() {
  local model_path="$1"
  local served_model="$2"
  local context_length="$3"
  shift 3
  SERVER_LOG="${OUT}/sglang_${served_model}.log"
  log "starting ${served_model} from ${model_path}"
  setsid env \
    -u OPENAI_API_KEY \
    -u OPENAI_API_BASE \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path "${model_path}" \
      --served-model-name "${served_model}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --tp 1 \
      --context-length "${context_length}" \
      --mem-fraction-static "${ATOMIC_GAP_MEM_FRACTION:-0.90}" \
      --max-running-requests 4 \
      "$@" \
      >"${SERVER_LOG}" 2>&1 &
  SERVER_PID="$!"
  wait_for_server
}

run_prepare() {
  log "running CPU tests"
  "${PYTHON_BIN}" "${TEST_PY}"
  log "pairing completed evaluations and preparing review packets"
  "${PYTHON_BIN}" "${ANALYSIS_PY}" prepare \
    --output-dir "${OUT}" \
    --tokenizer "${JUDGE_MODEL_PATH}" \
    --expected-cells 788 \
    --expected-tasks 197
}

run_select_replay() {
  [[ -f "${OUT}/agent_adjudications.jsonl" ]] || \
    die "missing ${OUT}/agent_adjudications.jsonl"
  "${PYTHON_BIN}" "${ANALYSIS_PY}" select-replay \
    --case-packets "${OUT}/case_packets.jsonl" \
    --agent-queue "${OUT}/agent_review_queue.jsonl" \
    --qwen-reviews "${OUT}/qwen36_reviews.jsonl" \
    --agent-reviews "${OUT}/agent_adjudications.jsonl" \
    --output-dir "${OUT}"
}

mkdir -p "${OUT}"

case "${STAGE}" in
  prepare)
    run_prepare
    ;;
  judge)
    [[ -f "${OUT}/inventory.json" ]] || run_prepare
    start_server "${JUDGE_MODEL_PATH}" "Qwen3.6-27B-atomic-gap-judge" 32768 \
      --reasoning-parser qwen3
    log "running two-pass diagnostic calibration"
    "${PYTHON_BIN}" "${ANALYSIS_PY}" judge \
      --input "${OUT}/calibration_payloads.jsonl" \
      --output "${OUT}/calibration_reviews.jsonl" \
      --base-url "http://${HOST}:${PORT}/v1" \
      --model "Qwen3.6-27B-atomic-gap-judge" \
      --primary-passes 2 \
      --concurrency 4 \
      --enable-thinking false
    log "reviewing canonical task packets"
    "${PYTHON_BIN}" "${ANALYSIS_PY}" judge \
      --input "${OUT}/judge_payloads.jsonl" \
      --output "${OUT}/qwen36_reviews.jsonl" \
      --base-url "http://${HOST}:${PORT}/v1" \
      --model "Qwen3.6-27B-atomic-gap-judge" \
      --primary-passes 1 \
      --concurrency 4 \
      --enable-thinking false \
      --case-packets "${OUT}/case_packets.jsonl" \
      --base-agent-queue "${OUT}/agent_review_queue.jsonl" \
      --calibration-reviews "${OUT}/calibration_reviews.jsonl" \
      --agent-queue-output "${OUT}/agent_review_queue.jsonl"
    ;;
  replay-qwen3)
    [[ -f "${OUT}/replay_manifest.jsonl" ]] || run_select_replay
    start_server "${QWEN3_MODEL_PATH}" "Qwen3-4B-Instruct-2507-replay" 65536 \
      --tool-call-parser qwen
    "${PYTHON_BIN}" "${ANALYSIS_PY}" replay \
      --manifest "${OUT}/replay_manifest.jsonl" \
      --model-label qwen3 \
      --served-model "Qwen3-4B-Instruct-2507-replay" \
      --model-path "${QWEN3_MODEL_PATH}" \
      --base-url "http://${HOST}:${PORT}/v1" \
      --output "${OUT}/replay_qwen3.jsonl" \
      --review-queue-output "${OUT}/replay_review_queue_qwen3.jsonl" \
      --max-tokens 1200 \
      --context-tokens 65536 \
      --enable-thinking unset
    ;;
  replay-qwen35)
    [[ -f "${OUT}/replay_manifest.jsonl" ]] || run_select_replay
    start_server "${QWEN35_MODEL_PATH}" "Qwen3.5-4B-nonthinking-replay" 65536 \
      --tool-call-parser qwen3_coder
    "${PYTHON_BIN}" "${ANALYSIS_PY}" replay \
      --manifest "${OUT}/replay_manifest.jsonl" \
      --model-label qwen35 \
      --served-model "Qwen3.5-4B-nonthinking-replay" \
      --model-path "${QWEN35_MODEL_PATH}" \
      --base-url "http://${HOST}:${PORT}/v1" \
      --output "${OUT}/replay_qwen35.jsonl" \
      --review-queue-output "${OUT}/replay_review_queue_qwen35.jsonl" \
      --max-tokens 8192 \
      --context-tokens 65536 \
      --enable-thinking false
    ;;
  summarize)
    [[ -f "${OUT}/replay_adjudications.jsonl" ]] || \
      die "missing ${OUT}/replay_adjudications.jsonl"
    "${PYTHON_BIN}" "${ANALYSIS_PY}" summarize \
      --adjudicated-cases "${OUT}/adjudicated_cases.jsonl" \
      --case-packets "${OUT}/case_packets.jsonl" \
      --replay-qwen3 "${OUT}/replay_qwen3.jsonl" \
      --replay-qwen35 "${OUT}/replay_qwen35.jsonl" \
      --replay-adjudications "${OUT}/replay_adjudications.jsonl" \
      --inventory "${OUT}/inventory.json" \
      --output-dir "${OUT}"
    ;;
esac

log "stage ${STAGE} complete"
