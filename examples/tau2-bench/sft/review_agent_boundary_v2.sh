#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
SERVER_LOG="${SERVER_LOG:-${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2/qwen3_6_27b_review_server.log}"
SERVER_PID=""

export PYTHONPATH="${PROJECT_ROOT}:${SERVICE_AGENT_ROOT}/tau2-bench/src:${PYTHONPATH:-}"
mkdir -p "$(dirname "${SERVER_LOG}")"

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name Qwen3.6-27B \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp 1 \
  --context-length 32768 \
  --mem-fraction-static 0.90 \
  --max-running-requests "${REVIEW_WORKERS:-8}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

ready=0
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 160 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if python3 -c "import urllib.request; urllib.request.urlopen('http://${HOST}:${PORT}/health', timeout=3)" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "${ready}" != "1" ]]; then
  tail -n 160 "${SERVER_LOG}" >&2 || true
  echo "[ERROR] local Qwen3.6-27B review server did not become ready" >&2
  exit 1
fi

python3 "${PROJECT_ROOT}/examples/tau2-bench/sft/review_boundary_targets.py" \
  --base-url "http://${HOST}:${PORT}/v1" \
  --model Qwen3.6-27B \
  --workers "${REVIEW_WORKERS:-8}" \
  "$@"
