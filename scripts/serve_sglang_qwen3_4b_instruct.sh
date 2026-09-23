#!/usr/bin/env bash
# Serve a model with sglang on the cluster and report the node IP + port so
# off-cluster clients can call it. Runs until the job is stopped (job stop).
#
# Usage (via scripts/submit.sh):
#   bash scripts/submit.sh --experiment serve --gpus 1 scripts/serve_sglang_qwen3_4b_instruct.sh
#
# Optional env overrides:
#   MODEL_PATH   HF-format model dir (default Qwen3-4B-Instruct-2507)
#   PORT         server port (default 30000)
#   SERVED_NAME  --served-model-name (default qwen3-4b-instruct-2507)

set -euo pipefail

# job-manager runs a snapshot copy of this script from an arbitrary log dir but
# cds into PROJECT_ROOT first, so resolve the model relative to PWD, not BASH_SOURCE.
SERVICE_AGENT_ROOT="$(cd "${PROJECT_ROOT:-$PWD}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-30000}"
SERVED_NAME="${SERVED_NAME:-qwen3-4b-instruct-2507}"

# torch_dist (Megatron .distcp) is a training format; sglang needs the HF dir.
if [[ -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "[serve] model (HF safetensors): ${MODEL_PATH}"
else
  echo "[ERROR] ${MODEL_PATH} has no model.safetensors.index.json" >&2
  echo "        sglang serves the HF-format dir, not the torch_dist (.distcp) one." >&2
  exit 1
fi

HOST_IP="$(hostname -I | awk '{print $1}')"
echo "[serve] node host=$(hostname) ip=${HOST_IP} port=${PORT}"

# Keep the job alive for the lifetime of the server.
cleanup() {
  trap - EXIT TERM INT
  [[ -n "${SERVER_PID:-}" ]] && kill -TERM "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Qwen3-2507-Instruct is non-thinking: no reasoning parser at all (this sglang
# build has no `qwen3_non_thinking` choice; plain `qwen3` misclassifies normal
# content as reasoning_content and breaks tool-call parsing).
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp 1 \
  --dtype bfloat16 \
  --context-length 32768 \
  --mem-fraction-static 0.90 \
  --max-running-requests 32 \
  --tool-call-parser qwen \
  >"${SERVER_LOG:-/tmp/sglang_server.log}" 2>&1 &
SERVER_PID="$!"

echo "[serve] waiting for /health ..."
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] server exited early:" >&2
    tail -n 200 "${SERVER_LOG:-/tmp/sglang_server.log}" >&2 || true
    exit 1
  fi
  if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

if ! curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "[ERROR] server not healthy after timeout:" >&2
  tail -n 200 "${SERVER_LOG:-/tmp/sglang_server.log}" >&2 || true
  exit 1
fi

echo "=============================================================="
echo "SERVICE READY"
echo "  node      : $(hostname)"
echo "  base_url  : http://${HOST_IP}:${PORT}"
echo "  api       : ${HOST_IP}:${PORT}/v1"
echo "  model     : ${SERVED_NAME}"
echo "  python    : openai.Api base_url=http://${HOST_IP}:${PORT}/v1, api_key=EMPTY"
echo "=============================================================="

# Smoke test one completion, then idle while the server stays up.
curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${SERVED_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say ready in one word.\"}], \"max_tokens\": 16}" \
  | tee /tmp/sglang_smoke.json
echo

wait "${SERVER_PID}"
