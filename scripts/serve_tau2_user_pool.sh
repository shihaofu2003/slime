#!/usr/bin/env bash
# Four-card User service: two TP=2 replicas behind one OpenAI-compatible router.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SERVICE_AGENT_ROOT="$(dirname "${PROJECT_ROOT}")"
MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
SERVED_NAME="Qwen3.6-27B-tau2-user-nonthinking"
LOG_DIR="${PROJECT_ROOT}/output/experiments/tau2-external-user-pool/service/$(date +%Y%m%d_%H%M%S)"
USER_ENDPOINT_FILE="${USER_ENDPOINT_FILE:-${PROJECT_ROOT}/output/experiments/tau2-external-user-pool/user_endpoint.env}"
mkdir -p "${LOG_DIR}"
rm -f "${USER_ENDPOINT_FILE}"
HOST_IP="$(hostname -I | awk '{print $1}')"
pids=()
cleanup() {
  trap - EXIT TERM INT
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for replica in 0 1; do
  port=$((30001 + replica))
  CUDA_VISIBLE_DEVICES="$((replica * 2)),$((replica * 2 + 1))" \
    python3 -m sglang.launch_server \
      --model-path "${MODEL_PATH}" --served-model-name "${SERVED_NAME}" \
      --host 0.0.0.0 --port "${port}" --tp 2 --dtype bfloat16 \
      --mem-fraction-static 0.90 --context-length 65536 --language-only \
      --max-running-requests 32 --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
      >"${LOG_DIR}/user_${replica}.log" 2>&1 &
  pids+=("$!")
done
for replica in 0 1; do
  port=$((30001 + replica))
  ready=0
  for ((attempt=0; attempt<240; attempt++)); do
    if ! kill -0 "${pids[$replica]}" 2>/dev/null; then
      tail -n 80 "${LOG_DIR}/user_${replica}.log"
      exit 1
    fi
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  [[ "${ready}" == 1 ]] || { echo "User replica ${replica} readiness timed out"; exit 1; }
done
python3 -m sglang_router.launch_router --host 0.0.0.0 --port 30000 \
  --worker-urls http://127.0.0.1:30001 http://127.0.0.1:30002 \
  --policy round_robin >"${LOG_DIR}/router.log" 2>&1 &
pids+=("$!")
for ((attempt=0; attempt<60; attempt++)); do
  kill -0 "${pids[2]}" 2>/dev/null || { cat "${LOG_DIR}/router.log"; exit 1; }
  if curl -fsS --max-time 3 http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
    endpoint_tmp="${USER_ENDPOINT_FILE}.tmp.$$"
    mkdir -p "$(dirname "${USER_ENDPOINT_FILE}")"
    cat >"${endpoint_tmp}" <<EOF
TAU2_USER_API_BASE=http://${HOST_IP}:30000/v1
TAU2_USER_MODEL=${SERVED_NAME}
TAU2_USER_API_KEY=EMPTY
TAU2_USER_READY=1
EOF
    mv -f "${endpoint_tmp}" "${USER_ENDPOINT_FILE}"
    echo "USER SERVICE READY: TAU2_USER_API_BASE=http://${HOST_IP}:30000/v1"
    echo "Training: USER_SGLANG=0 TAU2_USER_API_KEY=EMPTY; model=${SERVED_NAME}"
    echo "Endpoint file: ${USER_ENDPOINT_FILE}"
    echo "Service logs: ${LOG_DIR}"
    # A child exit makes the service job fail rather than silently losing capacity.
    wait -n "${pids[@]}"
    exit 1
  fi
  sleep 2
done
echo "User router readiness timed out"
tail -n 80 "${LOG_DIR}/router.log"
exit 1
