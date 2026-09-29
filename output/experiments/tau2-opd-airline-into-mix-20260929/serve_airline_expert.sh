#!/usr/bin/env bash
# Two-card persistent Airline expert (teacher) service for pure OPD.
# Serves the selected domain-expert checkpoint with TP2 and publishes
# TAU2_OPD_AIRLINE_URL to this experiment's teacher_endpoints.env.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-opd-airline-into-mix-20260929"
EXPERT_MODEL="${EXPERT_MODEL:-${PROJECT_ROOT}/output/experiments/tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-airline-train/checkpoints/iter_0000029_hf}"
ENDPOINT_FILE="${EXPERIMENT_DIR}/teacher_endpoints.env"
LOG_DIR="${EXPERIMENT_DIR}/service/$(date +%Y%m%d_%H%M%S)"
PORT="${PORT:-31001}"
mkdir -p "${LOG_DIR}"

[[ -d "${EXPERT_MODEL}" ]] || { echo "[ERROR] Missing expert checkpoint: ${EXPERT_MODEL}" >&2; exit 1; }

HOST_IP="$(hostname -I | awk '{print $1}')"
[[ -n "${HOST_IP}" ]] || { echo "[ERROR] Cannot determine routable host" >&2; exit 1; }

rm -f "${ENDPOINT_FILE}"

CUDA_VISIBLE_DEVICES="0,1" python3 -m sglang.launch_server \
  --model-path "${EXPERT_MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp 2 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --context-length 32768 \
  --max-total-tokens 131072 \
  --max-running-requests 4 \
  >"${LOG_DIR}/airline_expert.log" 2>&1 &
EXPERT_PID=$!

cleanup() {
  trap - EXIT TERM INT
  kill "${EXPERT_PID}" 2>/dev/null || true
  wait "${EXPERT_PID}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

ready=0
for _ in $(seq 1 240); do
  if ! kill -0 "${EXPERT_PID}" 2>/dev/null; then
    echo "[ERROR] airline expert exited before readiness" >&2
    tail -n 100 "${LOG_DIR}/airline_expert.log" >&2 || true
    exit 1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
[[ "${ready}" == 1 ]] || { echo "[ERROR] airline expert readiness timed out" >&2; tail -n 100 "${LOG_DIR}/airline_expert.log" >&2 || true; exit 1; }

# Input-token-logprob preflight: the OPD student consumes input_token_logprobs.
PORT="${PORT}" python3 - <<'PY'
import json
import os
import urllib.request

payload = json.dumps({
    "input_ids": [1, 2, 3],
    "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
    "return_logprob": True,
    "logprob_start_len": 0,
}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{os.environ['PORT']}/generate",
    data=payload,
    headers={"content-type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=30) as response:
    body = json.load(response)
values = body.get("meta_info", {}).get("input_token_logprobs")
if not isinstance(values, list) or len(values) != 3:
    raise SystemExit(f"airline teacher preflight returned invalid input_token_logprobs: {values!r}")
print(f"OPD_TEACHER_PREFLIGHT label=airline tokens={len(values)}")
PY

endpoint_tmp="${ENDPOINT_FILE}.tmp.$$"
cat >"${endpoint_tmp}" <<EOF
TAU2_OPD_AIRLINE_URL=http://${HOST_IP}:${PORT}/generate
TAU2_OPD_TEACHER_HOST=${HOST_IP}
TAU2_OPD_TEACHER_READY=1
EOF
mv -f "${endpoint_tmp}" "${ENDPOINT_FILE}"
echo "AIRLINE EXPERT READY: TAU2_OPD_AIRLINE_URL=http://${HOST_IP}:${PORT}/generate"
echo "Endpoint file: ${ENDPOINT_FILE}"
echo "Service log: ${LOG_DIR}/airline_expert.log"

# A child exit makes the service job fail rather than silently losing the teacher.
wait "${EXPERT_PID}"
exit 1
