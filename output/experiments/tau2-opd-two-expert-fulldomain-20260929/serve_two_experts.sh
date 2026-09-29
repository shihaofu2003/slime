#!/usr/bin/env bash
# Four-card persistent two-expert (teacher) service for pure OPD:
#   GPUs 0,1: Airline domain expert iter29 (TP2, port 31001)
#   GPUs 2,3: progress-db-count-v1 mix RL iter129 anchor (TP2, port 31002)
# The mix RL checkpoint is torch_dist only, so it is converted to HF in-job
# (CPU) before serving. Publishes all four TAU2_OPD_*_URL entries to this
# experiment's teacher_endpoints.env: airline -> expert, the other three
# domains -> the mix RL anchor (anti-forgetting self-distillation target).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
SERVICE_AGENT_ROOT="$(dirname "${PROJECT_ROOT}")"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-opd-two-expert-fulldomain-20260929"
AIRLINE_MODEL="${AIRLINE_MODEL:-${PROJECT_ROOT}/output/experiments/tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-airline-train/checkpoints/iter_0000029_hf}"
MIX_CKPT_ROOT="${PROJECT_ROOT}/output/experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/arms/async/20260927_credit-signal-v1-train/checkpoints"
MIX_ITER_DIR="${MIX_CKPT_ROOT}/iter_0000129"
MIX_MODEL="${MIX_CKPT_ROOT}/iter_0000129_hf"
ENDPOINT_FILE="${EXPERIMENT_DIR}/teacher_endpoints.env"
LOG_DIR="${EXPERIMENT_DIR}/service/$(date +%Y%m%d_%H%M%S)"
AIRLINE_PORT="${AIRLINE_PORT:-31001}"
ANCHOR_PORT="${ANCHOR_PORT:-31002}"
mkdir -p "${LOG_DIR}"

[[ -d "${AIRLINE_MODEL}" ]] || { echo "[ERROR] Missing airline expert checkpoint: ${AIRLINE_MODEL}" >&2; exit 1; }

# Convert the mix RL iter129 torch_dist checkpoint to HF once (CPU-only).
if [[ ! -f "${MIX_MODEL}/model.safetensors.index.json" ]]; then
  echo "[INFO] Converting mix RL iter129 to HF: ${MIX_ITER_DIR} -> ${MIX_MODEL}"
  ITER_DIR="${MIX_ITER_DIR}" OUTPUT_DIR="${MIX_MODEL}" \
  ORIGIN_HF_DIR="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507" \
    bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" \
    >"${LOG_DIR}/convert_mix_iter129.log" 2>&1 || {
      echo "[ERROR] mix RL iter129 HF conversion failed" >&2
      tail -n 100 "${LOG_DIR}/convert_mix_iter129.log" >&2 || true
      exit 1
    }
fi
[[ -d "${MIX_MODEL}" ]] || { echo "[ERROR] Missing mix RL anchor checkpoint: ${MIX_MODEL}" >&2; exit 1; }

HOST_IP="$(hostname -I | awk '{print $1}')"
[[ -n "${HOST_IP}" ]] || { echo "[ERROR] Cannot determine routable host" >&2; exit 1; }

rm -f "${ENDPOINT_FILE}"

pids=()
cleanup() {
  trap - EXIT TERM INT
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT TERM INT

CUDA_VISIBLE_DEVICES="0,1" python3 -m sglang.launch_server \
  --model-path "${AIRLINE_MODEL}" \
  --host 0.0.0.0 \
  --port "${AIRLINE_PORT}" \
  --tp 2 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --context-length 32768 \
  --max-total-tokens 131072 \
  --max-running-requests 4 \
  >"${LOG_DIR}/airline_expert.log" 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES="2,3" python3 -m sglang.launch_server \
  --model-path "${MIX_MODEL}" \
  --host 0.0.0.0 \
  --port "${ANCHOR_PORT}" \
  --tp 2 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --context-length 32768 \
  --max-total-tokens 131072 \
  --max-running-requests 4 \
  >"${LOG_DIR}/mix_anchor.log" 2>&1 &
pids+=("$!")

wait_ready() {
  local label="$1" pid="$2" port="$3" log="$4"
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] ${label} exited before readiness" >&2
      tail -n 100 "${log}" >&2 || true
      exit 1
    fi
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "[ERROR] ${label} readiness timed out" >&2
  tail -n 100 "${log}" >&2 || true
  exit 1
}

wait_ready "airline expert" "${pids[0]}" "${AIRLINE_PORT}" "${LOG_DIR}/airline_expert.log"
wait_ready "mix RL anchor" "${pids[1]}" "${ANCHOR_PORT}" "${LOG_DIR}/mix_anchor.log"

# Input-token-logprob preflight on both teachers: the OPD student consumes
# input_token_logprobs.
logprob_preflight() {
  local label="$1" port="$2"
  LABEL="${label}" PORT="${port}" python3 - <<'PY'
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
    raise SystemExit(f"{os.environ['LABEL']} teacher preflight returned invalid input_token_logprobs: {values!r}")
print(f"OPD_TEACHER_PREFLIGHT label={os.environ['LABEL']} tokens={len(values)}")
PY
}

logprob_preflight "airline" "${AIRLINE_PORT}"
logprob_preflight "mix-anchor" "${ANCHOR_PORT}"

endpoint_tmp="${ENDPOINT_FILE}.tmp.$$"
cat >"${endpoint_tmp}" <<EOF
TAU2_OPD_AIRLINE_URL=http://${HOST_IP}:${AIRLINE_PORT}/generate
TAU2_OPD_RETAIL_URL=http://${HOST_IP}:${ANCHOR_PORT}/generate
TAU2_OPD_TELECOM_URL=http://${HOST_IP}:${ANCHOR_PORT}/generate
TAU2_OPD_BANKING_URL=http://${HOST_IP}:${ANCHOR_PORT}/generate
TAU2_OPD_TEACHER_HOST=${HOST_IP}
TAU2_OPD_TEACHER_READY=1
EOF
mv -f "${endpoint_tmp}" "${ENDPOINT_FILE}"
echo "TWO EXPERTS READY:"
echo "  TAU2_OPD_AIRLINE_URL=http://${HOST_IP}:${AIRLINE_PORT}/generate (airline expert iter29)"
echo "  TAU2_OPD_RETAIL/TELECOM/BANKING_URL=http://${HOST_IP}:${ANCHOR_PORT}/generate (mix RL iter129 anchor)"
echo "Endpoint file: ${ENDPOINT_FILE}"
echo "Service logs: ${LOG_DIR}"

# A child exit makes the service job fail rather than silently losing a teacher.
wait -n "${pids[@]}"
echo "[ERROR] a teacher server exited unexpectedly" >&2
exit 1
