#!/usr/bin/env bash
#
# tau2-bench evaluation runner: local sglang policy server + tau2-bench eval.
#
# One-job flow:
#   1. parse the OpenAI-compatible proxy creds from tau2-bench/.env,
#   2. launch an sglang policy server (the model being evaluated),
#   3. wait for /health,
#   4. run the archived legacy eval.py against it,
#   5. tear the server down on exit.
#
# The user simulator defaults to the proxy (gemini-2.5-flash) via the creds
# above; to use a local sglang user-sim instead, set TAU2_USER_API_BASE +
# TAU2_USER_MODEL (see README.md) and give the job a 2nd GPU.
#
# Submit a smoke (1 GPU, 1 task/domain x1 sample):
#   bash scripts/submit.sh --experiment tau2-eval --gpus 1 \
#       examples/tau2-bench/eval/legacy/run_eval.sh
#
# Submit a full run (override EVAL_ARGS):
#   EVAL_ARGS="--num-samples 4" bash scripts/submit.sh --experiment tau2-eval \
#       --gpus 1 examples/tau2-bench/eval/legacy/run_eval.sh

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXAMPLE_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/legacy"
export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PYTHONPATH:-}"

# --- policy server + eval config (all overridable via env) ---
MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP="${TP:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
DOMAINS="${DOMAINS:-airline,retail,telecom}"
TASK_SPLIT="${TASK_SPLIT:-test}"
OUTPUT="${OUTPUT:-${EXAMPLE_DIR}/outputs/eval/tau2_eval.json}"
# User-simulator model routed through the OpenAI-compatible proxy in
# tau2-bench/.env (override via env). Ignored when TAU2_USER_API_BASE is set.
USER_MODEL="${USER_MODEL:-gemini-2.5-flash}"
# smoke by default; override for a full run, e.g. EVAL_ARGS="--num-samples 4"
EVAL_ARGS="${EVAL_ARGS:---max-tasks-per-domain 1 --num-samples 1}"

log(){ echo "[tau2-eval] $*"; }

# --- proxy creds: parse (not source) tau2-bench/.env ---
# The .env contains literal "<your_key_here>" placeholders; their angle brackets
# break `source`, so extract only the two keys we need.
ENV_FILE="${SERVICE_AGENT_ROOT}/tau2-bench/.env"
get_env() {
  # $1 = KEY; prints its value, or empty if absent/unset
  local key="$1"
  grep -m1 -E "^${key}=" "${ENV_FILE}" 2>/dev/null | head -n1 | sed -E "s/^${key}=//; s/\r$//"
}
if [[ -f "${ENV_FILE}" ]]; then
  : "${OPENAI_API_KEY:=$(get_env OPENAI_API_KEY)}"; export OPENAI_API_KEY
  : "${OPENAI_API_BASE:=$(get_env OPENAI_API_BASE)}"; export OPENAI_API_BASE
else
  log "WARN: ${ENV_FILE} not found; user-sim proxy creds must come from the environment"
fi
if [[ -z "${OPENAI_API_KEY:-}" && -z "${TAU2_USER_API_BASE:-}" ]]; then
  log "ERROR: no user-sim credentials (set OPENAI_API_KEY/OPENAI_API_BASE or TAU2_USER_API_BASE)" >&2
  exit 1
fi

SGLANG_URL="http://${HOST}:${PORT}/generate"
SGLANG_LOG_DIR="$(dirname "${OUTPUT}")"
SGLANG_LOG="${SGLANG_LOG_DIR}/sglang_${PORT}.log"
mkdir -p "${SGLANG_LOG_DIR}"

# --- launch sglang policy server ---
# The eval drives the raw /generate endpoint (eval.py applies the chat template
# client-side and parses the <tool_call> client-side), so sglang does NOT need
# --reasoning-parser / --tool-call-parser here — those only affect the
# /v1/chat/completions endpoint. Text-only serving: tau2 sends no images.
log "starting sglang policy server: ${MODEL_PATH} (tp=${TP}) -> ${HOST}:${PORT}"
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" --port "${PORT}" \
  --tp "${TP}" --mem-fraction-static "${MEM_FRACTION}" \
  >"${SGLANG_LOG}" 2>&1 &
SGLANG_PID=$!
log "sglang pid=${SGLANG_PID}; log: ${SGLANG_LOG}"

cleanup() {
  log "cleaning up sglang (pid=${SGLANG_PID})"
  kill "${SGLANG_PID}" 2>/dev/null || true
  wait "${SGLANG_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# --- wait for readiness (or until sglang dies) ---
log "waiting for sglang /health ..."
ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    log "ERROR: sglang died before becoming ready; tail of ${SGLANG_LOG}:" >&2
    tail -n 40 "${SGLANG_LOG}" >&2 || true
    exit 1
  fi
  if python3 -c "import urllib.request,sys; urllib.request.urlopen('http://${HOST}:${PORT}/health', timeout=3); sys.exit(0)" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "${ready}" -ne 1 ]]; then
  log "ERROR: sglang did not become ready in time; tail of ${SGLANG_LOG}:" >&2
  tail -n 40 "${SGLANG_LOG}" >&2 || true
  exit 1
fi
log "sglang ready."

# --- run eval ---
log "running eval.py: domains=${DOMAINS} split=${TASK_SPLIT} args='${EVAL_ARGS}'"
cd "${PROJECT_ROOT}"
# shellcheck disable=SC2086  # EVAL_ARGS is intentionally word-split into flags
python3 "${EXAMPLE_DIR}/eval.py" \
  --hf-checkpoint "${MODEL_PATH}" \
  --sglang-url "${SGLANG_URL}" \
  --domains "${DOMAINS}" \
  --task-split "${TASK_SPLIT}" \
  --output "${OUTPUT}" \
  --user-model "${USER_MODEL}" \
  ${EVAL_ARGS}

log "eval complete; report: ${OUTPUT}"
