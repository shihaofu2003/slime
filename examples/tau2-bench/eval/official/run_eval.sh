#!/usr/bin/env bash
#
# Official tau2-bench eval runner for local sglang policy/user servers.

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXAMPLE_DIR="${PROJECT_ROOT}/examples/tau2-bench"
OFFICIAL_DIR="${EXAMPLE_DIR}/eval/official"
export PYTHONPATH="${SERVICE_AGENT_ROOT}/tau2-bench/src:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP="${TP:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
AGENT_CUDA_VISIBLE_DEVICES="${AGENT_CUDA_VISIBLE_DEVICES:-0}"
DOMAINS="${DOMAINS:-airline,retail,telecom}"
TASK_SPLIT="${TASK_SPLIT:-test}"
NUM_TASKS="${NUM_TASKS-1}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_ERRORS="${MAX_ERRORS:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
SEED="${SEED:-300}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.6}"
AGENT_TOP_P="${AGENT_TOP_P:-1.0}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1200}"
USER_SGLANG="${USER_SGLANG:-1}"
USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
USER_MODEL="${USER_MODEL:-$(basename "${USER_MODEL_PATH}")}"
USER_PORT="${USER_PORT:-30001}"
USER_TP="${USER_TP:-1}"
USER_MEM_FRACTION="${USER_MEM_FRACTION:-0.85}"
USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-1}"
USER_TEMPERATURE="${USER_TEMPERATURE:-0.0}"
USER_TOP_P="${USER_TOP_P:-}"
USER_MAX_TOKENS="${USER_MAX_TOKENS:-512}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
SAVE_PREFIX="${SAVE_PREFIX:-tau2_official_${MODEL_NAME}_${RUN_STAMP}}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${OFFICIAL_DIR}/outputs/${MODEL_NAME}/summary.json}"

AGENT_SGLANG_EXTRA_ARGS="${AGENT_SGLANG_EXTRA_ARGS:-${SGLANG_EXTRA_ARGS:-}}"
# Default the user sglang server to the Qwen tool-call parser so the trained
# user model's <tool_call> text is decoded into structured tool_calls (needed
# for telecom TelecomUserTools to execute). Override with USER_SGLANG_EXTRA_ARGS.
# The agent server is untouched: it uses raw /generate and parses in Python.
USER_SGLANG_EXTRA_ARGS="${USER_SGLANG_EXTRA_ARGS:---tool-call-parser qwen}"

log(){ echo "[tau2-official-eval] $*" >&2; }

ENV_FILE="${SERVICE_AGENT_ROOT}/tau2-bench/.env"
get_env() {
  local key="$1"
  grep -m1 -E "^${key}=" "${ENV_FILE}" 2>/dev/null | head -n1 | sed -E "s/^${key}=//; s/\r$//"
}

if [[ "${USER_SGLANG}" == "0" ]]; then
  if [[ -f "${ENV_FILE}" ]]; then
    : "${OPENAI_API_KEY:=$(get_env OPENAI_API_KEY)}"; export OPENAI_API_KEY
    : "${OPENAI_API_BASE:=$(get_env OPENAI_API_BASE)}"; export OPENAI_API_BASE
  else
    log "WARN: ${ENV_FILE} not found; user-sim credentials must come from the environment"
  fi

  if [[ -n "${TAU2_USER_API_BASE:-}" ]]; then
    USER_API_BASE="${TAU2_USER_API_BASE}"
    USER_API_KEY="${TAU2_USER_API_KEY:-dummy-key-for-local-server}"
  else
    USER_API_BASE="${USER_API_BASE:-${OPENAI_API_BASE:-}}"
    USER_API_KEY="${USER_API_KEY:-${OPENAI_API_KEY:-}}"
  fi

  if [[ -z "${USER_API_KEY}" ]]; then
    log "ERROR: no user-sim credentials; set OPENAI_API_KEY/OPENAI_API_BASE or TAU2_USER_API_BASE, or use USER_SGLANG=1" >&2
    exit 1
  fi
else
  USER_API_BASE="${USER_API_BASE:-http://${HOST}:${USER_PORT}/v1}"
  USER_API_KEY="${USER_API_KEY:-dummy-key-for-local-server}"
fi

mkdir -p "$(dirname "${SUMMARY_OUTPUT}")"
SGLANG_LOG="$(dirname "${SUMMARY_OUTPUT}")/sglang_${PORT}.log"
USER_SGLANG_LOG="$(dirname "${SUMMARY_OUTPUT}")/sglang_user_${USER_PORT}.log"

SGLANG_PID=""
USER_SGLANG_PID=""
SGLANG_STARTED_PID=""

start_sglang() {
  local label="$1"
  local model_path="$2"
  local port="$3"
  local tp="$4"
  local mem_fraction="$5"
  local cuda_devices="$6"
  local log_path="$7"
  local served_model_name="$8"
  shift 8

  log "starting sglang ${label} server: ${model_path} (tp=${tp}, cuda=${cuda_devices}) -> ${HOST}:${port}"
  if [[ -n "${served_model_name}" ]]; then
    # shellcheck disable=SC2086  # Extra args are intentionally word-split into flags.
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 -m sglang.launch_server \
      --model-path "${model_path}" \
      --served-model-name "${served_model_name}" \
      --host "${HOST}" --port "${port}" \
      --tp "${tp}" --mem-fraction-static "${mem_fraction}" \
      "$@" \
      >"${log_path}" 2>&1 &
  else
    # shellcheck disable=SC2086  # Extra args are intentionally word-split into flags.
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python3 -m sglang.launch_server \
      --model-path "${model_path}" \
      --host "${HOST}" --port "${port}" \
      --tp "${tp}" --mem-fraction-static "${mem_fraction}" \
      "$@" \
      >"${log_path}" 2>&1 &
  fi
  local pid=$!
  log "sglang ${label} pid=${pid}; log: ${log_path}"
  SGLANG_STARTED_PID="${pid}"
}

cleanup() {
  if [[ -n "${USER_SGLANG_PID}" ]]; then
    log "cleaning up sglang user (pid=${USER_SGLANG_PID})"
    kill "${USER_SGLANG_PID}" 2>/dev/null || true
    wait "${USER_SGLANG_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SGLANG_PID}" ]]; then
    log "cleaning up sglang agent (pid=${SGLANG_PID})"
    kill "${SGLANG_PID}" 2>/dev/null || true
    wait "${SGLANG_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_sglang() {
  local label="$1"
  local pid="$2"
  local port="$3"
  local log_path="$4"

  log "waiting for sglang ${label} /health ..."
  local ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      log "ERROR: sglang ${label} died before becoming ready; tail of ${log_path}:" >&2
      tail -n 80 "${log_path}" >&2 || true
      exit 1
    fi
    if python3 -c "import urllib.request,sys; urllib.request.urlopen('http://${HOST}:${port}/health', timeout=3); sys.exit(0)" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "${ready}" -ne 1 ]]; then
    log "ERROR: sglang ${label} did not become ready in time; tail of ${log_path}:" >&2
    tail -n 80 "${log_path}" >&2 || true
    exit 1
  fi
  log "sglang ${label} ready."
}

# shellcheck disable=SC2086  # Extra args are intentionally word-split into flags.
start_sglang agent "${MODEL_PATH}" "${PORT}" "${TP}" "${MEM_FRACTION}" "${AGENT_CUDA_VISIBLE_DEVICES}" "${SGLANG_LOG}" "" ${AGENT_SGLANG_EXTRA_ARGS}
SGLANG_PID="${SGLANG_STARTED_PID}"
if [[ "${USER_SGLANG}" != "0" ]]; then
  # shellcheck disable=SC2086  # Extra args are intentionally word-split into flags.
  start_sglang user "${USER_MODEL_PATH}" "${USER_PORT}" "${USER_TP}" "${USER_MEM_FRACTION}" "${USER_CUDA_VISIBLE_DEVICES}" "${USER_SGLANG_LOG}" "${USER_MODEL}" ${USER_SGLANG_EXTRA_ARGS}
  USER_SGLANG_PID="${SGLANG_STARTED_PID}"
fi

wait_for_sglang agent "${SGLANG_PID}" "${PORT}" "${SGLANG_LOG}"
if [[ "${USER_SGLANG}" != "0" ]]; then
  wait_for_sglang user "${USER_SGLANG_PID}" "${USER_PORT}" "${USER_SGLANG_LOG}"
fi

cd "${PROJECT_ROOT}"
RUN_ARGS=(
  --domains "${DOMAINS}"
  --task-split-name "${TASK_SPLIT}"
  --num-trials "${NUM_TRIALS}"
  --max-steps "${MAX_STEPS}"
  --max-errors "${MAX_ERRORS}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --seed "${SEED}"
  --save-prefix "${SAVE_PREFIX}"
  --summary-output "${SUMMARY_OUTPUT}"
  --agent-llm "${AGENT_LLM:-${MODEL_PATH}}"
  --agent-api-base "http://${HOST}:${PORT}/generate"
  --agent-api-key "${AGENT_API_KEY:-dummy-key-for-local-server}"
  --agent-temperature "${AGENT_TEMPERATURE}"
  --agent-top-p "${AGENT_TOP_P}"
  --agent-max-tokens "${AGENT_MAX_TOKENS}"
  --user-llm "${USER_MODEL}"
  --user-api-base "${USER_API_BASE}"
  --user-api-key "${USER_API_KEY}"
  --user-temperature "${USER_TEMPERATURE}"
)

if [[ -n "${NUM_TASKS}" ]]; then
  RUN_ARGS+=(--num-tasks "${NUM_TASKS}")
fi
if [[ -n "${TASK_IDS:-}" ]]; then
  RUN_ARGS+=(--task-ids "${TASK_IDS}")
fi
if [[ -n "${AGENT_LLM_ARGS_JSON:-}" ]]; then
  RUN_ARGS+=(--agent-llm-args-json "${AGENT_LLM_ARGS_JSON}")
fi
if [[ -n "${AGENT_EXTRA_BODY_JSON:-}" ]]; then
  RUN_ARGS+=(--agent-extra-body-json "${AGENT_EXTRA_BODY_JSON}")
fi
if [[ -n "${USER_LLM_ARGS_JSON:-}" ]]; then
  RUN_ARGS+=(--user-llm-args-json "${USER_LLM_ARGS_JSON}")
fi
if [[ -n "${USER_TOP_P}" ]]; then
  RUN_ARGS+=(--user-top-p "${USER_TOP_P}")
fi
if [[ -n "${USER_MAX_TOKENS}" ]]; then
  RUN_ARGS+=(--user-max-tokens "${USER_MAX_TOKENS}")
fi
if [[ -n "${USER_EXTRA_BODY_JSON:-}" ]]; then
  RUN_ARGS+=(--user-extra-body-json "${USER_EXTRA_BODY_JSON}")
fi
if [[ "${AUTO_RESUME:-0}" != "0" ]]; then
  RUN_ARGS+=(--auto-resume)
fi
if [[ "${VERBOSE_LOGS:-0}" != "0" ]]; then
  RUN_ARGS+=(--verbose-logs)
fi

log "running official eval: domains=${DOMAINS} split=${TASK_SPLIT} trials=${NUM_TRIALS} num_tasks=${NUM_TASKS:-all}"
python3 "${OFFICIAL_DIR}/run_eval.py" "${RUN_ARGS[@]}"
log "eval complete; summary: ${SUMMARY_OUTPUT}"
