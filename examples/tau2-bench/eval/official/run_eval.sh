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
AGENT_REPLICA_CUDA_GROUPS="${AGENT_REPLICA_CUDA_GROUPS:-}"
AGENT_WORKER_PORT_BASE="${AGENT_WORKER_PORT_BASE:-31000}"
DOMAINS="${DOMAINS:-airline,retail,telecom}"
TASK_SPLIT="${TASK_SPLIT:-test}"
NUM_TASKS="${NUM_TASKS-1}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_ERRORS="${MAX_ERRORS:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
DOMAIN_CONCURRENCY="${DOMAIN_CONCURRENCY:-}"
GLOBAL_CONCURRENCY="${GLOBAL_CONCURRENCY:-}"
BORROW_COMPLETED_DOMAIN_SLOTS="${BORROW_COMPLETED_DOMAIN_SLOTS:-0}"
PARALLEL_DOMAINS="${PARALLEL_DOMAINS:-0}"
SEED="${SEED:-300}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.6}"
AGENT_TOP_P="${AGENT_TOP_P:-1.0}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1200}"
AGENT_PROTOCOL_PROFILE="${AGENT_PROTOCOL_PROFILE:-}"
USER_SGLANG="${USER_SGLANG:-1}"
USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
USER_MODEL="${USER_MODEL:-$(basename "${USER_MODEL_PATH}")}"
USER_PORT="${USER_PORT:-30001}"
USER_TP="${USER_TP:-1}"
USER_MEM_FRACTION="${USER_MEM_FRACTION:-0.85}"
USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-1}"
USER_REPLICA_CUDA_GROUPS="${USER_REPLICA_CUDA_GROUPS:-}"
USER_WORKER_PORT_BASE="${USER_WORKER_PORT_BASE:-32000}"
USER_TEMPERATURE="${USER_TEMPERATURE:-0.0}"
USER_TOP_P="${USER_TOP_P:-}"
USER_MAX_TOKENS="${USER_MAX_TOKENS:-512}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
SAVE_PREFIX="${SAVE_PREFIX:-tau2_official_${MODEL_NAME}_${RUN_STAMP}}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${OFFICIAL_DIR}/outputs/${MODEL_NAME}/summary.json}"
NAMESPACE_PROBE_OUTPUT="${NAMESPACE_PROBE_OUTPUT:-}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
AGENT_ROUTER_POLICY="${AGENT_ROUTER_POLICY:-${ROUTER_POLICY}}"
USER_ROUTER_POLICY="${USER_ROUTER_POLICY:-${ROUTER_POLICY}}"
AGENT_ROUTER_PROMETHEUS_PORT="${AGENT_ROUTER_PROMETHEUS_PORT:-33000}"
USER_ROUTER_PROMETHEUS_PORT="${USER_ROUTER_PROMETHEUS_PORT:-33001}"

AGENT_SGLANG_EXTRA_ARGS="${AGENT_SGLANG_EXTRA_ARGS:-${SGLANG_EXTRA_ARGS:-}}"
# Default the user sglang server to the Qwen tool-call parser so the trained
# user model's <tool_call> text is decoded into structured tool_calls (needed
# for telecom TelecomUserTools to execute). Override with USER_SGLANG_EXTRA_ARGS.
# The agent server is untouched: it uses raw /generate and parses in Python.
USER_SGLANG_EXTRA_ARGS="${USER_SGLANG_EXTRA_ARGS:---tool-call-parser qwen}"
SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-0}"
if [[ "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE}" != "0" && "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE}" != "1" ]]; then
  echo "[tau2-official-eval] ERROR: SGLANG_ENABLE_DETERMINISTIC_INFERENCE must be 0 or 1" >&2
  exit 1
fi
if [[ "${PARALLEL_DOMAINS}" != "0" && "${PARALLEL_DOMAINS}" != "1" ]]; then
  echo "[tau2-official-eval] ERROR: PARALLEL_DOMAINS must be 0 or 1" >&2
  exit 1
fi
if [[ "${BORROW_COMPLETED_DOMAIN_SLOTS}" != "0" && "${BORROW_COMPLETED_DOMAIN_SLOTS}" != "1" ]]; then
  echo "[tau2-official-eval] ERROR: BORROW_COMPLETED_DOMAIN_SLOTS must be 0 or 1" >&2
  exit 1
fi
SGLANG_DETERMINISTIC_ARGS=()
if [[ "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE}" == "1" ]]; then
  SGLANG_DETERMINISTIC_ARGS+=(--enable-deterministic-inference)
fi

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
OUTPUT_DIR="$(dirname "${SUMMARY_OUTPUT}")"
SGLANG_LOG="${OUTPUT_DIR}/sglang_${PORT}_seed${SEED}_${RUN_STAMP}.log"
USER_SGLANG_LOG="${OUTPUT_DIR}/sglang_user_${USER_PORT}_seed${SEED}_${RUN_STAMP}.log"
AGENT_ROUTER_LOG="${OUTPUT_DIR}/router_agent_${PORT}_seed${SEED}_${RUN_STAMP}.log"
USER_ROUTER_LOG="${OUTPUT_DIR}/router_user_${USER_PORT}_seed${SEED}_${RUN_STAMP}.log"

SERVICE_LABELS=()
SERVICE_PIDS=()
SERVICE_LOGS=()
SGLANG_STARTED_PID=""

record_service() {
  SERVICE_LABELS+=("$1")
  SERVICE_PIDS+=("$2")
  SERVICE_LOGS+=("$3")
}

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
  record_service "${label}" "${pid}" "${log_path}"
}

start_router() {
  local label="$1"
  local port="$2"
  local prometheus_port="$3"
  local log_path="$4"
  local policy="$5"
  shift 5

  log "starting ${label} router: policy=${policy}, workers=$# -> ${HOST}:${port}"
  python3 -m sglang_router.launch_router \
    --host "${HOST}" \
    --port "${port}" \
    --prometheus-host "${HOST}" \
    --prometheus-port "${prometheus_port}" \
    --worker-urls "$@" \
    --policy "${policy}" \
    --log-level info \
    >"${log_path}" 2>&1 &
  local pid=$!
  log "${label} router pid=${pid}; log: ${log_path}"
  SGLANG_STARTED_PID="${pid}"
  record_service "${label}-router" "${pid}" "${log_path}"
}

validate_cuda_group() {
  local label="$1"
  local cuda_group="$2"
  local expected_tp="$3"
  local devices=()
  IFS=',' read -r -a devices <<< "${cuda_group}"
  if [[ -z "${cuda_group}" || "${#devices[@]}" -ne "${expected_tp}" ]]; then
    log "ERROR: ${label} CUDA group '${cuda_group}' must contain ${expected_tp} device(s)"
    exit 1
  fi
}

cleanup() {
  local timeout_seconds="${SGLANG_CLEANUP_TIMEOUT_SECONDS:-30}"
  local i pid any_alive
  for i in "${!SERVICE_PIDS[@]}"; do
    pid="${SERVICE_PIDS[$i]}"
    log "cleaning up ${SERVICE_LABELS[$i]} (pid=${pid})"
    kill "${pid}" 2>/dev/null || true
  done
  for _ in $(seq 1 "${timeout_seconds}"); do
    any_alive=0
    for pid in "${SERVICE_PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    [[ "${any_alive}" -eq 0 ]] && break
    sleep 1
  done
  for i in "${!SERVICE_PIDS[@]}"; do
    pid="${SERVICE_PIDS[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      log "WARN: ${SERVICE_LABELS[$i]} did not exit after ${timeout_seconds}s; sending SIGKILL"
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
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

if [[ -n "${USER_REPLICA_CUDA_GROUPS}" && "${USER_SGLANG}" == "0" ]]; then
  log "ERROR: USER_REPLICA_CUDA_GROUPS requires USER_SGLANG=1"
  exit 1
fi

if [[ -n "${AGENT_REPLICA_CUDA_GROUPS}" || -n "${USER_REPLICA_CUDA_GROUPS}" ]]; then
  python3 -c "import sglang_router; import sglang_router.launch_router; print('sglang-router', sglang_router.__version__)"
fi

services_started_epoch="$(date +%s)"
log "stage=services_start epoch=${services_started_epoch}"

AGENT_WORKER_LABELS=()
AGENT_WORKER_PIDS=()
AGENT_WORKER_PORTS=()
AGENT_WORKER_LOGS=()
AGENT_WORKER_URLS=()
if [[ -n "${AGENT_REPLICA_CUDA_GROUPS}" ]]; then
  IFS=';' read -r -a AGENT_CUDA_GROUP_LIST <<< "${AGENT_REPLICA_CUDA_GROUPS}"
  for i in "${!AGENT_CUDA_GROUP_LIST[@]}"; do
    cuda_group="${AGENT_CUDA_GROUP_LIST[$i]}"
    validate_cuda_group "agent replica ${i}" "${cuda_group}" "${TP}"
    worker_port=$((AGENT_WORKER_PORT_BASE + i))
    worker_log="${OUTPUT_DIR}/sglang_agent_worker${i}_${worker_port}_seed${SEED}_${RUN_STAMP}.log"
    # shellcheck disable=SC2086  # Extra args are intentionally word-split into flags.
    start_sglang "agent-worker${i}" "${MODEL_PATH}" "${worker_port}" "${TP}" "${MEM_FRACTION}" "${cuda_group}" "${worker_log}" "" "${SGLANG_DETERMINISTIC_ARGS[@]}" ${AGENT_SGLANG_EXTRA_ARGS}
    AGENT_WORKER_LABELS+=("agent-worker${i}")
    AGENT_WORKER_PIDS+=("${SGLANG_STARTED_PID}")
    AGENT_WORKER_PORTS+=("${worker_port}")
    AGENT_WORKER_LOGS+=("${worker_log}")
    AGENT_WORKER_URLS+=("http://${HOST}:${worker_port}")
  done
else
  # shellcheck disable=SC2086  # Extra args are intentionally word-split into flags.
  start_sglang agent "${MODEL_PATH}" "${PORT}" "${TP}" "${MEM_FRACTION}" "${AGENT_CUDA_VISIBLE_DEVICES}" "${SGLANG_LOG}" "" "${SGLANG_DETERMINISTIC_ARGS[@]}" ${AGENT_SGLANG_EXTRA_ARGS}
  AGENT_WORKER_LABELS+=("agent")
  AGENT_WORKER_PIDS+=("${SGLANG_STARTED_PID}")
  AGENT_WORKER_PORTS+=("${PORT}")
  AGENT_WORKER_LOGS+=("${SGLANG_LOG}")
fi

USER_WORKER_LABELS=()
USER_WORKER_PIDS=()
USER_WORKER_PORTS=()
USER_WORKER_LOGS=()
USER_WORKER_URLS=()
if [[ "${USER_SGLANG}" != "0" ]]; then
  if [[ -n "${USER_REPLICA_CUDA_GROUPS}" ]]; then
    IFS=';' read -r -a USER_CUDA_GROUP_LIST <<< "${USER_REPLICA_CUDA_GROUPS}"
    for i in "${!USER_CUDA_GROUP_LIST[@]}"; do
      cuda_group="${USER_CUDA_GROUP_LIST[$i]}"
      validate_cuda_group "user replica ${i}" "${cuda_group}" "${USER_TP}"
      worker_port=$((USER_WORKER_PORT_BASE + i))
      worker_log="${OUTPUT_DIR}/sglang_user_worker${i}_${worker_port}_seed${SEED}_${RUN_STAMP}.log"
      # shellcheck disable=SC2086  # Extra args are intentionally word-split into flags.
      start_sglang "user-worker${i}" "${USER_MODEL_PATH}" "${worker_port}" "${USER_TP}" "${USER_MEM_FRACTION}" "${cuda_group}" "${worker_log}" "${USER_MODEL}" "${SGLANG_DETERMINISTIC_ARGS[@]}" ${USER_SGLANG_EXTRA_ARGS}
      USER_WORKER_LABELS+=("user-worker${i}")
      USER_WORKER_PIDS+=("${SGLANG_STARTED_PID}")
      USER_WORKER_PORTS+=("${worker_port}")
      USER_WORKER_LOGS+=("${worker_log}")
      USER_WORKER_URLS+=("http://${HOST}:${worker_port}")
    done
  else
    # shellcheck disable=SC2086  # Extra args are intentionally word-split into flags.
    start_sglang user "${USER_MODEL_PATH}" "${USER_PORT}" "${USER_TP}" "${USER_MEM_FRACTION}" "${USER_CUDA_VISIBLE_DEVICES}" "${USER_SGLANG_LOG}" "${USER_MODEL}" "${SGLANG_DETERMINISTIC_ARGS[@]}" ${USER_SGLANG_EXTRA_ARGS}
    USER_WORKER_LABELS+=("user")
    USER_WORKER_PIDS+=("${SGLANG_STARTED_PID}")
    USER_WORKER_PORTS+=("${USER_PORT}")
    USER_WORKER_LOGS+=("${USER_SGLANG_LOG}")
  fi
fi

for i in "${!AGENT_WORKER_PIDS[@]}"; do
  wait_for_sglang "${AGENT_WORKER_LABELS[$i]}" "${AGENT_WORKER_PIDS[$i]}" "${AGENT_WORKER_PORTS[$i]}" "${AGENT_WORKER_LOGS[$i]}"
done
for i in "${!USER_WORKER_PIDS[@]}"; do
  wait_for_sglang "${USER_WORKER_LABELS[$i]}" "${USER_WORKER_PIDS[$i]}" "${USER_WORKER_PORTS[$i]}" "${USER_WORKER_LOGS[$i]}"
done

if [[ -n "${AGENT_REPLICA_CUDA_GROUPS}" ]]; then
  start_router agent "${PORT}" "${AGENT_ROUTER_PROMETHEUS_PORT}" "${AGENT_ROUTER_LOG}" "${AGENT_ROUTER_POLICY}" "${AGENT_WORKER_URLS[@]}"
  AGENT_ROUTER_PID="${SGLANG_STARTED_PID}"
  wait_for_sglang agent-router "${AGENT_ROUTER_PID}" "${PORT}" "${AGENT_ROUTER_LOG}"
fi
if [[ -n "${USER_REPLICA_CUDA_GROUPS}" ]]; then
  start_router user "${USER_PORT}" "${USER_ROUTER_PROMETHEUS_PORT}" "${USER_ROUTER_LOG}" "${USER_ROUTER_POLICY}" "${USER_WORKER_URLS[@]}"
  USER_ROUTER_PID="${SGLANG_STARTED_PID}"
  wait_for_sglang user-router "${USER_ROUTER_PID}" "${USER_PORT}" "${USER_ROUTER_LOG}"
fi

services_ready_epoch="$(date +%s)"
log "stage=services_ready epoch=${services_ready_epoch} elapsed_seconds=$((services_ready_epoch - services_started_epoch))"
log "topology=agent_groups:${AGENT_REPLICA_CUDA_GROUPS:-${AGENT_CUDA_VISIBLE_DEVICES}} user_groups:${USER_REPLICA_CUDA_GROUPS:-${USER_CUDA_VISIBLE_DEVICES}} agent_router_policy=${AGENT_ROUTER_POLICY} user_router_policy=${USER_ROUTER_POLICY} parallel_domains=${PARALLEL_DOMAINS} domain_concurrency=${DOMAIN_CONCURRENCY:-uniform:${MAX_CONCURRENCY}} global_concurrency=${GLOBAL_CONCURRENCY:-auto} borrow_completed_domain_slots=${BORROW_COMPLETED_DOMAIN_SLOTS}"

if [[ -n "${NAMESPACE_PROBE_OUTPUT}" ]]; then
  log "running fixed Agent/User namespace probes"
  NAMESPACE_PROBE_ARGS=()
  if [[ "${NAMESPACE_PROBE_ENFORCE:-0}" == "0" ]]; then
    # Record an unsafe model and continue to the matched official evaluation;
    # the promotion gate consumes both artifacts and makes the decision.
    NAMESPACE_PROBE_ARGS+=(--no-enforce)
  fi
  PYTHONPATH="${PROJECT_ROOT}:${EXAMPLE_DIR}/shared:${EXAMPLE_DIR}/analysis:${SERVICE_AGENT_ROOT}/tau2-bench/src:${PYTHONPATH:-}" \
    python3 "${EXAMPLE_DIR}/analysis/run_namespace_probes.py" \
      --checkpoint "${MODEL_PATH}" \
      --api-base "http://${HOST}:${PORT}/generate" \
      --output "${NAMESPACE_PROBE_OUTPUT}" \
      --concurrency "${NAMESPACE_PROBE_CONCURRENCY:-16}" \
      "${NAMESPACE_PROBE_ARGS[@]}"
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
if [[ -n "${AGENT_PROTOCOL_PROFILE}" ]]; then
  RUN_ARGS+=(--agent-protocol-profile "${AGENT_PROTOCOL_PROFILE}")
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
if [[ "${PARALLEL_DOMAINS}" == "1" ]]; then
  RUN_ARGS+=(--parallel-domains)
fi
if [[ -n "${DOMAIN_CONCURRENCY}" ]]; then
  RUN_ARGS+=(--domain-concurrency "${DOMAIN_CONCURRENCY}")
fi
if [[ -n "${GLOBAL_CONCURRENCY}" ]]; then
  RUN_ARGS+=(--global-concurrency "${GLOBAL_CONCURRENCY}")
fi
if [[ "${BORROW_COMPLETED_DOMAIN_SLOTS}" == "1" ]]; then
  RUN_ARGS+=(--borrow-completed-domain-slots)
fi

log "running official eval: domains=${DOMAINS} split=${TASK_SPLIT} trials=${NUM_TRIALS} num_tasks=${NUM_TASKS:-all}"
eval_started_epoch="$(date +%s)"
log "stage=eval_process_start epoch=${eval_started_epoch}"
python3 "${OFFICIAL_DIR}/run_eval.py" "${RUN_ARGS[@]}"
eval_finished_epoch="$(date +%s)"
log "stage=eval_process_end epoch=${eval_finished_epoch} elapsed_seconds=$((eval_finished_epoch - eval_started_epoch))"
log "eval complete; summary: ${SUMMARY_OUTPUT}"
