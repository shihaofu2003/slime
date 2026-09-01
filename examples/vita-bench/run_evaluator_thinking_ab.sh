#!/usr/bin/env bash

set -euo pipefail
umask 077

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
VENV_PYTHON="${VITA_REPLAY_PYTHON:-/tmp/serviceagent-vitabench-venv/bin/python}"
MODEL_PATH="${VITA_ROLE_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
MODEL_NAME="Qwen3.6-27B-evaluator"
SOURCE_ARTIFACT_DIR="${VITA_REPLAY_SOURCE_ARTIFACT_DIR:-${PROJECT_ROOT}/output/experiments/vitabench-qwen35-local-role-eval/artifacts-v2}"
ARTIFACT_DIR="${VITA_REPLAY_ARTIFACT_DIR:-${PROJECT_ROOT}/output/experiments/vitabench-evaluator-thinking-ab/artifacts}"
REPLAYER="${PROJECT_ROOT}/examples/vita-bench/replay_evaluator_thinking_ab.py"
RUNNER_PATH="$(realpath "${BASH_SOURCE[0]}")"
HOST="127.0.0.1"
PORT_BASE="${VITA_REPLAY_PORT_BASE:-32000}"
CONTEXT_LENGTH="${VITA_REPLAY_CONTEXT_LENGTH:-32768}"
MEM_FRACTION="${VITA_REPLAY_MEM_FRACTION:-0.90}"
MAX_RUNNING_REQUESTS="${VITA_REPLAY_MAX_RUNNING_REQUESTS:-4}"
PILOT_TASKS_PER_SUITE="${VITA_REPLAY_PILOT_TASKS_PER_SUITE:-2}"
RUN_MODE="${VITA_REPLAY_RUN_MODE:-full}"
if [[ "${RUN_MODE}" == "recovery" ]]; then
  DEFAULT_INSTANCE_COUNT=1
else
  DEFAULT_INSTANCE_COUNT=8
fi
INSTANCE_COUNT="${VITA_REPLAY_INSTANCE_COUNT:-${DEFAULT_INSTANCE_COUNT}}"
MAX_SEMANTIC_ATTEMPTS="${VITA_REPLAY_MAX_SEMANTIC_ATTEMPTS:-3}"
BASE_ARTIFACT_DIR="${VITA_REPLAY_BASE_ARTIFACT_DIR:-}"
RECOVERY_REASON="${VITA_REPLAY_RECOVERY_REASON:-}"
BASELINE_EVALUATOR_MEAN_SECONDS="${VITA_REPLAY_BASELINE_EVALUATOR_MEAN_SECONDS:-125.5}"
BASELINE_TOTAL_CRITICAL_PATH_HOURS="${VITA_REPLAY_BASELINE_TOTAL_CRITICAL_PATH_HOURS:-276}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

log() {
  printf '[vitabench-evaluator-thinking-ab] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

[[ "${RUN_MODE}" == "pilot" || "${RUN_MODE}" == "full" || "${RUN_MODE}" == "recovery" ]] || \
  die "VITA_REPLAY_RUN_MODE must be pilot, full, or recovery"
if [[ "${RUN_MODE}" == "recovery" ]]; then
  [[ -f "${BASE_ARTIFACT_DIR}/manifest.json" ]] || \
    die "recovery base manifest is unavailable: ${BASE_ARTIFACT_DIR}/manifest.json"
  [[ -n "${RECOVERY_REASON}" ]] || die "VITA_REPLAY_RECOVERY_REASON is required for recovery"
fi
[[ -x "${VENV_PYTHON}" ]] || die "VitaBench Python is unavailable: ${VENV_PYTHON}"
[[ -f "${REPLAYER}" ]] || die "replayer is unavailable: ${REPLAYER}"
[[ -f "${SOURCE_ARTIFACT_DIR}/manifest.json" ]] || \
  die "source manifest is unavailable: ${SOURCE_ARTIFACT_DIR}/manifest.json"
[[ -d "${MODEL_PATH}" ]] || die "role model is unavailable: ${MODEL_PATH}"
command -v flock >/dev/null 2>&1 || die "flock is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required"

mkdir -p "${ARTIFACT_DIR}" "${ARTIFACT_DIR}/sessions"
exec {RUN_LOCK_FD}>"${ARTIFACT_DIR}/.run.lock"
flock -n "${RUN_LOCK_FD}" || die "another replay job owns ${ARTIFACT_DIR}"

REPLAYER_SHA256="$(sha256sum "${REPLAYER}" | awk '{print $1}')"
RUNNER_SHA256="$(sha256sum "${RUNNER_PATH}" | awk '{print $1}')"

verify_sources_unchanged() {
  [[ "$(sha256sum "${REPLAYER}" | awk '{print $1}')" == "${REPLAYER_SHA256}" ]] || \
    die "replayer source changed while the experiment was running"
  [[ "$(sha256sum "${RUNNER_PATH}" | awk '{print $1}')" == "${RUNNER_SHA256}" ]] || \
    die "runner source changed while the experiment was running"
}

log "hashing the Qwen3.6-27B identity files"
MODEL_FILES_SHA256="$(${VENV_PYTHON} - "${MODEL_PATH}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
identity_suffixes = {
    ".json",
    ".jinja",
    ".model",
    ".safetensors",
    ".tiktoken",
    ".txt",
}
files = sorted(
    path
    for path in root.rglob("*")
    if path.is_file()
    and path.suffix in identity_suffixes
    and "__pycache__" not in path.parts
)
if not files or not any(path.suffix == ".safetensors" for path in files):
    raise SystemExit("model identity files or safetensor weights are missing")
digest = hashlib.sha256()
for path in files:
    digest.update(str(path.relative_to(root)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(path.stat().st_size).encode("ascii"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    digest.update(b"\0")
print(digest.hexdigest(), end="")
PY
)"

SOURCE_MODEL_FILES_SHA256="$(${VENV_PYTHON} - "${SOURCE_ARTIFACT_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = (manifest.get("local_role_serving") or {}).get("model_files_sha256")
if not isinstance(value, str) or not value:
    raise SystemExit("source manifest has no local evaluator model fingerprint")
print(value, end="")
PY
)"
[[ "${MODEL_FILES_SHA256}" == "${SOURCE_MODEL_FILES_SHA256}" ]] || \
  die "current Qwen3.6-27B files differ from the thinking baseline"

audit_source() {
  verify_sources_unchanged
  "${VENV_PYTHON}" "${REPLAYER}" audit \
    --source-artifact-dir "${SOURCE_ARTIFACT_DIR}" \
    --output-dir "${ARTIFACT_DIR}" \
    --verify-journals \
    --model-files-sha256 "${MODEL_FILES_SHA256}" \
    --replayer-sha256 "${REPLAYER_SHA256}" \
    --runner-sha256 "${RUNNER_SHA256}" \
    --context-length "${CONTEXT_LENGTH}" \
    --max-semantic-attempts "${MAX_SEMANTIC_ATTEMPTS}"
}

log "auditing all source shards, final windows, and evaluator response journals"
audit_source

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_SELECTORS <<<"${CUDA_VISIBLE_DEVICES}"
else
  mapfile -t GPU_SELECTORS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
[[ "${#GPU_SELECTORS[@]}" -eq "${INSTANCE_COUNT}" ]] || \
  die "the replay requires exactly ${INSTANCE_COUNT} visible GPUs; got ${#GPU_SELECTORS[@]}"
[[ "$(printf '%s\n' "${GPU_SELECTORS[@]}" | sort -u | wc -l)" -eq "${INSTANCE_COUNT}" ]] || \
  die "visible GPU selectors are not unique"

declare -a SERVER_PIDS=()
declare -a SERVER_PORTS=()
declare -a SERVER_LOGS=()

cleanup() {
  local status=$?
  trap - EXIT TERM INT HUP
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit "${status}"
}
trap cleanup EXIT TERM INT HUP

for ((index = 0; index < INSTANCE_COUNT; index++)); do
  port=$((PORT_BASE + index))
  server_log="${ARTIFACT_DIR}/sglang_evaluator-${index}_${RUN_STAMP}.log"
  log "starting evaluator=${index} gpu=${GPU_SELECTORS[index]} port=${port}"
  (
    trap - EXIT TERM INT HUP
    exec {RUN_LOCK_FD}>&-
    exec setsid env \
      -u WANDB_API_KEY \
      -u OPENAI_API_KEY \
      -u OPENAI_API_BASE \
      -u OPENROUTER_API_KEY \
      -u VITA_API_KEY \
      -u VITA_API_BASE_URL \
      CUDA_VISIBLE_DEVICES="${GPU_SELECTORS[index]}" \
      "${VENV_PYTHON}" -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --host "${HOST}" \
        --port "${port}" \
        --tp 1 \
        --dtype bfloat16 \
        --context-length "${CONTEXT_LENGTH}" \
        --mem-fraction-static "${MEM_FRACTION}" \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --reasoning-parser qwen3
  ) >"${server_log}" 2>&1 &
  SERVER_PIDS+=("$!")
  SERVER_PORTS+=("${port}")
  SERVER_LOGS+=("${server_log}")
done

for index in "${!SERVER_PIDS[@]}"; do
  pid="${SERVER_PIDS[index]}"
  port="${SERVER_PORTS[index]}"
  server_log="${SERVER_LOGS[index]}"
  ready=0
  for _ in $(seq 1 900); do
    kill -0 "${pid}" 2>/dev/null || {
      tail -n 100 "${server_log}" >&2 || true
      die "evaluator ${index} exited before becoming healthy"
    }
    if "${VENV_PYTHON}" - "${HOST}" "${port}" <<'PY' >/dev/null 2>&1
import sys
import requests

response = requests.get(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=2)
response.raise_for_status()
PY
    then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || {
    tail -n 100 "${server_log}" >&2 || true
    die "evaluator ${index} did not become healthy within 30 minutes"
  }
done
log "all ${INSTANCE_COUNT} non-thinking replay endpoints are healthy"

ENDPOINTS=()
for port in "${SERVER_PORTS[@]}"; do
  ENDPOINTS+=("http://${HOST}:${port}")
done

run_replay() {
  local mode="$1"
  shift
  verify_sources_unchanged
  "${VENV_PYTHON}" "${REPLAYER}" replay \
    --source-artifact-dir "${SOURCE_ARTIFACT_DIR}" \
    --output-dir "${ARTIFACT_DIR}" \
    --mode "${mode}" \
    --slots-per-endpoint "${MAX_RUNNING_REQUESTS}" \
    --max-semantic-attempts "${MAX_SEMANTIC_ATTEMPTS}" \
    --endpoints "${ENDPOINTS[@]}" \
    "$@"
}

analyze() {
  verify_sources_unchanged
  "${VENV_PYTHON}" "${REPLAYER}" analyze \
    --source-artifact-dir "${SOURCE_ARTIFACT_DIR}" \
    --output-dir "${ARTIFACT_DIR}" \
    --baseline-evaluator-mean-seconds "${BASELINE_EVALUATOR_MEAN_SECONDS}" \
    --baseline-total-critical-path-hours "${BASELINE_TOTAL_CRITICAL_PATH_HOURS}" \
    "$@"
}

if [[ "${RUN_MODE}" == "recovery" ]]; then
  log "recovering missing chained windows from base=${BASE_ARTIFACT_DIR}"
  run_replay chained --base-output-dir "${BASE_ARTIFACT_DIR}"
  analyze \
    --base-output-dir "${BASE_ARTIFACT_DIR}" \
    --recovery-reason "${RECOVERY_REASON}" \
    --report "${ARTIFACT_DIR}/report.json"
  log "re-auditing immutable source inputs after recovery"
  audit_source
  log "RECOVERY_REPLAY_OK report=${ARTIFACT_DIR}/report.json"
  exit 0
fi

log "running the four-suite frozen-prompt pilot"
run_replay frozen --pilot-tasks-per-suite "${PILOT_TASKS_PER_SUITE}"
log "running the four-suite chained-state pilot"
run_replay chained --pilot-tasks-per-suite "${PILOT_TASKS_PER_SUITE}"
analyze \
  --pilot-tasks-per-suite "${PILOT_TASKS_PER_SUITE}" \
  --bootstrap-samples 1000 \
  --fail-on-operational-gate \
  --report "${ARTIFACT_DIR}/pilot_report.json"
log "pilot operational gate passed"

if [[ "${RUN_MODE}" == "pilot" ]]; then
  log "PILOT_ONLY_OK report=${ARTIFACT_DIR}/pilot_report.json"
  exit 0
fi

log "running/resuming all 6,349 frozen evaluator windows"
run_replay frozen
analyze \
  --modes frozen \
  --report "${ARTIFACT_DIR}/frozen_report.json"

log "running/resuming all 918 trajectories with non-thinking rubric state propagation"
run_replay chained
analyze --report "${ARTIFACT_DIR}/report.json"

log "re-auditing immutable source inputs after the full replay"
audit_source
log "FULL_REPLAY_OK report=${ARTIFACT_DIR}/report.json"
