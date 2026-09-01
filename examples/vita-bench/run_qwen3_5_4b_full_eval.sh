#!/usr/bin/env bash

set -euo pipefail
umask 077

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
VITABENCH_DIR="${SERVICE_AGENT_ROOT}/vitabench"
VENV_DIR="/tmp/serviceagent-vitabench-venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
VITA_BIN="${VENV_DIR}/bin/vita"
MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
AGENT_ENABLE_THINKING="${VITA_AGENT_ENABLE_THINKING:-true}"
AGENT_TOOL_CALL_PARSER="${VITA_AGENT_TOOL_CALL_PARSER:-qwen3_coder}"
ROLE_BACKEND="${VITA_ROLE_BACKEND:-remote}"
ROLE_MODEL_PATH="${VITA_ROLE_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
if [[ "${ROLE_BACKEND}" == "local" ]]; then
  USER_MODEL="Qwen3.6-27B-user"
  EVALUATOR_MODEL="Qwen3.6-27B-evaluator"
else
  USER_MODEL="gpt-4.1-ca"
  EVALUATOR_MODEL="gemini-2.5-flash"
fi
HOST="127.0.0.1"
PORT_BASE=30000
ROLE_PORT_BASE=31000
MODEL_CONFIG="${PROJECT_ROOT}/examples/vita-bench/models_qwen35_full.yaml"
SUMMARIZER="${PROJECT_ROOT}/examples/vita-bench/summarize_full_eval.py"
SFT_EXPORTER="${PROJECT_ROOT}/examples/vita-bench/export_sft_archive.py"
EVAL_SCHEDULER="${PROJECT_ROOT}/examples/vita-bench/eval_scheduler.py"
RUNNER_PATH="$(realpath "${BASH_SOURCE[0]}")"
RUNNER_PROTOCOL_VERSION="8"
ENV_FILE="${TAU2_ENV_FILE:-${SERVICE_AGENT_ROOT}/tau2-bench/.env}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_PARALLELISM="${VITA_SUITE_PARALLELISM:-4}"
RUN_MODE="${VITA_RUN_MODE:-full}"
AB_SHARD_INDEX="${VITA_AB_SHARD_INDEX:-0}"
ARTIFACT_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-parallelism)
      [[ $# -ge 2 ]] || { echo "--suite-parallelism requires a value" >&2; exit 2; }
      SUITE_PARALLELISM="$2"
      shift 2
      ;;
    --run-mode)
      [[ $# -ge 2 ]] || { echo "--run-mode requires a value" >&2; exit 2; }
      RUN_MODE="$2"
      shift 2
      ;;
    --ab-shard-index)
      [[ $# -ge 2 ]] || { echo "--ab-shard-index requires a value" >&2; exit 2; }
      AB_SHARD_INDEX="$2"
      shift 2
      ;;
    --artifact-dir)
      [[ $# -ge 2 ]] || { echo "--artifact-dir requires a value" >&2; exit 2; }
      ARTIFACT_DIR_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--suite-parallelism 1|4] [--run-mode full|gate|ab|legacy-v2-resume] [--ab-shard-index 0-9] [--artifact-dir PATH]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
[[ "${SUITE_PARALLELISM}" == "1" || "${SUITE_PARALLELISM}" == "4" ]] || {
  echo "--suite-parallelism must be 1 or 4" >&2
  exit 2
}
[[ "${RUN_MODE}" == "full" || "${RUN_MODE}" == "gate" || "${RUN_MODE}" == "ab" || \
   "${RUN_MODE}" == "legacy-v2-resume" ]] || {
  echo "--run-mode must be full, gate, ab, or legacy-v2-resume" >&2
  exit 2
}
[[ "${AB_SHARD_INDEX}" =~ ^[0-9]+$ ]] && ((AB_SHARD_INDEX <= 9)) || {
  echo "--ab-shard-index must be between 0 and 9" >&2
  exit 2
}

if [[ -n "${ARTIFACT_DIR_OVERRIDE}" ]]; then
  ARTIFACT_DIR="${ARTIFACT_DIR_OVERRIDE}"
elif [[ -n "${VITA_RESULT_DIR:-}" ]]; then
  ARTIFACT_DIR="${VITA_RESULT_DIR}"
elif [[ "${ROLE_BACKEND}" == "local" ]]; then
  case "${RUN_MODE}" in
    full)
      ARTIFACT_DIR="${PROJECT_ROOT}/output/experiments/vitabench-qwen35-local-role-eval/artifacts-v4"
      ;;
    gate)
      ARTIFACT_DIR="${PROJECT_ROOT}/output/experiments/vitabench-qwen35-local-role-eval/artifacts-v4-gate"
      ;;
    ab)
      ARTIFACT_DIR="${PROJECT_ROOT}/output/experiments/vitabench-qwen35-local-role-eval/artifacts-v4-ab-shard${AB_SHARD_INDEX}"
      ;;
    legacy-v2-resume)
      ARTIFACT_DIR="${PROJECT_ROOT}/output/experiments/vitabench-qwen35-local-role-eval/artifacts-v2"
      ;;
  esac
elif [[ -n "${OUTPUT_ROOT:-}" && "${OUTPUT_ROOT}" == /* ]]; then
  ARTIFACT_DIR="${OUTPUT_ROOT}/artifacts"
elif [[ -n "${OUTPUT_ROOT:-}" ]]; then
  ARTIFACT_DIR="${PROJECT_ROOT}/${OUTPUT_ROOT}/artifacts"
else
  ARTIFACT_DIR="${PROJECT_ROOT}/output/experiments/vitabench-qwen35-full-eval/artifacts"
fi

MANIFEST="${ARTIFACT_DIR}/manifest.json"
RUN_PLAN="${ARTIFACT_DIR}/run_plan.tsv"
SUMMARY_JSON="${ARTIFACT_DIR}/summary.json"
SUMMARY_CSV="${ARTIFACT_DIR}/summary.csv"
SFT_DIR="${ARTIFACT_DIR}/sft"
API_JOURNAL_DIR="${ARTIFACT_DIR}/api_journal"
REMOTE_SLOT_DIR="${ARTIFACT_DIR}/remote_concurrency_slots"
LANGUAGE="chinese"
NUM_TRIALS=4
SEED=300
MAX_STEPS=300
MAX_ERRORS=10
EVALUATION_TYPE="trajectory"
if [[ "${SUITE_PARALLELISM}" == "4" ]]; then
  MAX_CONCURRENCY=4
else
  MAX_CONCURRENCY=8
fi
REMOTE_CONCURRENCY_LIMIT="${VITA_REMOTE_CONCURRENCY_LIMIT:-8}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
ROLE_MEM_FRACTION="${VITA_ROLE_MEM_FRACTION:-0.90}"
ROLE_CONTEXT_LENGTH="${VITA_ROLE_CONTEXT_LENGTH:-32768}"
WORKER_COUNT="${VITA_WORKER_COUNT:-4}"
AGENT_INSTANCE_COUNT="${VITA_AGENT_INSTANCE_COUNT:-2}"
USER_INSTANCE_COUNT="${VITA_USER_INSTANCE_COUNT:-2}"
EVALUATOR_INSTANCE_COUNT="${VITA_EVALUATOR_INSTANCE_COUNT:-4}"
USER_MAX_RUNNING_REQUESTS="${VITA_USER_MAX_RUNNING_REQUESTS:-8}"
EVALUATOR_MAX_RUNNING_REQUESTS="${VITA_EVALUATOR_MAX_RUNNING_REQUESTS:-4}"
BATCH_MAX_ATTEMPTS="${BATCH_MAX_ATTEMPTS:-8}"
if [[ "${RUN_MODE}" == "legacy-v2-resume" ]]; then
  WORKER_COUNT=4
  AGENT_INSTANCE_COUNT=4
  USER_INSTANCE_COUNT=2
  EVALUATOR_INSTANCE_COUNT=2
  USER_MAX_RUNNING_REQUESTS=4
  EVALUATOR_MAX_RUNNING_REQUESTS=4
fi

log() { echo "[vitabench-full] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

[[ -d "${MODEL_PATH}" ]] || die "model path does not exist: ${MODEL_PATH}"
[[ -n "${MODEL_NAME}" ]] || die "MODEL_NAME must not be empty"
[[ -n "${AGENT_TOOL_CALL_PARSER}" ]] || die "VITA_AGENT_TOOL_CALL_PARSER must not be empty"
[[ "${AGENT_ENABLE_THINKING}" == "true" || "${AGENT_ENABLE_THINKING}" == "false" ]] || \
  die "VITA_AGENT_ENABLE_THINKING must be true or false"
[[ "${ROLE_BACKEND}" == "remote" || "${ROLE_BACKEND}" == "local" ]] || \
  die "VITA_ROLE_BACKEND must be remote or local"
if [[ "${RUN_MODE}" == "ab" && "${ROLE_BACKEND}" != "local" ]]; then
  die "A/B shard mode requires the local 2/2/4 topology"
fi
if [[ "${RUN_MODE}" == "legacy-v2-resume" && "${ROLE_BACKEND}" != "local" ]]; then
  die "legacy-v2 resume requires the local 4/2/2 topology"
fi
if [[ "${RUN_MODE}" == "legacy-v2-resume" && \
      ("${MODEL_NAME}" != "Qwen3.5-4B" || "${AGENT_ENABLE_THINKING}" != "true") ]]; then
  die "legacy-v2 resume is locked to the Thinking-on Qwen3.5-4B baseline"
fi
if [[ "${ROLE_BACKEND}" == "local" ]]; then
  [[ -d "${ROLE_MODEL_PATH}" ]] || die "role model path does not exist: ${ROLE_MODEL_PATH}"
fi
[[ -d "${VITABENCH_DIR}" ]] || die "VitaBench checkout does not exist: ${VITABENCH_DIR}"
[[ -f "${MODEL_CONFIG}" ]] || die "model config does not exist: ${MODEL_CONFIG}"
[[ -f "${SUMMARIZER}" ]] || die "summary validator does not exist: ${SUMMARIZER}"
[[ -f "${SFT_EXPORTER}" ]] || die "SFT exporter does not exist: ${SFT_EXPORTER}"
[[ -f "${EVAL_SCHEDULER}" ]] || die "evaluation scheduler does not exist: ${EVAL_SCHEDULER}"
if [[ "${ROLE_BACKEND}" == "remote" ]]; then
  [[ -f "${ENV_FILE}" ]] || die "credential env file does not exist: ${ENV_FILE}"
fi
[[ "${REMOTE_CONCURRENCY_LIMIT}" =~ ^[0-9]+$ ]] || die "remote concurrency limit must be an integer"
((REMOTE_CONCURRENCY_LIMIT >= 1 && REMOTE_CONCURRENCY_LIMIT <= 32)) || \
  die "remote concurrency limit must be between 1 and 32"
for count_name in \
  WORKER_COUNT AGENT_INSTANCE_COUNT USER_INSTANCE_COUNT EVALUATOR_INSTANCE_COUNT \
  USER_MAX_RUNNING_REQUESTS EVALUATOR_MAX_RUNNING_REQUESTS; do
  count_value="${!count_name}"
  [[ "${count_value}" =~ ^[0-9]+$ ]] || die "${count_name} must be an integer"
  ((count_value >= 1)) || die "${count_name} must be positive"
done
if [[ "${ROLE_BACKEND}" == "local" ]]; then
  [[ "${SGLANG_MAX_RUNNING_REQUESTS}" == "16" ]] || \
    die "local agent max-running-requests must be 16"
  if [[ "${RUN_MODE}" == "legacy-v2-resume" ]]; then
    [[ "${USER_MAX_RUNNING_REQUESTS}" == "4" ]] || \
      die "legacy-v2 user max-running-requests must be 4"
  else
    [[ "${USER_MAX_RUNNING_REQUESTS}" == "8" ]] || \
      die "local user max-running-requests must be 8"
  fi
  [[ "${EVALUATOR_MAX_RUNNING_REQUESTS}" == "4" ]] || \
    die "local evaluator max-running-requests must be 4"
  [[ "${ROLE_CONTEXT_LENGTH}" == "32768" ]] || \
    die "local user/evaluator context length must remain 32768"
  AGGREGATE_MAX_CONCURRENCY="$((WORKER_COUNT * MAX_CONCURRENCY))"
else
  AGGREGATE_MAX_CONCURRENCY="$((SUITE_PARALLELISM * MAX_CONCURRENCY))"
fi
((REMOTE_CONCURRENCY_LIMIT <= AGGREGATE_MAX_CONCURRENCY)) || \
  die "remote concurrency limit cannot exceed aggregate simulation concurrency ${AGGREGATE_MAX_CONCURRENCY}"

if [[ "${ROLE_BACKEND}" == "remote" ]]; then
  ENV_FILE="${ENV_FILE}" python3 - <<'PY'
import os
import stat
from pathlib import Path

path = Path(os.environ["ENV_FILE"])
if stat.S_IMODE(path.stat().st_mode) & 0o077:
    raise SystemExit(f"credential env file must not be accessible by group/other: {path}")
PY
fi

if [[ ! -x "${VITA_BIN}" ]]; then
  bash "${SERVICE_AGENT_ROOT}/setup/setup_vitabench_env.sh" "${VITABENCH_DIR}"
fi
[[ -x "${VENV_PYTHON}" ]] || die "VitaBench Python is unavailable: ${VENV_PYTHON}"
[[ -x "${VITA_BIN}" ]] || die "VitaBench CLI is unavailable: ${VITA_BIN}"
MODEL_CONFIG="${MODEL_CONFIG}" MODEL_NAME="${MODEL_NAME}" \
  AGENT_ENABLE_THINKING="${AGENT_ENABLE_THINKING}" "${VENV_PYTHON}" - <<'PY'
import os
from pathlib import Path

import yaml

config = yaml.safe_load(Path(os.environ["MODEL_CONFIG"]).read_text(encoding="utf-8"))
matches = [model for model in config.get("models", []) if model.get("name") == os.environ["MODEL_NAME"]]
if len(matches) != 1:
    raise SystemExit(f"model config must contain exactly one {os.environ['MODEL_NAME']!r} entry")
model = matches[0]
if model.get("response_model_family") != os.environ["MODEL_NAME"]:
    raise SystemExit("agent response_model_family must match MODEL_NAME")
thinking = (model.get("chat_template_kwargs") or {}).get("enable_thinking")
expected_thinking = os.environ["AGENT_ENABLE_THINKING"] == "true"
if thinking is not expected_thinking:
    raise SystemExit("agent model config thinking mode does not match VITA_AGENT_ENABLE_THINKING")
PY
mkdir -p "${ARTIFACT_DIR}/gate" "${ARTIFACT_DIR}/shards" "${API_JOURNAL_DIR}"
command -v flock >/dev/null 2>&1 || die "flock is required to protect the shared artifact directory"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required to verify locked inputs"
RUN_LOCK_FILE="${ARTIFACT_DIR}/.run.lock"
exec {RUN_LOCK_FD}>"${RUN_LOCK_FILE}"
flock -n "${RUN_LOCK_FD}" || die "another full evaluation already owns ${ARTIFACT_DIR}"
if [[ "${RUN_MODE}" == "legacy-v2-resume" && ! -f "${MANIFEST}" ]]; then
  die "legacy-v2 resume requires the existing schema-5 manifest: ${MANIFEST}"
fi

if [[ ! -f "${MANIFEST}" ]]; then
  MANIFEST="${MANIFEST}" ARTIFACT_DIR="${ARTIFACT_DIR}" RUN_PLAN="${RUN_PLAN}" \
    SUMMARY_JSON="${SUMMARY_JSON}" SUMMARY_CSV="${SUMMARY_CSV}" "${VENV_PYTHON}" - <<'PY'
import os
from pathlib import Path

artifact_dir = Path(os.environ["ARTIFACT_DIR"])
semantic_artifacts = [
    Path(os.environ["RUN_PLAN"]),
    Path(os.environ["SUMMARY_JSON"]),
    Path(os.environ["SUMMARY_CSV"]),
]
for directory_name in ("gate", "shards", "api_journal", "sft"):
    directory = artifact_dir / directory_name
    if directory.is_dir():
        semantic_artifacts.extend(path for path in directory.rglob("*") if path.is_file())
existing = sorted(path for path in semantic_artifacts if path.is_file())
if existing:
    sample = ", ".join(str(path) for path in existing[:5])
    raise SystemExit(
        "manifest is missing but prior semantic artifacts exist; refusing to assign new "
        f"provenance ({sample})"
    )
PY
fi

compute_protocol_state() {
  VITABENCH_DIR="${VITABENCH_DIR}" MODEL_CONFIG="${MODEL_CONFIG}" \
    SUMMARIZER="${SUMMARIZER}" SFT_EXPORTER="${SFT_EXPORTER}" RUNNER_PATH="${RUNNER_PATH}" \
    EVAL_SCHEDULER="${EVAL_SCHEDULER}" \
    RUNNER_PROTOCOL_VERSION="${RUNNER_PROTOCOL_VERSION}" "${VENV_PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def digest_files(root, files):
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


vita_root = Path(os.environ["VITABENCH_DIR"]).resolve()
source_root = vita_root / "src" / "vita"
source_files = [
    path
    for path in source_root.rglob("*")
    if path.is_file()
    and "__pycache__" not in path.parts
    and path.suffix not in {".pyc", ".pyo"}
]
task_files = sorted((vita_root / "data" / "vita" / "domains").glob("*/tasks.json"))
pyproject = vita_root / "pyproject.toml"
if not source_files or len(task_files) != 4 or not pyproject.is_file():
    raise SystemExit("VitaBench protocol inputs are incomplete")

state = {
    "runner_protocol_version": int(os.environ["RUNNER_PROTOCOL_VERSION"]),
    "runner_sha256": digest_file(Path(os.environ["RUNNER_PATH"]).resolve()),
    "model_config_sha256": digest_file(Path(os.environ["MODEL_CONFIG"]).resolve()),
    "summarizer_sha256": digest_file(Path(os.environ["SUMMARIZER"]).resolve()),
    "sft_exporter_sha256": digest_file(Path(os.environ["SFT_EXPORTER"]).resolve()),
    "eval_scheduler_sha256": digest_file(Path(os.environ["EVAL_SCHEDULER"]).resolve()),
    "vitabench_source_sha256": digest_files(vita_root, source_files),
    "vitabench_task_data_sha256": digest_files(vita_root, task_files),
    "vitabench_pyproject_sha256": digest_file(pyproject),
}
bundle = hashlib.sha256(
    json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
state["protocol_bundle_sha256"] = bundle
print(json.dumps(state, sort_keys=True, separators=(",", ":")), end="")
PY
}

compute_model_fingerprint() {
  local fingerprint_path="$1"
  MODEL_PATH="${fingerprint_path}" "${VENV_PYTHON}" - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["MODEL_PATH"]).resolve()
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
}

PROTOCOL_STATE_JSON="$(compute_protocol_state)"
log "hashing local model identity for reproducible resume"
MODEL_FILES_SHA256="$(compute_model_fingerprint "${MODEL_PATH}")"
if [[ "${ROLE_BACKEND}" == "local" ]]; then
  log "hashing local user/evaluator model identity for reproducible resume"
  ROLE_MODEL_FILES_SHA256="$(compute_model_fingerprint "${ROLE_MODEL_PATH}")"
else
  ROLE_MODEL_FILES_SHA256=""
fi

verify_protocol_state() {
  local current_state
  current_state="$(compute_protocol_state)"
  [[ "${current_state}" == "${PROTOCOL_STATE_JSON}" ]] || \
    die "protocol inputs changed while the evaluation was running"
}

read_dotenv() {
  local key="$1"
  ENV_FILE="${ENV_FILE}" ENV_KEY="${key}" "${VENV_PYTHON}" - <<'PY'
import os
from dotenv import dotenv_values

value = dotenv_values(os.environ["ENV_FILE"]).get(os.environ["ENV_KEY"])
if value:
    print(value, end="")
PY
}

probe_chat_model() {
  local requested_model="$1"
  local require_exact_ok="${2:-true}"
  local probe_max_tokens="${3:-64}"
  REQUESTED_MODEL="${requested_model}" REQUIRE_EXACT_OK="${require_exact_ok}" \
    PROBE_MAX_TOKENS="${probe_max_tokens}" \
    "${VENV_PYTHON}" - <<'PY'
import contextlib
import io
import os
import sys

with contextlib.redirect_stdout(io.StringIO()):
    from vita.data_model.message import UserMessage
    from vita.utils import llm_utils

captured = {}
post_json = llm_utils._post_json_with_retries

def capture_response(*args, **kwargs):
    payload = post_json(*args, **kwargs)
    captured["model"] = payload.get("model")
    return payload

llm_utils._post_json_with_retries = capture_response
try:
    reply = llm_utils.generate(
        model=os.environ["REQUESTED_MODEL"],
        messages=[UserMessage(role="user", content="Reply with exactly OK.")],
        max_tokens=int(os.environ["PROBE_MAX_TOKENS"]),
        temperature=0,
    )
except Exception as exc:
    print(f"remote probe failed ({type(exc).__name__})", file=sys.stderr)
    raise SystemExit(1)

content = (reply.content or "").strip()
if not content:
    print("remote probe returned empty content", file=sys.stderr)
    raise SystemExit(1)
if os.environ["REQUIRE_EXACT_OK"] == "true" and content != "OK":
    print("remote probe returned unexpected content", file=sys.stderr)
    raise SystemExit(1)
actual_model = captured.get("model")
if not isinstance(actual_model, str) or not actual_model.strip():
    print("remote probe omitted the response model id", file=sys.stderr)
    raise SystemExit(1)
print(actual_model.strip(), end="")
PY
}

export VITA_MODEL_CONFIG_PATH="${MODEL_CONFIG}"
export VITA_AGENT_API_BASE_URL="http://${HOST}:${PORT_BASE}/v1/chat/completions"
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${PROJECT_ROOT}"
export GIT_CONFIG_KEY_1=safe.directory
export GIT_CONFIG_VALUE_1="${VITABENCH_DIR}"

log "running Vita credential/redaction regression tests"
"${VENV_PYTHON}" -m pytest -q \
  "${VITABENCH_DIR}/tests/test_llm_security.py" \
  "${VITABENCH_DIR}/tests/test_evaluator_output_validation.py"

SUITE_NAMES=(delivery instore ota cross_domain)
declare -A SUITE_GPU=()
declare -A SUITE_PORT=(
  [delivery]="$((PORT_BASE + 0))"
  [instore]="$((PORT_BASE + 1))"
  [ota]="$((PORT_BASE + 2))"
  [cross_domain]="$((PORT_BASE + 3))"
)
if [[ "${RUN_MODE}" == "legacy-v2-resume" ]]; then
  WORKER_AGENT_PORT=(
    "$((PORT_BASE + 0))" "$((PORT_BASE + 1))"
    "$((PORT_BASE + 2))" "$((PORT_BASE + 3))"
  )
  WORKER_USER_PORT=(
    "$((ROLE_PORT_BASE + 0))" "$((ROLE_PORT_BASE + 1))"
    "$((ROLE_PORT_BASE + 0))" "$((ROLE_PORT_BASE + 1))"
  )
  WORKER_EVALUATOR_PORT=(
    "$((ROLE_PORT_BASE + 2))" "$((ROLE_PORT_BASE + 3))"
    "$((ROLE_PORT_BASE + 2))" "$((ROLE_PORT_BASE + 3))"
  )
else
  WORKER_AGENT_PORT=(
    "$((PORT_BASE + 0))" "$((PORT_BASE + 1))"
    "$((PORT_BASE + 0))" "$((PORT_BASE + 1))"
  )
  WORKER_USER_PORT=(
    "$((ROLE_PORT_BASE + 0))" "$((ROLE_PORT_BASE + 1))"
    "$((ROLE_PORT_BASE + 0))" "$((ROLE_PORT_BASE + 1))"
  )
  WORKER_EVALUATOR_PORT=(
    "$((ROLE_PORT_BASE + 2))" "$((ROLE_PORT_BASE + 3))"
    "$((ROLE_PORT_BASE + 4))" "$((ROLE_PORT_BASE + 5))"
  )
fi
DYNAMIC_QUEUE_STATE="${ARTIFACT_DIR}/.dynamic_shard_queue.json"
DYNAMIC_QUEUE_LOCK="${ARTIFACT_DIR}/.dynamic_shard_queue.lock"
SGLANG_PIDS=()
SGLANG_LABELS=()
SGLANG_PORTS=()
SGLANG_LOGS=()
ACTIVE_WORKER_PIDS=()
ACTIVE_BATCH_PID=""

close_run_lock_fd_for_child() {
  if [[ -e "/proc/${BASHPID}/fd/${RUN_LOCK_FD}" ]]; then
    exec {RUN_LOCK_FD}>&-
  fi
}

terminate_child() {
  local pid="$1"
  local label="$2"
  local send_term="${3:-1}"
  local watchdog_pid
  if [[ "${send_term}" == "1" ]]; then
    kill -TERM "${pid}" 2>/dev/null || true
  fi
  (
    trap - EXIT TERM INT HUP
    close_run_lock_fd_for_child
    sleep 20
    kill -KILL "${pid}" 2>/dev/null || true
  ) &
  watchdog_pid="$!"
  wait "${pid}" 2>/dev/null || true
  kill "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true
  log "stopped ${label} pid=${pid}"
}

terminate_process_group() {
  local leader_pid="$1"
  local label="$2"
  local watchdog_pid
  kill -TERM -- "-${leader_pid}" 2>/dev/null || true
  (
    trap - EXIT TERM INT HUP
    close_run_lock_fd_for_child
    sleep 20
    kill -KILL -- "-${leader_pid}" 2>/dev/null || true
  ) &
  watchdog_pid="$!"
  wait "${leader_pid}" 2>/dev/null || true
  kill "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true
  kill -KILL -- "-${leader_pid}" 2>/dev/null || true
  log "stopped ${label} process_group=${leader_pid}"
}

cleanup() {
  local pid
  local index
  trap - EXIT TERM INT HUP
  if [[ -n "${ACTIVE_BATCH_PID}" ]]; then
    terminate_process_group "${ACTIVE_BATCH_PID}" "active Vita batch"
    ACTIVE_BATCH_PID=""
  fi
  for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
    terminate_child "${pid}" "suite worker" 0
  done
  for ((index = ${#SGLANG_PIDS[@]} - 1; index >= 0; index--)); do
    pid="${SGLANG_PIDS[index]}"
    log "stopping sglang label=${SGLANG_LABELS[index]} pid=${pid}"
    terminate_process_group "${pid}" "sglang ${SGLANG_LABELS[index]}"
  done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required for process-tree cleanup"
GPU_SELECTORS=()
VISIBLE_DEVICE_LIST="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "${VISIBLE_DEVICE_LIST}" && "${VISIBLE_DEVICE_LIST,,}" != "all" && \
      "${VISIBLE_DEVICE_LIST,,}" != "nodevfiles" && "${VISIBLE_DEVICE_LIST,,}" != "void" ]]; then
  IFS=',' read -r -a GPU_SELECTORS <<<"${VISIBLE_DEVICE_LIST}"
else
  mapfile -t GPU_SELECTORS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits
  )
fi
for index in "${!GPU_SELECTORS[@]}"; do
  GPU_SELECTORS[index]="${GPU_SELECTORS[index]//[[:space:]]/}"
  [[ -n "${GPU_SELECTORS[index]}" ]] || die "GPU selector ${index} is empty"
done
if [[ "${ROLE_BACKEND}" == "local" ]]; then
  [[ "${SUITE_PARALLELISM}" == "4" ]] || \
    die "local user/evaluator deployment requires suite parallelism 4"
  REQUIRED_GPU_COUNT=8
  [[ "${#GPU_SELECTORS[@]}" == "${REQUIRED_GPU_COUNT}" ]] || \
    die "local evaluation requires exactly ${REQUIRED_GPU_COUNT} visible GPU selectors"
  if [[ "${RUN_MODE}" == "legacy-v2-resume" ]]; then
    LOCAL_TOPOLOGY_JSON=""
  else
    topology_args=()
    for selector in "${GPU_SELECTORS[@]}"; do
      topology_args+=(--gpu-selector "${selector}")
    done
    if ! LOCAL_TOPOLOGY_JSON="$(
      "${VENV_PYTHON}" "${EVAL_SCHEDULER}" validate-topology \
        "${topology_args[@]}" \
        --worker-count "${WORKER_COUNT}" \
        --agent-instance-count "${AGENT_INSTANCE_COUNT}" \
        --user-instance-count "${USER_INSTANCE_COUNT}" \
        --evaluator-instance-count "${EVALUATOR_INSTANCE_COUNT}" \
        --agent-port-base "${PORT_BASE}" \
        --role-port-base "${ROLE_PORT_BASE}"
    )"; then
      die "invalid local 8-GPU topology"
    fi
  fi
else
  REQUIRED_GPU_COUNT="${SUITE_PARALLELISM}"
  LOCAL_TOPOLOGY_JSON=""
fi
(( ${#GPU_SELECTORS[@]} >= REQUIRED_GPU_COUNT )) || \
  die "this evaluation requires ${REQUIRED_GPU_COUNT} visible GPUs"
[[ "$(printf '%s\n' "${GPU_SELECTORS[@]:0:REQUIRED_GPU_COUNT}" | sort -u | wc -l)" == \
   "${REQUIRED_GPU_COUNT}" ]] || die "GPU selectors are not unique"
if [[ "${ROLE_BACKEND}" == "remote" ]]; then
  for index in "${!SUITE_NAMES[@]}"; do
    suite="${SUITE_NAMES[index]}"
    if [[ "${SUITE_PARALLELISM}" == "4" ]]; then
      SUITE_GPU["${suite}"]="${GPU_SELECTORS[index]}"
    else
      SUITE_GPU["${suite}"]="${GPU_SELECTORS[0]}"
      SUITE_PORT["${suite}"]="${PORT_BASE}"
    fi
  done
fi

start_sglang() {
  local label="$1"
  local device="$2"
  local port="$3"
  local server_log="${ARTIFACT_DIR}/sglang_${label}_${RUN_STAMP}.log"
  log "starting ${MODEL_NAME} label=${label} gpu=${device} port=${port} thinking=${AGENT_ENABLE_THINKING} with structured tools"
  (
    trap - EXIT TERM INT HUP
    close_run_lock_fd_for_child
    exec setsid env \
      -u WANDB_API_KEY \
      -u OPENAI_API_KEY \
      -u OPENAI_API_BASE \
      -u OPENROUTER_API_KEY \
      -u VITA_API_KEY \
      -u VITA_API_BASE_URL \
      -u VITA_EVALUATOR_API_KEY \
      -u VITA_EVALUATOR_API_URL \
      CUDA_VISIBLE_DEVICES="${device}" \
      "${VENV_PYTHON}" -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --host "${HOST}" \
        --port "${port}" \
        --tp 1 \
        --dtype bfloat16 \
        --mem-fraction-static "${MEM_FRACTION}" \
        --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}" \
        --reasoning-parser qwen3 \
        --tool-call-parser "${AGENT_TOOL_CALL_PARSER}"
  ) >"${server_log}" 2>&1 &
  SGLANG_PIDS+=("$!")
  SGLANG_LABELS+=("${label}")
  SGLANG_PORTS+=("${port}")
  SGLANG_LOGS+=("${server_log}")
}

start_role_sglang() {
  local label="$1"
  local served_model="$2"
  local device="$3"
  local port="$4"
  local max_running_requests="$5"
  local server_log="${ARTIFACT_DIR}/sglang_${label}_${RUN_STAMP}.log"
  log "starting ${served_model} label=${label} gpu=${device} port=${port}"
  (
    trap - EXIT TERM INT HUP
    close_run_lock_fd_for_child
    exec setsid env \
      -u WANDB_API_KEY \
      -u OPENAI_API_KEY \
      -u OPENAI_API_BASE \
      -u OPENROUTER_API_KEY \
      -u VITA_API_KEY \
      -u VITA_API_BASE_URL \
      CUDA_VISIBLE_DEVICES="${device}" \
      "${VENV_PYTHON}" -m sglang.launch_server \
        --model-path "${ROLE_MODEL_PATH}" \
        --served-model-name "${served_model}" \
        --host "${HOST}" \
        --port "${port}" \
        --tp 1 \
        --dtype bfloat16 \
        --context-length "${ROLE_CONTEXT_LENGTH}" \
        --mem-fraction-static "${ROLE_MEM_FRACTION}" \
        --max-running-requests "${max_running_requests}" \
        --reasoning-parser qwen3
  ) >"${server_log}" 2>&1 &
  SGLANG_PIDS+=("$!")
  SGLANG_LABELS+=("${label}")
  SGLANG_PORTS+=("${port}")
  SGLANG_LOGS+=("${server_log}")
}

if [[ "${ROLE_BACKEND}" == "local" ]]; then
  for ((index = 0; index < AGENT_INSTANCE_COUNT; index++)); do
    start_sglang "agent-${index}" "${GPU_SELECTORS[index]}" "$((PORT_BASE + index))"
  done
elif [[ "${SUITE_PARALLELISM}" == "4" ]]; then
  for suite in "${SUITE_NAMES[@]}"; do
    start_sglang "${suite}" "${SUITE_GPU[${suite}]}" "${SUITE_PORT[${suite}]}"
  done
else
  start_sglang shared "${SUITE_GPU[delivery]}" "${PORT_BASE}"
fi

if [[ "${ROLE_BACKEND}" == "local" ]]; then
  for ((index = 0; index < USER_INSTANCE_COUNT; index++)); do
    gpu_index=$((AGENT_INSTANCE_COUNT + index))
    start_role_sglang \
      "user-${index}" "${USER_MODEL}" "${GPU_SELECTORS[gpu_index]}" \
      "$((ROLE_PORT_BASE + index))" "${USER_MAX_RUNNING_REQUESTS}"
  done
  for ((index = 0; index < EVALUATOR_INSTANCE_COUNT; index++)); do
    gpu_index=$((AGENT_INSTANCE_COUNT + USER_INSTANCE_COUNT + index))
    start_role_sglang \
      "evaluator-${index}" "${EVALUATOR_MODEL}" "${GPU_SELECTORS[gpu_index]}" \
      "$((ROLE_PORT_BASE + USER_INSTANCE_COUNT + index))" \
      "${EVALUATOR_MAX_RUNNING_REQUESTS}"
  done
fi

for index in "${!SGLANG_PIDS[@]}"; do
  pid="${SGLANG_PIDS[index]}"
  label="${SGLANG_LABELS[index]}"
  port="${SGLANG_PORTS[index]}"
  server_log="${SGLANG_LOGS[index]}"
  log "waiting for sglang label=${label} port=${port}; log=${server_log}"
  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 100 "${server_log}" >&2 || true
      die "sglang ${label} exited before becoming ready"
    fi
    if HOST="${HOST}" PORT="${port}" "${VENV_PYTHON}" - <<'PY' 2>/dev/null
import os
import urllib.request

urllib.request.urlopen(
    f"http://{os.environ['HOST']}:{os.environ['PORT']}/health",
    timeout=3,
).read()
PY
    then
      ready=1
      break
    fi
    sleep 5
  done
  [[ "${ready}" == "1" ]] || {
    tail -n 100 "${server_log}" >&2 || true
    die "sglang ${label} health check timed out"
  }
done

probe_local_tool_call() {
  local label="$1"
  local port="$2"
  log "checking structured tool_calls label=${label} port=${port}"
  HOST="${HOST}" PORT="${port}" MODEL_NAME="${MODEL_NAME}" LABEL="${label}" \
    AGENT_ENABLE_THINKING="${AGENT_ENABLE_THINKING}" "${VENV_PYTHON}" - <<'PY'
import os
import requests

response = requests.post(
    f"http://{os.environ['HOST']}:{os.environ['PORT']}/v1/chat/completions",
    json={
        "model": os.environ["MODEL_NAME"],
        "messages": [
            {
                "role": "user",
                "content": "Call get_weather for Beijing now. Do not answer directly.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {
            "enable_thinking": os.environ["AGENT_ENABLE_THINKING"] == "true"
        },
    },
    timeout=(10, 300),
)
response.raise_for_status()
payload = response.json()
if payload.get("model") != os.environ["MODEL_NAME"]:
    raise RuntimeError("agent server returned an unexpected model id")
message = payload["choices"][0]["message"]
tool_calls = message.get("tool_calls") or []
if not tool_calls or tool_calls[0]["function"]["name"] != "get_weather":
    raise RuntimeError("agent did not return the expected structured tool call")
print(
    "[vitabench-full] LOCAL_TOOL_CALL_OK",
    f"label={os.environ['LABEL']}",
    "name=get_weather",
)
PY
}

if [[ "${ROLE_BACKEND}" == "local" ]]; then
  for ((index = 0; index < AGENT_INSTANCE_COUNT; index++)); do
    probe_local_tool_call "agent-${index}" "$((PORT_BASE + index))"
  done
elif [[ "${SUITE_PARALLELISM}" == "4" ]]; then
  for suite in "${SUITE_NAMES[@]}"; do
    probe_local_tool_call "${suite}" "${SUITE_PORT[${suite}]}"
  done
else
  probe_local_tool_call shared "${PORT_BASE}"
fi

# Do not expose credentials through shell tracing or source/eval. The agent
# servers were deliberately started before any remote values enter this shell.
set +x
SELECTED_EVALUATOR="${EVALUATOR_MODEL}"
PAPER_PROTOCOL_JUDGE_MATCH="false"
if [[ "${ROLE_BACKEND}" == "local" ]]; then
  export VITA_USER_API_BASE_URL="http://${HOST}:$((ROLE_PORT_BASE + 0))/v1/chat/completions"
  export VITA_EVALUATOR_API_BASE_URL="http://${HOST}:$((ROLE_PORT_BASE + 2))/v1/chat/completions"
  USER_PROVIDER="local_sglang_openai_compatible"
  EVALUATOR_PROVIDER="local_sglang_openai_compatible"
  USER_RESPONSE_MODEL_FAMILY="${USER_MODEL}"
  EVALUATOR_RESPONSE_MODEL_FAMILY="${EVALUATOR_MODEL}"
  if [[ "${RUN_MODE}" == "legacy-v2-resume" ]]; then
    USER_API_ENDPOINT="suite-routed localhost ports ${ROLE_PORT_BASE}-$((ROLE_PORT_BASE + USER_INSTANCE_COUNT - 1))"
    EVALUATOR_API_ENDPOINT="suite-routed localhost ports $((ROLE_PORT_BASE + USER_INSTANCE_COUNT))-$((ROLE_PORT_BASE + USER_INSTANCE_COUNT + EVALUATOR_INSTANCE_COUNT - 1))"
  else
    USER_API_ENDPOINT="worker-routed localhost ports ${ROLE_PORT_BASE}-$((ROLE_PORT_BASE + USER_INSTANCE_COUNT - 1))"
    EVALUATOR_API_ENDPOINT="worker-routed localhost ports $((ROLE_PORT_BASE + USER_INSTANCE_COUNT))-$((ROLE_PORT_BASE + USER_INSTANCE_COUNT + EVALUATOR_INSTANCE_COUNT - 1))"
  fi
  EVALUATOR_SELECTION_REASON="Qwen3.6-27B is the strongest compatible dense open-weight model selected for a single 80GB GPU; non-thinking evaluator mode is the accepted default after the full paired replay measured a 4.36x evaluator speedup with a 1.31 percentage-point Avg@4/pass@1 reduction."

  USER_ACTUAL_MODEL=""
  for ((index = 0; index < USER_INSTANCE_COUNT; index++)); do
    role_port="$((ROLE_PORT_BASE + index))"
    export VITA_USER_API_BASE_URL="http://${HOST}:${role_port}/v1/chat/completions"
    log "probing the local Qwen3.6 user simulator port=${role_port} through Vita"
    if ! role_actual_model="$(probe_chat_model "${USER_MODEL}" true)"; then
      die "local Qwen3.6 user model probe failed on port ${role_port}"
    fi
    [[ "${role_actual_model}" == "${USER_MODEL}" ]] || \
      die "local user server returned an unexpected model id"
    USER_ACTUAL_MODEL="${role_actual_model}"
  done

  EVALUATOR_ACTUAL_MODEL=""
  for ((index = 0; index < EVALUATOR_INSTANCE_COUNT; index++)); do
    role_port="$((ROLE_PORT_BASE + USER_INSTANCE_COUNT + index))"
    export VITA_EVALUATOR_API_BASE_URL="http://${HOST}:${role_port}/v1/chat/completions"
    log "probing the local Qwen3.6 evaluator port=${role_port} through Vita"
    if ! role_actual_model="$(probe_chat_model "${SELECTED_EVALUATOR}" false 1024)"; then
      die "local Qwen3.6 evaluator probe failed on port ${role_port}"
    fi
    [[ "${role_actual_model}" == "${EVALUATOR_MODEL}" ]] || \
      die "local evaluator server returned an unexpected model id"
    EVALUATOR_ACTUAL_MODEL="${role_actual_model}"
  done
  export VITA_USER_API_BASE_URL="http://${HOST}:$((ROLE_PORT_BASE + 0))/v1/chat/completions"
  export VITA_EVALUATOR_API_BASE_URL="http://${HOST}:$((ROLE_PORT_BASE + USER_INSTANCE_COUNT))/v1/chat/completions"
else
  VITA_API_KEY="$(read_dotenv OPENAI_API_KEY)"
  VITA_API_BASE="$(read_dotenv OPENAI_API_BASE)"
  [[ -n "${VITA_API_KEY}" ]] || die "OPENAI_API_KEY is empty in ${ENV_FILE}"
  [[ -n "${VITA_API_BASE}" ]] || die "OPENAI_API_BASE is empty in ${ENV_FILE}"
  REMOTE_API_ENDPOINT="$(VITA_API_BASE="${VITA_API_BASE}" "${VENV_PYTHON}" - <<'PY'
import os
from urllib.parse import urlsplit, urlunsplit

raw_url = os.environ["VITA_API_BASE"].strip()
parsed = urlsplit(raw_url)
if (
    parsed.scheme.lower() != "https"
    or parsed.hostname != "api.chatanywhere.tech"
    or parsed.username is not None
    or parsed.password is not None
    or parsed.port not in {None, 443}
    or parsed.query
    or parsed.fragment
    or parsed.path.rstrip("/") != "/v1"
):
    raise SystemExit("OPENAI_API_BASE must be the ChatAnywhere HTTPS /v1 endpoint")
print(urlunsplit(("https", "api.chatanywhere.tech", "/v1", "", "")), end="")
PY
  )"
  export VITA_API_KEY
  export VITA_API_BASE_URL="${REMOTE_API_ENDPOINT}/chat/completions"
  export REMOTE_API_ENDPOINT
  readonly VITA_API_KEY VITA_API_BASE_URL REMOTE_API_ENDPOINT
  USER_PROVIDER="chatanywhere_openai_compatible_endpoint"
  EVALUATOR_PROVIDER="chatanywhere_openai_compatible_endpoint"
  USER_RESPONSE_MODEL_FAMILY="gpt-4.1"
  EVALUATOR_RESPONSE_MODEL_FAMILY="gemini-2.5-flash"
  USER_API_ENDPOINT="${REMOTE_API_ENDPOINT}"
  EVALUATOR_API_ENDPOINT="${REMOTE_API_ENDPOINT}"
  EVALUATOR_SELECTION_REASON="User selected Gemini 2.5 Flash because Claude 3.7 Sonnet is absent from the current ChatAnywhere model list."

  log "probing the user-selected GPT-4.1 CA user simulator through Vita"
  if ! USER_ACTUAL_MODEL="$(probe_chat_model "${USER_MODEL}" true)"; then
    die "GPT-4.1 CA user model probe failed"
  fi
  [[ "${USER_ACTUAL_MODEL,,}" =~ ^gpt-4\.1([-\.].*)?$ ]] || \
    die "GPT-4.1 CA resolved to an unexpected backend model family"

  log "probing the user-selected Gemini 2.5 Flash evaluator through Vita"
  if ! EVALUATOR_ACTUAL_MODEL="$(probe_chat_model "${SELECTED_EVALUATOR}" false)"; then
    die "Gemini 2.5 Flash evaluator probe failed; refusing to switch judges"
  fi
  [[ "${EVALUATOR_ACTUAL_MODEL,,}" =~ ^gemini-2\.5-flash([-\.].*)?$ ]] || \
    die "Gemini 2.5 Flash resolved to an unexpected backend model family"
fi
log "USER_API_OK requested=${USER_MODEL} actual=${USER_ACTUAL_MODEL} backend=${ROLE_BACKEND}"

validate_legacy_v2_manifest() {
  MANIFEST="${MANIFEST}" MODEL_PATH="${MODEL_PATH}" USER_ACTUAL_MODEL="${USER_ACTUAL_MODEL}" \
    MODEL_FILES_SHA256="${MODEL_FILES_SHA256}" PROTOCOL_STATE_JSON="${PROTOCOL_STATE_JSON}" \
    REMOTE_CONCURRENCY_LIMIT="${REMOTE_CONCURRENCY_LIMIT}" PORT_BASE="${PORT_BASE}" \
    MEM_FRACTION="${MEM_FRACTION}" \
    SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS}" \
    USER_MODEL="${USER_MODEL}" USER_PROVIDER="${USER_PROVIDER}" \
    USER_RESPONSE_MODEL_FAMILY="${USER_RESPONSE_MODEL_FAMILY}" \
    USER_API_ENDPOINT="${USER_API_ENDPOINT}" SELECTED_EVALUATOR="${SELECTED_EVALUATOR}" \
    EVALUATOR_PROVIDER="${EVALUATOR_PROVIDER}" \
    EVALUATOR_RESPONSE_MODEL_FAMILY="${EVALUATOR_RESPONSE_MODEL_FAMILY}" \
    EVALUATOR_API_ENDPOINT="${EVALUATOR_API_ENDPOINT}" ROLE_MODEL_PATH="${ROLE_MODEL_PATH}" \
    ROLE_MODEL_FILES_SHA256="${ROLE_MODEL_FILES_SHA256}" ROLE_PORT_BASE="${ROLE_PORT_BASE}" \
    ROLE_MEM_FRACTION="${ROLE_MEM_FRACTION}" ROLE_CONTEXT_LENGTH="${ROLE_CONTEXT_LENGTH}" \
    USER_MAX_RUNNING_REQUESTS="${USER_MAX_RUNNING_REQUESTS}" \
    "${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise SystemExit(message)


manifest = json.loads(Path(os.environ["MANIFEST"]).read_text(encoding="utf-8"))
require(manifest.get("schema_version") == 5, "legacy resume requires schema version 5")

expected_protocol = {
    "language": "chinese",
    "num_trials": 4,
    "temperature": 0,
    "seed": 300,
    "max_steps": 300,
    "max_errors": 10,
    "evaluation_type": "trajectory",
    "enable_think": True,
    "max_concurrency": 4,
    "shard_size": 10,
    "suite_count": 4,
    "task_count": 400,
    "trajectory_count": 1600,
    "shard_count": 40,
}
require(manifest.get("protocol") == expected_protocol, "legacy manifest protocol mismatch")

suite_names = ("delivery", "instore", "ota", "cross_domain")
expected_execution = {
    "role_backend": "local",
    "suite_parallelism": 4,
    "per_suite_max_concurrency": 4,
    "aggregate_max_concurrency": 16,
    "remote_api_concurrency_limit": int(os.environ["REMOTE_CONCURRENCY_LIMIT"]),
    "api_response_journal_schema": "vita-llm-response-journal/v1",
    "journaled_models": [
        "Qwen3.5-4B",
        os.environ["USER_MODEL"],
        os.environ["SELECTED_EVALUATOR"],
    ],
    "gpu_binding": "four_agent_plus_four_role_replicas",
    "suite_workers": [
        {
            "suite": suite,
            "gpu_slot": index,
            "port": int(os.environ["PORT_BASE"]) + index,
        }
        for index, suite in enumerate(suite_names)
    ],
}
require(manifest.get("execution") == expected_execution, "legacy manifest execution mismatch")
require(manifest.get("model") == "Qwen3.5-4B", "legacy manifest agent model mismatch")

agent = manifest.get("agent") or {}
agent_expected = {
    "requested": "Qwen3.5-4B",
    "served": "Qwen3.5-4B",
    "response_model_family": "Qwen3.5-4B",
    "model_files_sha256": os.environ["MODEL_FILES_SHA256"],
    "reasoning_parser": "qwen3",
    "tool_call_parser": "qwen3_coder",
    "max_tokens": 8192,
    "max_input_tokens": 262144,
    "sglang_mem_fraction_static": float(os.environ["MEM_FRACTION"]),
    "sglang_max_running_requests": int(os.environ["SGLANG_MAX_RUNNING_REQUESTS"]),
    "sglang_instance_count": 4,
}
for key, value in agent_expected.items():
    require(agent.get(key) == value, f"legacy manifest agent mismatch for {key}")
require(
    Path(agent.get("path", "")).resolve() == Path(os.environ["MODEL_PATH"]).resolve(),
    "legacy manifest agent path mismatch",
)

user = manifest.get("user") or {}
user_expected = {
    "model": os.environ["USER_MODEL"],
    "requested": os.environ["USER_MODEL"],
    "actual": os.environ["USER_ACTUAL_MODEL"],
    "response_model_family": os.environ["USER_RESPONSE_MODEL_FAMILY"],
    "provider": os.environ["USER_PROVIDER"],
    "api_base": os.environ["USER_API_ENDPOINT"],
    "paper_protocol_user_match": False,
    "thinking_mode": False,
}
for key, value in user_expected.items():
    require(user.get(key) == value, f"legacy manifest user mismatch for {key}")

# Schema-5 artifacts-v2 is the immutable historical Thinking-on baseline.
evaluator = manifest.get("evaluator") or {}
evaluator_expected = {
    "model": os.environ["SELECTED_EVALUATOR"],
    "requested": os.environ["SELECTED_EVALUATOR"],
    "response_model_family": os.environ["EVALUATOR_RESPONSE_MODEL_FAMILY"],
    "provider": os.environ["EVALUATOR_PROVIDER"],
    "api_base": os.environ["EVALUATOR_API_ENDPOINT"],
    "evaluation_type": "trajectory",
    "thinking_mode": True,
    "paper_protocol_judge_match": False,
    "official_comparable": False,
}
for key, value in evaluator_expected.items():
    require(evaluator.get(key) == value, f"legacy manifest evaluator mismatch for {key}")
require(
    isinstance(evaluator.get("actual"), str) and evaluator["actual"],
    "legacy manifest evaluator response model is missing",
)

role = manifest.get("local_role_serving") or {}
role_expected = {
    "base_model": "Qwen3.6-27B",
    "model_files_sha256": os.environ["ROLE_MODEL_FILES_SHA256"],
    "context_length": int(os.environ["ROLE_CONTEXT_LENGTH"]),
    "reasoning_parser": "qwen3",
    "sglang_mem_fraction_static": float(os.environ["ROLE_MEM_FRACTION"]),
    "sglang_max_running_requests": int(os.environ["USER_MAX_RUNNING_REQUESTS"]),
    "user_instance_count": 2,
    "evaluator_instance_count": 2,
    "user_ports": [int(os.environ["ROLE_PORT_BASE"]) + index for index in range(2)],
    "evaluator_ports": [
        int(os.environ["ROLE_PORT_BASE"]) + 2 + index for index in range(2)
    ],
}
for key, value in role_expected.items():
    require(role.get(key) == value, f"legacy manifest local role mismatch for {key}")
require(
    Path(role.get("path", "")).resolve()
    == Path(os.environ["ROLE_MODEL_PATH"]).resolve(),
    "legacy manifest role model path mismatch",
)

# The recovery runner and validation/export helpers intentionally differ from the
# failed job. The model config, Vita implementation, and task data must not differ.
software = manifest.get("software") or {}
protocol_state = json.loads(os.environ["PROTOCOL_STATE_JSON"])
for key in (
    "model_config_sha256",
    "vitabench_source_sha256",
    "vitabench_task_data_sha256",
    "vitabench_pyproject_sha256",
):
    require(
        software.get(key) == protocol_state.get(key),
        f"legacy immutable protocol input mismatch for {key}",
    )
require(software.get("runner_protocol_version") == 5, "legacy runner version mismatch")

suite_specs = {
    "delivery": ("delivery", "delivery"),
    "instore": ("instore", "instore"),
    "ota": ("ota", "ota"),
    "cross_domain": ("delivery,instore,ota", "cross_domain"),
}
suites = manifest.get("suites") or []
require([suite.get("name") for suite in suites] == list(suite_names), "legacy suite order mismatch")
all_task_ids = []
for suite in suites:
    name = suite["name"]
    require(
        (suite.get("domain"), suite.get("task_set")) == suite_specs[name],
        f"legacy suite metadata mismatch for {name}",
    )
    task_ids = suite.get("task_ids") or []
    shards = suite.get("shards") or []
    require(len(task_ids) == len(set(task_ids)) == 100, f"legacy task IDs invalid for {name}")
    require(len(shards) == 10, f"legacy shard count invalid for {name}")
    for index, shard in enumerate(shards):
        require(shard.get("index") == index, f"legacy shard index mismatch for {name}")
        require(
            shard.get("result_file") == f"shards/{name}/shard_{index:02d}.json",
            f"legacy shard result path mismatch for {name}:{index}",
        )
        require(
            shard.get("task_ids") == task_ids[index * 10 : (index + 1) * 10],
            f"legacy shard task IDs mismatch for {name}:{index}",
        )
    all_task_ids.extend(task_ids)
require(len(all_task_ids) == len(set(all_task_ids)) == 400, "legacy task IDs are not globally unique")

gate_ids = {
    "delivery": "G0721005",
    "instore": "10826245",
    "ota": "D0812001",
    "cross_domain": "10808001",
}
gates = manifest.get("gate") or []
require([gate.get("name") for gate in gates] == list(suite_names), "legacy gate order mismatch")
for gate in gates:
    name = gate["name"]
    require(
        (gate.get("domain"), gate.get("task_set")) == suite_specs[name],
        f"legacy gate metadata mismatch for {name}",
    )
    require(gate.get("task_ids") == [gate_ids[name]], f"legacy gate task mismatch for {name}")
    require(gate.get("result_file") == f"gate/{name}.json", f"legacy gate path mismatch for {name}")

print(evaluator["actual"], end="")
PY
}

if [[ -f "${MANIFEST}" ]]; then
  log "loading the evaluator and semantic protocol locked by the existing manifest"
  if [[ "${RUN_MODE}" == "legacy-v2-resume" ]]; then
    if ! manifest_selection="$(validate_legacy_v2_manifest)"; then
      die "existing legacy-v2 manifest is invalid; refusing to mix protocols"
    fi
  elif ! manifest_selection="$(
    MANIFEST="${MANIFEST}" MODEL_PATH="${MODEL_PATH}" USER_ACTUAL_MODEL="${USER_ACTUAL_MODEL}" \
      MODEL_FILES_SHA256="${MODEL_FILES_SHA256}" PROTOCOL_STATE_JSON="${PROTOCOL_STATE_JSON}" \
      MODEL_NAME="${MODEL_NAME}" AGENT_ENABLE_THINKING="${AGENT_ENABLE_THINKING}" \
      AGENT_TOOL_CALL_PARSER="${AGENT_TOOL_CALL_PARSER}" \
      SUITE_PARALLELISM="${SUITE_PARALLELISM}" MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
      REMOTE_CONCURRENCY_LIMIT="${REMOTE_CONCURRENCY_LIMIT}" PORT_BASE="${PORT_BASE}" \
      MEM_FRACTION="${MEM_FRACTION}" \
      SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS}" ROLE_BACKEND="${ROLE_BACKEND}" \
      USER_MODEL="${USER_MODEL}" USER_PROVIDER="${USER_PROVIDER}" \
      USER_RESPONSE_MODEL_FAMILY="${USER_RESPONSE_MODEL_FAMILY}" USER_API_ENDPOINT="${USER_API_ENDPOINT}" \
      SELECTED_EVALUATOR="${SELECTED_EVALUATOR}" EVALUATOR_PROVIDER="${EVALUATOR_PROVIDER}" \
      EVALUATOR_RESPONSE_MODEL_FAMILY="${EVALUATOR_RESPONSE_MODEL_FAMILY}" \
      EVALUATOR_API_ENDPOINT="${EVALUATOR_API_ENDPOINT}" ROLE_MODEL_PATH="${ROLE_MODEL_PATH}" \
      ROLE_MODEL_FILES_SHA256="${ROLE_MODEL_FILES_SHA256}" ROLE_PORT_BASE="${ROLE_PORT_BASE}" \
      ROLE_MEM_FRACTION="${ROLE_MEM_FRACTION}" ROLE_CONTEXT_LENGTH="${ROLE_CONTEXT_LENGTH}" \
      WORKER_COUNT="${WORKER_COUNT}" AGENT_INSTANCE_COUNT="${AGENT_INSTANCE_COUNT}" \
      USER_INSTANCE_COUNT="${USER_INSTANCE_COUNT}" EVALUATOR_INSTANCE_COUNT="${EVALUATOR_INSTANCE_COUNT}" \
      USER_MAX_RUNNING_REQUESTS="${USER_MAX_RUNNING_REQUESTS}" \
      EVALUATOR_MAX_RUNNING_REQUESTS="${EVALUATOR_MAX_RUNNING_REQUESTS}" \
      AGGREGATE_MAX_CONCURRENCY="${AGGREGATE_MAX_CONCURRENCY}" \
      LOCAL_TOPOLOGY_JSON="${LOCAL_TOPOLOGY_JSON}" \
      RUN_MODE="${RUN_MODE}" AB_SHARD_INDEX="${AB_SHARD_INDEX}" \
      "${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["MANIFEST"]).read_text(encoding="utf-8"))
if manifest.get("schema_version") != 6:
    raise SystemExit("existing manifest has an unsupported schema version")
protocol = manifest.get("protocol") or {}
expected = {
    "language": "chinese",
    "num_trials": 4,
    "temperature": 0,
    "seed": 300,
    "max_steps": 300,
    "max_errors": 10,
    "evaluation_type": "trajectory",
    "enable_think": os.environ["AGENT_ENABLE_THINKING"] == "true",
    "max_concurrency": int(os.environ["MAX_CONCURRENCY"]),
    "shard_size": 10,
}
for key, value in expected.items():
    if protocol.get(key) != value:
        raise SystemExit(f"existing manifest protocol mismatch for {key}")

parallelism = int(os.environ["SUITE_PARALLELISM"])
role_backend = os.environ["ROLE_BACKEND"]
worker_count = int(os.environ["WORKER_COUNT"]) if role_backend == "local" else parallelism
execution = manifest.get("execution") or {}
execution_expected = {
    "role_backend": role_backend,
    "suite_parallelism": parallelism,
    "scheduler": "dynamic_shard_queue" if role_backend == "local" else "suite_static",
    "run_mode": os.environ["RUN_MODE"],
    "selected_shard_index": (
        int(os.environ["AB_SHARD_INDEX"])
        if os.environ["RUN_MODE"] == "ab"
        else None
    ),
    "worker_count": worker_count,
    "per_worker_max_concurrency": int(os.environ["MAX_CONCURRENCY"]),
    "aggregate_max_concurrency": int(os.environ["AGGREGATE_MAX_CONCURRENCY"]),
    "remote_api_concurrency_limit": int(os.environ["REMOTE_CONCURRENCY_LIMIT"]),
    "api_response_journal_schema": "vita-llm-response-journal/v1",
}
for key, value in execution_expected.items():
    if execution.get(key) != value:
        raise SystemExit(f"existing manifest execution mismatch for {key}")
if execution.get("journaled_models") != [
    os.environ["MODEL_NAME"],
    os.environ["USER_MODEL"],
    os.environ["SELECTED_EVALUATOR"],
]:
    raise SystemExit("existing manifest has a different journaled-model set")
expected_gpu_binding = (
    "eight_single_gpu_role_instances"
    if role_backend == "local"
    else "one_distinct_visible_device_selector_per_suite"
)
if execution.get("gpu_binding") != expected_gpu_binding:
    raise SystemExit("existing manifest has a different GPU binding policy")
if role_backend == "local":
    local_topology = json.loads(os.environ["LOCAL_TOPOLOGY_JSON"])
    expected_workers = local_topology["worker_endpoints"]
    if execution.get("worker_endpoints") != expected_workers:
        raise SystemExit("existing manifest has a different worker endpoint mapping")
    if execution.get("queue_lock") != local_topology["queue_lock"]:
        raise SystemExit("existing manifest has a different queue lock policy")
    if "suite_workers" in execution:
        raise SystemExit("local dynamic scheduling must not bind suites to workers")
else:
    expected_workers = [
        {
            "suite": name,
            "gpu_slot": index if parallelism == 4 else 0,
            "port": int(os.environ["PORT_BASE"]) + (index if parallelism == 4 else 0),
        }
        for index, name in enumerate(("delivery", "instore", "ota", "cross_domain"))
    ]
    if execution.get("suite_workers") != expected_workers:
        raise SystemExit("existing manifest has a different suite worker mapping")

if manifest.get("model") != os.environ["MODEL_NAME"]:
    raise SystemExit("existing manifest has a different agent model")
agent = manifest.get("agent") or {}
for field in ("requested", "served", "response_model_family"):
    if agent.get(field) != os.environ["MODEL_NAME"]:
        raise SystemExit(f"existing manifest has a different agent {field}")
if str(Path(agent.get("path", "")).resolve()) != str(Path(os.environ["MODEL_PATH"]).resolve()):
    raise SystemExit("existing manifest has a different model path")
if agent.get("model_files_sha256") != os.environ["MODEL_FILES_SHA256"]:
    raise SystemExit("local model files changed from the locked manifest")
if agent.get("dtype") != "bfloat16":
    raise SystemExit("existing manifest has a different agent dtype")
if agent.get("tool_call_parser") != os.environ["AGENT_TOOL_CALL_PARSER"]:
    raise SystemExit("existing manifest has a different agent tool-call parser")
if agent.get("sglang_mem_fraction_static") != float(os.environ["MEM_FRACTION"]):
    raise SystemExit("existing manifest has a different SGLang memory fraction")
if agent.get("sglang_max_running_requests") != int(
    os.environ["SGLANG_MAX_RUNNING_REQUESTS"]
):
    raise SystemExit("existing manifest has a different SGLang request limit")
expected_agent_instances = (
    int(os.environ["AGENT_INSTANCE_COUNT"])
    if role_backend == "local"
    else parallelism
)
if agent.get("sglang_instance_count") != expected_agent_instances:
    raise SystemExit("existing manifest has a different SGLang instance count")

user = manifest.get("user") or {}
if user.get("model") != os.environ["USER_MODEL"] or user.get("requested") != os.environ["USER_MODEL"]:
    raise SystemExit("existing manifest has a different user model")
if user.get("actual") != os.environ["USER_ACTUAL_MODEL"]:
    raise SystemExit("user backend model id changed from the locked manifest")
if user.get("provider") != os.environ["USER_PROVIDER"]:
    raise SystemExit("existing manifest has a different user-model provider")
if user.get("response_model_family") != os.environ["USER_RESPONSE_MODEL_FAMILY"]:
    raise SystemExit("existing manifest has a different user response model family")
if user.get("api_base") != os.environ["USER_API_ENDPOINT"]:
    raise SystemExit("existing manifest has a different user-model API endpoint")

software = manifest.get("software") or {}
protocol_state = json.loads(os.environ["PROTOCOL_STATE_JSON"])
for key, value in protocol_state.items():
    if software.get(key) != value:
        raise SystemExit(f"protocol input changed from the locked manifest: {key}")

evaluator = manifest.get("evaluator") or {}
if evaluator.get("model") != os.environ["SELECTED_EVALUATOR"] or evaluator.get("requested") != os.environ["SELECTED_EVALUATOR"]:
    raise SystemExit("existing manifest has a different evaluator model")
if not isinstance(evaluator.get("actual"), str) or not evaluator.get("actual"):
    raise SystemExit("existing manifest omits the evaluator response model id")
if evaluator.get("provider") != os.environ["EVALUATOR_PROVIDER"]:
    raise SystemExit("existing manifest has a different evaluator provider")
if evaluator.get("response_model_family") != os.environ["EVALUATOR_RESPONSE_MODEL_FAMILY"]:
    raise SystemExit("existing manifest has a different evaluator response model family")
if evaluator.get("api_base") != os.environ["EVALUATOR_API_ENDPOINT"]:
    raise SystemExit("existing manifest has a different evaluator API endpoint")
expected_evaluator_thinking = False if role_backend == "local" else None
if evaluator.get("thinking_mode") != expected_evaluator_thinking:
    raise SystemExit("existing manifest has a different evaluator thinking mode")
if evaluator.get("paper_protocol_judge_match") is not False:
    raise SystemExit("the selected evaluator must not be marked as matching the paper judge")
if evaluator.get("official_comparable") is not False:
    raise SystemExit("official_comparable must remain false because the Qwen report omits protocol details")
if role_backend == "local":
    role = manifest.get("local_role_serving") or {}
    if str(Path(role.get("path", "")).resolve()) != str(Path(os.environ["ROLE_MODEL_PATH"]).resolve()):
        raise SystemExit("existing manifest has a different role model path")
    role_expected = {
        "model_files_sha256": os.environ["ROLE_MODEL_FILES_SHA256"],
        "context_length": int(os.environ["ROLE_CONTEXT_LENGTH"]),
        "dtype": "bfloat16",
        "sglang_mem_fraction_static": float(os.environ["ROLE_MEM_FRACTION"]),
        "user_sglang_max_running_requests": int(os.environ["USER_MAX_RUNNING_REQUESTS"]),
        "evaluator_sglang_max_running_requests": int(os.environ["EVALUATOR_MAX_RUNNING_REQUESTS"]),
        "agent_instance_count": int(os.environ["AGENT_INSTANCE_COUNT"]),
        "user_instance_count": int(os.environ["USER_INSTANCE_COUNT"]),
        "evaluator_instance_count": int(os.environ["EVALUATOR_INSTANCE_COUNT"]),
        "total_instance_count": 8,
        "agent_ports": [
            instance["port"] for instance in local_topology["agent_instances"]
        ],
        "user_ports": [
            instance["port"] for instance in local_topology["user_instances"]
        ],
        "evaluator_ports": [
            instance["port"] for instance in local_topology["evaluator_instances"]
        ],
    }
    for key, value in role_expected.items():
        if role.get(key) != value:
            raise SystemExit(f"existing manifest local role serving mismatch for {key}")
print(evaluator["actual"], end="")
PY
  )"; then
    die "existing manifest is invalid; refusing to mix protocols"
  fi
  [[ "${manifest_selection}" == "${EVALUATOR_ACTUAL_MODEL}" ]] || \
    die "evaluator backend model id changed from ${manifest_selection} to ${EVALUATOR_ACTUAL_MODEL}"
  log "EVALUATOR_API_OK requested=${SELECTED_EVALUATOR} actual=${EVALUATOR_ACTUAL_MODEL} locked=true"
else
  log "EVALUATOR_API_OK requested=${SELECTED_EVALUATOR} actual=${EVALUATOR_ACTUAL_MODEL} locked=false"

  MANIFEST="${MANIFEST}" ARTIFACT_DIR="${ARTIFACT_DIR}" VITABENCH_DIR="${VITABENCH_DIR}" \
    PROJECT_ROOT="${PROJECT_ROOT}" MODEL_PATH="${MODEL_PATH}" USER_ACTUAL_MODEL="${USER_ACTUAL_MODEL}" \
    MODEL_FILES_SHA256="${MODEL_FILES_SHA256}" PROTOCOL_STATE_JSON="${PROTOCOL_STATE_JSON}" \
    MODEL_NAME="${MODEL_NAME}" AGENT_ENABLE_THINKING="${AGENT_ENABLE_THINKING}" \
    AGENT_TOOL_CALL_PARSER="${AGENT_TOOL_CALL_PARSER}" \
    SELECTED_EVALUATOR="${SELECTED_EVALUATOR}" EVALUATOR_ACTUAL_MODEL="${EVALUATOR_ACTUAL_MODEL}" \
    EVALUATOR_PROVIDER="${EVALUATOR_PROVIDER}" PAPER_PROTOCOL_JUDGE_MATCH="${PAPER_PROTOCOL_JUDGE_MATCH}" \
    EVALUATOR_SELECTION_REASON="${EVALUATOR_SELECTION_REASON}" MEM_FRACTION="${MEM_FRACTION}" \
    SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS}" \
    SUITE_PARALLELISM="${SUITE_PARALLELISM}" MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
    REMOTE_CONCURRENCY_LIMIT="${REMOTE_CONCURRENCY_LIMIT}" PORT_BASE="${PORT_BASE}" \
    ROLE_BACKEND="${ROLE_BACKEND}" USER_MODEL="${USER_MODEL}" USER_PROVIDER="${USER_PROVIDER}" \
    USER_RESPONSE_MODEL_FAMILY="${USER_RESPONSE_MODEL_FAMILY}" USER_API_ENDPOINT="${USER_API_ENDPOINT}" \
    EVALUATOR_RESPONSE_MODEL_FAMILY="${EVALUATOR_RESPONSE_MODEL_FAMILY}" \
    EVALUATOR_API_ENDPOINT="${EVALUATOR_API_ENDPOINT}" ROLE_MODEL_PATH="${ROLE_MODEL_PATH}" \
    ROLE_MODEL_FILES_SHA256="${ROLE_MODEL_FILES_SHA256}" ROLE_PORT_BASE="${ROLE_PORT_BASE}" \
    ROLE_MEM_FRACTION="${ROLE_MEM_FRACTION}" ROLE_CONTEXT_LENGTH="${ROLE_CONTEXT_LENGTH}" \
    WORKER_COUNT="${WORKER_COUNT}" AGENT_INSTANCE_COUNT="${AGENT_INSTANCE_COUNT}" \
    USER_INSTANCE_COUNT="${USER_INSTANCE_COUNT}" EVALUATOR_INSTANCE_COUNT="${EVALUATOR_INSTANCE_COUNT}" \
    USER_MAX_RUNNING_REQUESTS="${USER_MAX_RUNNING_REQUESTS}" \
    EVALUATOR_MAX_RUNNING_REQUESTS="${EVALUATOR_MAX_RUNNING_REQUESTS}" \
    AGGREGATE_MAX_CONCURRENCY="${AGGREGATE_MAX_CONCURRENCY}" \
    LOCAL_TOPOLOGY_JSON="${LOCAL_TOPOLOGY_JSON}" \
    RUN_MODE="${RUN_MODE}" AB_SHARD_INDEX="${AB_SHARD_INDEX}" \
    "${VENV_PYTHON}" - <<'PY'
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vita.run import load_tasks

artifact_dir = Path(os.environ["ARTIFACT_DIR"])
manifest_path = Path(os.environ["MANIFEST"])
suites_config = [
    ("delivery", "delivery", "delivery"),
    ("instore", "instore", "instore"),
    ("ota", "ota", "ota"),
    ("cross_domain", "delivery,instore,ota", "cross_domain"),
]
gate_ids = {
    "delivery": "G0721005",
    "instore": "10826245",
    "ota": "D0812001",
    "cross_domain": "10808001",
}

suites = []
all_ids = []
for name, domain, task_set in suites_config:
    task_ids = sorted(task.id for task in load_tasks(task_set, "chinese"))
    if len(task_ids) != 100 or len(set(task_ids)) != 100:
        raise SystemExit(f"{name} must contain exactly 100 unique Chinese task IDs")
    if gate_ids[name] not in task_ids:
        raise SystemExit(f"gate task {gate_ids[name]} is absent from {name}")
    shards = []
    for index in range(10):
        shard_ids = task_ids[index * 10:(index + 1) * 10]
        shards.append(
            {
                "index": index,
                "task_ids": shard_ids,
                "result_file": f"shards/{name}/shard_{index:02d}.json",
            }
        )
    suites.append(
        {
            "name": name,
            "domain": domain,
            "task_set": task_set,
            "task_ids": task_ids,
            "shards": shards,
        }
    )
    all_ids.extend(task_ids)

if len(all_ids) != 400 or len(set(all_ids)) != 400:
    raise SystemExit("the four suites must contain 400 globally unique task IDs")

def git_commit(path):
    checkout = Path(path).resolve()
    try:
        top_level = Path(
            subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
        if top_level != checkout:
            return "unavailable"
        return subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"

judge_match = os.environ["PAPER_PROTOCOL_JUDGE_MATCH"] == "true"
protocol_state = json.loads(os.environ["PROTOCOL_STATE_JSON"])
role_backend = os.environ["ROLE_BACKEND"]
agent_name = os.environ["MODEL_NAME"]
agent_enable_thinking = os.environ["AGENT_ENABLE_THINKING"] == "true"
suite_parallelism = int(os.environ["SUITE_PARALLELISM"])
max_concurrency = int(os.environ["MAX_CONCURRENCY"])
if agent_name == "Qwen3.5-4B":
    official_reference = {
        "model": agent_name,
        "available": True,
        "score": 22.0,
        "unit": "points",
        "metric": "four-suite macro Avg@4",
        "source_url": "https://huggingface.co/Qwen/Qwen3.5-4B",
        "strictly_comparable": False,
        "limitation": "The Qwen model card reports 22.0 but omits language, exact judge endpoint/version, and complete run settings.",
    }
else:
    official_reference = {
        "model": agent_name,
        "available": False,
        "score": None,
        "unit": "points",
        "metric": "four-suite macro Avg@4",
        "source_url": f"https://huggingface.co/Qwen/{agent_name}",
        "strictly_comparable": False,
        "limitation": "No published four-suite VITA-Bench reference is recorded for this exact model.",
    }
if role_backend == "local":
    local_topology = json.loads(os.environ["LOCAL_TOPOLOGY_JSON"])
    worker_count = local_topology["worker_count"]
    execution = {
        "role_backend": role_backend,
        "suite_parallelism": suite_parallelism,
        "scheduler": "dynamic_shard_queue",
        "run_mode": os.environ["RUN_MODE"],
        "selected_shard_index": (
            int(os.environ["AB_SHARD_INDEX"])
            if os.environ["RUN_MODE"] == "ab"
            else None
        ),
        "worker_count": worker_count,
        "per_worker_max_concurrency": max_concurrency,
        "aggregate_max_concurrency": int(os.environ["AGGREGATE_MAX_CONCURRENCY"]),
        "remote_api_concurrency_limit": int(os.environ["REMOTE_CONCURRENCY_LIMIT"]),
        "api_response_journal_schema": "vita-llm-response-journal/v1",
        "journaled_models": [
            agent_name,
            os.environ["USER_MODEL"],
            os.environ["SELECTED_EVALUATOR"],
        ],
        "gpu_binding": "eight_single_gpu_role_instances",
        "queue_lock": local_topology["queue_lock"],
        "worker_endpoints": local_topology["worker_endpoints"],
    }
    agent_instance_count = len(local_topology["agent_instances"])
else:
    worker_count = suite_parallelism
    execution = {
        "role_backend": role_backend,
        "suite_parallelism": suite_parallelism,
        "scheduler": "suite_static",
        "run_mode": os.environ["RUN_MODE"],
        "selected_shard_index": None,
        "worker_count": worker_count,
        "per_worker_max_concurrency": max_concurrency,
        "aggregate_max_concurrency": int(os.environ["AGGREGATE_MAX_CONCURRENCY"]),
        "remote_api_concurrency_limit": int(os.environ["REMOTE_CONCURRENCY_LIMIT"]),
        "api_response_journal_schema": "vita-llm-response-journal/v1",
        "journaled_models": [
            agent_name,
            os.environ["USER_MODEL"],
            os.environ["SELECTED_EVALUATOR"],
        ],
        "gpu_binding": "one_distinct_visible_device_selector_per_suite",
        "suite_workers": [
            {
                "suite": name,
                "gpu_slot": index if suite_parallelism == 4 else 0,
                "port": int(os.environ["PORT_BASE"]) + (
                    index if suite_parallelism == 4 else 0
                ),
            }
            for index, (name, _, _) in enumerate(suites_config)
        ],
    }
    agent_instance_count = suite_parallelism
manifest = {
    "schema_version": 6,
    "benchmark": "VITA-Bench",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "official_reference": official_reference,
    "protocol": {
        "language": "chinese",
        "num_trials": 4,
        "temperature": 0,
        "seed": 300,
        "max_steps": 300,
        "max_errors": 10,
        "evaluation_type": "trajectory",
        "enable_think": agent_enable_thinking,
        "max_concurrency": int(os.environ["MAX_CONCURRENCY"]),
        "shard_size": 10,
        "suite_count": 4,
        "task_count": 400,
        "trajectory_count": 1600,
        "shard_count": 40,
    },
    "execution": execution,
    "model": agent_name,
    "agent": {
        "requested": agent_name,
        "served": agent_name,
        "response_model_family": agent_name,
        "path": str(Path(os.environ["MODEL_PATH"]).resolve()),
        "model_files_sha256": os.environ["MODEL_FILES_SHA256"],
        "model_hash_scope": "all JSON, Jinja, tokenizer text/model, and safetensor files",
        "reasoning_parser": "qwen3",
        "tool_call_parser": os.environ["AGENT_TOOL_CALL_PARSER"],
        "max_tokens": 8192,
        "max_input_tokens": 262144,
        "dtype": "bfloat16",
        "sglang_mem_fraction_static": float(os.environ["MEM_FRACTION"]),
        "sglang_max_running_requests": int(os.environ["SGLANG_MAX_RUNNING_REQUESTS"]),
        "sglang_instance_count": agent_instance_count,
        "cost_accounting": "local inference; no API price configured",
    },
    "user": {
        "model": os.environ["USER_MODEL"],
        "requested": os.environ["USER_MODEL"],
        "actual": os.environ["USER_ACTUAL_MODEL"],
        "response_model_family": os.environ["USER_RESPONSE_MODEL_FAMILY"],
        "provider": os.environ["USER_PROVIDER"],
        "api_base": os.environ["USER_API_ENDPOINT"],
        "paper_protocol_user_match": False,
        "thinking_mode": False if os.environ["ROLE_BACKEND"] == "local" else None,
        "cost_accounting": (
            "local inference; no API price configured"
            if os.environ["ROLE_BACKEND"] == "local"
            else "not configured; serialized cost is null and provider spend is unavailable"
        ),
    },
    "evaluator": {
        "model": os.environ["SELECTED_EVALUATOR"],
        "requested": os.environ["SELECTED_EVALUATOR"],
        "actual": os.environ["EVALUATOR_ACTUAL_MODEL"],
        "response_model_family": os.environ["EVALUATOR_RESPONSE_MODEL_FAMILY"],
        "provider": os.environ["EVALUATOR_PROVIDER"],
        "api_base": os.environ["EVALUATOR_API_ENDPOINT"],
        "evaluation_type": "trajectory",
        "thinking_mode": False if os.environ["ROLE_BACKEND"] == "local" else None,
        "paper_protocol_judge_match": judge_match,
        "official_comparable": False,
        "selection_reason": os.environ["EVALUATOR_SELECTION_REASON"],
        "cost_accounting": (
            "local inference; no API price configured"
            if os.environ["ROLE_BACKEND"] == "local"
            else "unknown provider price; judge token usage is persisted when returned, cost is unavailable"
        ),
        "comparison_limitation": official_reference["limitation"],
    },
    "software": {
        "vitabench_git_commit": git_commit(os.environ["VITABENCH_DIR"]),
        "slime_git_commit": git_commit(os.environ["PROJECT_ROOT"]),
        **protocol_state,
    },
    "gate": [
        {
            "name": name,
            "domain": domain,
            "task_set": task_set,
            "worker_id": index,
            "task_ids": [gate_ids[name]],
            "result_file": f"gate/{name}.json",
        }
        for index, (name, domain, task_set) in enumerate(suites_config)
    ],
    "suites": suites,
}

if os.environ["ROLE_BACKEND"] == "local":
    manifest["local_role_serving"] = {
        "base_model": "Qwen3.6-27B",
        "path": str(Path(os.environ["ROLE_MODEL_PATH"]).resolve()),
        "model_files_sha256": os.environ["ROLE_MODEL_FILES_SHA256"],
        "model_hash_scope": "all JSON, Jinja, tokenizer text/model, and safetensor files",
        "context_length": int(os.environ["ROLE_CONTEXT_LENGTH"]),
        "dtype": "bfloat16",
        "reasoning_parser": "qwen3",
        "sglang_mem_fraction_static": float(os.environ["ROLE_MEM_FRACTION"]),
        "agent_instance_count": len(local_topology["agent_instances"]),
        "user_instance_count": len(local_topology["user_instances"]),
        "evaluator_instance_count": len(local_topology["evaluator_instances"]),
        "total_instance_count": local_topology["instance_count"],
        "user_sglang_max_running_requests": int(os.environ["USER_MAX_RUNNING_REQUESTS"]),
        "evaluator_sglang_max_running_requests": int(os.environ["EVALUATOR_MAX_RUNNING_REQUESTS"]),
        "agent_ports": [
            instance["port"] for instance in local_topology["agent_instances"]
        ],
        "user_ports": [
            instance["port"] for instance in local_topology["user_instances"]
        ],
        "evaluator_ports": [
            instance["port"] for instance in local_topology["evaluator_instances"]
        ],
    }

manifest_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
temporary_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
os.replace(temporary_path, manifest_path)
print(f"[vitabench-full] MANIFEST_CREATED path={manifest_path}")
PY
fi

# The plan is derived from the locked manifest. Result discovery is never glob
# based, so gate files and unrelated artifacts cannot enter the final score.
MANIFEST="${MANIFEST}" RUN_PLAN="${RUN_PLAN}" RUN_MODE="${RUN_MODE}" \
  "${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["MANIFEST"]).read_text(encoding="utf-8"))
lines = []
for gate in manifest["gate"]:
    lines.append(
        "\t".join(
            (
        "gate",
        gate["name"],
        gate["domain"],
        gate["task_set"],
        "0",
        gate["result_file"],
        " ".join(gate["task_ids"]),
            )
        )
    )
for suite in manifest["suites"]:
    for shard in suite["shards"]:
        lines.append(
            "\t".join(
                (
            "shard",
            suite["name"],
            suite["domain"],
            suite["task_set"],
            str(shard["index"]),
            shard["result_file"],
            " ".join(shard["task_ids"]),
                )
            )
        )
rendered = "\n".join(lines) + "\n"
run_plan_path = Path(os.environ["RUN_PLAN"])
if os.environ["RUN_MODE"] == "legacy-v2-resume":
    if not run_plan_path.is_file() or run_plan_path.read_text(encoding="utf-8") != rendered:
        raise SystemExit("legacy run plan differs from the schema-5 manifest")
else:
    temporary_path = run_plan_path.with_name(f".{run_plan_path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(rendered, encoding="utf-8")
    os.replace(temporary_path, run_plan_path)
PY

read -r MANIFEST_SHA256 _ < <(sha256sum -- "${MANIFEST}")
read -r RUN_PLAN_SHA256 _ < <(sha256sum -- "${RUN_PLAN}")

verify_locked_inputs() {
  local current_manifest_sha256
  local current_run_plan_sha256
  verify_protocol_state
  read -r current_manifest_sha256 _ < <(sha256sum -- "${MANIFEST}")
  read -r current_run_plan_sha256 _ < <(sha256sum -- "${RUN_PLAN}")
  [[ "${current_manifest_sha256}" == "${MANIFEST_SHA256}" ]] || \
    die "manifest changed while the evaluation was running"
  [[ "${current_run_plan_sha256}" == "${RUN_PLAN_SHA256}" ]] || \
    die "run plan changed while the evaluation was running"
}

export_sft_archive() {
  local stage="$1"
  verify_locked_inputs
  log "exporting lossless API journals as SFT data stage=${stage}"
  "${VENV_PYTHON}" "${SFT_EXPORTER}" \
    --artifact-dir "${ARTIFACT_DIR}" \
    --manifest "${MANIFEST}" \
    --output-dir "${SFT_DIR}"
  SFT_DIR="${SFT_DIR}" STAGE="${stage}" SFT_EXPORTER="${SFT_EXPORTER}" \
    "${VENV_PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

sft_dir = Path(os.environ["SFT_DIR"])
index = json.loads((sft_dir / "index.json").read_text(encoding="utf-8"))
if index.get("schema") != "vita-sft-archive/v1":
    raise SystemExit("SFT archive has an unsupported schema")
expected_exporter_sha256 = hashlib.sha256(
    Path(os.environ["SFT_EXPORTER"]).read_bytes()
).hexdigest()
if (index.get("exporter") or {}).get("sha256") != expected_exporter_sha256:
    raise SystemExit("SFT archive exporter provenance does not match the runner")
counts = index.get("counts") or {}
journal_files = counts.get("journal_files")
journal_ids = counts.get("journal_ids")
journal_accepted = counts.get("journal_accepted")
journal_rejected = counts.get("journal_rejected")
legacy_examples = counts.get("legacy_examples")
examples = counts.get("examples")
if not all(
    isinstance(value, int) and value >= 0
    for value in (
        journal_files,
        journal_ids,
        journal_accepted,
        journal_rejected,
        legacy_examples,
        examples,
    )
):
    raise SystemExit("SFT archive contains invalid journal counts")
if journal_files < 1 or journal_ids != journal_files:
    raise SystemExit("not every API journal file has a unique valid journal identity")
if journal_accepted + journal_rejected != journal_ids:
    raise SystemExit("SFT archive does not account for every journal response")
if examples != journal_accepted + legacy_examples:
    raise SystemExit("SFT example count does not match accepted journals")

files = index.get("files") or []
by_kind = {record.get("kind"): record for record in files if isinstance(record, dict)}
for kind in ("agent", "user_simulator", "evaluator"):
    record = by_kind.get(kind)
    if not isinstance(record, dict) or not isinstance(record.get("lines"), int):
        raise SystemExit(f"SFT archive is missing {kind}")
    if record["lines"] < 1:
        raise SystemExit(f"SFT archive contains no {kind} examples")
for record in files:
    path = sft_dir / record["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record.get("sha256") or path.stat().st_size != record.get("bytes"):
        raise SystemExit(f"SFT archive checksum mismatch: {path}")
print(
    "[vitabench-full] SFT_ARCHIVE_OK",
    f"stage={os.environ['STAGE']}",
    f"journals={journal_files}",
    f"examples={examples}",
    f"rejected_for_sft={journal_rejected}",
)
PY
  verify_locked_inputs
}

run_batch() {
  local kind="$1"
  local suite="$2"
  local domain="$3"
  local task_set="$4"
  local shard_index="$5"
  local result_relative="$6"
  local worker_id="$7"
  shift 7
  local -a task_ids=("$@")
  local result_file="${ARTIFACT_DIR}/${result_relative}"
  local label
  if [[ "${RUN_MODE}" == "legacy-v2-resume" ]]; then
    label="${kind}:${suite}:${shard_index}"
  else
    label="${kind}:${suite}:${shard_index}:worker-${worker_id}"
  fi
  local batch_started_at
  local batch_started_epoch
  local duration_seconds
  local check_status
  local attempt
  local run_status
  local sleep_seconds
  local preserved_file
  local journal_dir="${API_JOURNAL_DIR}/${suite}"
  mkdir -p "${journal_dir}"
  local -a check_command=(
    "${VENV_PYTHON}" "${SUMMARIZER}" check-shard
    --file "${result_file}"
    --task-ids "${task_ids[@]}"
    --num-trials "${NUM_TRIALS}"
    --base-seed "${SEED}"
    --expected-domain "${domain}"
    --expected-max-steps "${MAX_STEPS}"
    --expected-max-errors "${MAX_ERRORS}"
    --expected-agent-model "${MODEL_NAME}"
    --expected-user-model "${USER_MODEL}"
  )
  local -a vita_command=(
    "${VITA_BIN}" run
    --domain "${domain}"
    --task-set-name "${task_set}"
    --task-ids "${task_ids[@]}"
    --agent-llm "${MODEL_NAME}"
    --user-llm "${USER_MODEL}"
    --evaluator-llm "${SELECTED_EVALUATOR}"
  )
  if [[ "${AGENT_ENABLE_THINKING}" == "true" ]]; then
    vita_command+=(--enable-think)
  fi
  vita_command+=(
    --language "${LANGUAGE}"
    --num-trials "${NUM_TRIALS}"
    --max-steps "${MAX_STEPS}"
    --max-errors "${MAX_ERRORS}"
    --evaluation-type "${EVALUATION_TYPE}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --seed "${SEED}"
    --log-level INFO
    --save-to "${result_file}"
  )
  local -a vita_environment=(
    env
    VITA_LLM_JOURNAL_DIR="${journal_dir}"
    VITA_LLM_JOURNAL_MODELS="${MODEL_NAME},${USER_MODEL},${SELECTED_EVALUATOR}"
    VITA_LLM_JOURNAL_CONTEXT="${label}"
    VITA_LLM_JOURNAL_FSYNC=1
    VITA_REMOTE_CONCURRENCY_DIR="${REMOTE_SLOT_DIR}"
    VITA_REMOTE_CONCURRENCY_HOST=api.chatanywhere.tech
    VITA_REMOTE_CONCURRENCY_LIMIT="${REMOTE_CONCURRENCY_LIMIT}"
    VITA_REMOTE_CONCURRENCY_ACQUIRE_TIMEOUT=600
  )

  verify_locked_inputs
  mkdir -p "$(dirname "${result_file}")"
  if "${check_command[@]}"; then
    log "SKIP_COMPLETE worker=${worker_id} suite=${suite} shard=${shard_index} result=${result_file}"
    return
  else
    check_status=$?
  fi
  if [[ "${check_status}" -eq 2 && -f "${result_file}" ]]; then
    preserved_file="${result_file}.invalid.${RUN_STAMP}.$$"
    mv -- "${result_file}" "${preserved_file}"
    log "PRESERVED_INVALID ${label} path=${preserved_file}"
  elif [[ "${check_status}" -ne 1 ]]; then
    die "unexpected shard-check status ${check_status} for ${label}"
  fi

  batch_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  batch_started_epoch="$(date +%s)"
  log "BATCH_START worker=${worker_id} suite=${suite} shard=${shard_index} kind=${kind} start_time=${batch_started_at} tasks=${#task_ids[@]}"
  for ((attempt = 1; attempt <= BATCH_MAX_ATTEMPTS; attempt++)); do
    log "BATCH_ATTEMPT worker=${worker_id} suite=${suite} shard=${shard_index} attempt=${attempt}/${BATCH_MAX_ATTEMPTS}"
    if [[ -f "${result_file}" ]]; then
      printf 'y\n' | (
        trap - EXIT TERM INT HUP
        close_run_lock_fd_for_child
        exec setsid --wait "${vita_environment[@]}" "${vita_command[@]}"
      ) &
    else
      (
        trap - EXIT TERM INT HUP
        close_run_lock_fd_for_child
        exec setsid --wait "${vita_environment[@]}" "${vita_command[@]}"
      ) &
    fi
    ACTIVE_BATCH_PID="$!"
    if wait "${ACTIVE_BATCH_PID}"; then
      run_status=0
    else
      run_status=$?
    fi
    ACTIVE_BATCH_PID=""

    verify_locked_inputs
    if "${check_command[@]}"; then
      duration_seconds=$(($(date +%s) - batch_started_epoch))
      log "BATCH_OK worker=${worker_id} suite=${suite} shard=${shard_index} start_time=${batch_started_at} duration_seconds=${duration_seconds} result=${result_file}"
      return
    else
      check_status=$?
    fi
    if [[ "${check_status}" -eq 2 && -f "${result_file}" ]]; then
      preserved_file="${result_file}.invalid.${RUN_STAMP}.$$.${attempt}"
      mv -- "${result_file}" "${preserved_file}"
      log "PRESERVED_INVALID ${label} path=${preserved_file}"
    elif [[ "${check_status}" -ne 1 ]]; then
      die "unexpected shard-check status ${check_status} for ${label}"
    fi

    if ((attempt < BATCH_MAX_ATTEMPTS)); then
      sleep_seconds=$((5 * (2 ** (attempt - 1))))
      if ((sleep_seconds > 60)); then
        sleep_seconds=60
      fi
      log "RETRY ${label} vita_status=${run_status} wait_seconds=${sleep_seconds}"
      sleep "${sleep_seconds}"
    fi
  done
  duration_seconds=$(($(date +%s) - batch_started_epoch))
  die "BATCH_FAILED worker=${worker_id} suite=${suite} shard=${shard_index} start_time=${batch_started_at} duration_seconds=${duration_seconds} attempts=${BATCH_MAX_ATTEMPTS}"
}

run_suite_stage() {
  local requested_kind="$1"
  local requested_suite="$2"
  local failure_file="$3"
  local worker_id="$4"
  local kind
  local suite
  local domain
  local task_set
  local shard_index
  local result_relative
  local task_ids_text
  local -a task_ids

  if [[ "${ROLE_BACKEND}" == "local" ]]; then
    export VITA_AGENT_API_BASE_URL="http://${HOST}:${WORKER_AGENT_PORT[worker_id]}/v1/chat/completions"
    export VITA_USER_API_BASE_URL="http://${HOST}:${WORKER_USER_PORT[worker_id]}/v1/chat/completions"
    export VITA_EVALUATOR_API_BASE_URL="http://${HOST}:${WORKER_EVALUATOR_PORT[worker_id]}/v1/chat/completions"
  else
    export VITA_AGENT_API_BASE_URL="http://${HOST}:${SUITE_PORT[${requested_suite}]}/v1/chat/completions"
  fi
  log "SUITE_WORKER_START stage=${requested_kind} worker=${worker_id} suite=${requested_suite} agent_port=${VITA_AGENT_API_BASE_URL}"
  while IFS=$'\t' read -r kind suite domain task_set shard_index result_relative task_ids_text; do
    [[ "${kind}" == "${requested_kind}" && "${suite}" == "${requested_suite}" ]] || continue
    if [[ "${SUITE_PARALLELISM}" == "4" && -e "${failure_file}" ]]; then
      log "SUITE_WORKER_STOP stage=${requested_kind} worker=${worker_id} suite=${requested_suite} reason=peer_failure"
      return 75
    fi
    read -r -a task_ids <<<"${task_ids_text}"
    run_batch \
      "${kind}" "${suite}" "${domain}" "${task_set}" "${shard_index}" \
      "${result_relative}" "${worker_id}" "${task_ids[@]}"
  done <"${RUN_PLAN}"
  log "SUITE_WORKER_OK stage=${requested_kind} worker=${worker_id} suite=${requested_suite}"
}

initialize_dynamic_shard_queue() {
  local queue_summary
  local -a queue_selection_args=()
  if [[ "${RUN_MODE}" == "ab" ]]; then
    queue_selection_args=(--shard-index "${AB_SHARD_INDEX}")
  fi
  verify_locked_inputs
  queue_summary="$(
    "${VENV_PYTHON}" "${EVAL_SCHEDULER}" init-queue \
      --run-plan "${RUN_PLAN}" \
      --state "${DYNAMIC_QUEUE_STATE}" \
      --lock "${DYNAMIC_QUEUE_LOCK}" \
      --artifact-dir "${ARTIFACT_DIR}" \
      --summarizer "${SUMMARIZER}" \
      --python "${VENV_PYTHON}" \
      --worker-count "${WORKER_COUNT}" \
      --num-trials "${NUM_TRIALS}" \
      --base-seed "${SEED}" \
      --expected-max-steps "${MAX_STEPS}" \
      --expected-max-errors "${MAX_ERRORS}" \
      --expected-agent-model "${MODEL_NAME}" \
      --expected-user-model "${USER_MODEL}" \
      "${queue_selection_args[@]}"
  )" || die "failed to initialize the dynamic shard queue"
  log "DYNAMIC_QUEUE_READY scheduler=dynamic_shard_queue ${queue_summary}"
  verify_locked_inputs
}

run_dynamic_shard_worker() {
  local worker_id="$1"
  local failure_file="$2"
  local claim_status
  local claimed_line
  local kind
  local suite
  local domain
  local task_set
  local shard_index
  local result_relative
  local task_ids_text
  local -a task_ids

  export VITA_AGENT_API_BASE_URL="http://${HOST}:${WORKER_AGENT_PORT[worker_id]}/v1/chat/completions"
  export VITA_USER_API_BASE_URL="http://${HOST}:${WORKER_USER_PORT[worker_id]}/v1/chat/completions"
  export VITA_EVALUATOR_API_BASE_URL="http://${HOST}:${WORKER_EVALUATOR_PORT[worker_id]}/v1/chat/completions"
  log "DYNAMIC_WORKER_START worker=${worker_id} agent_port=${WORKER_AGENT_PORT[worker_id]} user_port=${WORKER_USER_PORT[worker_id]} evaluator_port=${WORKER_EVALUATOR_PORT[worker_id]}"

  while true; do
    if [[ -e "${failure_file}" ]]; then
      log "DYNAMIC_WORKER_STOP worker=${worker_id} reason=peer_failure"
      return 75
    fi
    if claimed_line="$(
      "${VENV_PYTHON}" "${EVAL_SCHEDULER}" claim \
        --state "${DYNAMIC_QUEUE_STATE}" \
        --lock "${DYNAMIC_QUEUE_LOCK}" \
        --worker-id "${worker_id}"
    )"; then
      claim_status=0
    else
      claim_status=$?
    fi
    if [[ "${claim_status}" -eq 3 ]]; then
      break
    fi
    [[ "${claim_status}" -eq 0 ]] || \
      die "dynamic shard claim failed for worker ${worker_id} with status ${claim_status}"
    IFS=$'\t' read -r \
      kind suite domain task_set shard_index result_relative task_ids_text \
      <<<"${claimed_line}"
    [[ "${kind}" == "shard" ]] || die "dynamic queue returned a non-shard entry"
    read -r -a task_ids <<<"${task_ids_text}"
    log "SHARD_CLAIM worker=${worker_id} suite=${suite} shard=${shard_index}"
    run_batch \
      "${kind}" "${suite}" "${domain}" "${task_set}" "${shard_index}" \
      "${result_relative}" "${worker_id}" "${task_ids[@]}"
  done
  log "DYNAMIC_WORKER_OK worker=${worker_id} reason=queue_empty"
}

run_stage() {
  local requested_kind="$1"
  local failure_file="${ARTIFACT_DIR}/.${requested_kind}_worker_failure"
  local suite
  local pid
  local status
  local index
  local failed=0
  local worker_id
  local stage_started_at
  local stage_started_epoch
  local stage_duration_seconds
  local -a worker_labels=()

  stage_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  stage_started_epoch="$(date +%s)"
  log "STAGE_START stage=${requested_kind} run_mode=${RUN_MODE} start_time=${stage_started_at}"
  unlink "${failure_file}" 2>/dev/null || true
  if [[ "${ROLE_BACKEND}" == "local" && "${requested_kind}" == "shard" && \
        "${RUN_MODE}" != "legacy-v2-resume" ]]; then
    initialize_dynamic_shard_queue
    ACTIVE_WORKER_PIDS=()
    for ((worker_id = 0; worker_id < WORKER_COUNT; worker_id++)); do
      (
        trap - EXIT TERM INT HUP
        close_run_lock_fd_for_child
        mark_worker_exit() {
          local worker_status=$?
          if ((worker_status != 0)); then
            : >"${failure_file}"
          fi
        }
        stop_worker() {
          local exit_status="$1"
          trap - TERM INT HUP
          if [[ -n "${ACTIVE_BATCH_PID}" ]]; then
            terminate_process_group "${ACTIVE_BATCH_PID}" \
              "Vita batch shard:worker-${worker_id}"
            ACTIVE_BATCH_PID=""
          fi
          exit "${exit_status}"
        }
        trap mark_worker_exit EXIT
        trap 'stop_worker 143' TERM
        trap 'stop_worker 130' INT
        trap 'stop_worker 129' HUP
        run_dynamic_shard_worker "${worker_id}" "${failure_file}"
      ) &
      ACTIVE_WORKER_PIDS+=("$!")
      worker_labels+=("worker-${worker_id}")
    done
  elif [[ "${SUITE_PARALLELISM}" == "4" ]]; then
    ACTIVE_WORKER_PIDS=()
    for index in "${!SUITE_NAMES[@]}"; do
      suite="${SUITE_NAMES[index]}"
      worker_id="${index}"
      (
        trap - EXIT TERM INT HUP
        close_run_lock_fd_for_child
        mark_worker_exit() {
          local worker_status=$?
          if ((worker_status != 0)); then
            : >"${failure_file}"
          fi
        }
        stop_worker() {
          local exit_status="$1"
          trap - TERM INT HUP
          if [[ -n "${ACTIVE_BATCH_PID}" ]]; then
            terminate_process_group "${ACTIVE_BATCH_PID}" \
              "Vita batch ${requested_kind}:${suite}"
            ACTIVE_BATCH_PID=""
          fi
          exit "${exit_status}"
        }
        trap mark_worker_exit EXIT
        trap 'stop_worker 143' TERM
        trap 'stop_worker 130' INT
        trap 'stop_worker 129' HUP
        run_suite_stage "${requested_kind}" "${suite}" "${failure_file}" "${worker_id}"
      ) &
      ACTIVE_WORKER_PIDS+=("$!")
      worker_labels+=("worker-${worker_id}:${suite}")
    done
  else
    for suite in "${SUITE_NAMES[@]}"; do
      run_suite_stage "${requested_kind}" "${suite}" "${failure_file}" 0
    done
  fi

  if (( ${#ACTIVE_WORKER_PIDS[@]} > 0 )); then
    for index in "${!ACTIVE_WORKER_PIDS[@]}"; do
      pid="${ACTIVE_WORKER_PIDS[index]}"
      if wait "${pid}"; then
        log "WORKER_JOIN_OK stage=${requested_kind} worker=${worker_labels[index]} pid=${pid}"
      else
        status=$?
        failed=1
        log "WORKER_JOIN_FAILED stage=${requested_kind} worker=${worker_labels[index]} pid=${pid} status=${status}"
      fi
    done
    ACTIVE_WORKER_PIDS=()
    ((failed == 0)) || die "one or more ${requested_kind} workers failed"
  fi
  unlink "${failure_file}" 2>/dev/null || true
  stage_duration_seconds=$(($(date +%s) - stage_started_epoch))
  log "STAGE_OK stage=${requested_kind} run_mode=${RUN_MODE} start_time=${stage_started_at} duration_seconds=${stage_duration_seconds}"
}

if [[ "${RUN_MODE}" == "legacy-v2-resume" ]]; then
  log "resuming the schema-5 artifacts-v2 shard stage without rerunning its completed gate"
else
  log "running the four-suite one-task x four-trial capability gate"
  run_stage gate

  verify_locked_inputs
  MANIFEST="${MANIFEST}" ARTIFACT_DIR="${ARTIFACT_DIR}" "${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from vita.utils.utils import evaluator_extracter

artifact_dir = Path(os.environ["ARTIFACT_DIR"])
manifest = json.loads(Path(os.environ["MANIFEST"]).read_text(encoding="utf-8"))
simulation_count = 0
judge_window_count = 0
usage_count = 0
valid_rubric_window_count = 0
suite_judge_windows = {}
premature_reasons = {"max_steps", "too_many_errors", "invalid_agent_message"}

for gate in manifest["gate"]:
    result_path = artifact_dir / gate["result_file"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    simulations = result.get("simulations") or []
    simulation_count += len(simulations)
    suite_judge_window_count = 0
    for simulation in simulations:
        reward_info = simulation.get("reward_info") or {}
        windows = reward_info.get("window_evaluations") or []
        if windows and simulation.get("termination_reason") in premature_reasons:
            raise SystemExit(f"premature gate simulation unexpectedly has judge windows: {result_path}")
        for window in windows:
            content = window.get("assistant_message_content")
            if not isinstance(content, str) or not content.strip():
                raise SystemExit(f"gate judge window has no valid output: {result_path}")
            parsed_rubrics = evaluator_extracter(content)
            expected_rubrics = window.get("expected_rubric_count")
            validated_rubrics = window.get("validated_rubric_count")
            if (
                not isinstance(expected_rubrics, int)
                or expected_rubrics < 1
                or validated_rubrics != expected_rubrics
                or not isinstance(parsed_rubrics, list)
                or len(parsed_rubrics) != expected_rubrics
            ):
                raise SystemExit(f"gate judge window output is not parseable rubric JSON: {result_path}")
            if all(
                isinstance(rubric, dict)
                and isinstance(rubric.get("meetExpectation"), bool)
                for rubric in parsed_rubrics
            ):
                valid_rubric_window_count += 1
            else:
                raise SystemExit(f"gate judge window has an invalid rubric decision: {result_path}")
            judge_window_count += 1
            suite_judge_window_count += 1
            if window.get("assistent_message_usage"):
                usage_count += 1
    suite_judge_windows[gate["name"]] = suite_judge_window_count

if simulation_count != 16:
    raise SystemExit(f"four-suite gate expected 16 simulations, found {simulation_count}")
if judge_window_count == 0:
    raise SystemExit("all gate simulations terminated without exercising the sliding-window judge")
if valid_rubric_window_count == 0:
    raise SystemExit("the gate judge never returned a parsed meetExpectation decision")
if manifest.get("local_role_serving"):
    worker_endpoints = {
        worker["worker_id"]: worker
        for worker in manifest["execution"]["worker_endpoints"]
    }
    role_models = {
        "agent": manifest["agent"]["requested"],
        "user": manifest["user"]["requested"],
        "evaluator": manifest["evaluator"]["requested"],
    }
    role_port_fields = {
        "agent": "agent_port",
        "user": "user_port",
        "evaluator": "evaluator_port",
    }
    endpoint_journal_counts = {}
    for gate in manifest["gate"]:
        suite = gate["name"]
        worker_id = gate["worker_id"]
        if suite_judge_windows.get(suite, 0) == 0:
            raise SystemExit(
                f"local gate never exercised evaluator-{worker_id} through {suite}"
            )
        runner_context = f"gate:{suite}:0:worker-{worker_id}"
        role_counts = {role: 0 for role in role_models}
        for journal_path in (artifact_dir / "api_journal" / suite).rglob("*.json"):
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if (journal.get("context") or {}).get("runner_context") != runner_context:
                continue
            for role, requested_model in role_models.items():
                expected_endpoint = (
                    f"http://127.0.0.1:"
                    f"{worker_endpoints[worker_id][role_port_fields[role]]}"
                    "/v1/chat/completions"
                )
                if (
                    journal.get("schema") == "vita-llm-response-journal/v1"
                    and journal.get("requested_model") == requested_model
                    and journal.get("response_model") == requested_model
                    and journal.get("endpoint") == expected_endpoint
                ):
                    role_counts[role] += 1
        missing_roles = [role for role, count in role_counts.items() if count == 0]
        if missing_roles:
            raise SystemExit(
                f"local gate worker {worker_id} has no valid journals for {missing_roles}"
            )
        endpoint_journal_counts[f"worker-{worker_id}:{suite}"] = role_counts
print(
    "[vitabench-full] FOUR_SUITE_JUDGE_GATE_OK",
    f"simulations={simulation_count}",
    f"judge_windows={judge_window_count}",
    f"valid_rubric_windows={valid_rubric_window_count}",
    f"windows_with_usage={usage_count}",
    f"suite_judge_windows={json.dumps(suite_judge_windows, sort_keys=True)}",
    f"endpoint_journals={json.dumps(endpoint_journal_counts, sort_keys=True) if manifest.get('local_role_serving') else 'remote'}",
)
PY
  log "FOUR_SUITE_GATE_OK trajectories=16 max_steps=${MAX_STEPS}"
  export_sft_archive gate
  if [[ "${RUN_MODE}" == "gate" ]]; then
    log "VITABENCH_GATE_ONLY_OK trajectories=16 artifact_dir=${ARTIFACT_DIR}"
    exit 0
  fi
fi

if [[ "${RUN_MODE}" == "ab" ]]; then
  log "running one fixed 10-task shard per suite for A/B performance validation index=${AB_SHARD_INDEX}"
elif [[ "${RUN_MODE}" == "legacy-v2-resume" ]]; then
  log "checking all legacy suite shards and running only incomplete artifacts"
else
  log "running 40 fixed full-evaluation shards (400 tasks x 4 trials)"
fi
run_stage shard

if [[ "${RUN_MODE}" == "ab" ]]; then
  validated_ab_shards=0
  while IFS=$'\t' read -r \
    kind suite domain task_set shard_index result_relative task_ids_text; do
    [[ "${kind}" == "shard" && "${shard_index}" == "${AB_SHARD_INDEX}" ]] || continue
    read -r -a task_ids <<<"${task_ids_text}"
    "${VENV_PYTHON}" "${SUMMARIZER}" check-shard \
      --file "${ARTIFACT_DIR}/${result_relative}" \
      --task-ids "${task_ids[@]}" \
      --num-trials "${NUM_TRIALS}" \
      --base-seed "${SEED}" \
      --expected-domain "${domain}" \
      --expected-max-steps "${MAX_STEPS}" \
      --expected-max-errors "${MAX_ERRORS}" \
      --expected-agent-model "${MODEL_NAME}" \
      --expected-user-model "${USER_MODEL}"
    validated_ab_shards=$((validated_ab_shards + 1))
  done <"${RUN_PLAN}"
  [[ "${validated_ab_shards}" == "4" ]] || \
    die "A/B mode expected four validated suite shards; found ${validated_ab_shards}"
  export_sft_archive ab
  log "VITABENCH_AB_SHARDS_OK suites=4 tasks=40 trajectories=160 shard_index=${AB_SHARD_INDEX} artifact_dir=${ARTIFACT_DIR}"
  exit 0
fi

log "validating the rectangular 400-task x four-trial result and computing metrics"
verify_locked_inputs
"${VENV_PYTHON}" "${SUMMARIZER}" summarize \
  --artifact-dir "${ARTIFACT_DIR}" \
  --manifest "${MANIFEST}" \
  --output-json "${SUMMARY_JSON}" \
  --output-csv "${SUMMARY_CSV}"
export_sft_archive full

log "scanning artifacts and experiment logs for exact credential leakage"
ARTIFACT_DIR="${ARTIFACT_DIR}" PROJECT_ROOT="${PROJECT_ROOT}" OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-}" \
  "${VENV_PYTHON}" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ARTIFACT_DIR"])
project_root = Path(os.environ["PROJECT_ROOT"])
output_root_value = os.environ.get("OUTPUT_ROOT_VALUE", "")
if output_root_value:
    output_root = Path(output_root_value)
    if not output_root.is_absolute():
        output_root = project_root / output_root
else:
    output_root = root.parent

scan_roots = [root]
try:
    root.resolve().relative_to(output_root.resolve())
except ValueError:
    pass
else:
    scan_roots = [output_root]

secrets = [
    value.encode()
    for value in (os.environ.get("VITA_API_KEY", ""),)
    if len(value) >= 8
]
for scan_root in scan_roots:
    for path in sorted(scan_root.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            overlap = b""
            while chunk := stream.read(1024 * 1024):
                haystack = overlap + chunk
                if any(secret in haystack for secret in secrets):
                    raise SystemExit(f"credential leaked into output file: {path}")
                overlap_size = max((len(secret) for secret in secrets), default=1) - 1
                overlap = haystack[-overlap_size:] if overlap_size else b""
print("[vitabench-full] SECRET_SCAN_OK")
PY

log "VITABENCH_FULL_EVAL_OK tasks=400 trials=4 trajectories=1600 summary=${SUMMARY_JSON}"
