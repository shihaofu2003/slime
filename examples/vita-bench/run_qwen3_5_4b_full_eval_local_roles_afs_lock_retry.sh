#!/usr/bin/env bash

set -euo pipefail
umask 077

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime}"
BASE_WRAPPER="${PROJECT_ROOT}/examples/vita-bench/run_qwen3_5_4b_full_eval_local_roles.sh"
BASE_RUNNER="${PROJECT_ROOT}/examples/vita-bench/run_qwen3_5_4b_full_eval.sh"
BASE_SCHEDULER="${PROJECT_ROOT}/examples/vita-bench/eval_scheduler.py"
ADAPTER_DIR="${PROJECT_ROOT}/examples/vita-bench/afs_flock_retry"
ADAPTER_PATH="${ADAPTER_DIR}/sitecustomize.py"
ADAPTER_RECORD_NAME="scheduler_runtime_adapter.json"

run_mode="${VITA_RUN_MODE:-full}"
shard_index="${VITA_AB_SHARD_INDEX:-0}"
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --run-mode)
      ((index + 1 < ${#arguments[@]})) || { echo "--run-mode requires a value" >&2; exit 2; }
      run_mode="${arguments[index + 1]}"
      ;;
    --ab-shard-index)
      ((index + 1 < ${#arguments[@]})) || { echo "--ab-shard-index requires a value" >&2; exit 2; }
      shard_index="${arguments[index + 1]}"
      ;;
  esac
done

[[ "${run_mode}" == "ab" ]] || {
  echo "the AFS lock retry wrapper is restricted to --run-mode ab" >&2
  exit 2
}
[[ "${shard_index}" =~ ^[0-9]+$ ]] && ((shard_index <= 9)) || {
  echo "--ab-shard-index must be between 0 and 9" >&2
  exit 2
}
[[ -f "${BASE_WRAPPER}" && -f "${BASE_RUNNER}" && -f "${BASE_SCHEDULER}" ]] || {
  echo "VitaBench runner inputs are incomplete" >&2
  exit 1
}
[[ -f "${ADAPTER_PATH}" ]] || { echo "AFS flock retry adapter is unavailable" >&2; exit 1; }

if [[ -n "${VITA_RESULT_DIR:-}" ]]; then
  artifact_dir="${VITA_RESULT_DIR}"
else
  artifact_dir="${PROJECT_ROOT}/output/experiments/vitabench-qwen35-local-role-eval/artifacts-v4-ab-shard${shard_index}"
fi
mkdir -p "${artifact_dir}"

ADAPTER_RECORD="${artifact_dir}/${ADAPTER_RECORD_NAME}" \
  ADAPTER_PATH="${ADAPTER_PATH}" ADAPTER_WRAPPER="$(realpath "${BASH_SOURCE[0]}")" \
  BASE_RUNNER="${BASE_RUNNER}" BASE_SCHEDULER="${BASE_SCHEDULER}" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


record_path = Path(os.environ["ADAPTER_RECORD"])
record = {
    "schema": "vitabench-afs-flock-retry/v1",
    "enabled": True,
    "scope": "eval_scheduler.py subprocesses only",
    "retry_interval_seconds": 0.05,
    "retry_timeout_seconds": 120.0,
    "retry_errno_names": ["EACCES", "EAGAIN"],
    "base_runner_sha256": digest(os.environ["BASE_RUNNER"]),
    "base_scheduler_sha256": digest(os.environ["BASE_SCHEDULER"]),
    "adapter_sha256": digest(os.environ["ADAPTER_PATH"]),
    "adapter_wrapper_sha256": digest(os.environ["ADAPTER_WRAPPER"]),
}
if record_path.is_file():
    existing = json.loads(record_path.read_text(encoding="utf-8"))
    if existing != record:
        raise SystemExit("existing AFS lock adapter record does not match this wrapper")
else:
    temporary = record_path.with_name(f".{record_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, record_path)
PY

export VITA_AFS_FLOCK_RETRY=1
export VITA_RESULT_DIR="${artifact_dir}"
export PYTHONPATH="${ADAPTER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec bash "${BASE_WRAPPER}" "${arguments[@]}"
