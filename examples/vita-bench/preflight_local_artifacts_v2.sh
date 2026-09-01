#!/usr/bin/env bash

set -euo pipefail
umask 077

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime}"
VENV_PYTHON="/tmp/serviceagent-vitabench-venv/bin/python"
ARTIFACT_DIR="${PROJECT_ROOT}/output/experiments/vitabench-qwen35-local-role-eval/artifacts-v2"
MANIFEST="${ARTIFACT_DIR}/manifest.json"
SFT_DIR="${ARTIFACT_DIR}/sft"

[[ -x "${VENV_PYTHON}" ]] || { echo "VitaBench Python is unavailable" >&2; exit 1; }
[[ -f "${MANIFEST}" ]] || { echo "schema-5 manifest is unavailable" >&2; exit 1; }

exec {lock_fd}>"${ARTIFACT_DIR}/.run.lock"
flock -n "${lock_fd}" || { echo "artifact directory is owned by another job" >&2; exit 1; }

python3 "${PROJECT_ROOT}/tests/test_vitabench_artifact_schema.py"
MANIFEST="${MANIFEST}" PROJECT_ROOT="${PROJECT_ROOT}" python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["MANIFEST"]).read_text())
software = manifest["software"]
protocol_keys = (
    "runner_protocol_version",
    "runner_sha256",
    "model_config_sha256",
    "summarizer_sha256",
    "sft_exporter_sha256",
    "vitabench_source_sha256",
    "vitabench_task_data_sha256",
    "vitabench_pyproject_sha256",
)
protocol_state = {key: software[key] for key in protocol_keys}
bundle = hashlib.sha256(
    json.dumps(protocol_state, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if bundle != software.get("protocol_bundle_sha256"):
    raise SystemExit("manifest protocol bundle is inconsistent")
root = Path(os.environ["PROJECT_ROOT"])
local_files = {
    "runner_sha256": root / "examples/vita-bench/run_qwen3_5_4b_full_eval.sh",
    "model_config_sha256": root / "examples/vita-bench/models_qwen35_full.yaml",
    "summarizer_sha256": root / "examples/vita-bench/summarize_full_eval.py",
    "sft_exporter_sha256": root / "examples/vita-bench/export_sft_archive.py",
}
for field, path in local_files.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != software.get(field):
        raise SystemExit(f"manifest {field} does not match {path}")
print("SCHEMA5_MANIFEST_PROTOCOL_LOCK_OK", bundle)
PY
"${VENV_PYTHON}" "${PROJECT_ROOT}/examples/vita-bench/export_sft_archive.py" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --manifest "${MANIFEST}" \
  --output-dir "${SFT_DIR}"

SFT_DIR="${SFT_DIR}" "${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

index = json.loads((Path(os.environ["SFT_DIR"]) / "index.json").read_text())
counts = index.get("counts") or {}
if index.get("schema") != "vita-sft-archive/v1":
    raise SystemExit("unexpected SFT archive schema")
if counts.get("journal_files", 0) < 1 or counts.get("examples", 0) < 1:
    raise SystemExit("gate journals did not produce SFT examples")
if counts.get("journal_rejected") != 0:
    raise SystemExit("gate SFT archive rejected API journals")
if counts.get("journal_accepted") != counts.get("journal_ids"):
    raise SystemExit("not every unique gate journal was accepted")
print("SCHEMA5_GATE_SFT_EXPORT_OK", json.dumps(counts, sort_keys=True))
PY
