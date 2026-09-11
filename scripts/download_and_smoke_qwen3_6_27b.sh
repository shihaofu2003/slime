#!/usr/bin/env bash

set -euo pipefail
umask 077

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
MODEL_PATH="${VITA_ROLE_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
STAGING_PATH="${MODEL_PATH}.download"
LOCK_PATH="${MODEL_PATH}.download.lock"
VENV_PYTHON="/tmp/serviceagent-vitabench-venv/bin/python"
VITABENCH_DIR="${SERVICE_AGENT_ROOT}/vitabench"
HOST=127.0.0.1
PORT=32000
SERVER_LOG="${MODEL_PATH}.smoke_sglang.log"
SERVER_PID=""

log() { echo "[qwen3.6-27b-smoke] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

if [[ ! -x "${VENV_PYTHON}" ]]; then
  bash "${SERVICE_AGENT_ROOT}/setup/setup_vitabench_env.sh" "${VITABENCH_DIR}"
fi
[[ -x "${VENV_PYTHON}" ]] || die "VitaBench Python is unavailable: ${VENV_PYTHON}"
command -v flock >/dev/null 2>&1 || die "flock is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"

cleanup() {
  trap - EXIT TERM INT HUP
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

exec {DOWNLOAD_LOCK_FD}>"${LOCK_PATH}"
flock "${DOWNLOAD_LOCK_FD}"

model_complete() {
  MODEL_PATH="${MODEL_PATH}" "${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_PATH"])
config = root / "config.json"
index = root / "model.safetensors.index.json"
if not config.is_file() or not index.is_file():
    raise SystemExit(1)
payload = json.loads(index.read_text(encoding="utf-8"))
weight_files = sorted(set((payload.get("weight_map") or {}).values()))
if not weight_files or any(not (root / name).is_file() for name in weight_files):
    raise SystemExit(1)
if any((root / name).stat().st_size <= 0 for name in weight_files):
    raise SystemExit(1)
print(f"MODEL_FILES_OK shards={len(weight_files)}")
PY
}

if ! model_complete; then
  [[ ! -e "${MODEL_PATH}" ]] || die "existing target model is incomplete: ${MODEL_PATH}"
  log "downloading Qwen/Qwen3.6-27B to ${STAGING_PATH}"
  mkdir -p "${STAGING_PATH}"
  if ! "${VENV_PYTHON}" -c 'import modelscope' 2>/dev/null; then
    "${VENV_PYTHON}" -m pip install --no-input \
      -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" modelscope
  fi
  STAGING_PATH="${STAGING_PATH}" "${VENV_PYTHON}" - <<'PY'
import os
from modelscope import snapshot_download

snapshot_download(
    model_id="Qwen/Qwen3.6-27B",
    local_dir=os.environ["STAGING_PATH"],
    revision="master",
)
PY
  if [[ -e "${MODEL_PATH}" ]]; then
    die "target appeared during download: ${MODEL_PATH}"
  fi
  mv -- "${STAGING_PATH}" "${MODEL_PATH}"
  model_complete || die "downloaded model is incomplete"
else
  log "using existing complete model at ${MODEL_PATH}"
fi
flock -u "${DOWNLOAD_LOCK_FD}"

nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

log "starting one-GPU BF16 smoke server"
setsid env \
  -u OPENAI_API_KEY \
  -u OPENAI_API_BASE \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${VENV_PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name Qwen3.6-27B-smoke \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp 1 \
    --context-length 32768 \
    --mem-fraction-static 0.90 \
    --max-running-requests 2 \
    --reasoning-parser qwen3 \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

ready=0
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 160 "${SERVER_LOG}" >&2 || true
    die "SGLang exited before becoming ready"
  fi
  if HOST="${HOST}" PORT="${PORT}" "${VENV_PYTHON}" - <<'PY' 2>/dev/null
import os
import urllib.request

urllib.request.urlopen(
    f"http://{os.environ['HOST']}:{os.environ['PORT']}/health", timeout=3
).read()
PY
  then
    ready=1
    break
  fi
  sleep 5
done
[[ "${ready}" == "1" ]] || {
  tail -n 160 "${SERVER_LOG}" >&2 || true
  die "SGLang health check timed out"
}

HOST="${HOST}" PORT="${PORT}" "${VENV_PYTHON}" - <<'PY'
import json
import os
import requests

endpoint = f"http://{os.environ['HOST']}:{os.environ['PORT']}/v1/chat/completions"

def complete(messages, enable_thinking):
    response = requests.post(
        endpoint,
        json={
            "model": "Qwen3.6-27B-smoke",
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
        timeout=(10, 300),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("model") != "Qwen3.6-27B-smoke":
        raise RuntimeError("unexpected served model id")
    content = (payload["choices"][0]["message"].get("content") or "").strip()
    if not content:
        raise RuntimeError("empty model output")
    return content

user_content = complete(
    [{"role": "user", "content": "Reply with exactly OK."}],
    enable_thinking=False,
)
if user_content != "OK":
    raise RuntimeError("non-thinking user probe did not return exactly OK")
print("LOCAL_USER_OUTPUT_OK")

judge_content = complete(
    [
        {
            "role": "system",
            "content": "Return only valid JSON with no markdown fence.",
        },
        {
            "role": "user",
            "content": (
                'Return [{"rubrics":"r1","reasoning":"brief",'
                '"meetExpectation":true}] exactly as a JSON array.'
            ),
        },
    ],
    enable_thinking=True,
)
parsed = json.loads(judge_content)
if not (
    isinstance(parsed, list)
    and len(parsed) == 1
    and parsed[0].get("rubrics") == "r1"
    and parsed[0].get("meetExpectation") is True
):
    raise RuntimeError("thinking evaluator probe returned the wrong JSON schema")
print("LOCAL_EVALUATOR_JSON_OK")
PY

log "QWEN3_6_27B_SINGLE_80GB_SMOKE_OK model=${MODEL_PATH}"
