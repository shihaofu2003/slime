#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
VITABENCH_DIR="${SERVICE_AGENT_ROOT}/vitabench"
VENV_DIR="/tmp/serviceagent-vitabench-venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
VITA_BIN="${VENV_DIR}/bin/vita"
MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B}"
MODEL_NAME="Qwen3.5-4B"
REMOTE_MODEL="gpt-4.1"
HOST="127.0.0.1"
PORT="30000"
MODEL_CONFIG="${PROJECT_ROOT}/examples/vita-bench/models_qwen35_smoke.yaml"
ENV_FILE="${TAU2_ENV_FILE:-${SERVICE_AGENT_ROOT}/tau2-bench/.env}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${VITA_RESULT_DIR:-${PROJECT_ROOT}/${OUTPUT_ROOT:-output}/artifacts}"
RESULT_FILE="${RESULT_DIR}/vitabench_qwen35_delivery_${RUN_STAMP}.json"
SGLANG_LOG="${RESULT_DIR}/sglang_qwen35_${RUN_STAMP}.log"

log() { echo "[vitabench-qwen35-smoke] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

[[ -d "${MODEL_PATH}" ]] || die "model path does not exist: ${MODEL_PATH}"
[[ -f "${MODEL_CONFIG}" ]] || die "model config does not exist: ${MODEL_CONFIG}"
[[ -f "${ENV_FILE}" ]] || die "credential env file does not exist: ${ENV_FILE}"
if [[ ! -x "${VITA_BIN}" ]]; then
  bash "${SERVICE_AGENT_ROOT}/setup/setup_vitabench_env.sh" "${VITABENCH_DIR}"
fi
mkdir -p "${RESULT_DIR}"

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

export VITA_MODEL_CONFIG_PATH="${MODEL_CONFIG}"

log "running Vita credential/redaction regression tests"
"${VENV_PYTHON}" -m pytest -q "${VITABENCH_DIR}/tests/test_llm_security.py"

SGLANG_PID=""
cleanup() {
  if [[ -n "${SGLANG_PID}" ]]; then
    log "stopping sglang pid=${SGLANG_PID}"
    kill "${SGLANG_PID}" 2>/dev/null || true
    wait "${SGLANG_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log "starting ${MODEL_NAME} with structured Qwen3.5 tool-call parsing"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${VENV_PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp 1 \
  --mem-fraction-static "${MEM_FRACTION:-0.85}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  >"${SGLANG_LOG}" 2>&1 &
SGLANG_PID=$!

log "waiting for sglang health endpoint; log=${SGLANG_LOG}"
ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    tail -n 100 "${SGLANG_LOG}" >&2 || true
    die "sglang exited before becoming ready"
  fi
  if HOST="${HOST}" PORT="${PORT}" "${VENV_PYTHON}" - <<'PY' 2>/dev/null
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
[[ "${ready}" == "1" ]] || { tail -n 100 "${SGLANG_LOG}" >&2 || true; die "sglang health check timed out"; }

log "checking OpenAI structured tool_calls from the local Qwen3.5 server"
HOST="${HOST}" PORT="${PORT}" MODEL_NAME="${MODEL_NAME}" "${VENV_PYTHON}" - <<'PY'
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
        "chat_template_kwargs": {"enable_thinking": True},
    },
    timeout=(10, 300),
)
response.raise_for_status()
message = response.json()["choices"][0]["message"]
tool_calls = message.get("tool_calls") or []
if not tool_calls or tool_calls[0]["function"]["name"] != "get_weather":
    raise RuntimeError(f"Qwen3.5 did not return the expected structured tool call: {message}")
print("[vitabench-qwen35-smoke] LOCAL_TOOL_CALL_OK name=get_weather")
PY

# Keep the remote credential out of the local SGLang process environment. It is
# only needed from this point onward for the remote user/evaluator requests.
VITA_API_KEY="$(read_dotenv OPENAI_API_KEY)"
VITA_API_BASE="$(read_dotenv OPENAI_API_BASE)"
[[ -n "${VITA_API_KEY}" ]] || die "OPENAI_API_KEY is empty in ${ENV_FILE}"
[[ -n "${VITA_API_BASE}" ]] || die "OPENAI_API_BASE is empty in ${ENV_FILE}"
export VITA_API_KEY
export VITA_API_BASE_URL="${VITA_API_BASE%/}/chat/completions"

log "checking the existing remote API through Vita's request path"
REMOTE_MODEL="${REMOTE_MODEL}" "${VENV_PYTHON}" - <<'PY'
import os

from vita.data_model.message import UserMessage
from vita.utils.llm_utils import generate

reply = generate(
    model=os.environ["REMOTE_MODEL"],
    messages=[UserMessage(role="user", content="Reply with exactly OK.")],
    max_tokens=8,
    temperature=0,
    num_retries=1,
)
if (reply.content or "").strip() != "OK":
    raise RuntimeError(f"unexpected remote API reply: {reply.content!r}")
print("[vitabench-qwen35-smoke] REMOTE_API_OK")
PY

log "running VitaBench delivery task 10711001 with Qwen3.5-4B agent"
"${VITA_BIN}" run \
  --domain delivery \
  --task-ids 10711001 \
  --agent-llm "${MODEL_NAME}" \
  --user-llm "${REMOTE_MODEL}" \
  --evaluator-llm "${REMOTE_MODEL}" \
  --enable-think \
  --language english \
  --num-trials 1 \
  --max-steps "${MAX_STEPS:-30}" \
  --max-concurrency 1 \
  --log-level INFO \
  --save-to "${RESULT_FILE}"

RESULT_FILE="${RESULT_FILE}" VITA_API_KEY="${VITA_API_KEY}" "${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RESULT_FILE"])
payload_text = path.read_text(encoding="utf-8")
if os.environ["VITA_API_KEY"] in payload_text:
    raise RuntimeError("API key leaked into the Vita result")
payload = json.loads(payload_text)
simulations = payload.get("simulations") or []
if len(simulations) != 1:
    raise RuntimeError(f"expected one simulation, found {len(simulations)}")
messages = simulations[0].get("messages") or []
agent_tool_calls = sum(
    len(message.get("tool_calls") or [])
    for message in messages
    if message.get("role") == "assistant"
)
tool_results = sum(message.get("role") == "tool" for message in messages)
if agent_tool_calls < 1 or tool_results < 1:
    raise RuntimeError(
        f"trajectory did not exercise tools: calls={agent_tool_calls}, results={tool_results}"
    )
reward_info = simulations[0].get("reward_info") or {}
print(
    "[vitabench-qwen35-smoke] VITABENCH_SMOKE_OK",
    f"task_id={simulations[0].get('task_id')}",
    f"termination={simulations[0].get('termination_reason')}",
    f"reward={reward_info.get('reward')}",
    f"agent_tool_calls={agent_tool_calls}",
    f"tool_results={tool_results}",
    f"result={path}",
)
PY
