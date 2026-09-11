#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${QWEN38_MODEL_PATH:-/mnt/afs/models/Qwen3.8-27B}"
HOST="127.0.0.1"
PORT="${QWEN38_PORT:-32038}"
SERVER_LOG="${QWEN38_SERVER_LOG:-/tmp/qwen3_8_27b_sglang_${$}.log}"
SERVER_PID=""

cleanup() {
  trap - EXIT TERM INT
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

MODEL_PATH="${MODEL_PATH}" python3 - <<'PY'
import json
import os
from pathlib import Path

import sglang
import torch
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

model_path = Path(os.environ["MODEL_PATH"])
index_path = model_path / "model.safetensors.index.json"
if not index_path.is_file():
    raise RuntimeError(f"missing model index: {index_path}")

index = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (model_path / name).is_file()]
if missing:
    raise RuntimeError(f"missing weight shards: {missing}")

config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)
print(
    "ENVIRONMENT "
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"sglang={sglang.__version__} cuda={torch.version.cuda}"
)
print(
    "MODEL_METADATA "
    f"config={type(config).__name__} model_type={config.model_type} "
    f"tokenizer={type(tokenizer).__name__} processor={type(processor).__name__} "
    f"shards={len(shards)} total_bytes={index['metadata']['total_size']}"
)
PY

nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader

setsid env \
  -u OPENAI_API_KEY \
  -u OPENAI_API_BASE \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name Qwen3.8-27B-smoke \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp 1 \
    --dtype bfloat16 \
    --context-length 32768 \
    --mem-fraction-static 0.90 \
    --max-running-requests 2 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

ready=0
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 200 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if HOST="${HOST}" PORT="${PORT}" python3 - <<'PY' 2>/dev/null
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

if [[ "${ready}" != "1" ]]; then
  tail -n 200 "${SERVER_LOG}" >&2 || true
  exit 1
fi

HOST="${HOST}" PORT="${PORT}" python3 - <<'PY'
import base64
import io
import json
import os

import requests
from PIL import Image

endpoint = f"http://{os.environ['HOST']}:{os.environ['PORT']}/v1/chat/completions"

text_response = requests.post(
    endpoint,
    json={
        "model": "Qwen3.8-27B-smoke",
        "messages": [
            {"role": "user", "content": "What is 6 multiplied by 7? Reply with only 42."}
        ],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=(10, 300),
)
text_response.raise_for_status()
text_payload = text_response.json()
text_content = (text_payload["choices"][0]["message"].get("content") or "").strip()
if text_content != "42":
    raise RuntimeError(f"unexpected text output: {text_content!r}")
print(f"TEXT_INFERENCE_OK output={text_content!r}")

xhigh_response = requests.post(
    endpoint,
    json={
        "model": "Qwen3.8-27B-smoke",
        "messages": [
            {
                "role": "user",
                "content": "What is 19 plus 23? Include 42 in the final answer.",
            }
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": 4096,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            # SGLang 0.5.13 forwards this value after request validation.
            "reasoning_effort": "xhigh",
        },
    },
    timeout=(10, 600),
)
if not xhigh_response.ok:
    raise RuntimeError(
        f"xhigh request failed: status={xhigh_response.status_code} "
        f"body={xhigh_response.text}"
    )
xhigh_payload = xhigh_response.json()
xhigh_message = xhigh_payload["choices"][0]["message"]
xhigh_reasoning = (
    xhigh_message.get("reasoning_content") or xhigh_message.get("reasoning") or ""
)
xhigh_answer = (xhigh_message.get("content") or "").strip()
if not xhigh_reasoning.strip():
    raise RuntimeError("xhigh returned no parsed reasoning content")
if "42" not in xhigh_answer:
    raise RuntimeError(f"unexpected xhigh final answer: {xhigh_answer!r}")
xhigh_usage = xhigh_payload["usage"]
print(
    "XHIGH_REASONING_OK "
    f"prompt_tokens={xhigh_usage['prompt_tokens']} "
    f"completion_tokens={xhigh_usage['completion_tokens']} "
    f"reasoning_chars={len(xhigh_reasoning)} answer={xhigh_answer!r}"
)

tool_response = requests.post(
    endpoint,
    json={
        "model": "Qwen3.8-27B-smoke",
        "messages": [
            {
                "role": "user",
                "content": "Call get_temperature for Beijing. Do not answer without using the tool.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_temperature",
                    "description": "Get the current temperature for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "required",
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=(10, 300),
)
tool_response.raise_for_status()
tool_payload = tool_response.json()
tool_calls = tool_payload["choices"][0]["message"].get("tool_calls") or []
if not tool_calls or tool_calls[0]["function"]["name"] != "get_temperature":
    raise RuntimeError(f"unexpected tool-call output: {tool_calls!r}")
arguments = json.loads(tool_calls[0]["function"]["arguments"])
if arguments.get("city", "").lower() != "beijing":
    raise RuntimeError(f"unexpected tool-call arguments: {arguments!r}")
print(f"TOOL_CALL_INFERENCE_OK name=get_temperature arguments={arguments!r}")

image = Image.new("RGB", (64, 64), color=(255, 0, 0))
buffer = io.BytesIO()
image.save(buffer, format="PNG")
image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
vision_response = requests.post(
    endpoint,
    json={
        "model": "Qwen3.8-27B-smoke",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "What is the dominant color of this image? Reply with one English lowercase word.",
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=(10, 300),
)
vision_response.raise_for_status()
vision_payload = vision_response.json()
vision_content = (vision_payload["choices"][0]["message"].get("content") or "").strip().lower()
if "red" not in vision_content:
    raise RuntimeError(f"unexpected vision output: {vision_content!r}")
print(f"VISION_INFERENCE_OK output={vision_content!r}")
PY

echo "SGLANG_STARTUP_SUMMARY"
grep -E "(Qwen3_5|model|GPU|memory|KV Cache|ready|listening)" "${SERVER_LOG}" | tail -n 80 || true
echo "QWEN3_8_27B_SINGLE_H100_SMOKE_OK model=${MODEL_PATH} context_length=32768"
