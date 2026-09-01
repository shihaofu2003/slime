#!/usr/bin/env bash

set -euo pipefail
umask 077

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
VITABENCH_DIR="${SERVICE_AGENT_ROOT}/vitabench"
VENV_PYTHON="${SFT_QUALITY_PYTHON:-/tmp/serviceagent-vitabench-venv/bin/python}"
MODEL_PATH="${SFT_QUALITY_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
MODEL_NAME="${SFT_QUALITY_MODEL_NAME:-Qwen3.6-27B}"
SOURCE_DATA="${SFT_QUALITY_SOURCE_DATA:-${SERVICE_AGENT_ROOT}/datasets/AReaL-tau2-data/tau2_sft_train.jsonl}"
TRAINING_DATA="${SFT_QUALITY_TRAINING_DATA:-${PROJECT_ROOT}/output/datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking_max8192.jsonl}"
TAU2_ROOT="${TAU2_ROOT:-${SERVICE_AGENT_ROOT}/tau2-bench}"
AUDIT_OUT="${SFT_QUALITY_OUT:-${PROJECT_ROOT}/output/experiments/tau2-sft-quality-audit}"
ENV_FILE="${SFT_QUALITY_ENV_FILE:-${TAU2_ROOT}/.env}"
HOST="${SFT_QUALITY_HOST:-127.0.0.1}"
PORT="${SFT_QUALITY_PORT:-30100}"
TP="${SFT_QUALITY_TP:-1}"
CONTEXT_LENGTH="${SFT_QUALITY_CONTEXT_LENGTH:-32768}"
MAX_RUNNING="${SFT_QUALITY_MAX_RUNNING_REQUESTS:-8}"
CONCURRENCY="${SFT_QUALITY_CONCURRENCY:-8}"
ENABLE_THINKING="${SFT_QUALITY_ENABLE_THINKING:-false}"
MAX_TOKENS="${SFT_QUALITY_MAX_TOKENS:-2048}"
FALLBACK_MAX_TOKENS="${SFT_QUALITY_FALLBACK_MAX_TOKENS:-2048}"
SERVER_LOG="${AUDIT_OUT}/sglang_${MODEL_NAME}.log"
SERVER_PID=""

log() { echo "[tau2-sft-quality] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

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

[[ -f "${SOURCE_DATA}" ]] || die "source SFT data not found: ${SOURCE_DATA}"
[[ -f "${TRAINING_DATA}" ]] || die "actual training data not found: ${TRAINING_DATA}"
[[ -d "${TAU2_ROOT}" ]] || die "tau2 root not found: ${TAU2_ROOT}"
[[ -d "${MODEL_PATH}" ]] || die "judge model not found: ${MODEL_PATH}"
[[ "${ENABLE_THINKING}" == "true" || "${ENABLE_THINKING}" == "false" ]] || \
  die "SFT_QUALITY_ENABLE_THINKING must be true or false"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  log "creating the Qwen3.6-compatible VitaBench environment"
  bash "${SERVICE_AGENT_ROOT}/setup/setup_vitabench_env.sh" "${VITABENCH_DIR}"
fi
[[ -x "${VENV_PYTHON}" ]] || die "Python environment unavailable: ${VENV_PYTHON}"
mkdir -p "${AUDIT_OUT}"

log "running CPU tests"
"${VENV_PYTHON}" "${PROJECT_ROOT}/tests/test_tau2_sft_quality.py"

log "reconstructing dialogs and deterministic audit"
"${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/audit_sft_quality.py" prepare \
  --source "${SOURCE_DATA}" \
  --training "${TRAINING_DATA}" \
  --tau2-root "${TAU2_ROOT}" \
  --out "${AUDIT_OUT}"

if [[ "${SFT_QUALITY_PREPARE_ONLY:-0}" == "1" ]]; then
  log "prepare-only audit complete"
  exit 0
fi

log "starting local open-source judge ${MODEL_NAME} (thinking=${ENABLE_THINKING}, max_tokens=${MAX_TOKENS})"
setsid env \
  -u OPENAI_API_KEY \
  -u OPENAI_API_BASE \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${VENV_PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp "${TP}" \
    --context-length "${CONTEXT_LENGTH}" \
    --mem-fraction-static "${SFT_QUALITY_MEM_FRACTION:-0.90}" \
    --max-running-requests "${MAX_RUNNING}" \
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

FALLBACK_ARGS=()
FALLBACK_BASE_URL="${SFT_QUALITY_FALLBACK_BASE_URL:-}"
FALLBACK_MODEL="${SFT_QUALITY_FALLBACK_MODEL:-}"
FALLBACK_KEY_ENV="${SFT_QUALITY_FALLBACK_API_KEY_ENV:-}"
if [[ -z "${FALLBACK_BASE_URL}" && -f "${ENV_FILE}" ]]; then
  AUTO_FALLBACK="$({ ENV_FILE="${ENV_FILE}" "${VENV_PYTHON}" - <<'PY'
import os
from pathlib import Path

values = {}
for raw_line in Path(os.environ["ENV_FILE"]).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    values[key.strip()] = value.strip().strip("'\"")

def usable(key):
    value = values.get(key, "")
    normalized = value.lower()
    return bool(value) and not normalized.startswith("<") and "your_key" not in normalized

if usable("OPENROUTER_API_KEY"):
    print("https://openrouter.ai/api/v1\tgoogle/gemini-2.5-flash\tOPENROUTER_API_KEY")
elif usable("OPENAI_API_KEY") and values.get("OPENAI_API_BASE"):
    print(f"{values['OPENAI_API_BASE'].rstrip('/')}\tgemini-2.5-flash\tOPENAI_API_KEY")
PY
  } 2>/dev/null)"
  if [[ -n "${AUTO_FALLBACK}" ]]; then
    IFS=$'\t' read -r FALLBACK_BASE_URL AUTO_MODEL AUTO_KEY_ENV <<<"${AUTO_FALLBACK}"
    FALLBACK_MODEL="${FALLBACK_MODEL:-${AUTO_MODEL}}"
    FALLBACK_KEY_ENV="${FALLBACK_KEY_ENV:-${AUTO_KEY_ENV}}"
  fi
fi
if [[ -n "${FALLBACK_BASE_URL}" ]]; then
  FALLBACK_MODEL="${FALLBACK_MODEL:-google/gemini-2.5-flash}"
  FALLBACK_KEY_ENV="${FALLBACK_KEY_ENV:-OPENROUTER_API_KEY}"
  FALLBACK_ARGS+=(
    --fallback-base-url "${FALLBACK_BASE_URL}"
    --fallback-model "${FALLBACK_MODEL}"
    --fallback-api-key-env "${FALLBACK_KEY_ENV}"
  )
  log "API fallback is configured for ambiguous or failed local reviews"
else
  log "API fallback is not configured; unresolved local reviews will remain review-tier"
fi

JUDGE_EXTRA_BODY="{\"chat_template_kwargs\":{\"enable_thinking\":${ENABLE_THINKING}}}"
LOCAL_JUDGE_ARGS=(
  --base-url "http://${HOST}:${PORT}/v1"
  --model "${MODEL_NAME}"
  --extra-body-json "${JUDGE_EXTRA_BODY}"
)
SHARED_JUDGE_ARGS=(
  --env-file "${ENV_FILE}"
  --concurrency "${CONCURRENCY}"
  --max-tokens "${MAX_TOKENS}"
  --fallback-max-tokens "${FALLBACK_MAX_TOKENS}"
)

log "calibrating the local judge with two blinded passes"
"${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/judge_sft_quality.py" \
  --input "${AUDIT_OUT}/calibration_payloads.jsonl" \
  --output "${AUDIT_OUT}/calibration_reviews.jsonl" \
  --primary-passes 2 \
  "${LOCAL_JUDGE_ARGS[@]}" \
  "${SHARED_JUDGE_ARGS[@]}"

set +e
"${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/audit_sft_quality.py" calibrate \
  --sample "${AUDIT_OUT}/calibration_sample.jsonl" \
  --reviews "${AUDIT_OUT}/calibration_reviews.jsonl" \
  --output "${AUDIT_OUT}/calibration_report.json" \
  --require-pass
CALIBRATION_STATUS="$?"
set -e

FULL_FALLBACK_ARGS=()
if [[ "${CALIBRATION_STATUS}" != "0" ]]; then
  if [[ -n "${FALLBACK_BASE_URL}" ]]; then
    log "local calibration failed; escalating the full audit to the API judge"
    FULL_FALLBACK_ARGS+=(--force-fallback)
  else
    die "local calibration failed and no API fallback is configured"
  fi
else
  log "local calibration passed; keeping local-first review for the full corpus"
fi

log "reviewing all reconstructed training dialogs"
"${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/judge_sft_quality.py" \
  --input "${AUDIT_OUT}/judge_payloads.jsonl" \
  --output "${AUDIT_OUT}/judge_reviews.jsonl" \
  --primary-passes 1 \
  "${FULL_FALLBACK_ARGS[@]}" \
  "${LOCAL_JUDGE_ARGS[@]}" \
  "${SHARED_JUDGE_ARGS[@]}" \
  "${FALLBACK_ARGS[@]}"

log "applying quality gates and writing the filtered training dataset"
"${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/audit_sft_quality.py" summarize \
  --features "${AUDIT_OUT}/features.jsonl" \
  --inventory "${AUDIT_OUT}/inventory.json" \
  --reviews "${AUDIT_OUT}/judge_reviews.jsonl" \
  --training "${TRAINING_DATA}" \
  --out "${AUDIT_OUT}"

log "audit complete: ${AUDIT_OUT}/REPORT.md"
