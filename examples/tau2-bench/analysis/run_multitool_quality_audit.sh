#!/usr/bin/env bash

set -euo pipefail
umask 077

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
VITABENCH_DIR="${SERVICE_AGENT_ROOT}/vitabench"
VENV_PYTHON="${SFT_MULTITOOL_PYTHON:-/tmp/serviceagent-vitabench-venv/bin/python}"
MODEL_PATH="${SFT_MULTITOOL_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
MODEL_NAME="${SFT_MULTITOOL_MODEL_NAME:-Qwen3.6-27B}"
SOURCE_DATA="${SFT_MULTITOOL_SOURCE_DATA:-${SERVICE_AGENT_ROOT}/datasets/AReaL-tau2-data/tau2_sft_train.jsonl}"
TRAINING_DATA="${SFT_MULTITOOL_TRAINING_DATA:-${PROJECT_ROOT}/output/datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking_max8192.jsonl}"
TAU2_ROOT="${TAU2_ROOT:-${SERVICE_AGENT_ROOT}/tau2-bench}"
AUDIT_OUT="${SFT_MULTITOOL_OUT:-${PROJECT_ROOT}/output/experiments/tau2-sft-multitool-quality-audit}"
ENV_FILE="${SFT_MULTITOOL_ENV_FILE:-${TAU2_ROOT}/.env}"
STAGE="${SFT_MULTITOOL_STAGE:-all}"
HOST="${SFT_MULTITOOL_HOST:-127.0.0.1}"
PORT="${SFT_MULTITOOL_PORT:-30101}"
TP="${SFT_MULTITOOL_TP:-1}"
CONTEXT_LENGTH="${SFT_MULTITOOL_CONTEXT_LENGTH:-32768}"
MAX_RUNNING="${SFT_MULTITOOL_MAX_RUNNING_REQUESTS:-8}"
CONCURRENCY="${SFT_MULTITOOL_CONCURRENCY:-8}"
FALLBACK_CONCURRENCY="${SFT_MULTITOOL_FALLBACK_CONCURRENCY:-2}"
FALLBACK_TRANSPORT_RETRIES="${SFT_MULTITOOL_FALLBACK_TRANSPORT_RETRIES:-5}"
ENABLE_THINKING="${SFT_MULTITOOL_ENABLE_THINKING:-false}"
MAX_TOKENS="${SFT_MULTITOOL_MAX_TOKENS:-2048}"
FALLBACK_MAX_TOKENS="${SFT_MULTITOOL_FALLBACK_MAX_TOKENS:-2048}"
LONG_FALLBACK_MAX_TOKENS="${SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_MAX_TOKENS:-4096}"
SERVER_LOG="${AUDIT_OUT}/sglang_${MODEL_NAME}.log"
SERVER_PID=""

log() { echo "[tau2-sft-multitool] $*"; }
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ $# -ge 2 ]] || die "--stage requires a value"
      STAGE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--stage all|prepare|calibrate|quality|failure|summarize]"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

RUN_PREPARE=0
RUN_QUALITY_CALIBRATION=0
RUN_FAILURE_CALIBRATION=0
RUN_QUALITY=0
RUN_FAILURE=0
RUN_SUMMARIZE=0
case "${STAGE}" in
  all)
    RUN_PREPARE=1
    RUN_QUALITY_CALIBRATION=1
    RUN_FAILURE_CALIBRATION=1
    RUN_QUALITY=1
    RUN_FAILURE=1
    RUN_SUMMARIZE=1
    ;;
  prepare) RUN_PREPARE=1 ;;
  calibrate)
    RUN_QUALITY_CALIBRATION=1
    RUN_FAILURE_CALIBRATION=1
    ;;
  quality)
    RUN_QUALITY_CALIBRATION=1
    RUN_QUALITY=1
    ;;
  failure)
    RUN_FAILURE_CALIBRATION=1
    RUN_FAILURE=1
    ;;
  summarize) RUN_SUMMARIZE=1 ;;
  *) die "SFT_MULTITOOL_STAGE must be all, prepare, calibrate, quality, failure, or summarize" ;;
esac

[[ "${ENABLE_THINKING}" == "true" || "${ENABLE_THINKING}" == "false" ]] || \
  die "SFT_MULTITOOL_ENABLE_THINKING must be true or false"
if [[ "${RUN_PREPARE}" == "1" ]]; then
  [[ -f "${SOURCE_DATA}" ]] || die "source SFT data not found: ${SOURCE_DATA}"
  [[ -f "${TRAINING_DATA}" ]] || die "actual training data not found: ${TRAINING_DATA}"
  [[ -d "${TAU2_ROOT}" ]] || die "tau2 root not found: ${TAU2_ROOT}"
  [[ -d "${MODEL_PATH}" ]] || die "judge model/tokenizer not found: ${MODEL_PATH}"
elif [[ "${RUN_QUALITY_CALIBRATION}" == "1" || "${RUN_FAILURE_CALIBRATION}" == "1" || "${RUN_QUALITY}" == "1" || "${RUN_FAILURE}" == "1" ]]; then
  [[ -d "${MODEL_PATH}" ]] || die "judge model not found: ${MODEL_PATH}"
fi
if [[ ! -x "${VENV_PYTHON}" ]]; then
  log "creating the Qwen3.6-compatible VitaBench environment"
  bash "${SERVICE_AGENT_ROOT}/setup/setup_vitabench_env.sh" "${VITABENCH_DIR}"
fi
[[ -x "${VENV_PYTHON}" ]] || die "Python environment unavailable: ${VENV_PYTHON}"
mkdir -p "${AUDIT_OUT}"

log "running CPU tests"
"${VENV_PYTHON}" "${PROJECT_ROOT}/tests/test_tau2_sft_quality.py"

if [[ "${RUN_PREPARE}" == "1" ]]; then
  log "preparing full-source dependency-safe-multi audit views"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/audit_sft_multitool.py" prepare \
    --source "${SOURCE_DATA}" \
    --training "${TRAINING_DATA}" \
    --tau2-root "${TAU2_ROOT}" \
    --protocol-profile dependency-safe-multi \
    --local-context-limit "${CONTEXT_LENGTH}" \
    --judge-max-tokens "${MAX_TOKENS}" \
    --tokenizer "${MODEL_PATH}" \
    --runner-script "${BASH_SOURCE[0]}" \
    --out "${AUDIT_OUT}"
fi

if [[ "${STAGE}" == "prepare" ]]; then
  log "prepare stage complete; no judge server was started"
  exit 0
fi

FALLBACK_BASE_URL="${SFT_MULTITOOL_FALLBACK_BASE_URL:-}"
FALLBACK_MODEL="${SFT_MULTITOOL_FALLBACK_MODEL:-}"
FALLBACK_KEY_ENV="${SFT_MULTITOOL_FALLBACK_API_KEY_ENV:-}"
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
    print("https://openrouter.ai/api/v1\tgpt-5.6-luna\tOPENROUTER_API_KEY")
elif usable("OPENAI_API_KEY") and values.get("OPENAI_API_BASE"):
    print(f"{values['OPENAI_API_BASE'].rstrip('/')}\tgpt-5.6-luna\tOPENAI_API_KEY")
PY
  } 2>/dev/null)"
  if [[ -n "${AUTO_FALLBACK}" ]]; then
    IFS=$'\t' read -r FALLBACK_BASE_URL AUTO_MODEL AUTO_KEY_ENV <<<"${AUTO_FALLBACK}"
    FALLBACK_MODEL="${FALLBACK_MODEL:-${AUTO_MODEL}}"
    FALLBACK_KEY_ENV="${FALLBACK_KEY_ENV:-${AUTO_KEY_ENV}}"
  fi
fi

FALLBACK_MODEL="${FALLBACK_MODEL:-gpt-5.6-luna}"
FALLBACK_KEY_ENV="${FALLBACK_KEY_ENV:-OPENROUTER_API_KEY}"
LONG_FALLBACK_BASE_URL="${SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_BASE_URL:-${FALLBACK_BASE_URL}}"
LONG_FALLBACK_MODEL="${SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_MODEL:-${FALLBACK_MODEL}}"
LONG_FALLBACK_KEY_ENV="${SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_API_KEY_ENV:-${FALLBACK_KEY_ENV}}"

if [[ "${RUN_QUALITY_CALIBRATION}" == "1" || "${RUN_FAILURE_CALIBRATION}" == "1" || "${RUN_QUALITY}" == "1" || "${RUN_FAILURE}" == "1" ]]; then
  [[ -n "${FALLBACK_BASE_URL}" ]] || \
    die "dual review requires SFT_MULTITOOL_FALLBACK_BASE_URL (or usable credentials in ${ENV_FILE}); use SFT_MULTITOOL_STAGE=prepare to stop before judging"
  [[ -n "${LONG_FALLBACK_BASE_URL}" ]] || \
    die "long-context review requires SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_BASE_URL"

  log "starting local open-source judge ${MODEL_NAME} (context=${CONTEXT_LENGTH}, thinking=${ENABLE_THINKING})"
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
      --mem-fraction-static "${SFT_MULTITOOL_MEM_FRACTION:-0.90}" \
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
fi

JUDGE_EXTRA_BODY="{\"chat_template_kwargs\":{\"enable_thinking\":${ENABLE_THINKING}}}"
LOCAL_JUDGE_ARGS=(
  --base-url "http://${HOST}:${PORT}/v1"
  --model "${MODEL_NAME}"
  --extra-body-json "${JUDGE_EXTRA_BODY}"
  --env-file "${ENV_FILE}"
  --concurrency "${CONCURRENCY}"
  --max-tokens "${MAX_TOKENS}"
  --protocol-profile dependency-safe-multi
)
FALLBACK_ARGS=(
  --concurrency "${FALLBACK_CONCURRENCY}"
  --transport-retries "${FALLBACK_TRANSPORT_RETRIES}"
  --fallback-base-url "${FALLBACK_BASE_URL}"
  --fallback-model "${FALLBACK_MODEL}"
  --fallback-api-key-env "${FALLBACK_KEY_ENV}"
  --fallback-max-tokens "${FALLBACK_MAX_TOKENS}"
)
LONG_FALLBACK_ARGS=(
  --concurrency "${FALLBACK_CONCURRENCY}"
  --transport-retries "${FALLBACK_TRANSPORT_RETRIES}"
  --fallback-base-url "${LONG_FALLBACK_BASE_URL}"
  --fallback-model "${LONG_FALLBACK_MODEL}"
  --fallback-api-key-env "${LONG_FALLBACK_KEY_ENV}"
  --fallback-max-tokens "${LONG_FALLBACK_MAX_TOKENS}"
)
if [[ -n "${SFT_MULTITOOL_FALLBACK_EXTRA_BODY_JSON:-}" ]]; then
  FALLBACK_ARGS+=(--fallback-extra-body-json "${SFT_MULTITOOL_FALLBACK_EXTRA_BODY_JSON}")
fi
if [[ -n "${SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_EXTRA_BODY_JSON:-}" ]]; then
  LONG_FALLBACK_ARGS+=(
    --fallback-extra-body-json "${SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_EXTRA_BODY_JSON}"
  )
fi

run_review_partitions() {
  local input_stem="$1"
  local output_stem="$2"
  local mode="$3"
  local local_full="${AUDIT_OUT}/${input_stem}_judge_payloads_local.jsonl"
  local local_windows="${AUDIT_OUT}/${input_stem}_local_windows.jsonl"
  local long_full="${AUDIT_OUT}/${input_stem}_judge_payloads_long.jsonl"
  local local_reviews="${AUDIT_OUT}/${output_stem}_local_reviews.jsonl"
  local fallback_reviews="${AUDIT_OUT}/${output_stem}_fallback_reviews.jsonl"
  local scope_local_full="${AUDIT_OUT}/${output_stem}_judge_payloads_local.jsonl"
  local scope_local_windows="${AUDIT_OUT}/${output_stem}_local_windows.jsonl"
  local scope_long_full="${AUDIT_OUT}/${output_stem}_judge_payloads_long.jsonl"

  for input in "${local_full}" "${local_windows}" "${long_full}"; do
    [[ -e "${input}" ]] || die "${input_stem} input not found: ${input}; run the prepare stage first"
  done
  for input in "${scope_local_full}" "${scope_local_windows}" "${scope_long_full}"; do
    [[ -e "${input}" ]] || die "${output_stem} cache scope not found: ${input}; run the prepare stage first"
  done

  local cache_scope_args=(
    --cache-scope-input "${scope_local_full}"
    --cache-scope-input "${scope_local_windows}"
  )
  local fallback_cache_scope_args=(
    --cache-scope-input "${scope_local_full}"
    --cache-scope-input "${scope_long_full}"
  )

  log "${input_stem} local full-trajectory review"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/judge_sft_quality.py" \
      --input "${local_full}" \
      --output "${local_reviews}" \
      --review-mode "${mode}" \
      --primary-passes 1 \
      "${cache_scope_args[@]}" \
      "${LOCAL_JUDGE_ARGS[@]}"

  log "${input_stem} local structured-window review for long trajectories"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/judge_sft_quality.py" \
      --input "${local_windows}" \
      --output "${local_reviews}" \
      --review-mode "${mode}" \
      --primary-passes 1 \
      "${cache_scope_args[@]}" \
      "${LOCAL_JUDGE_ARGS[@]}"

  log "${input_stem} independent fallback full-trajectory review"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/judge_sft_quality.py" \
      --input "${local_full}" \
      --output "${fallback_reviews}" \
      --review-mode "${mode}" \
      --force-fallback \
      "${fallback_cache_scope_args[@]}" \
      "${LOCAL_JUDGE_ARGS[@]}" \
      "${FALLBACK_ARGS[@]}"

  log "${input_stem} long-context fallback review (full trajectory, no truncation)"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/judge_sft_quality.py" \
      --input "${long_full}" \
      --output "${fallback_reviews}" \
      --review-mode "${mode}" \
      --force-fallback \
      "${fallback_cache_scope_args[@]}" \
      "${LOCAL_JUDGE_ARGS[@]}" \
      "${LONG_FALLBACK_ARGS[@]}"
}

if [[ "${RUN_QUALITY_CALIBRATION}" == "1" ]]; then
  run_review_partitions calibration_quality quality quality
  log "validating the 72-dialog quality calibration gate"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/audit_sft_multitool.py" calibrate \
    --features "${AUDIT_OUT}/features.jsonl" \
    --inventory "${AUDIT_OUT}/inventory.json" \
    --prepare-manifest "${AUDIT_OUT}/prepare_manifest.json" \
    --selection "${AUDIT_OUT}/comparison_selection.jsonl" \
    --payload-manifest "${AUDIT_OUT}/payload_manifest.jsonl" \
    --calibration "${AUDIT_OUT}/calibration_selection.jsonl" \
    --quality-local-reviews "${AUDIT_OUT}/quality_local_reviews.jsonl" \
    --quality-fallback-reviews "${AUDIT_OUT}/quality_fallback_reviews.jsonl" \
    --require-pass \
    --out "${AUDIT_OUT}"
fi

if [[ "${RUN_FAILURE_CALIBRATION}" == "1" ]]; then
  run_review_partitions calibration_failure failure failure
  log "validating the sealed 72-dialog failure-attribution calibration gate"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/audit_sft_multitool.py" calibrate-failure \
    --features "${AUDIT_OUT}/features.jsonl" \
    --inventory "${AUDIT_OUT}/inventory.json" \
    --prepare-manifest "${AUDIT_OUT}/prepare_manifest.json" \
    --payload-manifest "${AUDIT_OUT}/payload_manifest.jsonl" \
    --failure-calibration "${AUDIT_OUT}/failure_calibration_selection.jsonl" \
    --failure-local-reviews "${AUDIT_OUT}/failure_local_reviews.jsonl" \
    --failure-fallback-reviews "${AUDIT_OUT}/failure_fallback_reviews.jsonl" \
    --require-pass \
    --out "${AUDIT_OUT}"
fi

if [[ "${RUN_QUALITY}" == "1" ]]; then
  log "quality calibration passed; starting the full quality review"
  run_review_partitions quality quality quality
fi
if [[ "${RUN_FAILURE}" == "1" ]]; then
  log "failure calibration passed; starting the full failure review"
  run_review_partitions failure failure failure
fi

if [[ "${RUN_SUMMARIZE}" == "1" ]]; then
  log "summarizing comparison and failure attribution; no training JSONL is emitted"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/audit_sft_multitool.py" summarize \
    --features "${AUDIT_OUT}/features.jsonl" \
    --inventory "${AUDIT_OUT}/inventory.json" \
    --prepare-manifest "${AUDIT_OUT}/prepare_manifest.json" \
    --selection "${AUDIT_OUT}/comparison_selection.jsonl" \
    --pairs "${AUDIT_OUT}/comparison_pairs.jsonl" \
    --payload-manifest "${AUDIT_OUT}/payload_manifest.jsonl" \
    --calibration "${AUDIT_OUT}/calibration_selection.jsonl" \
    --failure-calibration "${AUDIT_OUT}/failure_calibration_selection.jsonl" \
    --quality-local-reviews "${AUDIT_OUT}/quality_local_reviews.jsonl" \
    --quality-fallback-reviews "${AUDIT_OUT}/quality_fallback_reviews.jsonl" \
    --failure-local-reviews "${AUDIT_OUT}/failure_local_reviews.jsonl" \
    --failure-fallback-reviews "${AUDIT_OUT}/failure_fallback_reviews.jsonl" \
    --out "${AUDIT_OUT}"
fi

log "stage ${STAGE} complete: ${AUDIT_OUT}"
