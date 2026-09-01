#!/usr/bin/env bash

set -euo pipefail
umask 077

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
VITABENCH_DIR="${SERVICE_AGENT_ROOT}/vitabench"
VENV_PYTHON="${TAU2_LOCAL_FILTER_PYTHON:-/tmp/serviceagent-vitabench-venv/bin/python}"
MODEL_PATH="${TAU2_LOCAL_FILTER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
MODEL_NAME="${TAU2_LOCAL_FILTER_MODEL_NAME:-Qwen3.6-27B}"
TRAINING_TOKENIZER="${TAU2_LOCAL_FILTER_TRAINING_TOKENIZER:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
SOURCE_DATA="${TAU2_LOCAL_FILTER_SOURCE_DATA:-${SERVICE_AGENT_ROOT}/datasets/AReaL-tau2-data/tau2_sft_train.jsonl}"
TRAINING_DATA="${TAU2_LOCAL_FILTER_TRAINING_DATA:-${PROJECT_ROOT}/output/datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking_max8192.jsonl}"
TAU2_ROOT="${TAU2_ROOT:-${SERVICE_AGENT_ROOT}/tau2-bench}"
REFERENCE_AUDIT="${TAU2_LOCAL_FILTER_REFERENCE_AUDIT:-${PROJECT_ROOT}/output/experiments/tau2-sft-multitool-quality-audit}"
FILTER_OUT="${TAU2_LOCAL_FILTER_OUT:-${PROJECT_ROOT}/output/experiments/tau2-sft-local-first-relaxed}"
STAGE="${TAU2_LOCAL_FILTER_STAGE:-all}"
HOST="${TAU2_LOCAL_FILTER_HOST:-127.0.0.1}"
PORT="${TAU2_LOCAL_FILTER_PORT:-30101}"
TP="${TAU2_LOCAL_FILTER_TP:-1}"
CONTEXT_LENGTH="${TAU2_LOCAL_FILTER_CONTEXT_LENGTH:-32768}"
MAX_TOKENS="${TAU2_LOCAL_FILTER_MAX_TOKENS:-2048}"
MAX_RUNNING="${TAU2_LOCAL_FILTER_MAX_RUNNING_REQUESTS:-8}"
CONCURRENCY="${TAU2_LOCAL_FILTER_CONCURRENCY:-8}"
BUDGET_ROWS="${TAU2_LOCAL_FILTER_BUDGET_ROWS:-19318}"
BUDGET_SEED="${TAU2_LOCAL_FILTER_BUDGET_SEED:-20260803}"
SERVER_LOG="${FILTER_OUT}/sglang_${MODEL_NAME}.log"
LOCAL_REVIEWS="${FILTER_OUT}/success_local_reviews.jsonl"
SERVER_PID=""

log() { echo "[tau2-local-first-filter] $*"; }
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
      echo "Usage: $0 [--stage all|prepare|review|build]"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

RUN_PREPARE=0
RUN_REVIEW=0
RUN_BUILD=0
case "${STAGE}" in
  all)
    RUN_PREPARE=1
    RUN_REVIEW=1
    RUN_BUILD=1
    ;;
  prepare) RUN_PREPARE=1 ;;
  review) RUN_REVIEW=1 ;;
  build) RUN_BUILD=1 ;;
  *) die "stage must be all, prepare, review, or build" ;;
esac

[[ -f "${SOURCE_DATA}" ]] || die "source data not found: ${SOURCE_DATA}"
[[ -f "${TRAINING_DATA}" ]] || die "training data not found: ${TRAINING_DATA}"
[[ -d "${TAU2_ROOT}" ]] || die "tau2 root not found: ${TAU2_ROOT}"
[[ -d "${REFERENCE_AUDIT}" ]] || die "reference audit not found: ${REFERENCE_AUDIT}"
if [[ "${RUN_PREPARE}" == "1" || "${RUN_REVIEW}" == "1" ]]; then
  [[ -d "${MODEL_PATH}" ]] || die "local judge model not found: ${MODEL_PATH}"
fi
if [[ ! -x "${VENV_PYTHON}" ]]; then
  log "creating the Qwen3.6-compatible VitaBench environment"
  bash "${SERVICE_AGENT_ROOT}/setup/setup_vitabench_env.sh" "${VITABENCH_DIR}"
fi
[[ -x "${VENV_PYTHON}" ]] || die "Python environment unavailable: ${VENV_PYTHON}"
mkdir -p "${FILTER_OUT}"

log "running CPU tests"
"${VENV_PYTHON}" "${PROJECT_ROOT}/tests/test_tau2_sft_quality.py"

if [[ "${RUN_PREPARE}" == "1" ]]; then
  log "preparing all consensus-success payloads for local-only review"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/build_local_first_sft.py" prepare \
    --source "${SOURCE_DATA}" \
    --training "${TRAINING_DATA}" \
    --tau2-root "${TAU2_ROOT}" \
    --tokenizer "${MODEL_PATH}" \
    --reference-audit "${REFERENCE_AUDIT}" \
    --local-context-limit "${CONTEXT_LENGTH}" \
    --judge-max-tokens "${MAX_TOKENS}" \
    --runner-script "${BASH_SOURCE[0]}" \
    --out "${FILTER_OUT}"
fi

if [[ "${RUN_REVIEW}" == "1" ]]; then
  for input in \
    "${FILTER_OUT}/success_judge_payloads_local.jsonl" \
    "${FILTER_OUT}/success_local_windows.jsonl" \
    "${FILTER_OUT}/success_payload_manifest.jsonl"; do
    [[ -f "${input}" ]] || die "prepared input not found: ${input}"
  done
  if [[ ! -f "${LOCAL_REVIEWS}" ]]; then
    log "seeding the cache with sealed local reviews from the dual-judge audit"
    cp "${REFERENCE_AUDIT}/quality_local_reviews.jsonl" "${LOCAL_REVIEWS}"
  fi

  log "starting local Qwen3.6-27B SGLang judge; no fallback/API is configured"
  setsid "${VENV_PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp "${TP}" \
    --context-length "${CONTEXT_LENGTH}" \
    --mem-fraction-static "${TAU2_LOCAL_FILTER_MEM_FRACTION:-0.90}" \
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
  [[ "${ready}" == "1" ]] || die "SGLang health check timed out"

  JUDGE_EXTRA_BODY="{\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  JUDGE_ARGS=(
    --output "${LOCAL_REVIEWS}"
    --review-mode quality
    --protocol-profile dependency-safe-multi
    --base-url "http://${HOST}:${PORT}/v1"
    --model "${MODEL_NAME}"
    --extra-body-json "${JUDGE_EXTRA_BODY}"
    --primary-passes 1
    --concurrency "${CONCURRENCY}"
    --max-tokens "${MAX_TOKENS}"
    --cache-scope-input "${FILTER_OUT}/success_judge_payloads_local.jsonl"
    --cache-scope-input "${FILTER_OUT}/success_local_windows.jsonl"
  )
  log "reviewing full local-context trajectories"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/judge_sft_quality.py" \
    --input "${FILTER_OUT}/success_judge_payloads_local.jsonl" \
    "${JUDGE_ARGS[@]}"
  log "reviewing structured evidence windows for long trajectories"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/judge_sft_quality.py" \
    --input "${FILTER_OUT}/success_local_windows.jsonl" \
    "${JUDGE_ARGS[@]}"
fi

if [[ "${RUN_BUILD}" == "1" ]]; then
  [[ -f "${LOCAL_REVIEWS}" ]] || die "local reviews not found: ${LOCAL_REVIEWS}"
  log "building target-level canonical and row-budget-matched SFT datasets"
  "${VENV_PYTHON}" "${PROJECT_ROOT}/examples/tau2-bench/analysis/build_local_first_sft.py" build \
    --source "${SOURCE_DATA}" \
    --training "${TRAINING_DATA}" \
    --tau2-root "${TAU2_ROOT}" \
    --reference-audit "${REFERENCE_AUDIT}" \
    --local-reviews "${LOCAL_REVIEWS}" \
    --training-tokenizer "${TRAINING_TOKENIZER}" \
    --expected-local-model "${MODEL_NAME}" \
    --budget-rows "${BUDGET_ROWS}" \
    --budget-seed "${BUDGET_SEED}" \
    --out "${FILTER_OUT}"
fi

log "stage ${STAGE} complete: ${FILTER_OUT}"
