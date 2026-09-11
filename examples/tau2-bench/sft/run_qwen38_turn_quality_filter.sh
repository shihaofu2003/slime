#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_PATH="${TAU2_TURN_JUDGE_MODEL_PATH:-/mnt/afs/models/Qwen3.8-27B}"
MODEL_NAME="${TAU2_TURN_JUDGE_MODEL_NAME:-Qwen3.8-27B-turn-quality}"
OUTPUT_ROOT="${TAU2_TURN_FILTER_OUTPUT_ROOT:-${PROJECT_ROOT}/output/experiments/tau2-sft-turn-quality-qwen38-xhigh-v1}"
PYTHON_BIN="${TAU2_TURN_FILTER_PYTHON:-python3}"
HOST="127.0.0.1"
PORT_BASE="${TAU2_TURN_JUDGE_PORT_BASE:-33100}"
STAGE=""
SHARD_START=0
SHARD_COUNT=8
CONCURRENCY="${TAU2_TURN_JUDGE_CONCURRENCY:-8}"
TIMEOUT="${TAU2_TURN_JUDGE_TIMEOUT:-900}"
RETRIES="${TAU2_TURN_JUDGE_RETRIES:-1}"
declare -a SERVER_PIDS=()
declare -a JUDGE_PIDS=()

log() {
  echo "[tau2-turn-quality] $*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

cleanup() {
  trap - EXIT TERM INT HUP
  local pid
  for pid in "${JUDGE_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  for _ in $(seq 1 20); do
    local alive=0
    for pid in "${SERVER_PIDS[@]:-}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        alive=1
      fi
    done
    [[ "${alive}" == "0" ]] && break
    sleep 1
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ $# -ge 2 ]] || die "--stage requires pilot or full"
      STAGE="$2"
      shift 2
      ;;
    --shard-start)
      [[ $# -ge 2 ]] || die "--shard-start requires an integer"
      SHARD_START="$2"
      shift 2
      ;;
    --shard-count)
      [[ $# -ge 2 ]] || die "--shard-count requires an integer"
      SHARD_COUNT="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --stage pilot|full [--shard-start N --shard-count N]"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "${STAGE}" == "pilot" || "${STAGE}" == "full" ]] || die "--stage must be pilot or full"
[[ "${SHARD_START}" =~ ^[0-9]+$ ]] || die "--shard-start must be a non-negative integer"
[[ "${SHARD_COUNT}" =~ ^[1-8]$ ]] || die "--shard-count must be between 1 and 8"
[[ "${PORT_BASE}" =~ ^[0-9]+$ ]] || die "TAU2_TURN_JUDGE_PORT_BASE must be an integer"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || die "concurrency must be positive"
[[ -d "${MODEL_PATH}" ]] || die "model not found: ${MODEL_PATH}"
[[ -f "${PROJECT_ROOT}/examples/tau2-bench/sft/filter_official_native_turns.py" ]] || die "turn filter Python entrypoint is missing"
[[ -f "${OUTPUT_ROOT}/shards/manifest.json" ]] || die "prepare manifest is missing"

if [[ "${STAGE}" == "pilot" ]]; then
  SHARD_START=0
  SHARD_COUNT=1
fi

visible_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "${visible_gpus}" -ge "${SHARD_COUNT}" ]] || die "requested ${SHARD_COUNT} local shards but only ${visible_gpus} GPUs are visible"

RUNTIME_DIR="${OUTPUT_ROOT}/runtime/${STAGE}_${SHARD_START}_$((SHARD_START + SHARD_COUNT - 1))"
mkdir -p "${RUNTIME_DIR}"

log "starting ${SHARD_COUNT} TP1 Qwen3.8 judges from ${MODEL_PATH}"
for local_index in $(seq 0 $((SHARD_COUNT - 1))); do
  global_shard=$((SHARD_START + local_index))
  port=$((PORT_BASE + local_index))
  server_log="${RUNTIME_DIR}/sglang_$(printf '%02d' "${global_shard}").log"
  setsid env \
    -u OPENAI_API_KEY \
    -u OPENAI_API_BASE \
    CUDA_VISIBLE_DEVICES="${local_index}" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${MODEL_NAME}" \
      --host "${HOST}" \
      --port "${port}" \
      --tp 1 \
      --dtype bfloat16 \
      --context-length 32768 \
      --mem-fraction-static 0.90 \
      --max-running-requests "${CONCURRENCY}" \
      --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder \
      >"${server_log}" 2>&1 &
  SERVER_PIDS+=("$!")
done

for local_index in $(seq 0 $((SHARD_COUNT - 1))); do
  global_shard=$((SHARD_START + local_index))
  port=$((PORT_BASE + local_index))
  pid="${SERVER_PIDS[local_index]}"
  server_log="${RUNTIME_DIR}/sglang_$(printf '%02d' "${global_shard}").log"
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 160 "${server_log}" >&2 || true
      die "SGLang shard ${global_shard} exited before becoming ready"
    fi
    if HOST="${HOST}" PORT="${port}" "${PYTHON_BIN}" - <<'PY' 2>/dev/null
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
  [[ "${ready}" == "1" ]] || die "SGLang shard ${global_shard} health check timed out"
  log "SGLang shard ${global_shard} ready on local GPU ${local_index}, port ${port}"
done

for local_index in $(seq 0 $((SHARD_COUNT - 1))); do
  global_shard=$((SHARD_START + local_index))
  port=$((PORT_BASE + local_index))
  if [[ "${STAGE}" == "pilot" ]]; then
    input_path="${OUTPUT_ROOT}/shards/pilot.jsonl"
    output_a="${OUTPUT_ROOT}/reviews/pilot/pass_a.jsonl"
    output_b="${OUTPUT_ROOT}/reviews/pilot/pass_b.jsonl"
    judge_log="${RUNTIME_DIR}/judge_pilot.log"
  else
    input_path="${OUTPUT_ROOT}/shards/shard_$(printf '%02d' "${global_shard}").jsonl"
    output_a="${OUTPUT_ROOT}/reviews/pass_a/shard_$(printf '%02d' "${global_shard}").jsonl"
    output_b="${OUTPUT_ROOT}/reviews/pass_b/shard_$(printf '%02d' "${global_shard}").jsonl"
    judge_log="${RUNTIME_DIR}/judge_$(printf '%02d' "${global_shard}").log"
  fi
  [[ -f "${input_path}" ]] || die "judge input missing: ${input_path}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/tau2-bench/sft/filter_official_native_turns.py" judge \
    --input "${input_path}" \
    --output-a "${output_a}" \
    --output-b "${output_b}" \
    --base-url "http://${HOST}:${port}/v1" \
    --model "${MODEL_NAME}" \
    --concurrency "${CONCURRENCY}" \
    --timeout "${TIMEOUT}" \
    --retries "${RETRIES}" \
    >"${judge_log}" 2>&1 &
  JUDGE_PIDS+=("$!")
done

judge_failed=0
for local_index in $(seq 0 $((SHARD_COUNT - 1))); do
  global_shard=$((SHARD_START + local_index))
  if ! wait "${JUDGE_PIDS[local_index]}"; then
    judge_log="${RUNTIME_DIR}/judge_$(printf '%02d' "${global_shard}").log"
    [[ "${STAGE}" == "pilot" ]] && judge_log="${RUNTIME_DIR}/judge_pilot.log"
    tail -n 160 "${judge_log}" >&2 || true
    judge_failed=1
  fi
done
[[ "${judge_failed}" == "0" ]] || die "one or more judge shards failed"
JUDGE_PIDS=()

if [[ "${STAGE}" == "pilot" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/examples/tau2-bench/sft/filter_official_native_turns.py" check-pilot \
    --output-root "${OUTPUT_ROOT}"
fi

log "stage ${STAGE} complete"
