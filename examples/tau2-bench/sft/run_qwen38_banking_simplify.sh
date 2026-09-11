#!/usr/bin/env bash
set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${SERVICE_AGENT_ROOT}/slime"
OUTPUT_ROOT="${BANKING_SIMPLIFY_OUTPUT:-${PROJECT_ROOT}/output/experiments/tau2-banking-simplify-qwen38-v1}"
STAGE="${1:---stage}"
if [[ "${STAGE}" == "--stage" ]]; then STAGE="${2:-pilot}"; fi
case "${STAGE}" in
  pilot) GPU_COUNT="${BANKING_SIMPLIFY_GPUS:-2}" ;;
  full) GPU_COUNT="${BANKING_SIMPLIFY_GPUS:-8}" ;;
  *) echo "Usage: $0 --stage pilot|full" >&2; exit 2 ;;
esac
TP="${BANKING_SIMPLIFY_TP:-2}"
CONCURRENCY="${BANKING_SIMPLIFY_CONCURRENCY:-1}"
GPU_OFFSET="${BANKING_SIMPLIFY_GPU_OFFSET:-0}"
REPLICAS=$((GPU_COUNT / TP))
[[ "${REPLICAS}" -gt 0 && $((GPU_COUNT % TP)) -eq 0 && "${CONCURRENCY}" -gt 0 ]]
WORKERS=$((REPLICAS * CONCURRENCY))
PORT_BASE="${BANKING_SIMPLIFY_PORT:-33200}"
export PYTHONPATH="${PROJECT_ROOT}:${SERVICE_AGENT_ROOT}/tau2-bench/src:${PYTHONPATH:-}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/sft/qwen_simplify_banking.py"
RUN_DIR="${OUTPUT_ROOT}/runtime/${STAGE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
python3 tests/test_tau2_banking_expert_sft.py
if [[ ! -f "${OUTPUT_ROOT}/inventory.json" ]]; then
  python3 "${PIPELINE}" --stage prepare --output-dir "${OUTPUT_ROOT}"
fi
EXTRA_ARGS=()
if [[ "${BANKING_SIMPLIFY_RETRY_ERRORS:-0}" == "1" ]]; then EXTRA_ARGS+=(--retry-errors); fi
IFS=, read -ra RETRY_IDS <<< "${BANKING_SIMPLIFY_RETRY_IDS:-}"
for id in "${RETRY_IDS[@]}"; do EXTRA_ARGS+=(--retry-id "${id}"); done
python3 "${PIPELINE}" --stage assign --scope "${STAGE}" --output-dir "${OUTPUT_ROOT}" \
  --num-shards "${WORKERS}" --work-ids "${RUN_DIR}/work_ids.json" "${EXTRA_ARGS[@]}"

SERVER_PIDS=()
WORKER_PIDS=()
cleanup() {
  trap - EXIT TERM INT HUP
  for pid in "${WORKER_PIDS[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
  for _ in $(seq 1 15); do
    alive=0
    for pid in "${SERVER_PIDS[@]}"; do if kill -0 "${pid}" 2>/dev/null; then alive=1; fi; done
    [[ "${alive}" == "0" ]] && break
    sleep 1
  done
  for pid in "${SERVER_PIDS[@]}"; do
    kill -KILL -- "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
nvidia-smi --query-gpu=index,name,memory.total --format=csv
for ((worker=0; worker<REPLICAS; worker++)); do
  gpu_start=$((GPU_OFFSET + worker * TP))
  gpu_end=$((gpu_start + TP - 1))
  gpu_ids=$(seq -s, "${gpu_start}" "${gpu_end}")
  setsid env CUDA_VISIBLE_DEVICES="${gpu_ids}" python3 -m sglang.launch_server \
    --model-path /mnt/afs/models/Qwen3.8-27B \
    --served-model-name Qwen3.8-27B-banking-simplifier \
    --host 127.0.0.1 --port "$((PORT_BASE + worker))" \
    --tp "${TP}" --dtype bfloat16 --context-length 262144 \
    --max-running-requests "${CONCURRENCY}" --mem-fraction-static 0.95 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    >"${RUN_DIR}/server_${worker}.log" 2>&1 &
  SERVER_PIDS+=("$!")
done
for ((worker=0; worker<REPLICAS; worker++)); do
  ready=0
  for _ in $(seq 1 360); do
    if ! kill -0 "${SERVER_PIDS[worker]}" 2>/dev/null; then
      tail -n 100 "${RUN_DIR}/server_${worker}.log"
      exit 1
    fi
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$((PORT_BASE + worker))/health', timeout=3)" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "${ready}" != "1" ]]; then
    tail -n 100 "${RUN_DIR}/server_${worker}.log"
    exit 1
  fi
  for ((slot=0; slot<CONCURRENCY; slot++)); do
    client=$((worker * CONCURRENCY + slot))
    python3 "${PIPELINE}" --stage "${STAGE}" --output-dir "${OUTPUT_ROOT}" \
    --endpoint "http://127.0.0.1:$((PORT_BASE + worker))/v1" \
    --num-shards "${WORKERS}" --shard-index "${client}" --work-ids "${RUN_DIR}/work_ids.json" \
    "${EXTRA_ARGS[@]}" >"${RUN_DIR}/worker_${client}.log" 2>&1 &
    WORKER_PIDS+=("$!")
    echo "STARTED replica=${worker} client=${client} concurrency=${CONCURRENCY} log=${RUN_DIR}/worker_${client}.log"
  done
done
for pid in "${WORKER_PIDS[@]}"; do wait "${pid}"; done
python3 "${PIPELINE}" --stage finalize --scope "${STAGE}" --output-dir "${OUTPUT_ROOT}"
echo "BANKING_SIMPLIFICATION_FINISHED stage=${STAGE} run_dir=${RUN_DIR}"
