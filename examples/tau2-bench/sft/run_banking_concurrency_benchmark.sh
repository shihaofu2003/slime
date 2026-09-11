#!/usr/bin/env bash
set -euo pipefail
cd /mnt/afs/users/fush/projects/ServiceAgent/slime
SOURCE=output/experiments/tau2-banking-simplify-qwen38-v1
OUTPUT=output/experiments/tau2-banking-simplify-concurrency-v1
HELPER=examples/tau2-bench/sft/benchmark_banking_concurrency.py
python3 "${HELPER}" prepare --source "${SOURCE}" --output "${OUTPUT}"
PIDS=()
cleanup() {
  trap - EXIT TERM INT
  for pid in "${PIDS[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ARMS=(c1 c2 c4 c4_repeat)
CONCURRENCIES=(1 2 4 4)
for i in 0 1 2 3; do
  env BANKING_SIMPLIFY_OUTPUT="$(pwd)/${OUTPUT}/${ARMS[i]}" \
    BANKING_SIMPLIFY_GPUS=2 BANKING_SIMPLIFY_TP=2 \
    BANKING_SIMPLIFY_GPU_OFFSET="$((i * 2))" BANKING_SIMPLIFY_PORT="$((34200 + i))" \
    BANKING_SIMPLIFY_CONCURRENCY="${CONCURRENCIES[i]}" \
    bash examples/tau2-bench/sft/run_qwen38_banking_simplify.sh --stage pilot \
    >"${OUTPUT}/${ARMS[i]}/run.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "${pid}" || FAILED=1; done
python3 "${HELPER}" report --source "${SOURCE}" --output "${OUTPUT}"
exit "${FAILED}"
