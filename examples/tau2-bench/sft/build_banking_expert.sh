#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OUTPUT_DIR="${BANKING_EXPERT_SFT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-expert-sft/data}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
INPUT_PATH="${BANKING_EXPERT_INPUT:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/successful_trajectories.jsonl}"
TASKS_PATH="${BANKING_EXPERT_TASKS:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/train_tasks.json}"
CONTRACTS_PATH="${BANKING_EXPERT_CONTRACTS:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/train_contracts.jsonl}"
ALLOWED_DOCUMENTS_PATH="${BANKING_EXPERT_ALLOWED_DOCUMENTS:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1/allowed_documents.json}"
DOCUMENTS_DIR="${BANKING_EXPERT_DOCUMENTS_DIR:-${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge/documents}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"

MODE="${1:-full}"
case "${MODE}" in
  smoke)
    LIMIT_ARGS=(--limit-trajectories "${SMOKE_TRAJECTORIES:-32}")
    OUTPUT_PATH="${OUTPUT_DIR}/banking_expert_sft_smoke32.jsonl"
    STATS_PATH="${OUTPUT_DIR}/banking_expert_sft_smoke32.stats.json"
    ;;
  full)
    LIMIT_ARGS=()
    OUTPUT_PATH="${OUTPUT_DIR}/banking_expert_sft.jsonl"
    STATS_PATH="${OUTPUT_DIR}/banking_expert_sft.stats.json"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

python3 "${PROJECT_ROOT}/examples/tau2-bench/sft/build_banking_expert.py" \
  --input "${INPUT_PATH}" \
  --tasks "${TASKS_PATH}" \
  --contracts "${CONTRACTS_PATH}" \
  --allowed-documents "${ALLOWED_DOCUMENTS_PATH}" \
  --documents-dir "${DOCUMENTS_DIR}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --output "${OUTPUT_PATH}" \
  --stats "${STATS_PATH}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  "${LIMIT_ARGS[@]}"

test -s "${OUTPUT_PATH}"
echo "[banking-expert-sft] wrote ${OUTPUT_PATH}"
