#!/usr/bin/env bash
set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
SCRIPT_DIR="${PROJECT_ROOT}/examples/tau2-bench/sft"

AREAL_SFT_PATH="${AREAL_SFT_PATH:-${SERVICE_AGENT_ROOT}/datasets/AReaL-tau2-data/tau2_sft_train.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/output/experiments/tau2-sft-official-native-expanded/data/agent_official_native_expanded.jsonl}"
SMOKE_OUTPUT_PATH="${SMOKE_OUTPUT_PATH:-${PROJECT_ROOT}/output/experiments/tau2-sft-official-native-expanded/data/agent_official_native_expanded_longest32.jsonl}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"
SMOKE_SIZE="${SMOKE_SIZE:-32}"

python3 "${SCRIPT_DIR}/build_agent_official_native_expanded.py" build \
  --input "${AREAL_SFT_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --output "${OUTPUT_PATH}" \
  --smoke-output "${SMOKE_OUTPUT_PATH}" \
  --max-tokens "${MAX_TOTAL_TOKENS}" \
  --smoke-size "${SMOKE_SIZE}"

python3 "${SCRIPT_DIR}/build_agent_official_native_expanded.py" validate \
  --input "${OUTPUT_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --max-tokens "${MAX_TOTAL_TOKENS}"

python3 "${SCRIPT_DIR}/build_agent_official_native_expanded.py" validate \
  --input "${SMOKE_OUTPUT_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --max-tokens "${MAX_TOTAL_TOKENS}"
