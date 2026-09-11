#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <checkpoint-dir> [300|301]" >&2
  exit 2
fi

CHECKPOINT="$1"
SEED_VALUE="${2:-300}"
if [[ ! "${SEED_VALUE}" =~ ^(300|301)$ ]]; then
  echo "[ERROR] seed must be 300 or 301" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_BASE="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-official-native-expanded-turn-filtered"
RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_turn_filtered_20260908"

case "${CHECKPOINT}" in
  iter_0000399|iter_0000799|iter_0001199|iter_0001599|iter_0001855|iter_0001999|iter_0002399|iter_0002799|iter_0003199|iter_0003599)
    HF_BASENAME="${CHECKPOINT}_hf"
    ;;
  iter_0003711)
    HF_BASENAME=final_hf
    ;;
  *)
    echo "[ERROR] unsupported checkpoint: ${CHECKPOINT}" >&2
    exit 2
    ;;
esac

MODEL_PATH="${RUN_ROOT}/${HF_BASENAME}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Hugging Face checkpoint is not ready: ${MODEL_PATH}" >&2
  exit 1
fi

MODEL_NAME="Qwen3-4B-Instruct-2507-sft-official-native-expanded-turn-filtered-${CHECKPOINT}"
export EXPERIMENT_DIR
export EVAL_LABEL="checkpoint-${CHECKPOINT}"
export MODEL_PATH
export MODEL_NAME
export AGENT_SERVED_MODEL_NAME="${MODEL_NAME}"
export AGENT_EVAL_MODE=official-native
export AGENT_TOOL_CALL_PARSER=qwen
export DOMAINS=airline,retail,telecom
export DOMAIN_CONCURRENCY=airline:2,retail:2,telecom:5
export TASK_SPLIT=test
export NUM_TASKS=""
export NUM_TRIALS=4
export SEED="${SEED_VALUE}"
export MAX_STEPS=200
export AGENT_TEMPERATURE=0.6
export AGENT_TOP_P=1.0
export AGENT_MAX_TOKENS=1200

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
