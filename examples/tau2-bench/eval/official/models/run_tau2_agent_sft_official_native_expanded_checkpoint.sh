#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <smoke|full> <checkpoint-dir> [300|301]" >&2
  exit 2
fi

MODE="$1"
CHECKPOINT="$2"
SEED_VALUE="${3:-300}"
if [[ ! "${SEED_VALUE}" =~ ^(300|301)$ ]]; then
  echo "[ERROR] seed must be 300 or 301" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_BASE="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-official-native-expanded"

case "${MODE}" in
  smoke)
    if [[ "${CHECKPOINT}" != "iter_0000001" ]]; then
      echo "[ERROR] smoke evaluation requires iter_0000001" >&2
      exit 2
    fi
    RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_longest32_smoke_v2_20260903"
    HF_BASENAME=final_hf
    EVAL_LABEL=smoke-e2e
    NUM_TASKS_VALUE=1
    NUM_TRIALS_VALUE=1
    ;;
  full)
    case "${CHECKPOINT}" in
      iter_0000399|iter_0000799|iter_0001199|iter_0001599|iter_0001897|iter_0001999|iter_0002399|iter_0002799|iter_0003199|iter_0003599)
        HF_BASENAME="${CHECKPOINT}_hf"
        ;;
      iter_0003795)
        HF_BASENAME=final_hf
        ;;
      *)
        echo "[ERROR] unsupported full checkpoint: ${CHECKPOINT}" >&2
        exit 2
        ;;
    esac
    RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903"
    EVAL_LABEL="checkpoint-${CHECKPOINT}"
    NUM_TASKS_VALUE=""
    NUM_TRIALS_VALUE=4
    ;;
  *)
    echo "[ERROR] mode must be smoke or full" >&2
    exit 2
    ;;
esac

MODEL_PATH="${RUN_ROOT}/${HF_BASENAME}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Hugging Face checkpoint is not ready: ${MODEL_PATH}" >&2
  exit 1
fi

MODEL_NAME="Qwen3-4B-Instruct-2507-sft-official-native-expanded-${CHECKPOINT}"
export EXPERIMENT_DIR
export EVAL_LABEL
export MODEL_PATH
export MODEL_NAME
export AGENT_SERVED_MODEL_NAME="${MODEL_NAME}"
export AGENT_EVAL_MODE=official-native
export AGENT_TOOL_CALL_PARSER=qwen
export DOMAINS=airline,retail,telecom
export DOMAIN_CONCURRENCY=airline:2,retail:2,telecom:5
export TASK_SPLIT=test
export NUM_TASKS="${NUM_TASKS_VALUE}"
export NUM_TRIALS="${NUM_TRIALS_VALUE}"
export SEED="${SEED_VALUE}"
export MAX_STEPS=200
export AGENT_TEMPERATURE=0.6
export AGENT_TOP_P=1.0
export AGENT_MAX_TOKENS=1200

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
