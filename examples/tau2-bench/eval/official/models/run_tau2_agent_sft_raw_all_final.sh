#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(300|301)$ ]]; then
  echo "Usage: $0 <300|301>" >&2
  exit 2
fi

SEED_VALUE="$1"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_BASE="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-raw-all-max16384"
RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903"
MODEL_PATH="${RUN_ROOT}/final_hf"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Hugging Face checkpoint is not ready: ${MODEL_PATH}" >&2
  exit 1
fi

export EXPERIMENT_DIR
export EVAL_LABEL=final
export MODEL_PATH
export MODEL_NAME=Qwen3-4B-Instruct-2507-sft-raw-all-max16384-final
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
