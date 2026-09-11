#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 || ! "$1" =~ ^(raw-all|processed)$ || ! "$2" =~ ^(9|39|69|99)$ || ! "$3" =~ ^(300|301)$ ]]; then
  echo "Usage: $0 <raw-all|processed> <9|39|69|99> <300|301>" >&2
  exit 2
fi

ARM="$1"
ITERATION="$2"
SEED_VALUE="$3"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_BASE="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-sft-raw-vs-processed-vanilla-grpo"

case "${ARM}" in
  raw-all)
    RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_rl_raw_all_vanilla_grpo_qwen36_20260903"
    ;;
  processed)
    RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_rl_official_native_expanded_vanilla_grpo_qwen36_20260903"
    ;;
esac

printf -v iteration_padded '%07d' "${ITERATION}"
MODEL_PATH="${RUN_ROOT}/iter_${iteration_padded}_hf"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Hugging Face checkpoint is not ready: ${MODEL_PATH}" >&2
  exit 1
fi

export EXPERIMENT_DIR
export EVAL_LABEL="${ARM}-iter${ITERATION}"
export MODEL_PATH
export MODEL_NAME="Qwen3-4B-Instruct-2507-${ARM}-vanilla-grpo-iter${ITERATION}"
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
