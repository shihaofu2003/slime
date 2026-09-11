#!/usr/bin/env bash
set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
OFFICIAL_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/official"

if [[ -z "${MODEL_PATH:-}" || -z "${MODEL_NAME:-}" ]]; then
  echo "[ERROR] MODEL_PATH and MODEL_NAME are required" >&2
  exit 1
fi

export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf"
export USER_MODEL="Qwen3-4B-tau2-user-sft-stop-iter0006899"
export TP="1"
export AGENT_CUDA_VISIBLE_DEVICES="0"
export USER_TP="1"
export USER_CUDA_VISIBLE_DEVICES="1"
export DOMAINS="airline,retail,telecom"
export NUM_TASKS=""
export NUM_TRIALS="4"
export TASK_SPLIT="test"
export SEED="300"
export AGENT_TEMPERATURE="0.6"
export AGENT_TOP_P="1.0"
export AGENT_MAX_TOKENS="8192"
export USER_TEMPERATURE="0.0"
export USER_MAX_TOKENS="512"
export USER_SGLANG_EXTRA_ARGS="--tool-call-parser qwen"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="${SAVE_PREFIX:-tau2_official_${MODEL_NAME}_user_stop_parser_${RUN_STAMP}}"
export SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${OFFICIAL_DIR}/outputs/${MODEL_NAME}/user_stop_parser_pass4_summary.json}"

exec bash "${OFFICIAL_DIR}/run_eval.sh"
