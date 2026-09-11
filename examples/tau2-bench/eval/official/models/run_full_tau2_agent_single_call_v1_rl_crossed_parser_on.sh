#!/usr/bin/env bash

set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-agent-single-call-v1-rl-crossed-parser-on"

MODEL_KIND="${1:-}"
SEED="${2:-300}"

if [[ $# -gt 2 || ! "${MODEL_KIND}" =~ ^(v1rl99-raw|rawrl99-v1|rawrl99max120-v1|rawrl199-v1|rawrl199max120-v1)$ || ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 <v1rl99-raw|rawrl99-v1|rawrl99max120-v1|rawrl199-v1|rawrl199max120-v1> [seed]" >&2
  exit 2
fi

export MODEL_PATH=""
export MODEL_NAME=""
export USER_MODEL_PATH=""
export USER_MODEL=""

case "${MODEL_KIND}" in
  v1rl99-raw)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_20260809/iter_0000099_hf"
    export MODEL_NAME="Qwen3-4B-tau2-agent-v1-rl-iter99-raw-user-parser-on"
    export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
    export USER_MODEL="Qwen3-4B-Instruct-2507-raw-user"
    ;;
  rawrl99-v1)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_20260812/iter_0000099_hf"
    export MODEL_NAME="Qwen3-4B-tau2-agent-raw-rl-iter99-v1-user-parser-on"
    export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf"
    export USER_MODEL="Qwen3-4B-tau2-user-sft-stop-iter0006899"
    ;;
  rawrl99max120-v1)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812/iter_0000099_hf"
    export MODEL_NAME="Qwen3-4B-tau2-agent-raw-rl-iter99-max120-v1-user-parser-on"
    export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf"
    export USER_MODEL="Qwen3-4B-tau2-user-sft-stop-iter0006899"
    ;;
  rawrl199-v1)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_20260812/iter_0000199_hf"
    export MODEL_NAME="Qwen3-4B-tau2-agent-raw-rl-iter199-v1-user-parser-on"
    export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf"
    export USER_MODEL="Qwen3-4B-tau2-user-sft-stop-iter0006899"
    ;;
  rawrl199max120-v1)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812/iter_0000199_hf"
    export MODEL_NAME="Qwen3-4B-tau2-agent-raw-rl-iter199-max120-v1-user-parser-on"
    export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf"
    export USER_MODEL="Qwen3-4B-tau2-user-sft-stop-iter0006899"
    ;;
esac

# Keep the evaluation parser explicit for both RAW and SFT User checkpoints.
export USER_SGLANG_EXTRA_ARGS="--tool-call-parser qwen"
export USER_TEMPERATURE="0.0"
export USER_MAX_TOKENS="512"
export DOMAINS="airline,retail,telecom"
export TASK_SPLIT="test"
export NUM_TASKS=""
export NUM_TRIALS="4"
export SEED
export MAX_STEPS="200"
export MAX_ERRORS="10"
export MAX_CONCURRENCY="1"
export AGENT_TEMPERATURE="0.6"
export AGENT_TOP_P="1.0"
export AGENT_MAX_TOKENS="8192"
export AGENT_PROTOCOL_PROFILE="strict-single-v1"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${MODEL_KIND}/seed${SEED}_summary.json"
export NAMESPACE_PROBE_OUTPUT=""

exec bash "${OFFICIAL_DIR}/run_eval.sh"
