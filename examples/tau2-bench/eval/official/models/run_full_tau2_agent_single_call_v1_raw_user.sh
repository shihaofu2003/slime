#!/usr/bin/env bash

set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_RUNNER="${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_tau2_agent_single_call_v1_user_stop_parser.sh"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-agent-single-call-v1-user-ablation"

if [[ $# -ne 2 || ! "$1" =~ ^(sft|rl199|rawrl99|rawrl199|rawrl120)$ || ! "$2" =~ ^(300|301)$ ]]; then
  echo "Usage: $0 <sft|rl199|rawrl99|rawrl199|rawrl120> <300|301>" >&2
  exit 2
fi
MODEL_KIND="$1"
SEED="$2"
BASE_MODEL_KIND="${MODEL_KIND}"

export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
export USER_MODEL="Qwen3-4B-Instruct-2507"

case "${MODEL_KIND}" in
  sft)
    export MODEL_NAME="Qwen3-4B-tau2-agent-sft-single-call-v1-raw-user"
    export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/raw-user-sft/seed${SEED}_summary.json"
    ;;
  rl199)
    export MODEL_NAME="Qwen3-4B-tau2-agent-rl-single-call-v1-iter199-raw-user"
    export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/raw-user/seed${SEED}_summary.json"
    ;;
  rawrl99)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_20260812/iter_0000099_hf"
    export MODEL_NAME="Qwen3-4B-tau2-agent-rl-single-call-v1-raw-user-trained-iter99"
    export SUMMARY_OUTPUT="${PROJECT_ROOT}/output/experiments/tau2-agent-single-call-v1-raw-user-rl100/eval/raw-user/seed${SEED}_summary.json"
    BASE_MODEL_KIND="rl"
    ;;
  rawrl199)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_20260812/iter_0000199_hf"
    export MODEL_NAME="Qwen3-4B-tau2-agent-rl-single-call-v1-raw-user-trained-iter199"
    export SUMMARY_OUTPUT="${PROJECT_ROOT}/output/experiments/tau2-agent-single-call-v1-raw-user-rl200/eval/raw-user/seed${SEED}_summary.json"
    BASE_MODEL_KIND="rl199"
    ;;
  rawrl120)
    export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812/iter_0000099_hf"
    export MODEL_NAME="Qwen3-4B-tau2-agent-rl-single-call-v1-raw-user-trained-maxsteps120-iter99"
    export SUMMARY_OUTPUT="${PROJECT_ROOT}/output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl100/eval/raw-user/seed${SEED}_summary.json"
    BASE_MODEL_KIND="rl"
    ;;
esac

exec bash "${MODEL_RUNNER}" "${BASE_MODEL_KIND}" "${SEED}"
