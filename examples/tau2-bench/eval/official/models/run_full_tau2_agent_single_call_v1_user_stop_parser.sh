#!/usr/bin/env bash

set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-agent-single-call-v1"

usage() {
  echo "Usage:" >&2
  echo "  $0 <sft|rl|rl199> <300|301>" >&2
  echo "  MODEL_PATH=/absolute/checkpoint MODEL_NAME=name SUMMARY_OUTPUT=/absolute/summary.json \\" >&2
  echo "    $0 candidate <300|301>" >&2
}

if [[ $# -ne 2 || ! "$1" =~ ^(sft|rl|rl199|candidate)$ || ! "$2" =~ ^(300|301)$ ]]; then
  usage
  exit 2
fi
MODEL_KIND="$1"
SEED="$2"

case "${MODEL_KIND}" in
  sft)
    export MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809/final_hf}"
    export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-tau2-agent-sft-single-call-v1}"
    ;;
  rl)
    export MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_20260809/iter_0000099_hf}"
    export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-tau2-agent-rl-single-call-v1-iter99}"
    ;;
  rl199)
    export MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_20260809/iter_0000199_hf}"
    export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-tau2-agent-rl-single-call-v1-iter199}"
    ;;
  candidate)
    if [[ -z "${MODEL_PATH:-}" ]]; then
      echo "[ERROR] candidate mode requires MODEL_PATH" >&2
      usage
      exit 2
    fi
    if [[ -z "${MODEL_NAME:-}" ]]; then
      echo "[ERROR] candidate mode requires MODEL_NAME" >&2
      usage
      exit 2
    fi
    if [[ -z "${SUMMARY_OUTPUT:-}" ]]; then
      echo "[ERROR] candidate mode requires SUMMARY_OUTPUT" >&2
      usage
      exit 2
    fi
    if [[ "${MODEL_PATH}" != /* || ! -d "${MODEL_PATH}" ]]; then
      echo "[ERROR] MODEL_PATH must be an existing absolute directory: ${MODEL_PATH}" >&2
      exit 2
    fi
    if [[ ! "${MODEL_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
      echo "[ERROR] MODEL_NAME must contain only letters, digits, '.', '_', or '-'" >&2
      exit 2
    fi
    if [[ "${SUMMARY_OUTPUT}" != /* ]]; then
      echo "[ERROR] SUMMARY_OUTPUT must be an absolute path: ${SUMMARY_OUTPUT}" >&2
      exit 2
    fi
    if [[ -e "${SUMMARY_OUTPUT}" ]]; then
      echo "[ERROR] refusing to overwrite existing candidate summary: ${SUMMARY_OUTPUT}" >&2
      exit 2
    fi
    export SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-0}"
    export MODEL_PATH MODEL_NAME SUMMARY_OUTPUT
    ;;
esac

export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"
export DOMAINS="${DOMAINS:-airline,retail,telecom}"
export TASK_SPLIT="test"
export NUM_TASKS=""
export NUM_TRIALS=4
export SEED
export MAX_STEPS=200
export AGENT_TEMPERATURE=0.6
export AGENT_TOP_P=1.0
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-8192}"
export AGENT_PROTOCOL_PROFILE="strict-single-v1"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_strict_single_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${EXPERIMENT_DIR}/eval/${MODEL_KIND}/seed${SEED}_summary.json}"
export NAMESPACE_PROBE_OUTPUT=""

exec bash "${OFFICIAL_DIR}/run_eval.sh"
