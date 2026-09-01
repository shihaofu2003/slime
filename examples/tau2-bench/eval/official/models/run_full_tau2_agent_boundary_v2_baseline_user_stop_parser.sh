#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2"
BASELINE="${1:-${TAU2_BOUNDARY_BASELINE:-raw-instruct}}"
SEED="${2:-${SEED:-300}}"
PROBE_MODE="${3:-no-probe}"

if [[ $# -gt 3 || ! "${SEED}" =~ ^[0-9]+$ || ! "${PROBE_MODE}" =~ ^(probe|no-probe)$ ]]; then
  echo "Usage: $0 [raw-instruct|old-sft|current-sft] [seed] [probe|no-probe]" >&2
  exit 2
fi

case "${BASELINE}" in
  raw-instruct)
    DEFAULT_MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
    DEFAULT_MODEL_NAME="Qwen3-4B-Instruct-2507-agent-owned"
    LABEL="raw_instruct"
    ;;
  old-sft)
    DEFAULT_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714/iter_0002413_hf"
    DEFAULT_MODEL_NAME="Qwen3-4B-tau2-agent-old-sft-agent-owned"
    LABEL="old_sft"
    ;;
  current-sft)
    DEFAULT_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_local_first_relaxed_20260803/final_hf"
    DEFAULT_MODEL_NAME="Qwen3-4B-tau2-agent-current-sft-agent-owned"
    LABEL="current_sft"
    ;;
  *)
    echo "[ERROR] invalid TAU2_BOUNDARY_BASELINE=${BASELINE}" >&2
    echo "        expected raw-instruct, old-sft, or current-sft" >&2
    exit 2
    ;;
esac

export MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
export MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"
export NUM_TASKS=""
export NUM_TRIALS=4
export SEED
export TASK_SPLIT="${TASK_SPLIT:-test}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-8192}"
export AGENT_PROTOCOL_PROFILE="agent-owned-dependency-safe-multi"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_agent_owned_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${EXPERIMENT_DIR}/eval/${LABEL}/seed${SEED}_summary.json}"
if [[ "${PROBE_MODE}" == "probe" ]]; then
  export NAMESPACE_PROBE_OUTPUT="${NAMESPACE_PROBE_OUTPUT:-${EXPERIMENT_DIR}/eval/${LABEL}/namespace_probe.json}"
else
  export NAMESPACE_PROBE_OUTPUT=""
fi

exec bash "${OFFICIAL_DIR}/run_eval.sh"
