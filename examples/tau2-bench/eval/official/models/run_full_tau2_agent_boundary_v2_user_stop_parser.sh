#!/usr/bin/env bash

set -euo pipefail

export AGENT_EVAL_MODE="legacy-custom"

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2"
ARM="${1:-${TAU2_BOUNDARY_SFT_ARM:-contract-only}}"
SEED="${2:-${SEED:-300}}"
if [[ $# -ge 3 ]]; then
  PROBE_MODE="$3"
elif [[ "${SEED}" == "300" ]]; then
  PROBE_MODE="probe"
else
  PROBE_MODE="no-probe"
fi

if [[ $# -gt 3 || ! "${SEED}" =~ ^[0-9]+$ || ! "${PROBE_MODE}" =~ ^(probe|no-probe)$ ]]; then
  echo "Usage: $0 [contract-only|contract-boundary] [seed] [probe|no-probe]" >&2
  exit 2
fi

case "${ARM}" in
  contract-only)
    SUFFIX="contract_only"
    LABEL="contract-only"
    ;;
  contract-boundary)
    SUFFIX="contract_boundary"
    LABEL="contract-boundary"
    ;;
  *)
    echo "[ERROR] invalid TAU2_BOUNDARY_SFT_ARM=${ARM}" >&2
    exit 2
    ;;
esac

export MODEL_PATH="${MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_${SUFFIX}_20260804/final_hf}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-tau2-agent-sft-boundary-v2-${SUFFIX}}"
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
