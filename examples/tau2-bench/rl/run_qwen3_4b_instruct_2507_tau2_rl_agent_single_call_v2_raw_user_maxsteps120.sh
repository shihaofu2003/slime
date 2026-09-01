#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
BASE_RUNNER="${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_single_call_v1.sh"
ARM="${1:-}"
MODE="${2:-all}"

if [[ $# -gt 2 || ! "${ARM}" =~ ^(mixed|airline|retail|telecom)$ || ! "${MODE}" =~ ^(stage-a|stage-b|stage-final|all)$ ]]; then
  echo "Usage: $0 <mixed|airline|retail|telecom> [stage-a|stage-b|stage-final|all]" >&2
  exit 2
fi

case "${ARM}" in
  mixed)
    unset TAU2_RL_FIXED_DOMAIN_QUOTA
    ;;
  airline)
    export TAU2_RL_FIXED_DOMAIN_QUOTA="telecom:0,airline:6,retail:0"
    ;;
  retail)
    export TAU2_RL_FIXED_DOMAIN_QUOTA="telecom:0,airline:0,retail:6"
    ;;
  telecom)
    export TAU2_RL_FIXED_DOMAIN_QUOTA="telecom:6,airline:0,retail:0"
    ;;
esac

export EXPERIMENT_NAME="tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl"
export TAU2_RL_ARM="${ARM}"
export RL_CKPT_ROOT="${RL_CKPT_ROOT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v2_raw_user_maxsteps120_${ARM}_20260815}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-Instruct-2507}"
export TAU2_TURN_CREDIT_VERSION="turn-credit-v2"
export TAU2_TURN_CREDIT_REALLOCATION_WEIGHT="0.1"
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS="0"
export TAU2_MAX_STEPS="120"
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE="1"
export WANDB_GROUP="${WANDB_GROUP:-tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl}"

if [[ "${MODE}" == "all" ]]; then
  for stage in stage-a stage-b stage-final; do
    echo "[single-call-v2-raw-user-maxsteps120] arm=${ARM} stage=${stage}"
    bash "${BASE_RUNNER}" "${stage}"
  done
else
  exec bash "${BASE_RUNNER}" "${MODE}"
fi
