#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
BASE_RUNNER="${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_single_call_v1.sh"
ARM="${1:-}"

if [[ $# -ne 1 || ! "${ARM}" =~ ^(mixed|airline|retail|telecom)$ ]]; then
  echo "Usage: $0 <mixed|airline|retail|telecom>" >&2
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

export EXPERIMENT_NAME="tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130"
export TAU2_RL_ARM="${ARM}"
export RL_CKPT_ROOT="${RL_CKPT_ROOT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_from_iter99_domain_${ARM}_20260816}"
export TAU2_RL_INIT_HF_CHECKPOINT="${TAU2_RL_INIT_HF_CHECKPOINT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812/iter_0000099_hf}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-Instruct-2507}"
export TAU2_TURN_CREDIT_VERSION="turn-credit-v1"
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS=1
export TAU2_MAX_STEPS=120
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0
export WANDB_GROUP="${WANDB_GROUP:-tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130}"

exec bash "${BASE_RUNNER}" stage-iter99-to-129
