#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
BASE_RUNNER="${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_single_call_v1.sh"
MODE="${1:-all}"

if [[ $# -gt 1 || ! "${MODE}" =~ ^(stage-a|stage-b|stage-final|all)$ ]]; then
  echo "Usage: $0 <stage-a|stage-b|stage-final|all>" >&2
  exit 2
fi

export EXPERIMENT_NAME="tau2-agent-single-call-v1-raw-user-maxsteps120-rl100"
export RL_CKPT_ROOT="${RL_CKPT_ROOT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-Instruct-2507}"
export TAU2_MAX_STEPS=120
export WANDB_GROUP="${WANDB_GROUP:-tau2-agent-single-call-v1-raw-user-maxsteps120-rl100}"

if [[ "${MODE}" == "all" ]]; then
  for stage in stage-a stage-b stage-final; do
    echo "[raw-user-maxsteps120-rl100] starting ${stage}"
    bash "${BASE_RUNNER}" "${stage}"
  done
else
  exec bash "${BASE_RUNNER}" "${MODE}"
fi
