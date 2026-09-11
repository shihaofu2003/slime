#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime}"
export TAU2_RL_RUN_ID="${TAU2_RL_RUN_ID:-stability_k2_fieldreward_lr2e6_$(date +%Y%m%d_%H%M%S)}"
export LR="2e-6"
export KL_LOSS_TYPE="k2"
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_real_8g_airline.sh"
