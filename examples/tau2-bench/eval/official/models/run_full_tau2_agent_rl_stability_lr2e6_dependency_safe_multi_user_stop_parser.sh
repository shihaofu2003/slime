#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_stability_k2_fieldreward_lr2e6_20260729_065304/iter_0000099_hf"
export MODEL_NAME="Qwen3-4B-tau2-agent-rl-stability-lr2e6-iter0000099"
export AGENT_PROTOCOL_PROFILE="dependency-safe-multi"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_dependency_safe_multi_user_stop_parser_${RUN_STAMP}"
export SUMMARY_OUTPUT="${OFFICIAL_DIR}/outputs/${MODEL_NAME}/dependency_safe_multi_user_stop_parser_pass4_summary.json"

exec bash "${OFFICIAL_DIR}/models/run_full_tau2_agent_rl_stability_user_stop_parser.sh"
