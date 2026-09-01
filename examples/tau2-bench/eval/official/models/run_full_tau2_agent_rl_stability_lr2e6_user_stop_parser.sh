#!/usr/bin/env bash
set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_stability_k2_fieldreward_lr2e6_20260729_065304/iter_0000099_hf"
export MODEL_NAME="Qwen3-4B-tau2-agent-rl-stability-lr2e6-iter0000099"

exec bash "${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}/examples/tau2-bench/eval/official/models/run_full_tau2_agent_rl_stability_user_stop_parser.sh"
