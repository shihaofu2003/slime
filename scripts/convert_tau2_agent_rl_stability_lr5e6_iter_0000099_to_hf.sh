#!/usr/bin/env bash
set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export CHECKPOINT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_stability_k2_fieldreward_lr5e6_20260729_064904"

exec bash "${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}/scripts/convert_tau2_agent_rl_stability_iter_0000099_to_hf.sh" "$@"
