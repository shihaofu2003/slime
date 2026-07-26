#!/usr/bin/env bash
set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"

export CHECKPOINT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714"
export ITER_DIR="${CHECKPOINT_ROOT}/iter_0002413"
export OUTPUT_DIR="${CHECKPOINT_ROOT}/iter_0002413_hf"

exec bash "${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
