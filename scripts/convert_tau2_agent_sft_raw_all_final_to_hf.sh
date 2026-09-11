#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_BASE="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903"

export PROJECT_ROOT
export ITER_DIR="${RUN_ROOT}/iter_0004067"
export OUTPUT_DIR="${RUN_ROOT}/final_hf"
export ORIGIN_HF_DIR="${MODEL_ROOT}/Qwen3-4B-Instruct-2507"

exec bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
