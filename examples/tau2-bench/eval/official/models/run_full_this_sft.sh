#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
OFFICIAL_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/official"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_sft_areal_strict_no_thinking_epoch2_20260713/iter_0000881_hf"
export MODEL_NAME="Qwen3-4B-Instruct-2507-SFT"
export NUM_TASKS=""
export NUM_TRIALS="4"
export TASK_SPLIT="${TASK_SPLIT:-test}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}"
export SUMMARY_OUTPUT="${OFFICIAL_DIR}/outputs/${MODEL_NAME}/pass4_summary.json"

exec bash "${OFFICIAL_DIR}/run_eval.sh"
