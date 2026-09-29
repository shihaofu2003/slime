#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT}"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
RL_DIR="${PROJECT_ROOT}/examples/tau2-bench/rl_db_count_v1"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}:${RL_DIR}:${PROJECT_ROOT}/examples/tau2-bench/shared:${SERVICE_AGENT_ROOT}/tau2-bench/src:${PYTHONPATH:-}"
export HF_CHECKPOINT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903/final_hf"
export TAU2_AGENT_PROTOCOL_PROFILE=official-native
export TAU2_TURN_CREDIT_VERSION=progress-db-count-v1
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
export TAU2_PROGRESS_DIAGNOSTICS_PATH=""
cd "${PROJECT_ROOT}"
python3 -m pytest "${RL_DIR}/test_progress.py" "${RL_DIR}/test_db_count.py" -q -p no:cacheprovider
python3 "${RL_DIR}/test_rollout_logic.py"
