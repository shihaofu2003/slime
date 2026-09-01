#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
RL_DIR="${PROJECT_ROOT}/examples/tau2-bench/rl"
SHARED_DIR="${PROJECT_ROOT}/examples/tau2-bench/shared"
TAU2_SRC="${SERVICE_AGENT_ROOT}/tau2-bench/src"

export HF_CHECKPOINT="${HF_CHECKPOINT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_boundary_20260804/final_hf}"
export TAU2_AGENT_PROTOCOL_PROFILE="agent-owned-dependency-safe-multi"
export TAU2_TURN_CREDIT_VERSION="turn-credit-v1"
export PYTHONPATH="${PROJECT_ROOT}:${RL_DIR}:${SHARED_DIR}:${TAU2_SRC}:${PYTHONPATH:-}"

exec python3 "${RL_DIR}/test_rollout_logic.py"
