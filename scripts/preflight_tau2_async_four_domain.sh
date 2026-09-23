#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TAU2_DATA_DIR="$(dirname "${PROJECT_ROOT}")/tau2-bench/data"
export TAU2_SRC="${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl/dependencies/tau2/src"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/examples/tau2-bench/rl:${PROJECT_ROOT}/examples/tau2-bench/shared:${TAU2_SRC}:/root/Megatron-LM:${PYTHONPATH:-}"
export HF_CHECKPOINT="$(dirname "${PROJECT_ROOT}")/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909/iter_0004673_hf"
export TAU2_AGENT_PROTOCOL_PROFILE=official-native
export TAU2_RAW_TOKENS=0
export TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0
cd "${PROJECT_ROOT}"
python3 examples/tau2-bench/rl/prepare_four_domain_rl_data.py
python3 -m pytest examples/tau2-bench/rl/test_progress.py examples/tau2-bench/rl/test_db_count.py tests/test_tau2_continuous.py -q
TAU2_TURN_CREDIT_VERSION=progress-db-count-v1 python3 examples/tau2-bench/rl/test_rollout_logic.py
