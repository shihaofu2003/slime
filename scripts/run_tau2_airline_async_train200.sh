#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TAU2_RUN_NAME="airline-unbounded-p20-train200-20260914"
export PREPARED_RL_DATA="${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl/data/airline_tasks.jsonl"
export TAU2_POOL_CAPACITY=20
export TAU2_MAX_POLICY_LAG=-1
exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_tau2_areal_async.sh" async train200
