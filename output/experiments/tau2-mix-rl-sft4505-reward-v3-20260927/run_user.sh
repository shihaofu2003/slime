#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
cd "${PROJECT_ROOT}"
export MODEL_PATH="$(dirname "${PROJECT_ROOT}")/models/Qwen3.6-27B"
export USER_ENDPOINT_FILE="${PROJECT_ROOT}/output/experiments/tau2-mix-rl-sft4505-reward-v3-20260927/user_endpoint.env"
python3 output/experiments/tau2-mix-rl-sft4505-reward-v3-20260927/verify_semantic_rewards.py --workers 8
exec bash scripts/serve_tau2_user_pool.sh
