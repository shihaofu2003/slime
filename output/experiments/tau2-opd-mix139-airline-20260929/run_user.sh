#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
cd "${PROJECT_ROOT}"
export MODEL_PATH="$(dirname "${PROJECT_ROOT}")/models/Qwen3.6-27B"
export USER_ENDPOINT_FILE="${PROJECT_ROOT}/output/experiments/tau2-opd-mix139-airline-20260929/user_endpoint.env"
exec bash scripts/serve_tau2_user_pool.sh
