#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TAU2_RUN_NAME="${TAU2_RUN_NAME:-serial-train100-20260912-v1}"
export TAU2_ITERATION="${TAU2_ITERATION:-99}"
export TAU2_HF_SUFFIX="${TAU2_HF_SUFFIX:-_parallel_eval}"
export TAU2_EVAL_SUFFIX="${TAU2_EVAL_SUFFIX:--${TAU2_RUN_NAME}-parallel}"

CONVERT_SCRIPT="${PROJECT_ROOT}/scripts/convert_tau2_areal_async_to_hf.sh"
EVAL_SCRIPT="${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_tau2_areal_async.sh"

bash "${CONVERT_SCRIPT}" async
bash "${EVAL_SCRIPT}" async 300
bash "${EVAL_SCRIPT}" async 301
