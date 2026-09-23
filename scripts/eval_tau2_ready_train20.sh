#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^(sync|async)$ ]]; then
  echo "Usage: $0 <sync|async>" >&2
  exit 2
fi
ARM="$1"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TAU2_RUN_NAME=ready-train20
export TAU2_ITERATION=19
export TAU2_EVAL_SUFFIX="${TAU2_EVAL_SUFFIX:--judge-env-fix}"
if [[ "${TAU2_SKIP_CONVERSION:-0}" != "1" ]]; then
  bash "${PROJECT_ROOT}/scripts/convert_tau2_areal_async_to_hf.sh" "${ARM}"
fi
for seed in 300 301; do
  bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_tau2_areal_async.sh" "${ARM}" "${seed}"
done
