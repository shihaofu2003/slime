#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^(mixed|airline)$ ]]; then
  echo "Usage: $0 <mixed|airline>" >&2
  exit 2
fi
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TAU2_ITERATION="${TAU2_ITERATION:-99}"
export TAU2_HF_SUFFIX=""
export AUTO_RESUME=1
if [[ "$1" == airline ]]; then
  export TAU2_RUN_NAME=airline-unbounded-p20-train200-20260914
else
  export TAU2_RUN_NAME=unbounded-p20-train200-20260914
fi
export TAU2_EVAL_SUFFIX="-${TAU2_RUN_NAME}-${1}-all-domains"
printf -v iteration_padded '%07d' "${TAU2_ITERATION}"
model_dir="${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl/arms/async/${TAU2_RUN_NAME}/checkpoints/iter_${iteration_padded}_hf"
if [[ ! -f "${model_dir}/model.safetensors.index.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/convert_tau2_areal_async_to_hf.sh" async --force
fi
exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_tau2_areal_async.sh" async 300
