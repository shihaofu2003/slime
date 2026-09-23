#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TAU2_RUN_NAME=airline-unbounded-p20-train200-20260914
export TAU2_ITERATION=19
export TAU2_EVAL_SUFFIX="-${TAU2_RUN_NAME}-airline"
export TAU2_EVAL_DOMAINS=airline
export TAU2_EVAL_DOMAIN_CONCURRENCY=airline:9
export BORROW_COMPLETED_DOMAIN_SLOTS=0
export AUTO_RESUME=1
model_dir="${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl/arms/async/${TAU2_RUN_NAME}/checkpoints/iter_0000019_hf"
if [[ ! -f "${model_dir}/model.safetensors.index.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/convert_tau2_areal_async_to_hf.sh" async --force
fi
exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_tau2_areal_async.sh" async 300
