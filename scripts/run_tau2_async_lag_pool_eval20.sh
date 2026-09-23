#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TAU2_ITERATION=19
export TAU2_HF_SUFFIX="-eval20"
arms=(
  "lag-pool-20-20260913-101740-unbounded-p10"
  "lag-pool-20-20260913-101740-unbounded-p20"
)
for run_name in "${arms[@]}"; do
  export TAU2_RUN_NAME="${run_name}"
  export TAU2_EVAL_SUFFIX="-${run_name}-seed300"
  model_dir="${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl/arms/async/${run_name}/checkpoints/iter_0000019_hf${TAU2_HF_SUFFIX}"
  if [[ ! -f "${model_dir}/model.safetensors.index.json" ]]; then
    bash "${PROJECT_ROOT}/scripts/convert_tau2_areal_async_to_hf.sh" async
  fi
  for seed in 300; do
    bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_tau2_areal_async.sh" async "${seed}"
  done
done
