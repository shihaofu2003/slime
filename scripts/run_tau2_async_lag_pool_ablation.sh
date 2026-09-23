#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_NAME="${TAU2_ABLATION_NAME:-lag-pool-20-$(date +%Y%m%d-%H%M%S)}"
TRAIN_SCRIPT="${PROJECT_ROOT}/examples/tau2-bench/rl/run_tau2_areal_async.sh"

# One 8-GPU job runs the controlled arms serially so every arm uses the same
# hardware allocation. The first arm preserves the current recipe; later arms
# isolate lag removal and pool enlargement.
configs=(
  "baseline-p10-lag1|10|1"
  "unbounded-p10|10|-1"
  "unbounded-p20|20|-1"
)
for config in "${configs[@]}"; do
  IFS='|' read -r label pool lag <<<"${config}"
  export TAU2_RUN_NAME="${BASE_NAME}-${label}"
  export TAU2_POOL_CAPACITY="${pool}"
  export TAU2_MAX_POLICY_LAG="${lag}"
  echo "tau2_ablation start run=${TAU2_RUN_NAME} pool=${pool} max_policy_lag=${lag}"
  bash "${TRAIN_SCRIPT}" async train20
  echo "tau2_ablation end run=${TAU2_RUN_NAME}"
done
