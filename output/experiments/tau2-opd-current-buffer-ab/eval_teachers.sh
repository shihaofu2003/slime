#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
for model_key in airline retail; do
  bash "${PROJECT_ROOT}/output/experiments/tau2-opd-current-buffer-ab/eval_model.sh" "${model_key}" both
done
