#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${TAU2_SMOKE_RUN_NAME:?Set the completed smoke run name}"
: "${TAU2_SMOKE_LOG:?Set the completed smoke run log path}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-async-db-count-four-domain"
SOURCE_ROOT="${EXPERIMENT_DIR}/arms/async/${TAU2_SMOKE_RUN_NAME}"
export TAU2_RUN_NAME="${TAU2_RUN_NAME:-$(date +%Y%m%d_%H%M%S)-progress-db-count-v1-resume-smoke}"
RESUME_ROOT="${EXPERIMENT_DIR}/arms/async/${TAU2_RUN_NAME}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/examples/tau2-bench/analysis:${PYTHONPATH:-}"
mkdir -p "${EXPERIMENT_DIR}/reports"
python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/inspect_tau2_ready_smoke.py" \
  --logs "${TAU2_SMOKE_LOG}" --trajectories "${SOURCE_ROOT}/trajectories/smoke.jsonl" \
  --output "${EXPERIMENT_DIR}/reports/smoke.json"
python3 - "${EXPERIMENT_DIR}/reports/smoke.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert not report["violations"], report["violations"]
assert report["accepted_groups"] == 25 and report["accepted_trajectories"] == 200
PY
export LOAD_DIR="${SOURCE_ROOT}/checkpoints"
export CKPT_STEP=2
export SMOKE_UPDATES=5
bash "${PROJECT_ROOT}/scripts/run_tau2_async_four_domain.sh" progress-db-count-v1 smoke
python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/inspect_tau2_ready_resume.py" \
  --initial-state "${SOURCE_ROOT}/checkpoints/rollout/global_dataset_state_dict_2.pt" \
  --final-state "${RESUME_ROOT}/checkpoints/rollout/global_dataset_state_dict_4.pt" \
  --trajectories "${RESUME_ROOT}/trajectories/smoke.jsonl" \
  --dataset "${EXPERIMENT_DIR}/data/train.jsonl" --resume-version 3 \
  --output "${EXPERIMENT_DIR}/reports/resume-smoke.json"
python3 - "${EXPERIMENT_DIR}/reports/resume-smoke.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert not report["violations"], report["violations"]
assert report["consumed_groups_in_recovery_run"] == 10
PY
