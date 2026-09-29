#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python3 tests/test_tau2_opd_training.py
