#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export EXPERIMENT_NAME=tau2-opd-four-domain-20260923
export TAU2_OPD_DOMAINS=airline,retail,telecom,banking
export TAU2_OPD_RUN_TAG=teachers
export TAU2_OPD_TEACHER_PERSISTENT=1
cd "${PROJECT_ROOT}"
python3 -m pytest examples/tau2-bench/opd/test_tau2_opd.py -q
exec bash examples/tau2-bench/opd/run_tau2_opd_airline_retail.sh pilot teacher
