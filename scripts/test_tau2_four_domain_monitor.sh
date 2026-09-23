#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
python3 -m pytest tests/test_tau2_continuous.py -k four_domain_monitor -q
