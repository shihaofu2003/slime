#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?}"
python3 -u scripts/watch_tau2_four_domain_rewards.py "$@"
