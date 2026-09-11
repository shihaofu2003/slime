#!/usr/bin/env bash
set -euo pipefail
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${SERVICE_AGENT_ROOT}/slime"
export PYTHONPATH="${PROJECT_ROOT}:${SERVICE_AGENT_ROOT}/tau2-bench/src:${PYTHONPATH:-}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TOKENIZERS_PARALLELISM=false
export BANKING_SIMPLIFY_INTEGRATION=1
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"
python3 -c 'from importlib.metadata import version; print({name:version(name) for name in ["transformers","tokenizers","sglang","torch"]})'
python3 tests/test_tau2_banking_expert_sft.py
python3 tests/test_tau2_sft_quality.py
python3 tests/test_tau2_banking_synthetic.py
echo BANKING_SIMPLIFICATION_CPU_CHECKS_OK
