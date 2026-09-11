#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"

cd "${PROJECT_ROOT}"
python3 -c "import sglang_router; import sglang_router.launch_router; print('sglang-router', sglang_router.__version__)"
python3 -m sglang_router.launch_router --help >/dev/null
python3 tests/test_tau2_official_async_eval.py
python3 examples/tau2-bench/rl/test_rollout_logic.py
python3 -m py_compile \
  "${OFFICIAL_DIR}/run_eval.py" \
  "${OFFICIAL_DIR}/sglang_agent.py" \
  "${OFFICIAL_DIR}/timed_user.py" \
  "${OFFICIAL_DIR}/model_request.py"
bash -n \
  "${OFFICIAL_DIR}/run_eval.sh" \
  "${OFFICIAL_DIR}/models/run_full_qwen3_4b_qwen36_user_async_timed.sh" \
  "${OFFICIAL_DIR}/models/run_qwen3_4b_qwen36_user_async_four_domain.sh" \
  "${OFFICIAL_DIR}/models/run_qwen3_5_4b_qwen36_user_async_four_domain.sh"
