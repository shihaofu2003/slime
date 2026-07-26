#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
SFT_DIR="${PROJECT_ROOT}/examples/tau2-bench/sft"

APIGEN_PATH="${APIGEN_PATH:-${SERVICE_AGENT_ROOT}/datasets/APIGen-MT-5k/apigen-mt_5k.json}"
AREAL_PATH="${AREAL_PATH:-${SERVICE_AGENT_ROOT}/datasets/AReaL-tau2-data/tau2_sft_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SERVICE_AGENT_ROOT}/datasets/tau2-bench-user-sft}"
USER_GUIDELINES_PATH="${USER_GUIDELINES_PATH:-${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/user_simulator/simulation_guidelines.md}"

ARGS=(
  --apigen-path "${APIGEN_PATH}"
  --areal-path "${AREAL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --user-guidelines-path "${USER_GUIDELINES_PATH}"
)

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi
if [[ "${SKIP_APIGEN:-0}" != "0" ]]; then
  ARGS+=(--skip-apigen)
fi
if [[ "${SKIP_AREAL:-0}" != "0" ]]; then
  ARGS+=(--skip-areal)
fi

cd "${PROJECT_ROOT}"
python3 "${SFT_DIR}/prepare_user_sft_data.py" "${ARGS[@]}"
