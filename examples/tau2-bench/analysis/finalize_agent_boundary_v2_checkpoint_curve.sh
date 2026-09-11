#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CURVE_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-checkpoint-curve"
LONG100_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
HISTORICAL_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2"
SFT_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2"
ANALYSIS_DIR="${PROJECT_ROOT}/examples/tau2-bench/analysis"

MODEL_ARGS=(
  --model "selected_sft=${SFT_EXPERIMENT}/eval/contract-boundary/seed300_summary.json"
  --model "lineage_iter9=${CURVE_EXPERIMENT}/eval/iter9/seed300_summary.json"
  --model "lineage_iter39=${CURVE_EXPERIMENT}/eval/iter39/seed300_summary.json"
  --model "lineage_iter69=${CURVE_EXPERIMENT}/eval/iter69/seed300_summary.json"
  --model "lineage_iter89=${CURVE_EXPERIMENT}/eval/iter89/seed300_summary.json"
  --model "lineage_iter99=${LONG100_EXPERIMENT}/eval/iter99/seed300_summary.json"
  --model "historical_iter9=${HISTORICAL_EXPERIMENT}/eval/iter9/seed300_summary.json"
  --model "raw_instruct=${SFT_EXPERIMENT}/eval/raw_instruct/seed300_summary.json"
)
REQUIRED_PATHS=(
  "${CURVE_EXPERIMENT}/authorizations/curve_iter9.json"
  "${CURVE_EXPERIMENT}/authorizations/curve_iter39.json"
  "${CURVE_EXPERIMENT}/authorizations/curve_iter69.json"
  "${CURVE_EXPERIMENT}/authorizations/curve_iter89.json"
)
for argument in "${MODEL_ARGS[@]}"; do
  if [[ "${argument}" == *=* ]]; then
    REQUIRED_PATHS+=("${argument#*=}")
  fi
done
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing checkpoint-curve artifact: ${path}" >&2
    exit 1
  fi
done

exec python3 "${ANALYSIS_DIR}/summarize_agent_boundary_checkpoint_curve.py" \
  --results-root "${SERVICE_AGENT_ROOT}/tau2-bench" \
  "${MODEL_ARGS[@]}" \
  --json-output "${CURVE_EXPERIMENT}/CHECKPOINT_CURVE.json" \
  --markdown-output "${CURVE_EXPERIMENT}/CHECKPOINT_CURVE.md"
