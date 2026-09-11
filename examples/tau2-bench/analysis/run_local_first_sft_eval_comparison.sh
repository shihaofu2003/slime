#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_OUTPUTS="${PROJECT_ROOT}/examples/tau2-bench/eval/official/outputs"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-local-first-relaxed"

RAW_SUMMARY="${RAW_SUMMARY:-${OFFICIAL_OUTPUTS}/Qwen3-4B-Instruct-2507/dependency_safe_multi_user_stop_parser_pass4_summary.json}"
OLD_SFT_SUMMARY="${OLD_SFT_SUMMARY:-${OFFICIAL_OUTPUTS}/Qwen3-4B-tau2-agent-sft-multitool-iter0002413/dependency_safe_multi_user_stop_parser_pass4_summary.json}"
NEW_SFT_SUMMARY="${NEW_SFT_SUMMARY:-${OFFICIAL_OUTPUTS}/Qwen3-4B-tau2-agent-sft-local-first-relaxed/dependency_safe_multi_user_stop_parser_pass4_summary.json}"

exec python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/compare_official_sft_evals.py" \
  --model "raw=${RAW_SUMMARY}" \
  --model "old_sft=${OLD_SFT_SUMMARY}" \
  --model "new_sft=${NEW_SFT_SUMMARY}" \
  --candidate new_sft \
  --results-root "${SERVICE_AGENT_ROOT}/tau2-bench" \
  --protocol-profile dependency-safe-multi \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-100000}" \
  --bootstrap-seed "${BOOTSTRAP_SEED:-20260803}" \
  --json-output "${EXPERIMENT_DIR}/EVAL_COMPARISON.json" \
  --markdown-output "${EXPERIMENT_DIR}/EVAL_COMPARISON.md"
