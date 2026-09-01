#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_OUTPUTS="${PROJECT_ROOT}/examples/tau2-bench/eval/official/outputs"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-local-first-dependency-safe-k2-fieldreward"
ITERATION="${1:-99}"

if [[ $# -gt 1 || ( "${ITERATION}" != "99" && "${ITERATION}" != "199" ) ]]; then
  echo "Usage: $0 [99|199]" >&2
  exit 2
fi

RAW_SUMMARY="${RAW_SUMMARY:-${OFFICIAL_OUTPUTS}/Qwen3-4B-Instruct-2507/dependency_safe_multi_user_stop_parser_pass4_summary.json}"
OLD_SFT_SUMMARY="${OLD_SFT_SUMMARY:-${OFFICIAL_OUTPUTS}/Qwen3-4B-tau2-agent-sft-multitool-iter0002413/dependency_safe_multi_user_stop_parser_pass4_summary.json}"
NEW_SFT_SUMMARY="${NEW_SFT_SUMMARY:-${OFFICIAL_OUTPUTS}/Qwen3-4B-tau2-agent-sft-local-first-relaxed/dependency_safe_multi_user_stop_parser_pass4_summary.json}"
OLD_RL_SUMMARY="${OLD_RL_SUMMARY:-${OFFICIAL_OUTPUTS}/Qwen3-4B-tau2-agent-rl-stability-lr2e6-iter0000099/user_stop_parser_pass4_summary.json}"
NEW_RL100_DEFAULT="${OFFICIAL_OUTPUTS}/Qwen3-4B-tau2-agent-rl-local-first-dependency-safe-lr2e6-iter0000099/dependency_safe_multi_user_stop_parser_pass4_summary.json"
if [[ ! -f "${NEW_RL100_DEFAULT}" ]]; then
  # The completed iter99 eval predates preservation of the model wrapper's
  # explicit output name.  Reuse that result instead of rerunning 400 sims.
  NEW_RL100_DEFAULT="${OFFICIAL_OUTPUTS}/Qwen3-4B-tau2-agent-rl-local-first-dependency-safe-lr2e6-iter0000099/user_stop_parser_pass4_summary.json"
fi
NEW_RL100_SUMMARY="${NEW_RL100_SUMMARY:-${NEW_RL100_DEFAULT}}"

MODELS=(
  --model "raw=${RAW_SUMMARY}"
  --model "old_sft=${OLD_SFT_SUMMARY}"
  --model "new_sft=${NEW_SFT_SUMMARY}"
  --model "old_rl100=${OLD_RL_SUMMARY}"
  --model "new_rl100=${NEW_RL100_SUMMARY}"
)
if [[ "${ITERATION}" == "99" ]]; then
  CANDIDATE="new_rl100"
  OUTPUT_STEM="ITER99_COMPARISON"
else
  NEW_RL200_DEFAULT="${OFFICIAL_OUTPUTS}/Qwen3-4B-tau2-agent-rl-local-first-dependency-safe-lr2e6-iter0000199/dependency_safe_multi_user_stop_parser_pass4_summary.json"
  if [[ ! -f "${NEW_RL200_DEFAULT}" ]]; then
    NEW_RL200_DEFAULT="${OFFICIAL_OUTPUTS}/Qwen3-4B-tau2-agent-rl-local-first-dependency-safe-lr2e6-iter0000199/user_stop_parser_pass4_summary.json"
  fi
  NEW_RL200_SUMMARY="${NEW_RL200_SUMMARY:-${NEW_RL200_DEFAULT}}"
  MODELS+=(--model "new_rl200=${NEW_RL200_SUMMARY}")
  CANDIDATE="new_rl200"
  OUTPUT_STEM="ITER199_COMPARISON"
fi

exec python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/compare_official_sft_evals.py" \
  "${MODELS[@]}" \
  --candidate "${CANDIDATE}" \
  --results-root "${SERVICE_AGENT_ROOT}/tau2-bench" \
  --protocol-profile dependency-safe-multi \
  --legacy-model raw \
  --legacy-model old_sft \
  --legacy-model new_sft \
  --legacy-model old_rl100 \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-100000}" \
  --bootstrap-seed "${BOOTSTRAP_SEED:-20260803}" \
  --json-output "${EXPERIMENT_DIR}/${OUTPUT_STEM}.json" \
  --markdown-output "${EXPERIMENT_DIR}/${OUTPUT_STEM}.md"
