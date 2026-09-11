#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2"
RESULTS_ROOT="${SERVICE_AGENT_ROOT}/tau2-bench"
ANALYSIS_DIR="${PROJECT_ROOT}/examples/tau2-bench/analysis"
SEEDS=(300 301)
ARMS=(contract-only contract-boundary)

summary_path() {
  local label="$1" seed="$2"
  printf '%s/eval/%s/seed%s_summary.json' "${EXPERIMENT_DIR}" "${label}" "${seed}"
}

mkdir -p "${EXPERIMENT_DIR}/eval/comparisons"

for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    python3 "${ANALYSIS_DIR}/compare_official_sft_evals.py" \
      --results-root "${RESULTS_ROOT}" \
      --protocol-profile agent-owned-dependency-safe-multi \
      --model "raw_instruct=$(summary_path raw_instruct "${seed}")" \
      --model "old_sft=$(summary_path old_sft "${seed}")" \
      --model "current_sft=$(summary_path current_sft "${seed}")" \
      --model "contract-only=$(summary_path contract-only "${seed}")" \
      --model "contract-boundary=$(summary_path contract-boundary "${seed}")" \
      --candidate "${arm}" \
      --bootstrap-seed "$((20260804 + seed))" \
      --json-output "${EXPERIMENT_DIR}/eval/comparisons/seed${seed}_${arm}.json" \
      --markdown-output "${EXPERIMENT_DIR}/eval/comparisons/seed${seed}_${arm}.md"
  done
done

GATE_ARGS=(
  --candidate contract-only
  --candidate contract-boundary
  --checkpoint "contract-only=${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_only_20260804"
  --checkpoint "contract-boundary=${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_boundary_20260804"
  --probe "contract-only=${EXPERIMENT_DIR}/eval/contract-only/namespace_probe.json"
  --probe "contract-boundary=${EXPERIMENT_DIR}/eval/contract-boundary/namespace_probe.json"
  --baseline old_sft
  --capability-baseline raw_instruct
  --capability-baseline old_sft
  --capability-baseline current_sft
  --output "${EXPERIMENT_DIR}/SFT_PROMOTION.json"
)
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    GATE_ARGS+=(
      --eval-summary "${arm}=$(summary_path "${arm}" "${seed}")"
      --comparison "${arm}=${EXPERIMENT_DIR}/eval/comparisons/seed${seed}_${arm}.json"
    )
  done
done

exec python3 "${ANALYSIS_DIR}/check_agent_boundary_sft_gate.py" "${GATE_ARGS[@]}"
