#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
SFT_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2"
LONG100_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
LONG200_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long200"
SFT_GATE="${TAU2_SFT_PROMOTION_GATE:-${SFT_EXPERIMENT}/SFT_PROMOTION.json}"
SOURCE_CKPT_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
LONG200_CKPT_ROOT="${TAU2_RL_LONG200_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805}"
RESULTS_ROOT="${SERVICE_AGENT_ROOT}/tau2-bench"
ANALYSIS_DIR="${PROJECT_ROOT}/examples/tau2-bench/analysis"
SEEDS=(300 301)

if [[ $# -eq 2 && "$1" == "early-stop" ]]; then
  EARLY_STOP_HEALTH="$2"
  if [[ ! -f "${EARLY_STOP_HEALTH}" ]]; then
    echo "[ERROR] missing early-stop health artifact: ${EARLY_STOP_HEALTH}" >&2
    exit 1
  fi
  exec python3 "${ANALYSIS_DIR}/check_agent_boundary_rl_long200.py" \
    --checkpoint-root "${LONG200_CKPT_ROOT}" \
    --source-checkpoint-root "${SOURCE_CKPT_ROOT}" \
    --early-stop-health "${EARLY_STOP_HEALTH}" \
    --candidate long200_iter199 \
    --baseline selected_sft \
    --prior long100_iter99 \
    --output "${LONG200_EXPERIMENT}/ITER199_LONG200_DECISION.json"
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [early-stop FAILED_PARTIAL_HEALTH_GATE]" >&2
  exit 2
fi

SELECTED_ARM="$(python3 - "${SFT_GATE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
gate = json.loads(path.read_text(encoding="utf-8"))
if (
    gate.get("status") != "pass"
    or gate.get("gate_version") != "turn-aware-rl-v3"
    or gate.get("selected") != "contract-boundary"
):
    raise SystemExit(f"[ERROR] invalid Contract + boundary SFT gate: {path}")
print(gate["selected"])
PY
)"

summary_path() {
  local root="$1" label="$2" seed="$3"
  printf '%s/eval/%s/seed%s_summary.json' "${root}" "${label}" "${seed}"
}

REQUIRED_PATHS=(
  "${LONG200_EXPERIMENT}/ITER199_HEALTH_GATE.json"
  "${LONG200_EXPERIMENT}/eval/iter199/namespace_probe.json"
)
for seed in "${SEEDS[@]}"; do
  REQUIRED_PATHS+=(
    "$(summary_path "${SFT_EXPERIMENT}" "${SELECTED_ARM}" "${seed}")"
    "$(summary_path "${LONG100_EXPERIMENT}" iter99 "${seed}")"
    "$(summary_path "${LONG200_EXPERIMENT}" iter199 "${seed}")"
  )
done
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing required long200 artifact: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${LONG200_EXPERIMENT}/eval/comparisons"
DECISION_ARGS=(
  --checkpoint-root "${LONG200_CKPT_ROOT}"
  --source-checkpoint-root "${SOURCE_CKPT_ROOT}"
  --iter199-health "${LONG200_EXPERIMENT}/ITER199_HEALTH_GATE.json"
  --namespace-probe "${LONG200_EXPERIMENT}/eval/iter199/namespace_probe.json"
  --candidate long200_iter199
  --baseline selected_sft
  --prior long100_iter99
  --output "${LONG200_EXPERIMENT}/ITER199_LONG200_DECISION.json"
)
for seed in "${SEEDS[@]}"; do
  sft_summary="$(summary_path "${SFT_EXPERIMENT}" "${SELECTED_ARM}" "${seed}")"
  iter99_summary="$(summary_path "${LONG100_EXPERIMENT}" iter99 "${seed}")"
  iter199_summary="$(summary_path "${LONG200_EXPERIMENT}" iter199 "${seed}")"
  comparison="${LONG200_EXPERIMENT}/eval/comparisons/iter199_seed${seed}.json"
  python3 "${ANALYSIS_DIR}/compare_official_sft_evals.py" \
    --results-root "${RESULTS_ROOT}" \
    --protocol-profile agent-owned-dependency-safe-multi \
    --model "selected_sft=${sft_summary}" \
    --model "long100_iter99=${iter99_summary}" \
    --model "long200_iter199=${iter199_summary}" \
    --candidate long200_iter199 \
    --bootstrap-seed "$((20261105 + seed))" \
    --json-output "${comparison}" \
    --markdown-output "${comparison%.json}.md"
  DECISION_ARGS+=(
    --sft-summary "${sft_summary}"
    --iter99-summary "${iter99_summary}"
    --iter199-summary "${iter199_summary}"
    --comparison "${comparison}"
  )
done

exec python3 "${ANALYSIS_DIR}/check_agent_boundary_rl_long200.py" "${DECISION_ARGS[@]}"
