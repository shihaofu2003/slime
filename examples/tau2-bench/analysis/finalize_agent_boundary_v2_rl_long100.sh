#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
SFT_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2"
HISTORICAL_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2"
LONG100_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
SFT_GATE="${TAU2_SFT_PROMOTION_GATE:-${SFT_EXPERIMENT}/SFT_PROMOTION.json}"
LONG100_CKPT_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
RESULTS_ROOT="${SERVICE_AGENT_ROOT}/tau2-bench"
ANALYSIS_DIR="${PROJECT_ROOT}/examples/tau2-bench/analysis"
SEEDS=(300 301)

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
  "${LONG100_EXPERIMENT}/ITER9_HEALTH_GATE.json"
  "${LONG100_EXPERIMENT}/ITER19_HEALTH_GATE.json"
  "${LONG100_EXPERIMENT}/ITER99_HEALTH_GATE.json"
  "${LONG100_EXPERIMENT}/eval/iter99/namespace_probe.json"
)
for seed in "${SEEDS[@]}"; do
  REQUIRED_PATHS+=(
    "$(summary_path "${SFT_EXPERIMENT}" "${SELECTED_ARM}" "${seed}")"
    "$(summary_path "${HISTORICAL_EXPERIMENT}" iter9 "${seed}")"
    "$(summary_path "${LONG100_EXPERIMENT}" iter99 "${seed}")"
  )
done
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing required long100 artifact: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${LONG100_EXPERIMENT}/eval/comparisons"
DECISION_ARGS=(
  --checkpoint-root "${LONG100_CKPT_ROOT}"
  --iter9-health "${LONG100_EXPERIMENT}/ITER9_HEALTH_GATE.json"
  --iter19-health "${LONG100_EXPERIMENT}/ITER19_HEALTH_GATE.json"
  --iter99-health "${LONG100_EXPERIMENT}/ITER99_HEALTH_GATE.json"
  --namespace-probe "${LONG100_EXPERIMENT}/eval/iter99/namespace_probe.json"
  --candidate long100_iter99
  --baseline selected_sft
  --prior historical_rl_iter9
  --output "${LONG100_EXPERIMENT}/ITER99_LONG100_DECISION.json"
)
for seed in "${SEEDS[@]}"; do
  sft_summary="$(summary_path "${SFT_EXPERIMENT}" "${SELECTED_ARM}" "${seed}")"
  iter9_summary="$(summary_path "${HISTORICAL_EXPERIMENT}" iter9 "${seed}")"
  iter99_summary="$(summary_path "${LONG100_EXPERIMENT}" iter99 "${seed}")"
  comparison="${LONG100_EXPERIMENT}/eval/comparisons/iter99_seed${seed}.json"
  python3 "${ANALYSIS_DIR}/compare_official_sft_evals.py" \
    --results-root "${RESULTS_ROOT}" \
    --protocol-profile agent-owned-dependency-safe-multi \
    --model "selected_sft=${sft_summary}" \
    --model "historical_rl_iter9=${iter9_summary}" \
    --model "long100_iter99=${iter99_summary}" \
    --candidate long100_iter99 \
    --bootstrap-seed "$((20261004 + seed))" \
    --json-output "${comparison}" \
    --markdown-output "${comparison%.json}.md"
  DECISION_ARGS+=(
    --sft-summary "${sft_summary}"
    --iter9-summary "${iter9_summary}"
    --iter99-summary "${iter99_summary}"
    --comparison "${comparison}"
  )
done

exec python3 "${ANALYSIS_DIR}/check_agent_boundary_rl_long100.py" "${DECISION_ARGS[@]}"
