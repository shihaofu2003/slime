#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 iter9 PILOT_A_RUN_LOG | $0 iter19 PILOT_A_RUN_LOG PILOT_B_RUN_LOG" >&2
  exit 2
fi

STAGE="$1"
shift
case "${STAGE}" in
  iter9)
    [[ $# -eq 1 ]] || { echo "[ERROR] iter9 requires one training log" >&2; exit 2; }
    EXPECTED_ITERATION=9
    LABEL="iter9"
    WINDOW=480
    PILOT_A_LOG="$1"
    PILOT_B_LOG=""
    ;;
  iter19)
    [[ $# -eq 2 ]] || { echo "[ERROR] iter19 requires Pilot A and Pilot B logs" >&2; exit 2; }
    EXPECTED_ITERATION=19
    LABEL="iter19"
    WINDOW=960
    PILOT_A_LOG="$1"
    PILOT_B_LOG="$2"
    ;;
  *)
    echo "[ERROR] stage must be iter9 or iter19" >&2
    exit 2
    ;;
esac

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
SFT_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2"
RL_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2"
SFT_GATE="${TAU2_SFT_PROMOTION_GATE:-${SFT_EXPERIMENT}/SFT_PROMOTION.json}"
RL_CKPT_ROOT="${RL_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_20260804}"
RESULTS_ROOT="${SERVICE_AGENT_ROOT}/tau2-bench"
ANALYSIS_DIR="${PROJECT_ROOT}/examples/tau2-bench/analysis"
SEEDS=(300 301)

REQUIRED_PATHS=("${SFT_GATE}" "${PILOT_A_LOG}")
if [[ -n "${PILOT_B_LOG}" ]]; then
  REQUIRED_PATHS+=("${PILOT_B_LOG}")
fi
for path in "${REQUIRED_PATHS[@]}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] missing required artifact: ${path}" >&2; exit 1; }
done

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
    raise SystemExit(f"[ERROR] invalid v3 SFT promotion gate: {path}")
print(gate["selected"])
PY
)"

summary_path() {
  local root="$1" label="$2" seed="$3"
  printf '%s/eval/%s/seed%s_summary.json' "${root}" "${label}" "${seed}"
}

mkdir -p "${RL_EXPERIMENT}/eval/comparisons"
for seed in "${SEEDS[@]}"; do
  sft_summary="$(summary_path "${SFT_EXPERIMENT}" "${SELECTED_ARM}" "${seed}")"
  rl_summary="$(summary_path "${RL_EXPERIMENT}" "${LABEL}" "${seed}")"
  comparison="${RL_EXPERIMENT}/eval/comparisons/${LABEL}_seed${seed}.json"
  python3 "${ANALYSIS_DIR}/compare_official_sft_evals.py" \
    --results-root "${RESULTS_ROOT}" \
    --protocol-profile agent-owned-dependency-safe-multi \
    --model "selected_sft=${sft_summary}" \
    --model "rl_${LABEL}=${rl_summary}" \
    --candidate "rl_${LABEL}" \
    --bootstrap-seed "$((20260804 + EXPECTED_ITERATION * 100 + seed))" \
    --json-output "${comparison}" \
    --markdown-output "${comparison%.json}.md"
done

GATE_ARGS=(
  --stage "${STAGE}"
  --checkpoint-root "${RL_CKPT_ROOT}"
  --pilot-a-training-log "${PILOT_A_LOG}"
  --pilot-a-trajectories "${RL_EXPERIMENT}/trajectories/iter0000-0009.jsonl"
  --namespace-probe "${RL_EXPERIMENT}/eval/${LABEL}/namespace_probe.json"
  --candidate "rl_${LABEL}"
  --baseline selected_sft
  --window "${WINDOW}"
  --output "${RL_EXPERIMENT}/$(printf 'ITER%s_SAFETY_GATE.json' "${EXPECTED_ITERATION}")"
)
if [[ "${STAGE}" == "iter19" ]]; then
  GATE_ARGS+=(
    --pilot-b-training-log "${PILOT_B_LOG}"
    --pilot-b-trajectories "${RL_EXPERIMENT}/trajectories/iter0010-0019.jsonl"
  )
fi
for seed in "${SEEDS[@]}"; do
  GATE_ARGS+=(
    --sft-summary "$(summary_path "${SFT_EXPERIMENT}" "${SELECTED_ARM}" "${seed}")"
    --pilot-summary "$(summary_path "${RL_EXPERIMENT}" "${LABEL}" "${seed}")"
    --comparison "${RL_EXPERIMENT}/eval/comparisons/${LABEL}_seed${seed}.json"
  )
done

exec python3 "${ANALYSIS_DIR}/check_agent_boundary_rl_pilot.py" "${GATE_ARGS[@]}"
