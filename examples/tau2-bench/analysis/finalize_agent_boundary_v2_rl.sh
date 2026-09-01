#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 TRAIN80_RUN_LOG" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
SFT_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2"
RL_EXPERIMENT="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2"
SFT_GATE="${TAU2_SFT_PROMOTION_GATE:-${SFT_EXPERIMENT}/SFT_PROMOTION.json}"
RL_PILOT_GATE="${TAU2_RL_PILOT_GATE:-${RL_EXPERIMENT}/ITER19_SAFETY_GATE.json}"
RL_CKPT_ROOT="${RL_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_20260804}"
RESULTS_ROOT="${SERVICE_AGENT_ROOT}/tau2-bench"
ANALYSIS_DIR="${PROJECT_ROOT}/examples/tau2-bench/analysis"
TRAINING_LOG="$1"
SEEDS=(300 301)

for path in "${SFT_GATE}" "${RL_PILOT_GATE}" "${TRAINING_LOG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing required artifact: ${path}" >&2
    exit 1
  fi
done

SELECTED_ARM="$(python3 - "${SFT_GATE}" "${RL_PILOT_GATE}" "${RL_CKPT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

sft_path, pilot_path, checkpoint = map(Path, sys.argv[1:])
sft = json.loads(sft_path.read_text(encoding="utf-8"))
pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
selected = sft.get("selected")
if sft.get("status") != "pass" or selected not in {"contract-only", "contract-boundary"}:
    raise SystemExit(f"[ERROR] invalid SFT promotion gate: {sft_path}")
if (
    sft.get("gate_version") != "turn-aware-rl-v3"
    or selected != "contract-boundary"
    or
    pilot.get("status") != "pass"
    or pilot.get("stage") != "iter19"
    or pilot.get("expected_latest") != 19
    or pilot.get("checkpoint_root") != str(checkpoint.resolve())
    or pilot.get("namespace_non_regression") is not True
):
    raise SystemExit(f"[ERROR] invalid iter19 safety gate: {pilot_path}")
print(selected)
PY
)"

summary_path() {
  local root="$1" label="$2" seed="$3"
  printf '%s/eval/%s/seed%s_summary.json' "${root}" "${label}" "${seed}"
}

mkdir -p "${RL_EXPERIMENT}/eval/comparisons"
for seed in "${SEEDS[@]}"; do
  sft_summary="$(summary_path "${SFT_EXPERIMENT}" "${SELECTED_ARM}" "${seed}")"
  iter19_summary="$(summary_path "${RL_EXPERIMENT}" iter19 "${seed}")"
  iter99_summary="$(summary_path "${RL_EXPERIMENT}" iter99 "${seed}")"
  comparison="${RL_EXPERIMENT}/eval/comparisons/iter99_seed${seed}.json"
  python3 "${ANALYSIS_DIR}/compare_official_sft_evals.py" \
    --results-root "${RESULTS_ROOT}" \
    --protocol-profile agent-owned-dependency-safe-multi \
    --model "selected_sft=${sft_summary}" \
    --model "rl_iter19=${iter19_summary}" \
    --model "rl_iter99=${iter99_summary}" \
    --candidate rl_iter99 \
    --bootstrap-seed "$((20260904 + seed))" \
    --json-output "${comparison}" \
    --markdown-output "${comparison%.json}.md"
done

GATE_ARGS=(
  --checkpoint-root "${RL_CKPT_ROOT}"
  --training-log "${TRAINING_LOG}"
  --trajectories "${RL_EXPERIMENT}/trajectories/iter0020-0099.jsonl"
  --candidate rl_iter99
  --baseline selected_sft
  --prior rl_iter19
  --namespace-probe "${RL_EXPERIMENT}/eval/iter99/namespace_probe.json"
  --window 3840
  --output "${RL_EXPERIMENT}/ITER99_PROMOTION_GATE.json"
)
for seed in "${SEEDS[@]}"; do
  GATE_ARGS+=(
    --sft-summary "$(summary_path "${SFT_EXPERIMENT}" "${SELECTED_ARM}" "${seed}")"
    --iter19-summary "$(summary_path "${RL_EXPERIMENT}" iter19 "${seed}")"
    --iter99-summary "$(summary_path "${RL_EXPERIMENT}" iter99 "${seed}")"
    --comparison "${RL_EXPERIMENT}/eval/comparisons/iter99_seed${seed}.json"
  )
done

exec python3 "${ANALYSIS_DIR}/check_agent_boundary_rl_final.py" "${GATE_ARGS[@]}"
