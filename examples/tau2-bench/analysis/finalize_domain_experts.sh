#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"
DOMAIN="${2:-}"
if [[ ! "${MODE}" =~ ^(curve|decision|summary)$ ]]; then
  echo "Usage: $0 curve | decision <airline|retail|telecom> | summary" >&2
  exit 2
fi
if [[ "${MODE}" == "decision" && ! "${DOMAIN}" =~ ^(airline|retail|telecom)$ ]]; then
  echo "[ERROR] decision mode requires airline|retail|telecom" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts"
LONG100_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
RESULTS_ROOT="${SERVICE_AGENT_ROOT}/tau2-bench"
ANALYZER="${PROJECT_ROOT}/examples/tau2-bench/analysis/summarize_domain_experts.py"

case "${MODE}" in
  curve)
    ARGS=(
      curve
      --baseline-summary "${LONG100_DIR}/eval/iter99/seed300_summary.json"
      --results-root "${RESULTS_ROOT}"
      --json-output "${EXPERIMENT_DIR}/DOMAIN_EXPERT_CURVE.json"
      --markdown-output "${EXPERIMENT_DIR}/DOMAIN_EXPERT_CURVE.md"
    )
    for iteration in 109 119 129; do
      ARGS+=(
        --mixed-summary "${iteration}=${EXPERIMENT_DIR}/eval/mixed/iter${iteration}/seed300_summary.json"
        --health-gate "mixed:${iteration}=${EXPERIMENT_DIR}/ITER${iteration}_PREFIX_HEALTH_GATE.json"
      )
      for domain in airline retail telecom; do
        ARGS+=(
          --expert-summary "${domain}:${iteration}=${EXPERIMENT_DIR}/eval/expert/${domain}/iter${iteration}/target/seed300_summary.json"
          --health-gate "${domain}:${iteration}=${EXPERIMENT_DIR}/${domain^^}_ITER${iteration}_HEALTH_GATE.json"
        )
      done
    done
    exec python3 "${ANALYZER}" "${ARGS[@]}"
    ;;
  decision)
    ITERATION="$(python3 - "${EXPERIMENT_DIR}/DOMAIN_EXPERT_CURVE.json" "${DOMAIN}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoints"][sys.argv[2]]["iteration"])
PY
)"
    UPPER_DOMAIN="${DOMAIN^^}"
    exec python3 "${ANALYZER}" decision \
      --domain "${DOMAIN}" \
      --iteration "${ITERATION}" \
      --expert-summary "300=${EXPERIMENT_DIR}/eval/expert/${DOMAIN}/iter${ITERATION}/target/seed300_summary.json" \
      --expert-summary "301=${EXPERIMENT_DIR}/eval/expert/${DOMAIN}/iter${ITERATION}/target/seed301_summary.json" \
      --baseline-summary "300=${LONG100_DIR}/eval/iter99/seed300_summary.json" \
      --baseline-summary "301=${LONG100_DIR}/eval/iter99/seed301_summary.json" \
      --mixed-summary "300=${EXPERIMENT_DIR}/eval/mixed/iter${ITERATION}/seed300_summary.json" \
      --mixed-summary "301=${EXPERIMENT_DIR}/eval/mixed/iter${ITERATION}/seed301_summary.json" \
      --forgetting-summary "${EXPERIMENT_DIR}/eval/expert/${DOMAIN}/iter${ITERATION}/forgetting/seed300_summary.json" \
      --health-gate "${EXPERIMENT_DIR}/${UPPER_DOMAIN}_ITER${ITERATION}_HEALTH_GATE.json" \
      --namespace-probe "${EXPERIMENT_DIR}/eval/expert/${DOMAIN}/iter${ITERATION}/target/namespace_probe.json" \
      --results-root "${RESULTS_ROOT}" \
      --json-output "${EXPERIMENT_DIR}/${UPPER_DOMAIN}_EXPERT_DECISION.json" \
      --markdown-output "${EXPERIMENT_DIR}/${UPPER_DOMAIN}_EXPERT_DECISION.md"
    ;;
  summary)
    exec python3 "${ANALYZER}" summary \
      --decision "airline=${EXPERIMENT_DIR}/AIRLINE_EXPERT_DECISION.json" \
      --decision "retail=${EXPERIMENT_DIR}/RETAIL_EXPERT_DECISION.json" \
      --decision "telecom=${EXPERIMENT_DIR}/TELECOM_EXPERT_DECISION.json" \
      --json-output "${EXPERIMENT_DIR}/DOMAIN_EXPERT_SUMMARY.json" \
      --markdown-output "${EXPERIMENT_DIR}/DOMAIN_EXPERT_SUMMARY.md"
    ;;
esac
