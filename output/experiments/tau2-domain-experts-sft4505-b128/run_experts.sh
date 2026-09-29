#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2
cd "${PROJECT_ROOT}"
export EXPERIMENT_NAME=tau2-domain-experts-sft4505-b128
export RUN_STAMP=20260920_sft4505
export SFT_CKPT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/iter_0004505_hf"
export REF_LOAD="${SFT_CKPT_ROOT}"
export REF_CKPT_STEP=4505
case "${1:-}" in
  three-domain) export DOMAIN_SPECS='airline:60 retail:30 telecom:20' ;;
  banking) export DOMAIN_SPECS='banking:30' ;;
  *) echo 'Usage: run_experts.sh <three-domain|banking>' >&2; exit 2 ;;
esac
export SMOKE_DOMAINS='airline retail telecom banking'
user_log="${PROJECT_ROOT}/output/experiments/tau2-external-user-pool/jobs/22031-tau2-user-pool-refresh-sft4505-0920-230958466/run_0_20260920_230958466.log"
# The replacement service has a new compute IP, printed after both replicas load.
for ((attempt=0; attempt<240; attempt++)); do
  if [[ -f "${user_log}" ]]; then
    export TAU2_USER_API_BASE="$(sed -n 's/.*USER SERVICE READY: TAU2_USER_API_BASE=\(http[^[:space:]]*\).*/\1/p' "${user_log}" | tail -n 1)"
    if [[ -n "${TAU2_USER_API_BASE}" ]] && curl -fsS --max-time 5 "${TAU2_USER_API_BASE}/models" >/dev/null; then
      echo "EXPERT_USER_READY job=22031 api=${TAU2_USER_API_BASE}"
      exec bash scripts/run_tau2_domain_experts_serial.sh
    fi
  fi
  if (( attempt % 10 == 0 )); then echo 'Waiting for User job22031 readiness.'; fi
  sleep 30
done
echo 'User job22031 did not become reachable within two hours.' >&2
exit 1
