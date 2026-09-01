#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
SFT_GATE="${TAU2_SFT_PROMOTION_GATE:-${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2/SFT_PROMOTION.json}"
VARIANT="${VARIANT:-legacy}"
ITERATION="${ITERATION:-9}"
EXPERT_DOMAIN="${EXPERT_DOMAIN:-}"

if [[ $# -gt 0 && "$1" =~ ^(legacy|long100|curve|long200|long200-kl-waiver|domain-expert|mixed-control)$ ]]; then
  VARIANT="$1"
  shift
fi
if [[ "${VARIANT}" == "domain-expert" && $# -gt 0 ]]; then
  EXPERT_DOMAIN="$1"
  shift
fi
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  ITERATION="$1"
  shift
fi
if [[ ! "${VARIANT}" =~ ^(legacy|long100|curve|long200|long200-kl-waiver|domain-expert|mixed-control)$ || ! "${ITERATION}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 domain-expert <airline|retail|telecom> <109|119|129> [--force|--dry-run-only]" >&2
  echo "       $0 mixed-control <109|119|129> [--force|--dry-run-only]" >&2
  echo "       $0 [legacy|long100|curve|long200|long200-kl-waiver] ITERATION [--force|--dry-run-only]" >&2
  exit 2
fi
case "${VARIANT}" in
  legacy|long100)
    if [[ ! "${ITERATION}" =~ ^(9|19|99)$ ]]; then
      echo "[ERROR] ${VARIANT} preserves the historical 9|19|99 iteration set" >&2
      exit 2
    fi
    ;;
  curve)
    if (( ITERATION < 9 || ITERATION > 99 || (ITERATION - 9) % 10 != 0 )); then
      echo "[ERROR] curve requires an authorized iter9/19/.../99 checkpoint" >&2
      exit 2
    fi
    ;;
  long200|long200-kl-waiver)
    if [[ "${ITERATION}" != "199" ]]; then
      echo "[ERROR] long200 conversion requires iter199" >&2
      exit 2
    fi
    ;;
  domain-expert)
    if [[ ! "${EXPERT_DOMAIN}" =~ ^(airline|retail|telecom)$ || ! "${ITERATION}" =~ ^(109|119|129)$ ]]; then
      echo "[ERROR] domain-expert requires airline|retail|telecom and iter109|119|129" >&2
      exit 2
    fi
    ;;
  mixed-control)
    if [[ -n "${EXPERT_DOMAIN}" || ! "${ITERATION}" =~ ^(109|119|129)$ ]]; then
      echo "[ERROR] mixed-control requires iter109|119|129 and no domain" >&2
      exit 2
    fi
    ;;
esac

LONG100_CKPT_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
if [[ "${VARIANT}" == "long100" ]]; then
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
  HF_OUTPUT_ROOT="${CHECKPOINT_ROOT}"
elif [[ "${VARIANT}" == "curve" ]]; then
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${LONG100_CKPT_ROOT}}"
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-checkpoint-curve"
  HF_OUTPUT_ROOT="${TAU2_RL_CURVE_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_checkpoint_curve_hf_20260805}"
elif [[ "${VARIANT}" == "long200" || "${VARIANT}" == "long200-kl-waiver" ]]; then
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805}"
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long200"
  if [[ "${VARIANT}" == "long200-kl-waiver" ]]; then
    HF_OUTPUT_ROOT="${TAU2_RL_LONG200_KL_WAIVER_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_kl_waiver_hf_20260805}"
  else
    HF_OUTPUT_ROOT="${TAU2_RL_LONG200_HF_ROOT:-${CHECKPOINT_ROOT}}"
  fi
elif [[ "${VARIANT}" == "domain-expert" ]]; then
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts"
  case "${EXPERT_DOMAIN}" in
    airline)
      CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${TAU2_RL_AIRLINE_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_airline_expert_20260806}}"
      HF_OUTPUT_ROOT="${TAU2_RL_AIRLINE_EXPERT_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_airline_expert_hf_20260806}"
      ;;
    retail)
      CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${TAU2_RL_RETAIL_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_retail_expert_20260806}}"
      HF_OUTPUT_ROOT="${TAU2_RL_RETAIL_EXPERT_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_retail_expert_hf_20260806}"
      ;;
    telecom)
      CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${TAU2_RL_TELECOM_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_telecom_expert_20260806}}"
      HF_OUTPUT_ROOT="${TAU2_RL_TELECOM_EXPERT_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_telecom_expert_hf_20260806}"
      ;;
  esac
elif [[ "${VARIANT}" == "mixed-control" ]]; then
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805}"
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts"
  HF_OUTPUT_ROOT="${TAU2_RL_DOMAIN_EXPERT_MIXED_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_domain_expert_mixed_control_hf_20260806}"
else
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_20260804}"
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2"
  HF_OUTPUT_ROOT="${CHECKPOINT_ROOT}"
fi

SFT_CKPT_ROOT="$(python3 - "${SFT_GATE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
gate = json.loads(path.read_text(encoding="utf-8"))
if gate.get("status") != "pass" or not gate.get("selected_checkpoint_root"):
    raise SystemExit(f"[ERROR] invalid SFT promotion gate: {path}")
print(str(Path(gate["selected_checkpoint_root"]).resolve()))
PY
)"

printf -v ITER_PADDED '%07d' "${ITERATION}"
LATEST_FILE="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
if [[ ! -f "${LATEST_FILE}" ]]; then
  echo "[ERROR] missing checkpoint marker: ${LATEST_FILE}" >&2
  exit 1
fi
LATEST="$(tr -d '[:space:]' < "${LATEST_FILE}")"
if [[ ! "${LATEST}" =~ ^[0-9]+$ || "${LATEST}" -lt "${ITERATION}" ]]; then
  echo "[ERROR] iteration ${ITERATION} is not complete; latest=${LATEST}" >&2
  exit 1
fi
if [[ "${VARIANT}" == "long100" ]]; then
  HEALTH_GATE="${EXPERIMENT_DIR}/ITER${ITERATION}_HEALTH_GATE.json"
  python3 - "${HEALTH_GATE}" "${CHECKPOINT_ROOT}" "${ITERATION}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
iteration = int(sys.argv[3])
if not path.is_file():
    raise SystemExit(f"[ERROR] missing long100 health gate: {path}")
gate = json.loads(path.read_text(encoding="utf-8"))
if (
    gate.get("status") != "pass"
    or gate.get("gate_version") != "turn-aware-rl-health-v1"
    or gate.get("stage") != f"iter{iteration}"
    or gate.get("checkpoint_root") != checkpoint
    or gate.get("expected_latest") != iteration
):
    raise SystemExit(f"[ERROR] long100 health gate does not authorize conversion: {path}")
PY
elif [[ "${VARIANT}" =~ ^(curve|long200|long200-kl-waiver|domain-expert|mixed-control)$ ]]; then
  if [[ "${VARIANT}" == "curve" ]]; then
    if [[ "${ITERATION}" == "9" ]]; then
      GATE_ITERATION=9
    elif [[ "${ITERATION}" == "19" ]]; then
      GATE_ITERATION=19
    else
      GATE_ITERATION=99
    fi
    HEALTH_GATE="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100/ITER${GATE_ITERATION}_HEALTH_GATE.json"
    REFERENCE_ROOT="${LONG100_CKPT_ROOT}"
  elif [[ "${VARIANT}" == "long200" ]]; then
    HEALTH_GATE="${EXPERIMENT_DIR}/ITER199_HEALTH_GATE.json"
    REFERENCE_ROOT="${LONG100_CKPT_ROOT}"
  elif [[ "${VARIANT}" == "long200-kl-waiver" ]]; then
    HEALTH_GATE="${EXPERIMENT_DIR}/ITER199_KL_WAIVER_HEALTH_GATE.json"
    REFERENCE_ROOT="${LONG100_CKPT_ROOT}"
  elif [[ "${VARIANT}" == "domain-expert" ]]; then
    HEALTH_GATE="${EXPERIMENT_DIR}/${EXPERT_DOMAIN^^}_ITER${ITERATION}_HEALTH_GATE.json"
    REFERENCE_ROOT="${LONG100_CKPT_ROOT}"
  else
    HEALTH_GATE="${EXPERIMENT_DIR}/ITER${ITERATION}_PREFIX_HEALTH_GATE.json"
    REFERENCE_ROOT="${LONG100_CKPT_ROOT}"
  fi
  AUTHORIZATION_NAME="${VARIANT//-/_}${EXPERT_DOMAIN:+_${EXPERT_DOMAIN}}_iter${ITERATION}.json"
  AUTHORIZATION="${EXPERIMENT_DIR}/authorizations/${AUTHORIZATION_NAME}"
  AUTHORIZATION_ARGS=(
    --variant "${VARIANT}" \
    --iteration "${ITERATION}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --health-gate "${HEALTH_GATE}" \
    --reference-root "${REFERENCE_ROOT}" \
    --output "${AUTHORIZATION}"
  )
  if [[ "${VARIANT}" == "domain-expert" ]]; then
    AUTHORIZATION_ARGS+=(--domain "${EXPERT_DOMAIN}")
  fi
  python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/authorize_agent_boundary_checkpoint.py" "${AUTHORIZATION_ARGS[@]}"
fi

export CHECKPOINT_ROOT
export ITER_DIR="${CHECKPOINT_ROOT}/iter_${ITER_PADDED}"
export OUTPUT_DIR="${HF_OUTPUT_ROOT}/iter_${ITER_PADDED}_hf"
export ORIGIN_HF_DIR="${SFT_CKPT_ROOT}/final_hf"

exec bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
