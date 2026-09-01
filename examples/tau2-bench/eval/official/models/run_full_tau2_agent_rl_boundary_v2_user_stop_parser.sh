#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
VARIANT="${VARIANT:-legacy}"
EXPERT_DOMAIN="${EXPERT_DOMAIN:-}"
if [[ $# -gt 0 && "$1" =~ ^(legacy|long100|curve|long200|long200-kl-waiver|domain-expert|mixed-control)$ ]]; then
  VARIANT="$1"
  shift
fi
if [[ "${VARIANT}" == "domain-expert" && $# -gt 0 ]]; then
  EXPERT_DOMAIN="$1"
  shift
fi
ITERATION="${1:-${ITERATION:-9}}"
if [[ $# -gt 0 ]]; then
  shift
fi
SEED="${1:-${SEED:-300}}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  PROBE_MODE="$1"
  shift
elif [[ "${SEED}" == "300" ]]; then
  PROBE_MODE="probe"
else
  PROBE_MODE="no-probe"
fi
if [[ "${VARIANT}" == "domain-expert" && $# -gt 0 ]]; then
  EXPERT_EVAL_SCOPE="$1"
  shift
else
  EXPERT_EVAL_SCOPE="${TAU2_DOMAIN_EXPERT_EVAL_SCOPE:-target}"
fi

if [[ $# -gt 0 || ! "${VARIANT}" =~ ^(legacy|long100|curve|long200|long200-kl-waiver|domain-expert|mixed-control)$ || ! "${ITERATION}" =~ ^[0-9]+$ || ! "${SEED}" =~ ^[0-9]+$ || ! "${PROBE_MODE}" =~ ^(probe|no-probe)$ || ! "${EXPERT_EVAL_SCOPE}" =~ ^(target|forgetting|all)$ ]]; then
  echo "Usage: $0 domain-expert <airline|retail|telecom> <109|119|129> [seed] [probe|no-probe] [target|forgetting|all]" >&2
  echo "       $0 mixed-control <109|119|129> [seed] [probe|no-probe]" >&2
  echo "       $0 [legacy|long100|curve|long200|long200-kl-waiver] ITERATION [seed] [probe|no-probe]" >&2
  exit 2
fi
case "${VARIANT}" in
  legacy|long100)
    [[ "${ITERATION}" =~ ^(9|19|99)$ ]] || { echo "[ERROR] ${VARIANT} supports only 9|19|99" >&2; exit 2; }
    ;;
  curve)
    (( ITERATION >= 9 && ITERATION <= 99 && (ITERATION - 9) % 10 == 0 )) || { echo "[ERROR] curve requires iter9/19/.../99" >&2; exit 2; }
    ;;
  long200|long200-kl-waiver)
    [[ "${ITERATION}" == "199" ]] || { echo "[ERROR] long200 requires iter199" >&2; exit 2; }
    ;;
  domain-expert)
    [[ "${EXPERT_DOMAIN}" =~ ^(airline|retail|telecom)$ && "${ITERATION}" =~ ^(109|119|129)$ ]] || { echo "[ERROR] domain-expert requires a domain and iter109|119|129" >&2; exit 2; }
    ;;
  mixed-control)
    [[ -z "${EXPERT_DOMAIN}" && "${ITERATION}" =~ ^(109|119|129)$ ]] || { echo "[ERROR] mixed-control requires iter109|119|129" >&2; exit 2; }
    ;;
esac
printf -v ITER_PADDED '%07d' "${ITERATION}"
EVAL_LABEL="iter${ITERATION}"

if [[ "${VARIANT}" == "long100" ]]; then
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
  RL_CKPT_ROOT="${RL_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
  DEFAULT_MODEL_NAME="Qwen3-4B-tau2-agent-rl-boundary-v2-long100-iter${ITERATION}"
  HEALTH_GATE="${EXPERIMENT_DIR}/ITER${ITERATION}_HEALTH_GATE.json"
  python3 - "${HEALTH_GATE}" "${RL_CKPT_ROOT}" "${ITERATION}" <<'PY'
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
    raise SystemExit(f"[ERROR] long100 health gate does not authorize evaluation: {path}")
PY
elif [[ "${VARIANT}" == "curve" ]]; then
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-checkpoint-curve"
  SOURCE_CKPT_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
  RL_CKPT_ROOT="${RL_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_checkpoint_curve_hf_20260805}"
  DEFAULT_MODEL_NAME="Qwen3-4B-tau2-agent-rl-boundary-v2-curve-iter${ITERATION}"
  if [[ "${ITERATION}" == "9" ]]; then
    GATE_ITERATION=9
  elif [[ "${ITERATION}" == "19" ]]; then
    GATE_ITERATION=19
  else
    GATE_ITERATION=99
  fi
  HEALTH_GATE="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100/ITER${GATE_ITERATION}_HEALTH_GATE.json"
  python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/authorize_agent_boundary_checkpoint.py" \
    --variant curve \
    --iteration "${ITERATION}" \
    --checkpoint-root "${SOURCE_CKPT_ROOT}" \
    --health-gate "${HEALTH_GATE}" \
    --reference-root "${SOURCE_CKPT_ROOT}" \
    --output "${EXPERIMENT_DIR}/authorizations/curve_iter${ITERATION}.json"
elif [[ "${VARIANT}" == "long200" ]]; then
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long200"
  SOURCE_CKPT_ROOT="${TAU2_RL_LONG200_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805}"
  REFERENCE_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
  RL_CKPT_ROOT="${RL_CKPT_ROOT:-${TAU2_RL_LONG200_HF_ROOT:-${SOURCE_CKPT_ROOT}}}"
  DEFAULT_MODEL_NAME="Qwen3-4B-tau2-agent-rl-boundary-v2-long200-iter199"
  python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/authorize_agent_boundary_checkpoint.py" \
    --variant long200 \
    --iteration 199 \
    --checkpoint-root "${SOURCE_CKPT_ROOT}" \
    --health-gate "${EXPERIMENT_DIR}/ITER199_HEALTH_GATE.json" \
    --reference-root "${REFERENCE_ROOT}" \
    --output "${EXPERIMENT_DIR}/authorizations/long200_iter199.json"
elif [[ "${VARIANT}" == "long200-kl-waiver" ]]; then
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long200"
  SOURCE_CKPT_ROOT="${TAU2_RL_LONG200_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805}"
  REFERENCE_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
  RL_CKPT_ROOT="${RL_CKPT_ROOT:-${TAU2_RL_LONG200_KL_WAIVER_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_kl_waiver_hf_20260805}}"
  DEFAULT_MODEL_NAME="Qwen3-4B-tau2-agent-rl-boundary-v2-long200-kl-waiver-iter199"
  EVAL_LABEL="iter199-kl-waiver"
  python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/authorize_agent_boundary_checkpoint.py" \
    --variant long200-kl-waiver \
    --iteration 199 \
    --checkpoint-root "${SOURCE_CKPT_ROOT}" \
    --health-gate "${EXPERIMENT_DIR}/ITER199_KL_WAIVER_HEALTH_GATE.json" \
    --reference-root "${REFERENCE_ROOT}" \
    --output "${EXPERIMENT_DIR}/authorizations/long200_kl_waiver_iter199.json"
elif [[ "${VARIANT}" == "domain-expert" ]]; then
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts"
  SOURCE_CKPT_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
  case "${EXPERT_DOMAIN}" in
    airline)
      EXPERT_SOURCE_ROOT="${TAU2_RL_AIRLINE_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_airline_expert_20260806}"
      RL_CKPT_ROOT="${RL_CKPT_ROOT:-${TAU2_RL_AIRLINE_EXPERT_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_airline_expert_hf_20260806}}"
      ;;
    retail)
      EXPERT_SOURCE_ROOT="${TAU2_RL_RETAIL_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_retail_expert_20260806}"
      RL_CKPT_ROOT="${RL_CKPT_ROOT:-${TAU2_RL_RETAIL_EXPERT_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_retail_expert_hf_20260806}}"
      ;;
    telecom)
      EXPERT_SOURCE_ROOT="${TAU2_RL_TELECOM_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_telecom_expert_20260806}"
      RL_CKPT_ROOT="${RL_CKPT_ROOT:-${TAU2_RL_TELECOM_EXPERT_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_telecom_expert_hf_20260806}}"
      ;;
  esac
  DEFAULT_MODEL_NAME="Qwen3-4B-tau2-agent-rl-boundary-v2-${EXPERT_DOMAIN}-expert-iter${ITERATION}"
  EVAL_SCOPE="${EXPERT_EVAL_SCOPE}"
  case "${EVAL_SCOPE}" in
    target) DEFAULT_DOMAINS="${EXPERT_DOMAIN}" ;;
    forgetting)
      case "${EXPERT_DOMAIN}" in
        airline) DEFAULT_DOMAINS="retail,telecom" ;;
        retail) DEFAULT_DOMAINS="airline,telecom" ;;
        telecom) DEFAULT_DOMAINS="airline,retail" ;;
      esac
      ;;
    all) DEFAULT_DOMAINS="airline,retail,telecom" ;;
    *) echo "[ERROR] expert evaluation scope must be target, forgetting, or all" >&2; exit 2 ;;
  esac
  export DOMAINS="${DOMAINS:-${DEFAULT_DOMAINS}}"
  EVAL_LABEL="expert/${EXPERT_DOMAIN}/iter${ITERATION}/${EVAL_SCOPE}"
  python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/authorize_agent_boundary_checkpoint.py" \
    --variant domain-expert \
    --domain "${EXPERT_DOMAIN}" \
    --iteration "${ITERATION}" \
    --checkpoint-root "${EXPERT_SOURCE_ROOT}" \
    --health-gate "${EXPERIMENT_DIR}/${EXPERT_DOMAIN^^}_ITER${ITERATION}_HEALTH_GATE.json" \
    --reference-root "${SOURCE_CKPT_ROOT}" \
    --output "${EXPERIMENT_DIR}/authorizations/domain_expert_${EXPERT_DOMAIN}_iter${ITERATION}.json"
elif [[ "${VARIANT}" == "mixed-control" ]]; then
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts"
  SOURCE_CKPT_ROOT="${TAU2_RL_LONG200_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805}"
  REFERENCE_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
  RL_CKPT_ROOT="${RL_CKPT_ROOT:-${TAU2_RL_DOMAIN_EXPERT_MIXED_HF_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_domain_expert_mixed_control_hf_20260806}}"
  DEFAULT_MODEL_NAME="Qwen3-4B-tau2-agent-rl-boundary-v2-mixed-control-iter${ITERATION}"
  EVAL_LABEL="mixed/iter${ITERATION}"
  export DOMAINS="${DOMAINS:-airline,retail,telecom}"
  python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/authorize_agent_boundary_checkpoint.py" \
    --variant mixed-control \
    --iteration "${ITERATION}" \
    --checkpoint-root "${SOURCE_CKPT_ROOT}" \
    --health-gate "${EXPERIMENT_DIR}/ITER${ITERATION}_PREFIX_HEALTH_GATE.json" \
    --reference-root "${REFERENCE_ROOT}" \
    --output "${EXPERIMENT_DIR}/authorizations/mixed_control_iter${ITERATION}.json"
else
  EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2"
  RL_CKPT_ROOT="${RL_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_20260804}"
  DEFAULT_MODEL_NAME="Qwen3-4B-tau2-agent-rl-boundary-v2-iter${ITERATION}"
fi

export MODEL_PATH="${MODEL_PATH:-${RL_CKPT_ROOT}/iter_${ITER_PADDED}_hf}"
export MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"
export NUM_TASKS=""
export NUM_TRIALS=4
export SEED
export TASK_SPLIT="${TASK_SPLIT:-test}"
export AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.6}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-8192}"
export MAX_STEPS="${MAX_STEPS:-200}"
export AGENT_PROTOCOL_PROFILE="agent-owned-dependency-safe-multi"
export RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
export SAVE_PREFIX="tau2_official_${MODEL_NAME}_agent_owned_seed${SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed${SEED}_summary.json}"
if [[ "${PROBE_MODE}" == "probe" ]]; then
  export NAMESPACE_PROBE_OUTPUT="${NAMESPACE_PROBE_OUTPUT:-${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/namespace_probe.json}"
else
  export NAMESPACE_PROBE_OUTPUT=""
fi

exec bash "${OFFICIAL_DIR}/run_eval.sh"
