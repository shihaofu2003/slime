#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
SFT_GATE="${TAU2_SFT_PROMOTION_GATE:-${PROJECT_ROOT}/output/experiments/tau2-sft-agent-user-boundary-v2/SFT_PROMOTION.json}"
RL_CKPT_ROOT="${RL_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_20260804}"
LONG100_CKPT_ROOT="${TAU2_RL_LONG100_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804}"
LONG200_CKPT_ROOT="${TAU2_RL_LONG200_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805}"
SMOKE_CKPT_ROOT="${SMOKE_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_turn_credit_smoke_r3_20260804}"
STAGE="${1:-pilot-a}"
EXPERT_DOMAIN="${2:-}"

if [[ "${STAGE}" == "domain-expert" ]]; then
  if [[ $# -ne 2 || ! "${EXPERT_DOMAIN}" =~ ^(airline|retail|telecom)$ ]]; then
    echo "Usage: $0 domain-expert <airline|retail|telecom>" >&2
    exit 2
  fi
elif [[ $# -gt 1 ]]; then
  echo "Usage: $0 [smoke|pilot-a|pilot-b|train80|long100-a|long100-b|long100-final|long200-final|long200-final-kl-waiver]" >&2
  exit 2
fi

if [[ "${STAGE}" == "domain-expert" ]]; then
  EXPERIMENT_NAME="tau2-rl-agent-user-boundary-v2-domain-experts"
  case "${EXPERT_DOMAIN}" in
    airline) EXPERT_CKPT_ROOT="${TAU2_RL_AIRLINE_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_airline_expert_20260806}" ;;
    retail) EXPERT_CKPT_ROOT="${TAU2_RL_RETAIL_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_retail_expert_20260806}" ;;
    telecom) EXPERT_CKPT_ROOT="${TAU2_RL_TELECOM_EXPERT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_telecom_expert_20260806}" ;;
  esac
  ACTIVE_CKPT_ROOT="${EXPERT_CKPT_ROOT}"
elif [[ "${STAGE}" == "long200-final" || "${STAGE}" == "long200-final-kl-waiver" ]]; then
  EXPERIMENT_NAME="tau2-rl-agent-user-boundary-v2-long200"
  ACTIVE_CKPT_ROOT="${LONG200_CKPT_ROOT}"
elif [[ "${STAGE}" == long100-* ]]; then
  EXPERIMENT_NAME="tau2-rl-agent-user-boundary-v2-long100"
  ACTIVE_CKPT_ROOT="${LONG100_CKPT_ROOT}"
else
  EXPERIMENT_NAME="tau2-rl-agent-user-boundary-v2"
  if [[ "${STAGE}" == "smoke" ]]; then
    ACTIVE_CKPT_ROOT="${SMOKE_CKPT_ROOT}"
  else
    ACTIVE_CKPT_ROOT="${RL_CKPT_ROOT}"
  fi
fi
EXPERIMENT_NAME="${TAU2_RL_EXPERIMENT_NAME:-${EXPERIMENT_NAME}}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
LONG100_EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-rl-agent-user-boundary-v2-long100"
RL_ITER9_GATE="${TAU2_RL_ITER9_GATE:-${EXPERIMENT_DIR}/ITER9_SAFETY_GATE.json}"
RL_ITER19_GATE="${TAU2_RL_ITER19_GATE:-${EXPERIMENT_DIR}/ITER19_SAFETY_GATE.json}"
LONG100_ITER9_GATE="${TAU2_RL_LONG100_ITER9_GATE:-${EXPERIMENT_DIR}/ITER9_HEALTH_GATE.json}"
LONG100_ITER19_GATE="${TAU2_RL_LONG100_ITER19_GATE:-${EXPERIMENT_DIR}/ITER19_HEALTH_GATE.json}"
LONG100_ITER99_GATE="${TAU2_RL_LONG100_ITER99_GATE:-${LONG100_EXPERIMENT_DIR}/ITER99_HEALTH_GATE.json}"

read_sft_checkpoint() {
  python3 - "${SFT_GATE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[ERROR] missing SFT promotion gate: {path}")
gate = json.loads(path.read_text(encoding="utf-8"))
if (
    gate.get("status") != "pass"
    or gate.get("gate_version") != "turn-aware-rl-v3"
    or gate.get("selected") != "contract-boundary"
):
    raise SystemExit(f"[ERROR] v3 gate did not select Contract + boundary: {path}")
checkpoint = gate.get("selected_checkpoint_root")
if not isinstance(checkpoint, str) or not checkpoint:
    raise SystemExit(f"[ERROR] SFT gate has no selected_checkpoint_root: {path}")
print(str(Path(checkpoint).resolve()))
PY
}

SFT_CKPT_ROOT="$(read_sft_checkpoint)"
if [[ ! -d "${SFT_CKPT_ROOT}/final_hf" ]]; then
  echo "[ERROR] selected SFT has no final_hf conversion: ${SFT_CKPT_ROOT}/final_hf" >&2
  exit 1
fi

LATEST_FILE="${ACTIVE_CKPT_ROOT}/latest_checkpointed_iteration.txt"
latest=""
if [[ -f "${LATEST_FILE}" ]]; then
  latest="$(tr -d '[:space:]' < "${LATEST_FILE}")"
  if [[ ! "${latest}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] invalid latest checkpoint iteration: ${latest}" >&2
    exit 1
  fi
fi

require_long100_health_gate() {
  local path="$1" expected_stage="$2" expected_iteration="$3"
  python3 - "${path}" "${LONG100_CKPT_ROOT}" "${expected_stage}" "${expected_iteration}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
expected_stage = sys.argv[3]
expected_iteration = int(sys.argv[4])
if not path.is_file():
    raise SystemExit(f"[ERROR] missing long100 health gate: {path}")
gate = json.loads(path.read_text(encoding="utf-8"))
if (
    gate.get("status") != "pass"
    or gate.get("gate_version") != "turn-aware-rl-health-v1"
    or gate.get("stage") != expected_stage
    or gate.get("checkpoint_root") != checkpoint
    or gate.get("expected_checkpoint_root") != checkpoint
    or gate.get("expected_latest") != expected_iteration
    or gate.get("turn_credit_version") != "turn-credit-v1"
):
    raise SystemExit(f"[ERROR] long100 health gate did not pass for this checkpoint: {path}")
PY
}

prepare_long200_load() {
  python3 - \
    "${LONG100_ITER99_GATE}" \
    "${LONG100_CKPT_ROOT}" \
    "${LONG200_CKPT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

gate_path = Path(sys.argv[1])
source = Path(sys.argv[2]).resolve()
destination = Path(sys.argv[3]).resolve()
if not gate_path.is_file():
    raise SystemExit(f"[ERROR] missing long100 iter99 health gate: {gate_path}")
gate_bytes = gate_path.read_bytes()
gate = json.loads(gate_bytes)
source_string = str(source)
if (
    gate.get("status") != "pass"
    or gate.get("gate_version") != "turn-aware-rl-health-v1"
    or gate.get("stage") != "iter99"
    or gate.get("checkpoint_root") != source_string
    or gate.get("expected_checkpoint_root") != source_string
    or gate.get("expected_latest") != 99
    or gate.get("turn_credit_version") != "turn-credit-v1"
):
    raise SystemExit(f"[ERROR] long100 iter99 gate does not authorize continuation: {gate_path}")

source_latest = source / "latest_checkpointed_iteration.txt"
if not source_latest.is_file() or source_latest.read_text(encoding="utf-8").strip() != "99":
    raise SystemExit(f"[ERROR] long100 source must remain exactly at iter99: {source_latest}")

def require_checkpoint(root: Path, iteration: int) -> None:
    iteration_dir = root / f"iter_{iteration:07d}"
    required = (
        iteration_dir / ".metadata",
        iteration_dir / "common.pt",
        iteration_dir / "metadata.json",
        root / "rollout" / f"global_dataset_state_dict_{iteration}.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    shards = [path for path in iteration_dir.glob("__*.distcp") if path.stat().st_size > 0]
    if missing or not shards:
        raise SystemExit(
            f"[ERROR] incomplete checkpoint {root}@{iteration}; "
            f"missing={missing}, nonempty_shards={len(shards)}"
        )

require_checkpoint(source, 99)
gate_sha256 = hashlib.sha256(gate_bytes).hexdigest()
lineage = {
    "lineage_version": "boundary-v2-long200-v1",
    "source_checkpoint_root": source_string,
    "destination_checkpoint_root": str(destination),
    "source_iteration": 99,
    "target_iteration": 199,
    "source_health_gate": str(gate_path.resolve()),
    "source_health_gate_sha256": gate_sha256,
    "scheduler_horizon_override": True,
}
destination.mkdir(parents=True, exist_ok=True)
lineage_path = destination / "LONG200_LINEAGE.json"
if lineage_path.is_file():
    observed = json.loads(lineage_path.read_text(encoding="utf-8"))
    if observed != lineage:
        raise SystemExit(f"[ERROR] long200 lineage marker mismatch: {lineage_path}")
else:
    unexpected = [path.name for path in destination.iterdir()]
    if unexpected:
        raise SystemExit(
            f"[ERROR] long200 destination has no lineage marker but is non-empty: {unexpected}"
        )
    temporary = lineage_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(lineage_path)

latest_path = destination / "latest_checkpointed_iteration.txt"
if not latest_path.is_file():
    print(source_string)
    raise SystemExit(0)
raw_latest = latest_path.read_text(encoding="utf-8").strip()
if not raw_latest.isdigit():
    raise SystemExit(f"[ERROR] invalid long200 latest marker: {latest_path}")
latest = int(raw_latest)
if latest >= 199:
    raise SystemExit(f"[ERROR] long200-final already completed at iteration {latest}")
if latest < 109 or (latest - 9) % 10 != 0:
    raise SystemExit(
        f"[ERROR] long200 resume requires a saved iter109/119/.../189 checkpoint; latest={latest}"
    )
require_checkpoint(destination, latest)
print(str(destination))
PY
}

case "${STAGE}" in
  smoke)
    if [[ -n "${latest}" ]]; then
      echo "[ERROR] smoke root must be fresh; latest=${latest}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=2
    SAVE_INTERVAL_VALUE=2
    DOMAIN_QUOTA="telecom:3,airline:2,retail:1"
    TRAJECTORY_STAGE="smoke_r3_iter0000-0001"
    ;;
  pilot-a)
    if [[ -n "${latest}" && "${latest}" -ge 9 ]]; then
      echo "[ERROR] pilot-a already completed at iteration ${latest}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=10
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:3,airline:2,retail:1"
    TRAJECTORY_STAGE="iter0000-0009"
    ;;
  pilot-b)
    if [[ -z "${latest}" || "${latest}" -lt 9 || "${latest}" -ge 19 ]]; then
      echo "[ERROR] pilot-b requires completed iter9 (or interrupted iter10-18); latest=${latest:-missing}" >&2
      exit 1
    fi
    python3 - "${RL_ITER9_GATE}" "${RL_CKPT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
if not path.is_file():
    raise SystemExit(f"[ERROR] missing iter9 safety gate: {path}")
gate = json.loads(path.read_text(encoding="utf-8"))
if (
    gate.get("status") != "pass"
    or gate.get("stage") != "iter9"
    or gate.get("checkpoint_root") != checkpoint
    or gate.get("expected_latest") != 9
    or gate.get("namespace_non_regression") is not True
    or gate.get("namespace_strict_reduction") is not True
):
    raise SystemExit(f"[ERROR] iter9 safety gate did not pass: {path}")
PY
    NUM_ROLLOUT_VALUE=20
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:3,airline:1,retail:2"
    TRAJECTORY_STAGE="iter0010-0019"
    ;;
  train80)
    if [[ -z "${latest}" || "${latest}" -lt 19 || "${latest}" -ge 99 ]]; then
      echo "[ERROR] train80 requires completed iter19 (or interrupted iter20-98); latest=${latest:-missing}" >&2
      exit 1
    fi
    python3 - "${RL_ITER19_GATE}" "${RL_CKPT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
if not path.is_file():
    raise SystemExit(f"[ERROR] missing iter19 safety gate: {path}")
gate = json.loads(path.read_text(encoding="utf-8"))
if (
    gate.get("status") != "pass"
    or gate.get("stage") != "iter19"
    or gate.get("checkpoint_root") != checkpoint
    or gate.get("expected_latest") != 19
    or gate.get("namespace_non_regression") is not True
    or gate.get("namespace_strict_reduction") is not True
):
    raise SystemExit(f"[ERROR] iter19 safety gate did not pass: {path}")
PY
    NUM_ROLLOUT_VALUE=100
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:2,airline:2,retail:2"
    TRAJECTORY_STAGE="iter0020-0099"
    ;;
  long100-a)
    if [[ -n "${latest}" ]]; then
      echo "[ERROR] long100-a requires a fresh checkpoint root; latest=${latest}" >&2
      exit 1
    fi
    if [[ -d "${LONG100_CKPT_ROOT}" ]] && [[ -n "$(find "${LONG100_CKPT_ROOT}" -mindepth 1 -print -quit)" ]]; then
      echo "[ERROR] long100-a checkpoint root is non-empty without a valid checkpoint marker: ${LONG100_CKPT_ROOT}" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=10
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:3,airline:2,retail:1"
    TRAJECTORY_STAGE="long100-a_iter0000-0009"
    OVERRIDE_OPT_PARAM_SCHEDULER_VALUE=0
    ;;
  long100-b)
    if [[ -z "${latest}" || "${latest}" -lt 9 || "${latest}" -ge 19 ]]; then
      echo "[ERROR] long100-b requires completed iter9 (or interrupted iter10-18); latest=${latest:-missing}" >&2
      exit 1
    fi
    require_long100_health_gate "${LONG100_ITER9_GATE}" iter9 9
    NUM_ROLLOUT_VALUE=20
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:3,airline:1,retail:2"
    TRAJECTORY_STAGE="long100-b_iter0010-0019"
    OVERRIDE_OPT_PARAM_SCHEDULER_VALUE=1
    ;;
  long100-final)
    if [[ -z "${latest}" || "${latest}" -lt 19 || "${latest}" -ge 99 ]]; then
      echo "[ERROR] long100-final requires completed iter19 (or interrupted iter20-98); latest=${latest:-missing}" >&2
      exit 1
    fi
    require_long100_health_gate "${LONG100_ITER19_GATE}" iter19 19
    NUM_ROLLOUT_VALUE=100
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:2,airline:2,retail:2"
    TRAJECTORY_STAGE="long100-final_iter0020-0099"
    OVERRIDE_OPT_PARAM_SCHEDULER_VALUE=1
    ;;
  long200-final|long200-final-kl-waiver)
    LOAD_DIR_VALUE="$(prepare_long200_load)"
    ATTEMPT_ID="${TAU2_RL_ATTEMPT_ID:-$(date +%Y%m%d_%H%M%S)}"
    if [[ ! "${ATTEMPT_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
      echo "[ERROR] TAU2_RL_ATTEMPT_ID may contain only letters, digits, dot, underscore, and dash" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=200
    SAVE_INTERVAL_VALUE=10
    DOMAIN_QUOTA="telecom:2,airline:2,retail:2"
    TRAJECTORY_STAGE="${STAGE}_iter0100-0199_${ATTEMPT_ID}"
    OVERRIDE_OPT_PARAM_SCHEDULER_VALUE=1
    ;;
  domain-expert)
    LOAD_DIR_VALUE="$(python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/domain_expert_checkpoint.py" \
      --domain "${EXPERT_DOMAIN}" \
      --source-root "${LONG100_CKPT_ROOT}" \
      --destination-root "${ACTIVE_CKPT_ROOT}" \
      --source-gate "${LONG100_ITER99_GATE}")"
    ATTEMPT_ID="${TAU2_RL_ATTEMPT_ID:-$(date +%Y%m%d_%H%M%S)}"
    if [[ ! "${ATTEMPT_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
      echo "[ERROR] TAU2_RL_ATTEMPT_ID may contain only letters, digits, dot, underscore, and dash" >&2
      exit 1
    fi
    NUM_ROLLOUT_VALUE=130
    SAVE_INTERVAL_VALUE=10
    case "${EXPERT_DOMAIN}" in
      airline) DOMAIN_QUOTA="airline:6,retail:0,telecom:0" ;;
      retail) DOMAIN_QUOTA="airline:0,retail:6,telecom:0" ;;
      telecom) DOMAIN_QUOTA="airline:0,retail:0,telecom:6" ;;
    esac
    TRAJECTORY_STAGE="domain-expert_${EXPERT_DOMAIN}_iter0100-0129_${ATTEMPT_ID}"
    OVERRIDE_OPT_PARAM_SCHEDULER_VALUE=1
    ;;
  *)
    echo "[ERROR] unknown stage ${STAGE}; expected smoke, pilot-a, pilot-b, train80, long100-a, long100-b, long100-final, long200-final, long200-final-kl-waiver, or domain-expert <domain>" >&2
    exit 2
    ;;
esac

mkdir -p "${EXPERIMENT_DIR}/trajectories"

export SFT_CKPT_ROOT
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/final_hf"
export REF_LOAD="${SFT_CKPT_ROOT}"
export SAVE_DIR="${ACTIVE_CKPT_ROOT}"
export LOAD_DIR="${LOAD_DIR_VALUE:-${ACTIVE_CKPT_ROOT}}"
if [[ "${STAGE}" == "domain-expert" ]]; then
  export PREPARED_RL_DATA="${EXPERIMENT_DIR}/data/areal_tau2_rl_train_all_${EXPERT_DOMAIN}.jsonl"
else
  export PREPARED_RL_DATA="${EXPERIMENT_DIR}/areal_tau2_rl_train_all.jsonl"
fi
export TAU2_AGENT_PROTOCOL_PROFILE="agent-owned-dependency-safe-multi"
export LOSS_MASK_TYPE="qwen3_full"
export TAU2_TURN_CREDIT_VERSION="${TAU2_TURN_CREDIT_VERSION:-turn-credit-v1}"
if [[ "${STAGE}" == "long200-final-kl-waiver" ]]; then
  export TAU2_RL_HEALTH_POLICY="kl-waiver-v1"
elif [[ "${STAGE}" == "domain-expert" ]]; then
  export TAU2_RL_HEALTH_POLICY="domain-expert-kl-diagnostic-v1"
fi
export DATA_SOURCE_PATH="filters.DomainQuotaDataSource"
export TAU2_RL_DOMAIN_QUOTA="${DOMAIN_QUOTA}"
export TAU2_RL_DOMAIN=""
export SKIP_PREFLIGHT=0

export TOTAL_GPUS=8
export RAY_GPUS=6
export AGENT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
export USER_CUDA_VISIBLE_DEVICES="7"
export ROLLOUT_BATCH_SIZE=6
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=48
export NUM_ROLLOUT="${NUM_ROLLOUT_VALUE}"
export SAVE_INTERVAL="${SAVE_INTERVAL_VALUE}"
export OVERRIDE_OPT_PARAM_SCHEDULER="${OVERRIDE_OPT_PARAM_SCHEDULER_VALUE:-0}"
export USE_DYNAMIC_FILTER=1

export LR=2e-6
export KL_LOSS_TYPE=k2
export KL_LOSS_COEF=0.01
export KL_COEF=0
export ENTROPY_COEF=0
export EPS_CLIP=0.4
export EPS_CLIP_HIGH=0.4

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export AGENT_MAX_TOKENS=1024
export TAU2_MAX_STEPS=60
export TAU2_MAX_ERRORS=10
export TAU2_USER_TEMPERATURE=0.0
export TAU2_USER_MAX_TOKENS=512
export TAU2_RL_MAX_TRAIN_TOKENS=16384
export TAU2_RL_MAX_ROLLOUT_RETRIES=2

export USE_REWARD_SHAPING=1
export TAU2_REWARD_ALPHA=0.25
export TAU2_PARTIAL_TOOL_NAME_WEIGHT=0.25
export TAU2_PARTIAL_ARGUMENT_WEIGHT=0.35
export TAU2_PARTIAL_DB_WEIGHT=0.25
export TAU2_PARTIAL_ENV_ASSERTION_WEIGHT=0.10
export TAU2_PARTIAL_COMMUNICATE_WEIGHT=0.05
export TAU2_PENALTY_MALFORMED_JSON=0.15
export TAU2_PENALTY_NONEXISTENT_TOOL=0.15
export TAU2_PENALTY_WRONG_ARGUMENT_FIELD=0.10
export TAU2_PENALTY_TOOL_EXECUTION_ERROR=0.25
export TAU2_PENALTY_REPETITION=0.05
export TAU2_PENALTY_MAX_STEPS=0.20

export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"
export TAU2_RL_RUN_ID="boundary_v2_${STAGE}${EXPERT_DOMAIN:+_${EXPERT_DOMAIN}}${ATTEMPT_ID:+_${ATTEMPT_ID}}"
export TAU2_RL_TRAJECTORY_DUMP_PATH="${EXPERIMENT_DIR}/trajectories/${TRAJECTORY_STAGE}.jsonl"
export USER_SGLANG_LOG="${EXPERIMENT_DIR}/user_sglang_${STAGE}${EXPERT_DOMAIN:+_${EXPERT_DOMAIN}}${ATTEMPT_ID:+_${ATTEMPT_ID}}.log"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
export WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
