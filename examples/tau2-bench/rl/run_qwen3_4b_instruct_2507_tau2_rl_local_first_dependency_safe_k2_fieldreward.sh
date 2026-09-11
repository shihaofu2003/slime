#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
EXPERIMENT_NAME="tau2-rl-local-first-dependency-safe-k2-fieldreward"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
SFT_CKPT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_local_first_relaxed_20260803"
RL_CKPT_ROOT="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_local_first_dependency_safe_k2_fieldreward_lr2e6_20260803"
LATEST_FILE="${RL_CKPT_ROOT}/latest_checkpointed_iteration.txt"
PROMOTION_GATE="${EXPERIMENT_DIR}/ITER99_PROMOTION.json"
STAGE="${1:-train100}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [train100|resume200]" >&2
  exit 2
fi

case "${STAGE}" in
  train100)
    if [[ -f "${LATEST_FILE}" ]]; then
      latest="$(tr -d '[:space:]' < "${LATEST_FILE}")"
      if [[ ! "${latest}" =~ ^[0-9]+$ || "${latest}" -ge 99 ]]; then
        echo "[ERROR] train100 requires no completed iter99 checkpoint; latest=${latest}" >&2
        exit 1
      fi
      echo "[RESUME] continuing interrupted train100 stage from iteration ${latest}"
    fi
    NUM_ROLLOUT_VALUE=100
    TRAJECTORY_STAGE="iter0000-0099"
    ;;
  resume200)
    if [[ ! -f "${LATEST_FILE}" ]]; then
      echo "[ERROR] resume200 requires ${LATEST_FILE}" >&2
      exit 1
    fi
    latest="$(tr -d '[:space:]' < "${LATEST_FILE}")"
    if [[ ! "${latest}" =~ ^[0-9]+$ || "${latest}" -lt 99 || "${latest}" -ge 199 ]]; then
      echo "[ERROR] resume200 requires latest=99 (or an interrupted 100-199 stage); latest=${latest}" >&2
      exit 1
    fi
    python3 - "${PROMOTION_GATE}" "${RL_CKPT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

gate_path = Path(sys.argv[1])
checkpoint_root = str(Path(sys.argv[2]).resolve())
if not gate_path.is_file():
    raise SystemExit(f"[ERROR] missing iter99 promotion gate: {gate_path}")
gate = json.loads(gate_path.read_text(encoding="utf-8"))
if gate.get("status") != "pass" or gate.get("candidate") != "new_rl100":
    raise SystemExit(f"[ERROR] iter99 promotion gate did not pass: {gate_path}")
if gate.get("checkpoint_root") != checkpoint_root or gate.get("expected_latest") != 99:
    raise SystemExit(f"[ERROR] iter99 promotion gate targets a different checkpoint: {gate_path}")
PY
    if [[ "${latest}" -ne 99 ]]; then
      echo "[RESUME] continuing interrupted resume200 stage from iteration ${latest}"
    fi
    NUM_ROLLOUT_VALUE=200
    TRAJECTORY_STAGE="iter0100-0199"
    ;;
  *)
    echo "[ERROR] unknown stage ${STAGE}; expected train100 or resume200" >&2
    exit 2
    ;;
esac

mkdir -p "${EXPERIMENT_DIR}/trajectories"

export SFT_CKPT_ROOT
export HF_CHECKPOINT="${SFT_CKPT_ROOT}/final_hf"
export REF_LOAD="${SFT_CKPT_ROOT}"
export SAVE_DIR="${RL_CKPT_ROOT}"
export TAU2_AGENT_PROTOCOL_PROFILE="dependency-safe-multi"

export TOTAL_GPUS="8"
export RAY_GPUS="6"
export AGENT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
export USER_CUDA_VISIBLE_DEVICES="7"
export TAU2_RL_DOMAIN="airline"
export ROLLOUT_BATCH_SIZE="6"
export N_SAMPLES_PER_PROMPT="8"
export GLOBAL_BATCH_SIZE="48"
export NUM_ROLLOUT="${NUM_ROLLOUT_VALUE}"
export SAVE_INTERVAL="20"
export USE_DYNAMIC_FILTER="1"

export LR="2e-6"
export KL_LOSS_TYPE="k2"
export KL_LOSS_COEF="0.01"
export KL_COEF="0"
export ENTROPY_COEF="0"
export EPS_CLIP="0.4"
export EPS_CLIP_HIGH="0.4"

export ROLLOUT_TEMPERATURE="1.0"
export ROLLOUT_TOP_P="1.0"
export AGENT_MAX_TOKENS="1024"
export TAU2_MAX_STEPS="60"
export TAU2_MAX_ERRORS="10"
export TAU2_USER_TEMPERATURE="0.0"
export TAU2_USER_MAX_TOKENS="512"
export TAU2_RL_MAX_TRAIN_TOKENS="16384"
export TAU2_RL_MAX_ROLLOUT_RETRIES="2"

export USE_REWARD_SHAPING="1"
export TAU2_REWARD_ALPHA="0.25"
export TAU2_PARTIAL_TOOL_NAME_WEIGHT="0.25"
export TAU2_PARTIAL_ARGUMENT_WEIGHT="0.35"
export TAU2_PARTIAL_DB_WEIGHT="0.25"
export TAU2_PARTIAL_ENV_ASSERTION_WEIGHT="0.10"
export TAU2_PARTIAL_COMMUNICATE_WEIGHT="0.05"
export TAU2_PENALTY_MALFORMED_JSON="0.15"
export TAU2_PENALTY_NONEXISTENT_TOOL="0.15"
export TAU2_PENALTY_REPETITION="0.05"
export TAU2_PENALTY_MAX_STEPS="0.20"

export USER_MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf"
export USER_MODEL="Qwen3-4B-tau2-user-sft-stop-iter0006899"
export TAU2_RL_RUN_ID="local_first_dependency_safe_k2_fieldreward_lr2e6_${STAGE}"
export TAU2_RL_TRAJECTORY_DUMP_PATH="${EXPERIMENT_DIR}/trajectories/${TRAJECTORY_STAGE}.jsonl"
export USER_SGLANG_LOG="${EXPERIMENT_DIR}/user_sglang_${STAGE}.log"

export USE_WANDB="1"
export WANDB_PROJECT="slime-dev"
export WANDB_GROUP="${EXPERIMENT_NAME}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
