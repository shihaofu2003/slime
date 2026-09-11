#!/usr/bin/env bash

set -euo pipefail

if [[ "${TAU2_SFT_CLEANUP:-1}" != "0" ]]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 3
  pkill -9 ray || true
  pkill -9 python || true
fi

set -ex

export PYTHONUNBUFFERED=1

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
DATASET_DIR="${DATASET_DIR:-${SERVICE_AGENT_ROOT}/datasets/tau2-bench-sft}"
SFT_DATA_PATH="${SFT_DATA_PATH:-${DATASET_DIR}/areal_tau2_sft_strict_no_thinking.jsonl}"

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist}"
LOAD_DIR="${LOAD_DIR:-${TORCH_DIST_DIR}}"
SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_sft}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
NUM_EPOCH="${NUM_EPOCH:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-qwen3}"
TOOL_KEY="${TOOL_KEY:-tools}"
ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-slime.rollout.sft_rollout.generate_rollout}"
LABEL_KEY="${LABEL_KEY:-}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
TRAIN_SEED="${TRAIN_SEED:-1234}"
START_ROLLOUT_ID="${START_ROLLOUT_ID:-}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-4B-tau2-sft}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"

if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
else
  DETECTED_GPUS=0
fi
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -le 0 ]]; then
  NUM_GPUS=1
fi

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"
echo "NUM_GPUS: ${NUM_GPUS}"

source "${PROJECT_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh"
cd "${PROJECT_ROOT}"

verify_resolves_under() {
  local pkg="$1" expected resolved
  expected="$(cd "${2}" && pwd)"
  resolved="$(IMPORT_NAME="${pkg}" python3 -c 'import importlib.util,os; s=importlib.util.find_spec(os.environ["IMPORT_NAME"]); exit(1) if s is None else print(os.path.realpath((s.submodule_search_locations or [os.path.dirname(s.origin)])[0]))' 2>/dev/null)" || resolved=""
  case "${resolved}" in
    "${expected}"|"${expected}"/*)
      echo "[VERIFY] OK: ${pkg} -> ${resolved}" ;;
    *)
      echo "[VERIFY] FAIL: ${pkg} -> '${resolved:-<unresolvable>}', expected under '${expected}'" >&2
      return 1 ;;
  esac
}

verify_resolves_under slime "${PROJECT_ROOT}" \
  || { echo "[VERIFY] slime is not our checkout; run setup/setup_slime.sh first." >&2; exit 1; }

DATA_CHECK_PATH="${SFT_DATA_PATH}"
case "${DATA_CHECK_PATH}" in
  *@\[*\]) DATA_CHECK_PATH="${DATA_CHECK_PATH%@*}" ;;
esac
if [[ ! -f "${DATA_CHECK_PATH}" ]]; then
  echo "[ERROR] Missing SFT data: ${SFT_DATA_PATH}" >&2
  echo "        Run examples/tau2-bench/sft/prepare_data.sh first." >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${TORCH_DIST_DIR}"
  --load "${LOAD_DIR}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)

SFT_ARGS=(
  --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
  --prompt-data "${SFT_DATA_PATH}"
  --input-key messages
  --tool-key "${TOOL_KEY}"
  --loss-mask-type "${LOSS_MASK_TYPE}"
  --seed "${TRAIN_SEED}"
  --rollout-shuffle
  --num-epoch "${NUM_EPOCH}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)
if [[ -n "${LABEL_KEY}" ]]; then
  SFT_ARGS+=(--label-key "${LABEL_KEY}")
fi
if [[ -n "${START_ROLLOUT_ID}" ]]; then
  SFT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style cosine
  --min-lr "${MIN_LR}"
  --lr-warmup-fraction 0.1
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)

load_wandb_api_key() {
  local trace_was_on=0
  case "$-" in
    *x*) trace_was_on=1; set +x ;;
  esac

  if [[ -n "${WANDB_KEY:-}" ]]; then
    export WANDB_API_KEY="${WANDB_KEY}"
  elif [[ -z "${WANDB_API_KEY:-}" && -n "${WANDB_API_KEY_FILE}" ]]; then
    if [[ ! -r "${WANDB_API_KEY_FILE}" ]]; then
      echo "[ERROR] WANDB_API_KEY_FILE is not readable: ${WANDB_API_KEY_FILE}" >&2
      exit 1
    fi
    WANDB_API_KEY="$(head -n 1 "${WANDB_API_KEY_FILE}" | tr -d '[:space:]')"
    export WANDB_API_KEY
  fi

  if [[ "${trace_was_on}" == "1" ]]; then
    set -x
  fi
}

WANDB_ARGS=()
_wandb_trace_was_on=0
case "$-" in
  *x*) _wandb_trace_was_on=1; set +x ;;
esac
if [[ "${USE_WANDB:-0}" != "0" || -n "${WANDB_API_KEY:-}" || -n "${WANDB_KEY:-}" || -n "${WANDB_API_KEY_FILE}" ]]; then
  load_wandb_api_key
  if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE}" != "offline" && "${WANDB_MODE}" != "disabled" ]]; then
    echo "[ERROR] W&B is enabled but no API key is configured." >&2
    echo "        Set WANDB_API_KEY, WANDB_KEY, or WANDB_API_KEY_FILE; or set WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
  )
  if [[ -n "${WANDB_MODE}" ]]; then
    WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
  fi
fi
if [[ "${_wandb_trace_was_on}" == "1" ]]; then
  set -x
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

_trace_was_on=0
case "$-" in
  *x*) _trace_was_on=1; set +x ;;
esac

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${PROJECT_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
  }
}"
if [[ -n "${WANDB_MODE}" ]]; then
  RUNTIME_ENV_JSON="$(
    RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}" WANDB_MODE="${WANDB_MODE}" python3 - <<'PY'
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime_env["env_vars"]["WANDB_MODE"] = os.environ["WANDB_MODE"]
print(json.dumps(runtime_env))
PY
  )"
fi

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}"

if [[ "${_trace_was_on}" == "1" ]]; then
  set -x
fi
