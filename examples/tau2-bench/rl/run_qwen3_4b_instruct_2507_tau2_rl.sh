#!/usr/bin/env bash

set -euo pipefail

if [[ "${TAU2_RL_CLEANUP:-1}" != "0" ]]; then
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
DATA_ROOT="${DATA_ROOT:-${SERVICE_AGENT_ROOT}/datasets}"
TAU2_SRC="${SERVICE_AGENT_ROOT}/tau2-bench/src"
RL_DIR="${PROJECT_ROOT}/examples/tau2-bench/rl"

SOURCE_RL_DATA="${SOURCE_RL_DATA:-${DATA_ROOT}/AReaL-tau2-data/tau2_rl_train.jsonl}"
PREPARED_RL_DATA="${PREPARED_RL_DATA:-${PROJECT_ROOT}/output/datasets/tau2-bench-rl/areal_tau2_rl_train.jsonl}"

SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${SFT_CKPT_ROOT}/iter_0002413_hf}"
REF_LOAD="${REF_LOAD:-${SFT_CKPT_ROOT}}"
SAVE_DIR="${SAVE_DIR:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo}"

USER_SGLANG="${USER_SGLANG:-1}"
USER_MODEL_PATH="${USER_MODEL_PATH:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"
HOST="${HOST:-127.0.0.1}"
USER_PORT="${USER_PORT:-30001}"
USER_TP="${USER_TP:-1}"
USER_MEM_FRACTION="${USER_MEM_FRACTION:-0.82}"
USER_SGLANG_EXTRA_ARGS="${USER_SGLANG_EXTRA_ARGS:---tool-call-parser qwen}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1200}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-80}"
TAU2_MAX_ERRORS="${TAU2_MAX_ERRORS:-10}"
TAU2_USER_TEMPERATURE="${TAU2_USER_TEMPERATURE:-0.0}"
TAU2_USER_MAX_TOKENS="${TAU2_USER_MAX_TOKENS:-512}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-8}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-12288}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}"
LR="${LR:-1e-6}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-tau2-agent-rl-grpo}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-}"
USE_WANDB="${USE_WANDB:-0}"

if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
else
  DETECTED_GPUS=0
fi
TOTAL_GPUS="${TOTAL_GPUS:-${DETECTED_GPUS}}"
if [[ "${TOTAL_GPUS}" != "4" && "${TOTAL_GPUS}" != "8" ]]; then
  echo "[ERROR] tau2 RL scripts only support 4-card or 8-card jobs; got TOTAL_GPUS=${TOTAL_GPUS}" >&2
  echo "        Submit with scripts/submit.sh --gpus 4 or --gpus 8." >&2
  exit 1
fi

if [[ "${USER_SGLANG}" == "0" ]]; then
  RAY_GPUS="${RAY_GPUS:-${TOTAL_GPUS}}"
  USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-}"
else
  if [[ "${TOTAL_GPUS}" == "8" ]]; then
    RAY_GPUS="${RAY_GPUS:-6}"
    USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-7}"
  else
    RAY_GPUS="${RAY_GPUS:-2}"
    USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-3}"
  fi
fi

join_gpu_range() {
  local end="$1"
  local out="" i
  for ((i=0; i<end; i++)); do
    if [[ -n "${out}" ]]; then
      out+=","
    fi
    out+="${i}"
  done
  echo "${out}"
}

AGENT_CUDA_VISIBLE_DEVICES="${AGENT_CUDA_VISIBLE_DEVICES:-$(join_gpu_range "${RAY_GPUS}")}"

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"
echo "TOTAL_GPUS: ${TOTAL_GPUS}; RAY_GPUS: ${RAY_GPUS}; AGENT_CUDA_VISIBLE_DEVICES=${AGENT_CUDA_VISIBLE_DEVICES}; USER_CUDA_VISIBLE_DEVICES=${USER_CUDA_VISIBLE_DEVICES:-external}"

export PYTHONPATH="${PROJECT_ROOT}:${RL_DIR}:${TAU2_SRC}:${PYTHONPATH:-}"

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
verify_resolves_under tau2 "${TAU2_SRC}/tau2" \
  || { echo "[VERIFY] tau2 is not our checkout; run setup/setup_tau2_bench.sh first." >&2; exit 1; }

if [[ ! -f "${SOURCE_RL_DATA}" ]]; then
  echo "[ERROR] Missing source RL data: ${SOURCE_RL_DATA}" >&2
  exit 1
fi
if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing HF checkpoint: ${HF_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
  echo "[ERROR] Missing torch_dist SFT checkpoint root: ${REF_LOAD}" >&2
  exit 1
fi

PREPARE_ARGS=(
  --input "${SOURCE_RL_DATA}"
  --output "${PREPARED_RL_DATA}"
)
if [[ -n "${PREPARE_RL_LIMIT:-}" ]]; then
  PREPARE_ARGS+=(--limit "${PREPARE_RL_LIMIT}")
fi
python3 "${RL_DIR}/prepare_rl_data.py" "${PREPARE_ARGS[@]}"

mkdir -p "${SAVE_DIR}" "$(dirname "${TAU2_RL_TRAJECTORY_DUMP_PATH:-${PROJECT_ROOT}/output/tau2-rl-trajectories/unused.jsonl}")"

USER_SGLANG_PID=""
USER_SGLANG_LOG="${USER_SGLANG_LOG:-${PROJECT_ROOT}/output/tau2-rl-user-sglang-${USER_PORT}.log}"

cleanup() {
  if [[ -n "${USER_SGLANG_PID}" ]]; then
    kill "${USER_SGLANG_PID}" 2>/dev/null || true
    wait "${USER_SGLANG_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_sglang() {
  local label="$1"
  local pid="$2"
  local port="$3"
  local log_path="$4"
  local ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] sglang ${label} died before readiness; tail of ${log_path}:" >&2
      tail -n 80 "${log_path}" >&2 || true
      exit 1
    fi
    if python3 -c "import urllib.request,sys; urllib.request.urlopen('http://${HOST}:${port}/health', timeout=3); sys.exit(0)" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "[ERROR] sglang ${label} did not become ready; tail of ${log_path}:" >&2
    tail -n 80 "${log_path}" >&2 || true
    exit 1
  fi
}

if [[ "${USER_SGLANG}" != "0" ]]; then
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES}" python3 -m sglang.launch_server \
    --model-path "${USER_MODEL_PATH}" \
    --served-model-name "${USER_MODEL}" \
    --host "${HOST}" \
    --port "${USER_PORT}" \
    --tp "${USER_TP}" \
    --mem-fraction-static "${USER_MEM_FRACTION}" \
    ${USER_SGLANG_EXTRA_ARGS} \
    >"${USER_SGLANG_LOG}" 2>&1 &
  USER_SGLANG_PID="$!"
  wait_for_sglang user "${USER_SGLANG_PID}" "${USER_PORT}" "${USER_SGLANG_LOG}"
  export TAU2_USER_API_BASE="${TAU2_USER_API_BASE:-http://${HOST}:${USER_PORT}/v1}"
  export TAU2_USER_API_KEY="${TAU2_USER_API_KEY:-dummy-key-for-local-server}"
else
  : "${TAU2_USER_API_BASE:?TAU2_USER_API_BASE is required when USER_SGLANG=0}"
  : "${TAU2_USER_API_KEY:?TAU2_USER_API_KEY is required when USER_SGLANG=0}"
fi
export TAU2_USER_MODEL="${TAU2_USER_MODEL:-${USER_MODEL}}"
export TAU2_USER_TEMPERATURE
export TAU2_USER_MAX_TOKENS
export TAU2_MAX_STEPS
export TAU2_MAX_ERRORS
export TAU2_AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS}"

source "${PROJECT_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh"
cd "${PROJECT_ROOT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD}"
  --load "${SAVE_DIR}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
  --prompt-data "${PREPARED_RL_DATA}"
  --input-key prompt
  --metadata-key metadata
  --loss-mask-type qwen3
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --rollout-max-response-len "${AGENT_MAX_TOKENS}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-top-p "${ROLLOUT_TOP_P}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --balance-data
)
if [[ "${USE_DYNAMIC_FILTER:-0}" != "0" ]]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-2}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --kl-coef "${KL_COEF:-0}"
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
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
if [[ "${USE_WANDB}" != "0" || -n "${WANDB_API_KEY:-}" || -n "${WANDB_KEY:-}" || -n "${WANDB_API_KEY_FILE}" ]]; then
  load_wandb_api_key
  if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE}" != "offline" && "${WANDB_MODE}" != "disabled" ]]; then
    echo "[ERROR] W&B is enabled but no API key is configured." >&2
    echo "        Set WANDB_API_KEY, WANDB_KEY, or WANDB_API_KEY_FILE; or set WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
  WANDB_ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}" --disable-wandb-random-suffix)
  if [[ -n "${WANDB_MODE}" ]]; then
    WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
  fi
fi
if [[ "${_wandb_trace_was_on}" == "1" ]]; then
  set -x
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
  --sglang-mem-fraction-static "${AGENT_MEM_FRACTION:-0.68}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-generate-function-path rollout.generate
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export CUDA_VISIBLE_DEVICES="${AGENT_CUDA_VISIBLE_DEVICES}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir /root/shared/ray_temp

_trace_was_on=0
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  case "$-" in
    *x*) _trace_was_on=1; set +x ;;
  esac
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${PROJECT_ROOT}:${RL_DIR}:${TAU2_SRC}\",
    \"CUDA_VISIBLE_DEVICES\": \"${AGENT_CUDA_VISIBLE_DEVICES}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TAU2_USER_MODEL\": \"${TAU2_USER_MODEL}\",
    \"TAU2_USER_API_BASE\": \"${TAU2_USER_API_BASE}\",
    \"TAU2_USER_API_KEY\": \"${TAU2_USER_API_KEY}\",
    \"TAU2_USER_TEMPERATURE\": \"${TAU2_USER_TEMPERATURE}\",
    \"TAU2_USER_MAX_TOKENS\": \"${TAU2_USER_MAX_TOKENS}\",
    \"TAU2_MAX_STEPS\": \"${TAU2_MAX_STEPS}\",
    \"TAU2_MAX_ERRORS\": \"${TAU2_MAX_ERRORS}\",
    \"TAU2_AGENT_MAX_TOKENS\": \"${TAU2_AGENT_MAX_TOKENS}\",
    \"TAU2_RL_TRAJECTORY_DUMP_PATH\": \"${TAU2_RL_TRAJECTORY_DUMP_PATH:-}\",
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
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${RAY_GPUS}" \
  --rollout-num-gpus "${RAY_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"

if [[ "${_trace_was_on}" == "1" ]]; then
  set -x
fi
