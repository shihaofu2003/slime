#!/bin/bash
#
# tau-bench (tau1, retail) RL run-through for Qwen3-4B-Instruct-2507 with slime.
#
# Adapted from examples/tau-bench/run_qwen3_4B.sh (the tau reference) onto our
# /mnt/afs paths and the serviceagent conventions (env roots, editable-install
# verify, ray + sglang). Differs from run-qwen3-4B-serviceagent.sh (math/dapo)
# in: model -> Instruct-2507, data -> tau-bench retail task indices, a custom
# tau rollout fn, and a reward-nonzero-std dynamic filter.
#
# The tau rollout's user simulator calls the Gemini API (litellm), so
# GEMINI_API_KEY must be set (defaults to NONE -> rollouts will abort). It is
# propagated to the ray workers via RUNTIME_ENV_JSON below.

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

export PYTHONUNBUFFERED=1

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${SERVICE_AGENT_ROOT}/models}"
DATA_ROOT="${TAU_DATA_ROOT:-${SERVICE_AGENT_ROOT}/datasets/tau-bench}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
TAU_EXAMPLES="${PROJECT_ROOT}/examples/tau-bench"
# Per-trajectory JSONL dump (one line per completed rollout trajectory). Each RUN
# gets its own timestamped file under the experiment's trajectories/ dir so runs
# don't clobber each other (OUTPUT_ROOT is set by submit.sh). Propagated to the
# Ray rollout workers via RUNTIME_ENV_JSON below. Unset/empty -> no dumping.
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TRAJECTORY_DUMP_DIR="${TRAJECTORY_DUMP_DIR:-${PROJECT_ROOT}/${OUTPUT_ROOT:-output}/trajectories}"
TRAJECTORY_DUMP_PATH="${TRAJECTORY_DUMP_PATH:-${TRAJECTORY_DUMP_DIR}/trajectories_${RUN_TIMESTAMP}.jsonl}"
mkdir -p "${TRAJECTORY_DUMP_DIR}"
WANDB_KEY="${WANDB_KEY:-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}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-4B-tau-bench}"
# tau user simulator: OpenAI-compatible endpoint (litellm "openai" provider).
TAU_USER_API_KEY="${TAU_USER_API_KEY:-sk-BYKhT7a1luGBZg2gcjxFhA3r4GOgLaJAsXsmmqhTuvYvlrwL}"
TAU_USER_API_BASE="${TAU_USER_API_BASE:-https://api.chatanywhere.tech/v1}"
TAU_USER_MODEL="${TAU_USER_MODEL:-gemini-3-flash-preview}"
TAU_USER_MODEL_PROVIDER="${TAU_USER_MODEL_PROVIDER:-openai}"
# litellm's openai provider reads these for the user-sim calls.
export OPENAI_API_KEY="${TAU_USER_API_KEY}"
export OPENAI_API_BASE="${TAU_USER_API_BASE}"

# Run-through defaults (kept small/conservative to bound simultaneous user-sim
# API calls). The user sim (gemini-3-flash-preview) is fast, so these can be
# scaled up. The upstream tau reference uses ROLLOUT_BATCH_SIZE=32,
# N_SAMPLES_PER_PROMPT=8 (global-batch 256), NUM_ROLLOUT=500.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS}}
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ]; then
    NUM_GPUS=8
fi
echo "NUM_GPUS: $NUM_GPUS"

if [[ ! -f "${PROJECT_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh" ]]; then
    echo "[ERROR] Missing model config script: ${PROJECT_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh" >&2
    exit 1
fi

source "${PROJECT_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh"
cd "${PROJECT_ROOT}"

# tau_bench.envs imports litellm at module load, which the image lacks. Install
# tau_bench's runtime deps (skips if already importable) before any tau import.
bash "${SERVICE_AGENT_ROOT}/setup/setup_tau_bench_deps.sh"

# ---------------------------------------------------------------------------
# Verify editable installs resolve to OUR /mnt/afs checkout (the image ships its
# own slime/tau_bench under /root). find_spec locates the package without
# importing it, so this stays cheap.
# ---------------------------------------------------------------------------
verify_resolves_under() {
    local pkg="$1" expected resolved
    if [ ! -d "${2}" ]; then
        echo "[VERIFY] FAIL: expected source dir for '${pkg}' missing: ${2}" >&2
        return 1
    fi
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
verify_resolves_under tau_bench "${SERVICE_AGENT_ROOT}/tau-bench" \
    || { echo "[VERIFY] tau_bench is not our checkout; run setup/setup_tau_bench.sh first." >&2; exit 1; }

if [ -z "${TAU_USER_API_KEY}" ]; then
    echo "[WARN] TAU_USER_API_KEY is empty -- the tau user simulator will fail and all rollouts will abort." >&2
fi

# Pre-flight: the tau data must exist (run run-tau-bench-mock-data.sh first).
if [ ! -f "${DATA_ROOT}/retail_train_tasks.jsonl" ]; then
    echo "[ERROR] Missing tau-bench data: ${DATA_ROOT}/retail_train_tasks.jsonl" >&2
    echo "        Run scripts/run-tau-bench-mock-data.sh first." >&2
    exit 1
fi

mkdir -p "${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_slime"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-4B-Instruct-2507"
   --ref-load "${MODEL_ROOT}/Qwen3-4B-Instruct-2507_torch_dist"
   --load "${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_slime"
   --save "${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_slime"
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_ROOT}/retail_train_tasks.jsonl"
   --input-key index
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)
# Reward-variance dynamic filter (drops groups whose samples all share the same
# reward). OFF by default: a cold model rarely yields per-group reward variance,
# so this filter rejects ~90% of groups and resamples -> very slow batch fill
# (observed ~3h for one 32-sample batch). Enable (USE_DYNAMIC_FILTER=1) once the
# model is warm enough to produce reward variance.
if [ "${USE_DYNAMIC_FILTER:-0}" = "1" ]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
fi

# Eval runs full multi-turn rollouts over the dev set; skip for a quick smoke
# (SKIP_EVAL=1).
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    EVAL_ARGS=(
       --eval-interval 5
       --eval-prompt-data retail-dev "${DATA_ROOT}/retail_dev_tasks.jsonl"
       --n-samples-per-eval-prompt 1
       --eval-max-response-len 1024
       --eval-top-k 1
    )
else
    EVAL_ARGS=()
fi

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --kl-coef (reward-shaping KL) left at 0: the custom tau rollout now returns
   # per-token logprobs (generate_with_tau -> trainable_agents asolve sets
   # Sample.rollout_log_probs), so compute_advantages_and_returns finds a non-None
   # tensor at loss.py:700-703 and no longer crashes. NB this is NOT --kl-loss-coef
   # (the actor-loss KL term, left at 0 below); they are independent.
   # Fallback: if the loss.py:703 all-None crash ever returns, set KL_COEF=0.001
   # (a reward-shaping kl forces the actor log-prob pre-forward as a workaround).
   --kl-coef "${KL_COEF:-0}"
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${WANDB_GROUP}"
   --wandb-key "${WANDB_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   # Cap concurrent agent generations to keep simultaneous user-sim API calls
   # (and rate limits) bounded.
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-16}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau.generate
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export WANDB_API_KEY="${WANDB_KEY}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir /root/shared/ray_temp

# PYTHONPATH must include Megatron-LM and the tau examples dir: the custom rollout
# fn is imported as `generate_with_tau`, which in turn imports `trainable_agents`,
# `openai_tool_adapter`, ... all siblings under examples/tau-bench.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${TAU_EXAMPLES}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_API_KEY\": \"${WANDB_KEY}\",
    \"OPENAI_API_KEY\": \"${TAU_USER_API_KEY}\",
    \"OPENAI_API_BASE\": \"${TAU_USER_API_BASE}\",
    \"TAU_USER_MODEL\": \"${TAU_USER_MODEL}\",
    \"TAU_USER_MODEL_PROVIDER\": \"${TAU_USER_MODEL_PROVIDER}\",
    \"TRAJECTORY_DUMP_PATH\": \"${TRAJECTORY_DUMP_PATH}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
