#!/bin/bash

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
DATA_ROOT="${DATA_ROOT:-${SERVICE_AGENT_ROOT}/datasets}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
WANDB_KEY="${WANDB_KEY:-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}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-4B-serviceagent}"

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

if [[ ! -f "${PROJECT_ROOT}/scripts/models/qwen3-4B.sh" ]]; then
    echo "[ERROR] Missing model config script: ${PROJECT_ROOT}/scripts/models/qwen3-4B.sh" >&2
    exit 1
fi

source "${PROJECT_ROOT}/scripts/models/qwen3-4B.sh"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Verify the editable slime install resolves to OUR /mnt/afs checkout.
# The image ships its own slime under /root/slime; setup_slime.sh (run before
# this via submit.sh) re-points it at our source. Gate training on this so we
# never silently run against the image's copy. find_spec locates the package
# WITHOUT importing it, so this stays cheap. (tau_bench check intentionally
# omitted for now.)
# ---------------------------------------------------------------------------
SLIME_SRC="$(cd "${PROJECT_ROOT}" && pwd)"

verify_resolves_under() {
    # args: <python import name> <expected source dir>
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

verify_resolves_under slime "${SLIME_SRC}" || { echo "[VERIFY] slime is not our checkout; run setup/setup_slime.sh first." >&2; exit 1; }

# Confirm our slime actually imports (slime/__init__.py is empty -> cheap) and
# the training entrypoint exists, before spinning up ray.
if ! python3 -c 'import slime' 2>/dev/null; then
    echo "[VERIFY] FAIL: 'import slime' raised an error." >&2
    exit 1
fi
if [ ! -f "${PROJECT_ROOT}/train.py" ]; then
    echo "[VERIFY] FAIL: ${PROJECT_ROOT}/train.py not found." >&2
    exit 1
fi
echo "[VERIFY] OK: 'import slime' works; train.py present."

mkdir -p "${CHECKPOINT_ROOT}/Qwen3-4B_slime"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-4B"
   --ref-load "${MODEL_ROOT}/Qwen3-4B_torch_dist"
   --load "${CHECKPOINT_ROOT}/Qwen3-4B_slime"
   --save "${CHECKPOINT_ROOT}/Qwen3-4B_slime"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${DATA_ROOT}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

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
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export WANDB_API_KEY="${WANDB_KEY}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_API_KEY\": \"${WANDB_KEY}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
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
   ${MISC_ARGS[@]}
