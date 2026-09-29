#!/usr/bin/env bash
set -euo pipefail

update="${1:?usage: eval_checkpoint_seed300.sh <40|80|90|100|110|120>}"
case "${update}" in
  40) checkpoint_iteration=39 ;;
  80) checkpoint_iteration=79 ;;
  90) checkpoint_iteration=89 ;;
  100) checkpoint_iteration=99 ;;
  110) checkpoint_iteration=109 ;;
  120) checkpoint_iteration=119 ;;
  *)
    echo "Expected update 40, 80, 90, 100, 110, or 120; got ${update}." >&2
    exit 2
    ;;
esac

printf -v padded '%07d' "${checkpoint_iteration}"
export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-opd-four-domain-20260923"
checkpoint_root="${EXPERIMENT_DIR}/pilot-lr2e6-seed1235-43/checkpoints"
export ITER_DIR="${checkpoint_root}/iter_${padded}"
export OUTPUT_DIR="${checkpoint_root}/iter_${padded}_hf"
export ORIGIN_HF_DIR="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"

eval_dir="${EXPERIMENT_DIR}/eval/checkpoint-update${update}-seed300"
mkdir -p "${eval_dir}"
if [[ ! -f "${OUTPUT_DIR}/model.safetensors.index.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" \
    2>&1 | tee -a "${eval_dir}/convert.log"
fi

export MODEL_PATH="${OUTPUT_DIR}"
export MODEL_NAME="Qwen3-4B-opd-four-domain-lr2e6-update${update}"
export USER_SGLANG=1 AGENT_REPLICA_CUDA_GROUPS='0;1'
export AGENT_EVAL_MODE=official-native AGENT_MAX_TOKENS=1200
export AGENT_TEMPERATURE=0.6 AGENT_TOP_P=1.0 MAX_STEPS=200 MAX_ERRORS=10
export NUM_TRIALS=4 NUM_TASKS="" AUTO_RESUME=1 RETRIEVAL_CONFIG=bm25
export EVAL_LABEL="four-domain-lr2e6-update${update}"
export SEED=300
export RUN_STAMP="20260923_four_domain_lr2e6_update${update}_seed300"
export SAVE_PREFIX="${eval_dir}/trajectories/seed300_${RUN_STAMP}"
export SUMMARY_OUTPUT="${eval_dir}/summary.json"

cd "${PROJECT_ROOT}"
echo "OPD_CHECKPOINT_EVAL_START update=${update} checkpoint=iter_${padded} seed=300 time=$(date -Is)"
bash examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh full \
  2>&1 | tee -a "${eval_dir}/run.log"
echo "OPD_CHECKPOINT_EVAL_COMPLETE update=${update} checkpoint=iter_${padded} seed=300 summary=${SUMMARY_OUTPUT} time=$(date -Is)"
