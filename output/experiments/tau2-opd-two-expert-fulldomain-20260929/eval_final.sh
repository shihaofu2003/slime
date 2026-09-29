#!/usr/bin/env bash
# Official four-domain evaluation for the two-expert pure-OPD student.
# Converts the selected training checkpoint to HF first, then runs the
# unchanged seed300 protocol (197 tasks x 4 trials = 788 simulations).
# Submit only after run_opd.sh completes:
#   bash scripts/submit.sh --experiment tau2-opd-two-expert-fulldomain-20260929 \
#     --gpus 8 --name tau2-opd-two-expert-iter159-eval \
#     output/experiments/tau2-opd-two-expert-fulldomain-20260929/eval_final.sh
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-opd-two-expert-fulldomain-20260929"
checkpoint_root="${EXPERIMENT_DIR}/arms/async/20260929_two-expert-fulldomain-iter129-opd/checkpoints"
# Trainer iteration IDs start at zero: update 160 is checkpoint iter_0000159.
export ITER="${ITER:-159}"
printf -v ITER_PAD '%07d' "${ITER}"
export ITER_DIR="${checkpoint_root}/iter_${ITER_PAD}"
export OUTPUT_DIR="${checkpoint_root}/iter_${ITER_PAD}_hf"
export ORIGIN_HF_DIR="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
[[ -d "${ITER_DIR}" ]] || { echo "[ERROR] Missing training checkpoint: ${ITER_DIR}" >&2; exit 1; }
if [[ ! -f "${OUTPUT_DIR}/model.safetensors.index.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh"
fi

export MODEL_PATH="${OUTPUT_DIR}"
export MODEL_NAME="Qwen3-4B-opd-two-expert-fulldomain-iter${ITER}"
export EVAL_LABEL="opd-two-expert-fulldomain-iter${ITER}-four-domain"
export RUN_STAMP="20260929_opd_two_expert_fulldomain_iter${ITER}_seed300"
export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/trajectories/seed300_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed300_${RUN_STAMP}_summary.json"
export AGENT_EVAL_MODE=official-native
export AGENT_TOOL_CALL_PARSER=qwen
export TP=1
export AGENT_REPLICA_CUDA_GROUPS='0;1'
export USER_SGLANG=1
export USER_REPLICA_CUDA_GROUPS='2,3;4,5;6,7'
export DOMAINS=airline,retail,telecom,banking_knowledge
export DOMAIN_CONCURRENCY=airline:1,retail:2,telecom:2,banking_knowledge:4
export GLOBAL_CONCURRENCY=9 BORROW_COMPLETED_DOMAIN_SLOTS=1
export RETRIEVAL_CONFIG=bm25
export NUM_TASKS="" AGENT_MAX_TOKENS=1200 NUM_TRIALS=4 SEED=300

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh" full
