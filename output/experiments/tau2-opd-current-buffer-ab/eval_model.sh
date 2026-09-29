#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-opd-current-buffer-ab"
model_key="${1:?Usage: eval_model.sh <sft|airline|retail|A|B> [seed300|seed301|both]}"
expert_root="${PROJECT_ROOT}/output/experiments/tau2-domain-experts-sft4505-b128/arms/async"
case "${model_key}" in
  sft) export MODEL_PATH="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920/iter_0004505_hf" ;;
  airline) export MODEL_PATH="${expert_root}/20260920_sft4505-airline-train/checkpoints/iter_0000029_hf" ;;
  retail) export MODEL_PATH="${expert_root}/20260920_sft4505-retail-train/checkpoints/iter_0000009_hf" ;;
  A) export MODEL_PATH="${EXPERIMENT_DIR}/pilot-current-bounded-lr2e6/checkpoints/iter_0000059_hf" ;;
  B) export MODEL_PATH="${EXPERIMENT_DIR}/pilot-current-unbounded-lr2e6/checkpoints/iter_0000059_hf" ;;
  *) echo "Unknown model: ${model_key}" >&2; exit 2 ;;
esac
case "${2:-both}" in
  300|301) seeds=("$2") ;;
  both) seeds=(300 301) ;;
  *) echo 'Expected seed300, seed301 or both (arguments: 300|301|both)' >&2; exit 2 ;;
esac
export MODEL_NAME="Qwen3-4B-opd-buffer-${model_key}"
export USER_SGLANG=1 DOMAINS=airline,retail,telecom
export DOMAIN_CONCURRENCY=airline:2,retail:2,telecom:5 GLOBAL_CONCURRENCY=9 MAX_CONCURRENCY=3
export AGENT_REPLICA_CUDA_GROUPS='0;1' BORROW_COMPLETED_DOMAIN_SLOTS=1
export AGENT_EVAL_MODE=official-native AGENT_MAX_TOKENS=1200 NUM_TRIALS=4 NUM_TASKS=""
export AUTO_RESUME=1
mkdir -p "${EXPERIMENT_DIR}/eval/${model_key}"
cd "${PROJECT_ROOT}"
for seed in "${seeds[@]}"; do
  export SEED="${seed}" EVAL_LABEL="${model_key}"
  export RUN_STAMP="20260922_${model_key}_seed${seed}"
  export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${model_key}/trajectories/seed${seed}_${RUN_STAMP}"
  export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${model_key}/seed${seed}_summary.json"
  echo "OPD_AB_EVAL_START model=${model_key} seed=${seed} time=$(date -Is)"
  bash examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh \
    2>&1 | tee -a "${EXPERIMENT_DIR}/eval/${model_key}/seed${seed}.log"
  echo "OPD_AB_EVAL_COMPLETE model=${model_key} seed=${seed} summary=${SUMMARY_OUTPUT} time=$(date -Is)"
  printf '\n%s seed%s official evaluation completed. [Summary](eval/%s/seed%s_summary.json), [run log](eval/%s/seed%s.log).\n' \
    "${model_key}" "${seed}" "${model_key}" "${seed}" "${model_key}" "${seed}" >>"${EXPERIMENT_DIR}/README.md"
done
python3 "${EXPERIMENT_DIR}/summarize.py" || echo 'Comparison report failed; retain official results and rerun summarize.py.' >&2
