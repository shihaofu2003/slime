#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export EXPERIMENT_NAME=tau2-opd-bounded-lr-20260922
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
arm="${1:?Usage: run_distillation.sh <lr2e6|lr5e6>}"
case "${arm}" in
  lr2e6) export TAU2_OPD_LR=2e-6 ;;
  lr5e6) export TAU2_OPD_LR=5e-6 ;;
  *) echo 'Usage: run_distillation.sh <lr2e6|lr5e6>' >&2; exit 2 ;;
esac

export TRAIN_SEED=1235 ROLLOUT_SEED=43
export TAU2_OPD_RUN_TAG="${arm}-seed1235-43"
export TEACHER_ENDPOINT_FILE="${EXPERIMENT_DIR}/teacher_endpoints.env"
export TAU2_USER_ENDPOINT_FILE="${EXPERIMENT_DIR}/user_endpoint.env"
export TAU2_OPD_TEACHER_PERSISTENT=1
export TAU2_OPD_NUM_UPDATES=60 TAU2_OPD_USE_BEHAVIOR_LOGPROBS=0 OPD_POST_UPDATE_LOG_INTERVAL=10
export TAU2_MAX_BUFFERED_GROUPS=32 TAU2_POOL_CAPACITY=32 TAU2_MAX_PENDING_GROUPS=64
export TAU2_ENVIRONMENT_WORKERS=4 TAU2_SAMPLING_MODE=async TAU2_MAX_POLICY_LAG=-1
run_dir="${EXPERIMENT_DIR}/pilot-${TAU2_OPD_RUN_TAG}"
mkdir -p "${run_dir}"
cd "${PROJECT_ROOT}"
echo "OPD_LR_START arm=${arm} lr=${TAU2_OPD_LR} seeds=${TRAIN_SEED}/${ROLLOUT_SEED} time=$(date -Is)"
if bash examples/tau2-bench/opd/run_tau2_opd_airline_retail.sh pilot student 2>&1 | tee -a "${run_dir}/run.log"; then
  train_status=0
else
  train_status=$?
fi
# These are this student's local Ray workers. The persistent services run in
# separate jobs and are intentionally kept running throughout and afterwards.
ray stop --force >"${run_dir}/ray_stop.log" 2>&1
[[ "${train_status}" == 0 ]] || exit "${train_status}"

export ITER_DIR="${run_dir}/checkpoints/iter_0000059"
export OUTPUT_DIR="${run_dir}/checkpoints/iter_0000059_hf"
if [[ ! -f "${OUTPUT_DIR}/model.safetensors.index.json" ]]; then
  bash scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh 2>&1 | tee -a "${run_dir}/convert_iter59.log"
fi
echo "OPD_LR_TRAIN_COMPLETE arm=${arm} checkpoint=${OUTPUT_DIR} time=$(date -Is)"
printf '\n%s completed training and iter59 HF export. [Training log](pilot-%s/run.log), [conversion log](pilot-%s/convert_iter59.log).\n' \
  "${arm}" "${TAU2_OPD_RUN_TAG}" "${TAU2_OPD_RUN_TAG}" >>"${EXPERIMENT_DIR}/README.md"

bash "${EXPERIMENT_DIR}/eval_arm.sh" "${arm}"
