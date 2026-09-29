#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export EXPERIMENT_NAME=tau2-opd-current-buffer-ab
experiment_dir="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
service_dir="${PROJECT_ROOT}/output/experiments/tau2-opd-airline-retail-pilot"
export TEACHER_ENDPOINT_FILE="${service_dir}/teacher_endpoints.env"
export TAU2_USER_ENDPOINT_FILE="${service_dir}/user_endpoint.env"
export TAU2_OPD_LR=2e-6 TAU2_OPD_NUM_UPDATES=60 TAU2_OPD_USE_BEHAVIOR_LOGPROBS=0
export OPD_POST_UPDATE_LOG_INTERVAL=10
export TAU2_POOL_CAPACITY=32 TAU2_MAX_PENDING_GROUPS=64 TAU2_ENVIRONMENT_WORKERS=4
export TAU2_SAMPLING_MODE=async TAU2_MAX_POLICY_LAG=-1

case "${1:-both}" in
  A) arms=(A) ;;
  B) arms=(B) ;;
  both) arms=(A B) ;;
  *) echo 'Usage: train_ab.sh [A|B|both]' >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
for arm in "${arms[@]}"; do
  if [[ "${arm}" == A ]]; then
    export TAU2_OPD_RUN_TAG=current-bounded-lr2e6 TAU2_MAX_BUFFERED_GROUPS=32
  else
    export TAU2_OPD_RUN_TAG=current-unbounded-lr2e6 TAU2_MAX_BUFFERED_GROUPS=-1
  fi
  run_dir="${experiment_dir}/pilot-${TAU2_OPD_RUN_TAG}"
  mkdir -p "${run_dir}"
  echo "OPD_AB_START arm=${arm} tag=${TAU2_OPD_RUN_TAG} time=$(date -Is)"
  if bash examples/tau2-bench/opd/run_tau2_opd_airline_retail.sh pilot student 2>&1 | tee -a "${run_dir}/run.log"; then
    train_status=0
  else
    train_status=$?
  fi
  # Release this job's Ray workers before conversion and the next fresh arm.
  ray stop --force >"${run_dir}/ray_stop.log" 2>&1
  [[ "${train_status}" == 0 ]] || exit "${train_status}"

  export ITER_DIR="${run_dir}/checkpoints/iter_0000059"
  export OUTPUT_DIR="${run_dir}/checkpoints/iter_0000059_hf"
  if [[ ! -f "${OUTPUT_DIR}/model.safetensors.index.json" ]]; then
    bash scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh 2>&1 | tee -a "${run_dir}/convert_iter59.log"
  fi
  echo "OPD_AB_COMPLETE arm=${arm} checkpoint=${OUTPUT_DIR} time=$(date -Is)"
  printf '\nArm %s runner exited0; iter59 converted to HF. [Training log](pilot-%s/run.log), [conversion log](pilot-%s/convert_iter59.log).\n' \
    "${arm}" "${TAU2_OPD_RUN_TAG}" "${TAU2_OPD_RUN_TAG}" >>"${experiment_dir}/README.md"
done

# Both training arms finish before evaluation so their shared-service load is
# comparable. Evaluations deploy their own Agent/User services in this job.
for arm in "${arms[@]}"; do
  bash "${experiment_dir}/eval_model.sh" "${arm}" both
done
