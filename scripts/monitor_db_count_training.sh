#!/usr/bin/env bash
set -euo pipefail

: "${DB_COUNT_JOB_ID:?Set DB_COUNT_JOB_ID}"
: "${DB_COUNT_RUN_DIR:?Set DB_COUNT_RUN_DIR}"
: "${DB_COUNT_PROJECT_ROOT:?Set DB_COUNT_PROJECT_ROOT}"

PROJECT_ROOT="${DB_COUNT_PROJECT_ROOT}"
RUN_DIR="${DB_COUNT_RUN_DIR}"
EVAL_ROOT="${DB_COUNT_EVAL_ROOT:-${RUN_DIR}/eval}"
INTERVAL_SECONDS="${DB_COUNT_POLL_SECONDS:-300}"
REPORT_SECONDS="${DB_COUNT_REPORT_SECONDS:-3600}"
LAST_REPORT=0
mkdir -p "${EVAL_ROOT}" "${RUN_DIR}/monitor"
LOG_PATH="${RUN_DIR}/monitor/hourly_progress.log"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "${message}" | tee -a "${LOG_PATH}"
}

submit_checkpoint_eval() {
  local checkpoint_iteration="$1"
  local marker="${EVAL_ROOT}/iter_${checkpoint_iteration}.submitted"
  if [[ -e "${marker}" ]]; then
    return 0
  fi
  mkdir -p "${EVAL_ROOT}/iter_${checkpoint_iteration}"
  if bash "${PROJECT_ROOT}/scripts/submit.sh" \
      --experiment tau2-db-count-v1 \
      --name "db-count-four-domain-eval-iter${checkpoint_iteration}" \
      --gpus 8 --cpus 64 --memory 1024 \
      -e "DB_COUNT_PROJECT_ROOT=${PROJECT_ROOT}" \
      -e "DB_COUNT_RUN_DIR=${RUN_DIR}" \
      -e "DB_COUNT_EVAL_ROOT=${EVAL_ROOT}/iter_${checkpoint_iteration}" \
      "${PROJECT_ROOT}/../slime-async-tau2/scripts/eval_db_count_checkpoint.sh" "${checkpoint_iteration}"; then
    touch "${marker}"
    log "submitted four-domain evaluation for checkpoint iter_${checkpoint_iteration}"
  else
    log "submission failed for checkpoint iter_${checkpoint_iteration}; will retry"
  fi
}

log "monitor started job=${DB_COUNT_JOB_ID} run=${RUN_DIR} poll=${INTERVAL_SECONDS}s report=${REPORT_SECONDS}s"
while true; do
  now="$(date +%s)"
  latest="$(tr -d '[:space:]' < "${RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
  checkpoint_list="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -type d -name 'iter_???????' -printf '%f\n' 2>/dev/null | sort -V | tr '\n' ' ')"

  for checkpoint in ${checkpoint_list}; do
    number="${checkpoint#iter_}"
    if (( (10#${number} + 1) % 50 == 0 )); then
      submit_checkpoint_eval "${number}"
    fi
  done

  job_state="$(/mnt/afs/users/fush/job show "${DB_COUNT_JOB_ID}" 2>/dev/null | grep -E '"state"|STATE' | head -2 | tr '\n' ' ' || true)"
  if (( now - LAST_REPORT >= REPORT_SECONDS )); then
    log "job_state=${job_state:-unavailable} latest_checkpoint=${latest:-none} checkpoints=${checkpoint_list:-none}"
    LAST_REPORT="${now}"
  fi

  case "${job_state:-}" in
    *SUCCEEDED*|*FAILED*|*STOPPED*|*CANCELLED*|*SUSPENDED*)
      log "training job is terminal; monitor exiting"
      exit 0
      ;;
  esac
  sleep "${INTERVAL_SECONDS}"
done
