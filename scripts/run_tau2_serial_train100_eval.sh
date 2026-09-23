#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="serial-train100-$(date +%Y%m%d-%H%M%S)"
START_AT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --start-at) START_AT="$2"; shift 2 ;;
    *) echo "Usage: $0 [--run-name NAME] [--start-at STAGE]" >&2; exit 2 ;;
  esac
done
STAGES=(sync-train async-train sync-convert sync-eval300 sync-eval301 async-convert async-eval300 async-eval301 report)
if [[ -n "${START_AT}" && ! " ${STAGES[*]} " == *" ${START_AT} "* ]]; then
  echo "Unknown stage: ${START_AT}" >&2; exit 2
fi
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl"
RUN_DIR="${EXPERIMENT_DIR}/serial/${RUN_NAME}"
if [[ -d "${RUN_DIR}" && -z "${START_AT}" ]]; then
  echo "Run already exists; specify --start-at to resume: ${RUN_DIR}" >&2; exit 2
fi
mkdir -p "${RUN_DIR}"
export TAU2_RUN_NAME="${RUN_NAME}"
export TAU2_ITERATION=99
export TAU2_EVAL_SUFFIX="-${RUN_NAME}"
export TAU2_RL_CLEANUP=0
export TAU2_SERIAL_CLEANUP=1
# Every arm owns its own load/save path; do not inherit a previous experiment.
unset LOAD_DIR CKPT_STEP SAVE_DIR SMOKE_UPDATES SUMMARY_OUTPUT RUN_STAMP
RL_SCRIPT="${PROJECT_ROOT}/examples/tau2-bench/rl/run_tau2_areal_async.sh"
CONVERT_SCRIPT="${PROJECT_ROOT}/scripts/convert_tau2_areal_async_to_hf.sh"
EVAL_SCRIPT="${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_tau2_areal_async.sh"
stage_pid=""
stage=""
cleanup_stage() {
  [[ -n "${stage_pid}" ]] || return 0
  if [[ -n "${stage_pid}" ]]; then
    kill -TERM -- "-${stage_pid}" 2>/dev/null || true
    for ((i=0; i<30; i++)); do
      kill -0 -- "-${stage_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-${stage_pid}" 2>/dev/null; then
      kill -KILL -- "-${stage_pid}" 2>/dev/null || true
    fi
    wait "${stage_pid}" 2>/dev/null || true
    stage_pid=""
  fi
  if [[ "${stage}" == *-train ]]; then
    ray stop --force
  fi
}
trap cleanup_stage EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
enabled=0
for stage in "${STAGES[@]}"; do
  if [[ -z "${START_AT}" || "${START_AT}" == "${stage}" ]]; then enabled=1; fi
  [[ "${enabled}" == 1 ]] || continue
  case "${stage}" in
    sync-train|async-train) command=(bash "${RL_SCRIPT}" "${stage%%-*}" train100) ;;
    sync-convert|async-convert) command=(bash "${CONVERT_SCRIPT}" "${stage%%-*}") ;;
    *-eval300|*-eval301) command=(bash "${EVAL_SCRIPT}" "${stage%%-*}" "${stage##*eval}") ;;
    report) command=(python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/report_tau2_serial.py" "${RUN_NAME}") ;;
  esac
  started="$(date +%s)"
  log="${RUN_DIR}/${stage}_${started}.log"
  echo "tau2_serial stage=${stage} start=${started} run=${RUN_NAME} log=${log}" | tee -a "${RUN_DIR}/stages.log"
  setsid "${command[@]}" > >(tee "${log}") 2>&1 &
  stage_pid=$!
  status=0
  wait "${stage_pid}" || status=$?
  cleanup_stage
  ended="$(date +%s)"
  echo "tau2_serial stage=${stage} end=${ended} seconds=$((ended-started)) exit=${status}" | tee -a "${RUN_DIR}/stages.log"
  [[ "${status}" == 0 ]] || exit "${status}"
done
echo "tau2_serial complete run=${RUN_NAME}" | tee -a "${RUN_DIR}/stages.log"
