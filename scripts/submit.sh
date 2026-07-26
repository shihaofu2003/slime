#!/usr/bin/env bash
set -euo pipefail

# Generic SCO ACP submit helper for arbitrary training/evaluation scripts.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit.sh [options] <script>
  bash scripts/submit.sh <script> [options]

Arguments:
  <script>                   Bash script path to run (positional or --script).

Options:
  --name <name>              Job display name. Default: derived from script path (drop scripts/ + .sh, replace / with -).
  --workspace-name <name>    Default: project-one
  --aec2-name <name>         Default: computing-cluster-01e
  --gpus <num>               Default: 1
  --worker-spec <spec>       Default: auto-derived from --gpus
  --nodes <num>              Default: 1
  --priority <level>         Default: high
  --image <url>              Default: project image
  --storage-mount <mount>    Default: 019d2d50-6ac1-70b5-809c-c9e87fcc8899:/mnt/afs
  --project-root <path>      Default: repo root
  --experiment <name>     Experiment name (groups output under output/experiments/<name>/jobs)
  --output-root <path>       Default: <project_root>/output/jobs or <project_root>/output/experiments/$EXPERIMENT/jobs
  --log-dir <path>           Default: <output_root>/<job_name>
  --spot                     Submit as spot (idle-time) job
  --                         Separator: all args after this are passed to the script
  -t, --tail                 Wait for run log and tail -f after submission
  --dry-run                  Print command only, do not submit
  -h, --help                 Show this message

Examples:
  # Quick: script name is job name
  bash scripts/submit.sh scripts/eval.sh

  # With options
  bash scripts/submit.sh --gpus 8 --spot scripts/train/train_dense_encoder.sh

  # Explicit name
  bash scripts/submit.sh --name my-train --gpus 8 scripts/train/train_dense_encoder.sh

  # Under a named experiment
  bash scripts/submit.sh --experiment exp1 --gpus 8 scripts/train/train.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT="${EXPERIMENT:-}"
JOB_PREFIX="${JOB_PREFIX:-fsh}"
WORKSPACE_NAME="project-one"
AEC2_NAME="computing-cluster-01e"
JOB_NAME=""
GPU_NUMS="1"
WORKER_SPEC=""
WORKER_NODES="1"
PRIORITY="normal"
TRAINING_FRAMEWORK="pytorch"
CONTAINER_IMAGE_URL="registry.cn-sh-01.sensecore.cn/ccr-zhicheng-06/slimerl/slime"
STORAGE_MOUNT="019d2d50-6ac1-70b5-809c-c9e87fcc8899:/mnt/afs"

OUTPUT_ROOT=""
LOG_DIR=""
SCRIPT=""
DRY_RUN="0"
SPOT="0"
TAIL_LOG="0"
SCRIPT_ARGS=""


derive_job_name_from_script() {
  local script_path="$1"
  local rel_path="${script_path}"

  # Prefer deriving a stable path relative to project root when possible.
  if command -v realpath >/dev/null 2>&1; then
    local abs_path
    abs_path="$(realpath "${script_path}" 2>/dev/null || true)"
    if [[ -n "${abs_path}" ]]; then
      rel_path="$(realpath --relative-to="${PROJECT_ROOT}" "${abs_path}" 2>/dev/null || echo "${script_path}")"
    fi
  else
    rel_path="${rel_path#${PROJECT_ROOT}/}"
    rel_path="${rel_path#./}"
  fi

  # Drop leading scripts/ (repo convention)
  rel_path="${rel_path#scripts/}"
  # Drop extension
  rel_path="${rel_path%.sh}"
  # Replace path separators with dashes
  rel_path="${rel_path//\//-}"

  echo "${rel_path}"
}

derive_output_root() {
  if [[ -n "${EXPERIMENT}" ]]; then
    echo "${PROJECT_ROOT}/output/experiments/${EXPERIMENT}/jobs"
  else
    echo "${PROJECT_ROOT}/output/jobs"
  fi
}

derive_remote_output_root() {
  if [[ -n "${EXPERIMENT}" ]]; then
    echo "output/experiments/${EXPERIMENT}"
  else
    echo "output"
  fi
}


while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-name)
      WORKSPACE_NAME="$2"; shift 2 ;;
    --experiment)
      EXPERIMENT="$2"; shift 2 ;;
    --aec2-name)
      AEC2_NAME="$2"; shift 2 ;;
    --name)
      JOB_NAME="$2"; shift 2 ;;
    --gpus)
      GPU_NUMS="$2"; shift 2 ;;
    --worker-spec)
      WORKER_SPEC="$2"; shift 2 ;;
    --nodes)
      WORKER_NODES="$2"; shift 2 ;;
    --priority)
      PRIORITY="$2"; shift 2 ;;
    --training-framework)
      TRAINING_FRAMEWORK="$2"; shift 2 ;;
    --image)
      CONTAINER_IMAGE_URL="$2"; shift 2 ;;
    --storage-mount)
      STORAGE_MOUNT="$2"; shift 2 ;;
    --project-root)
      PROJECT_ROOT="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    --script)
      SCRIPT="$2"; shift 2 ;;
    --spot)
      SPOT="1"; shift 1 ;;
    --)
      shift 1; SCRIPT_ARGS="$*"; break ;;
    -t|--tail)
      TAIL_LOG="1"; shift 1 ;;
    --dry-run)
      DRY_RUN="1"; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    -*)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1 ;;
    *)
      # Positional argument: script path
      SCRIPT="$1"; shift 1 ;;
  esac
done

# Default job name from script path (drop scripts/ prefix, drop extension, replace / with -)
if [[ -z "${JOB_NAME}" && -n "${SCRIPT}" ]]; then
  JOB_NAME="$(derive_job_name_from_script "${SCRIPT}")"
fi

if [[ -z "${JOB_NAME}" ]]; then
  echo "[ERROR] Job name is required (use --name or provide a script path)" >&2
  usage
  exit 1
fi

# Append timestamp suffix to job name
TIMESTAMP="$(date +%m%d-%H%M%S)"
JOB_NAME="${JOB_PREFIX}-${JOB_NAME}-${TIMESTAMP}"

if [[ -z "${SCRIPT}" ]]; then
  echo "[ERROR] Script path is required" >&2
  usage
  exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[ERROR] Script not found: ${SCRIPT}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="$(derive_output_root)"
fi

if [[ -z "${WORKER_SPEC}" ]]; then
  if [[ "${GPU_NUMS}" == "8" ]]; then
    WORKER_SPEC="N6lS.Iu.I10.8.64c1024g"
  else
    WORKER_SPEC="N6lS.Iu.I10.${GPU_NUMS}"
  fi
fi

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${OUTPUT_ROOT}/${JOB_NAME}"
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"
SUBMIT_LOG="${LOG_DIR}/submit_${TIMESTAMP}.log"

# Copy the script to output directory so that later edits to the original don't
# affect the submitted job.
SCRIPT_COPY="${LOG_DIR}/$(basename "${SCRIPT}")"
cp "${SCRIPT}" "${SCRIPT_COPY}"

REMOTE_OUTPUT_ROOT="$(derive_remote_output_root)"

# Environment setup scripts to run before the training script. Each re-points an
# editable package install at our /mnt/afs checkout -- the image ships its own
# copies under /root (slime, tau_bench, ...), so without this `import <pkg>`
# resolves there instead of to our code. The scripts live under the same
# /mnt/afs mount, so the paths are identical inside the container; we check
# existence locally to decide whether to wire each one in. Derived from
# SCRIPT_DIR (not PROJECT_ROOT) so they survive a --project-root override.
SERVICE_AGENT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_SETUP_SCRIPT="${SERVICE_AGENT_ROOT}/setup/setup_env.sh"
SETUP_SCRIPTS=(
  "${SERVICE_AGENT_ROOT}/setup/setup_slime.sh"
  "${SERVICE_AGENT_ROOT}/setup/setup_tau_bench.sh"
  # tau2-bench: deps first (hatchling/editables are needed for the --no-deps
  # editable build), then the editable install itself. Both are idempotent.
  "${SERVICE_AGENT_ROOT}/setup/setup_tau2_bench_deps.sh"
  "${SERVICE_AGENT_ROOT}/setup/setup_tau2_bench.sh"
)

REMOTE_ENV_SETUP_CMD=""
if [[ -f "${ENV_SETUP_SCRIPT}" ]]; then
  REMOTE_ENV_SETUP_CMD=". ${ENV_SETUP_SCRIPT} && "
fi

REMOTE_SETUP_CMD=""
for _script in "${SETUP_SCRIPTS[@]}"; do
  if [[ -f "${_script}" ]]; then
    REMOTE_SETUP_CMD+="bash ${_script} && "
  fi
done

# Each present setup script must succeed -- the `&&` chain aborts before the
# training script if any fails, so we never run a job against a wrong checkout.
# Setup + training output all land in RUN_LOG via the brace-group redirect.
# NOTE: keep the brace-group body POSIX-safe (the container may run this via
# /bin/sh); bash-isms belong in the setup_*.sh scripts, invoked via `bash`.
REMOTE_CMD="export PROJECT_ROOT=${PROJECT_ROOT} && export OUTPUT_ROOT=${REMOTE_OUTPUT_ROOT} && { cd ${PROJECT_ROOT} && ${REMOTE_ENV_SETUP_CMD}${REMOTE_SETUP_CMD}bash ${SCRIPT_COPY} ${SCRIPT_ARGS}; } > ${RUN_LOG} 2>&1"

if [[ "${SPOT}" == "1" ]]; then
  PRIORITY="normal"
fi

SUBMIT_CMD=(
  sco acp jobs create
  --workspace-name "${WORKSPACE_NAME}"
  --aec2-name "${AEC2_NAME}"
  --job-name "${JOB_NAME}"
  --container-image-url "${CONTAINER_IMAGE_URL}"
  --training-framework "${TRAINING_FRAMEWORK}"
  --worker-nodes "${WORKER_NODES}"
  --worker-spec "${WORKER_SPEC}"
  --priority "${PRIORITY}"
  --storage-mount "${STORAGE_MOUNT}"
  --command "${REMOTE_CMD}"
)

if [[ "${SPOT}" == "1" ]]; then
  SUBMIT_CMD+=(--quota-type spot)
fi

echo "[INFO] Job name      : ${JOB_NAME}"
echo "[INFO] Workspace     : ${WORKSPACE_NAME}"
echo "[INFO] Cluster       : ${AEC2_NAME}"
echo "[INFO] Worker spec   : ${WORKER_SPEC}"
echo "[INFO] Experiment    : ${EXPERIMENT:-<none>}"
echo "[INFO] Project root  : ${PROJECT_ROOT}"
echo "[INFO] Log directory : ${LOG_DIR}"
echo "[INFO] Submit log    : ${SUBMIT_LOG}"
echo "[INFO] Run log       : ${RUN_LOG}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] Submission command:"
  printf '%q ' "${SUBMIT_CMD[@]}"
  echo
  exit 0
fi

"${SUBMIT_CMD[@]}" | tee -a "${SUBMIT_LOG}"

echo "[INFO] Submitted. Track with:"
echo "watch -n 0.5 \"sco acp jobs list --workspace-name=${WORKSPACE_NAME} --page-size 500 | awk 'NR<=3 || /${JOB_NAME}/'\""

if [[ "${TAIL_LOG}" == "1" ]]; then
  echo "[INFO] Waiting for run log: ${RUN_LOG}"
  # Wait up to 300 seconds for the log file to appear
  _wait=0
  while [[ ! -s "${RUN_LOG}" ]] && [[ $_wait -lt 300 ]]; do
    sleep 2
    _wait=$((_wait + 2))
  done
  if [[ -s "${RUN_LOG}" ]]; then
    echo "[INFO] Tailing run log (Ctrl-C to stop)..."
    tail -f "${RUN_LOG}"
  else
    echo "[WARN] Run log did not appear within 300s: ${RUN_LOG}" >&2
    exit 1
  fi
fi
