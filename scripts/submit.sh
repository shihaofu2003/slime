#!/usr/bin/env bash
set -euo pipefail

# Project-aware job-manager submit helper for training/evaluation scripts.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit.sh [options] <script>
  bash scripts/submit.sh <script> [options]

Arguments:
  <script>                   Bash script path to run (positional or --script).

Options:
  -N, --name <name>          Job name. Default: derived from the script path.
  -g, --gpus <num>           GPUs per node. Default: 1
  -c, --cpus <num>           CPU cores per node (use together with --memory).
  -m, --memory <num>         Memory in GB per node (use together with --cpus).
  -n, --nodes <num>          Node count. Default: 1
  -v, --version <version>    Store output below <output_root>/v<version>.
  -e, --env <key=value>      Export an environment variable; repeat as needed.
  --image <url>              Container image URL.
  --conda-env <name>         Conda environment to activate in the job.
  --project-root <path>      Default: repo root
  --experiment <name>       Experiment name (groups output under output/experiments/<name>/jobs)
  --with-vitabench          Build and activate the isolated VitaBench environment
  --output-root <path>       Default: <project_root>/output or <project_root>/output/experiments/$EXPERIMENT
  -s, --spot                 Submit as spot (idle-time) job
  --                         Separator: all args after this are passed to the script
  -t, --track, --tail        Track the run log after submission
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

  # VitaBench job (tau2-compatible dependency overlay)
  bash scripts/submit.sh --with-vitabench examples/vita-bench/run_qwen3_5_4b_smoke.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT="${EXPERIMENT:-}"
JOB_PREFIX="${JOB_PREFIX:-fsh}"
JOB_NAME=""
GPU_NUMS="1"
WORKER_NODES="1"
CPU_NUMS=""
MEMORY_GB=""
VERSION=""
CONDA_ENV=""
CONTAINER_IMAGE_URL="registry.cn-sh-01.sensecore.cn/ccr-zhicheng-06/slimerl/slime"

OUTPUT_ROOT=""
LOG_DIR=""
SCRIPT=""
DRY_RUN="0"
SPOT="0"
TAIL_LOG="0"
WITH_VITABENCH="0"
SCRIPT_ARGS=()
JOB_ENVS=()


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
    echo "${PROJECT_ROOT}/output/experiments/${EXPERIMENT}"
  else
    echo "${PROJECT_ROOT}/output"
  fi
}


while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-name|--aec2-name|--worker-spec|--training-framework|--storage-mount)
      echo "[WARN] Ignoring legacy SCO option: $1" >&2
      shift 2 ;;
    --experiment)
      EXPERIMENT="$2"; shift 2 ;;
    -N|--name)
      JOB_NAME="$2"; shift 2 ;;
    -g|--gpus)
      GPU_NUMS="$2"; shift 2 ;;
    -c|--cpus)
      CPU_NUMS="$2"; shift 2 ;;
    -m|--memory)
      MEMORY_GB="$2"; shift 2 ;;
    -n|--nodes)
      WORKER_NODES="$2"; shift 2 ;;
    -v|--version)
      VERSION="$2"; shift 2 ;;
    -e|--env)
      JOB_ENVS+=("$2"); shift 2 ;;
    --conda-env)
      CONDA_ENV="$2"; shift 2 ;;
    --priority)
      if [[ "$2" != "normal" ]]; then
        echo "[WARN] job-manager uses normal priority; ignoring --priority $2" >&2
      fi
      shift 2 ;;
    --image)
      CONTAINER_IMAGE_URL="$2"; shift 2 ;;
    --project-root)
      PROJECT_ROOT="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    --script)
      SCRIPT="$2"; shift 2 ;;
    -s|--spot)
      SPOT="1"; shift 1 ;;
    --with-vitabench)
      WITH_VITABENCH="1"; shift 1 ;;
    --)
      shift 1; SCRIPT_ARGS=("$@"); break ;;
    -t|--track|--tail)
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

# Keep the original hint for job-manager; the prefixed name is only for the
# local script snapshot directory created by this compatibility helper.
JOB_NAME_HINT="${JOB_NAME}"
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

if [[ "${SCRIPT}" != *.sh ]]; then
  echo "[ERROR] job-manager only accepts Bash scripts (*.sh): ${SCRIPT}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="$(derive_output_root)"
fi

if [[ -n "${VERSION}" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT}/v${VERSION}"
fi

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${OUTPUT_ROOT}/jobs/${JOB_NAME}"
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUBMIT_LOG="${LOG_DIR}/submit_${TIMESTAMP}.log"

# Copy the script to output directory so that later edits to the original don't
# affect the submitted job.
SCRIPT_COPY="${LOG_DIR}/$(basename "${SCRIPT}")"
cp "${SCRIPT}" "${SCRIPT_COPY}"

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

# VitaBench has a broad dependency surface, so opt in per submission instead of
# adding install latency to every slime training job. Its setup runs after tau2
# so shared packages can enforce tau2's versions inside an isolated overlay.
VITABENCH_VENV="/tmp/serviceagent-vitabench-venv"
if [[ "${WITH_VITABENCH}" == "1" ]]; then
  SETUP_SCRIPTS+=("${SERVICE_AGENT_ROOT}/setup/setup_vitabench_env.sh")
fi

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

REMOTE_VITABENCH_ENV_CMD=""
if [[ "${WITH_VITABENCH}" == "1" ]]; then
  REMOTE_VITABENCH_ENV_CMD="export VIRTUAL_ENV=${VITABENCH_VENV} && export PATH=${VITABENCH_VENV}/bin:\$PATH && "
fi

# Each present setup script must succeed -- the `&&` chain aborts before the
# training script if any fails, so we never run a job against a wrong checkout.
# job-manager snapshots this Bash entrypoint and owns the final run log. The
# original task script is already copied above, so later edits cannot affect it.
REMOTE_CMD="{ cd ${PROJECT_ROOT} && ${REMOTE_ENV_SETUP_CMD}${REMOTE_SETUP_CMD}${REMOTE_VITABENCH_ENV_CMD}bash ${SCRIPT_COPY} \"\$@\"; }"
JOB_SCRIPT="${LOG_DIR}/job_entrypoint.sh"
printf '#!/usr/bin/env bash\nset -euo pipefail\n%s\n' "${REMOTE_CMD}" > "${JOB_SCRIPT}"
chmod +x "${JOB_SCRIPT}"

JOB_CLIENT="${JOB_CLIENT:-/mnt/afs/users/fush/job}"
if [[ ! -x "${JOB_CLIENT}" ]]; then
  echo "[ERROR] job-manager client not found: ${JOB_CLIENT}" >&2
  echo "Run: bash /mnt/afs/users/admin/job-manager/bin/job install -u fush" >&2
  exit 1
fi

SUBMIT_CMD=(
  "${JOB_CLIENT}" submit "${JOB_SCRIPT}"
  --gpus "${GPU_NUMS}"
  --nodes "${WORKER_NODES}"
  --name "${JOB_NAME_HINT}"
  --project-root "${PROJECT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --image "${CONTAINER_IMAGE_URL}"
)

if [[ -n "${CPU_NUMS}" ]]; then
  SUBMIT_CMD+=(--cpus "${CPU_NUMS}")
fi
if [[ -n "${MEMORY_GB}" ]]; then
  SUBMIT_CMD+=(--memory "${MEMORY_GB}")
fi
if [[ -n "${CONDA_ENV}" ]]; then
  SUBMIT_CMD+=(--conda-env "${CONDA_ENV}")
fi
for _env in "${JOB_ENVS[@]}"; do
  SUBMIT_CMD+=(--env "${_env}")
done
if [[ "${SPOT}" == "1" ]]; then
  SUBMIT_CMD+=(--spot)
fi
if [[ "${TAIL_LOG}" == "1" ]]; then
  SUBMIT_CMD+=(--track)
fi
if (( ${#SCRIPT_ARGS[@]} > 0 )); then
  SUBMIT_CMD+=(-- "${SCRIPT_ARGS[@]}")
fi

echo "[INFO] Job name hint : ${JOB_NAME_HINT}"
echo "[INFO] Experiment    : ${EXPERIMENT:-<none>}"
echo "[INFO] VitaBench env : ${WITH_VITABENCH}"
echo "[INFO] Project root  : ${PROJECT_ROOT}"
echo "[INFO] Output root   : ${OUTPUT_ROOT}"
echo "[INFO] Snapshot dir  : ${LOG_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] Submission command:"
  printf '%q ' "${SUBMIT_CMD[@]}"
  echo
  exit 0
fi

"${SUBMIT_CMD[@]}" | tee -a "${SUBMIT_LOG}"

echo "[INFO] Use 'job list' to view jobs and 'job show <job_id>' for log paths."
