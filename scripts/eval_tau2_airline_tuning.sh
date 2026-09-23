#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <sft|run-name> [iteration=29]" >&2
  exit 2
fi
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_AGENT_ROOT="$(dirname "${PROJECT_ROOT}")"
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-airline-db-count-tuning"
SFT_HF="${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909/iter_0004673_hf"
if [[ "$1" == sft ]]; then
  export MODEL_PATH="${SFT_HF}"
  export EVAL_LABEL=sft-iter4673
else
  iteration="${2:-29}"
  printf -v padded '%07d' "${iteration}"
  checkpoint_root="${EXPERIMENT_DIR}/arms/async/$1/checkpoints"
  export MODEL_PATH="${checkpoint_root}/iter_${padded}_hf"
  export EVAL_LABEL="$1-iter${iteration}"
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    ITER_DIR="${checkpoint_root}/iter_${padded}" OUTPUT_DIR="${MODEL_PATH}" \
      ORIGIN_HF_DIR="${SFT_HF}" bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh"
  fi
fi
export MODEL_NAME="Qwen3-4B-airline-${EVAL_LABEL}"
# Official evaluation deploys its own User replicas within the eight-GPU job.
export USER_SGLANG=1
export DOMAINS=airline
export DOMAIN_CONCURRENCY=airline:9
export BORROW_COMPLETED_DOMAIN_SLOTS=0
export NUM_TASKS=""
export NUM_TRIALS=4
export SEED=300
export AUTO_RESUME=1
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/trajectories/seed300_${RUN_STAMP}"
mkdir -p "${EXPERIMENT_DIR}/eval/${EVAL_LABEL}"
bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh" \
  2>&1 | tee -a "${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/run_${RUN_STAMP}.log"
