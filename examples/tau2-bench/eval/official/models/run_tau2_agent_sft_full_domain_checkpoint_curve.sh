#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <iteration> [<iteration> ...]" >&2
  exit 2
fi

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-sft-full-domain-eval}"
FOUR_DOMAIN_WRAPPER="${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh"

for iteration in "$@"; do
  if [[ ! "${iteration}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] iteration must be numeric: ${iteration}" >&2
    exit 2
  fi

  printf -v iteration_padded '%07d' "$((10#${iteration}))"
  hf_dir="${CHECKPOINT_ROOT}/iter_${iteration_padded}_hf"
  if [[ ! -f "${hf_dir}/config.json" ]]; then
    echo "[ERROR] missing converted checkpoint: ${hf_dir}" >&2
    exit 1
  fi

  MODEL_PATH="${hf_dir}" \
  MODEL_NAME="tau2-sft-full-domain-iter${iteration_padded}" \
  EXPERIMENT_DIR="${EXPERIMENT_DIR}" \
  EVAL_LABEL="iter${iteration_padded}-four-domain-full-bm25" \
  RETRIEVAL_CONFIG="bm25" \
  SEED="300" \
    bash "${FOUR_DOMAIN_WRAPPER}" full
done
