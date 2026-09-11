#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || ! "$1" =~ ^(raw-all|processed)$ || ! "$2" =~ ^(9|39|69|99)$ ]]; then
  echo "Usage: $0 <raw-all|processed> <9|39|69|99> [--force|--dry-run-only]" >&2
  exit 2
fi

ARM="$1"
ITERATION="$2"
shift 2

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_BASE="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"

case "${ARM}" in
  raw-all)
    RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_rl_raw_all_vanilla_grpo_qwen36_20260903"
    ORIGIN_HF_DIR="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903/final_hf"
    ;;
  processed)
    RUN_ROOT="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_rl_official_native_expanded_vanilla_grpo_qwen36_20260903"
    ORIGIN_HF_DIR="${CHECKPOINT_BASE}/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903/final_hf"
    ;;
esac

latest_file="${RUN_ROOT}/latest_checkpointed_iteration.txt"
if [[ ! -f "${latest_file}" ]]; then
  echo "[ERROR] missing checkpoint marker: ${latest_file}" >&2
  exit 1
fi
latest="$(tr -d '[:space:]' < "${latest_file}")"
if [[ ! "${latest}" =~ ^[0-9]+$ || "${latest}" -lt "${ITERATION}" ]]; then
  echo "[ERROR] iteration ${ITERATION} is not complete; latest=${latest}" >&2
  exit 1
fi

printf -v iteration_padded '%07d' "${ITERATION}"
export PROJECT_ROOT
export ITER_DIR="${RUN_ROOT}/iter_${iteration_padded}"
export OUTPUT_DIR="${RUN_ROOT}/iter_${iteration_padded}_hf"
export ORIGIN_HF_DIR

exec bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
