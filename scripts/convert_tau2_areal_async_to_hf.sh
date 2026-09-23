#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^(sync|async)$ ]]; then
  echo "Usage: $0 <sync|async> [conversion options]" >&2
  exit 2
fi
ARM="$1"
shift
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl/arms/${ARM}/${TAU2_RUN_NAME:-ready-train100}/checkpoints"
printf -v iteration_padded '%07d' "${TAU2_ITERATION:-99}"
export ITER_DIR="${RUN_ROOT}/iter_${iteration_padded}"
export OUTPUT_DIR="${RUN_ROOT}/iter_${iteration_padded}_hf${TAU2_HF_SUFFIX:-}"
export ORIGIN_HF_DIR="$(dirname "${PROJECT_ROOT}")/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903/final_hf"
exec bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh" "$@"
