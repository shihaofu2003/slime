#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130"
BASE_EVAL="${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_full_tau2_agent_single_call_v1_user_stop_parser.sh"
ARM="${1:-}"
ITERATION="${2:-}"
SEED="${3:-}"

if [[ $# -ne 3 || ! "${ARM}" =~ ^(mixed|airline|retail|telecom)$ || ! "${ITERATION}" =~ ^(109|119|129)$ || ! "${SEED}" =~ ^(300|301)$ ]]; then
  echo "Usage: $0 <mixed|airline|retail|telecom> <109|119|129> <300|301>" >&2
  exit 2
fi

printf -v ITER_PADDED '%07d' "${ITERATION}"
RL_CKPT_ROOT="${TAU2_RL_HF_ROOT:-${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_from_iter99_domain_${ARM}_20260816}"
export MODEL_PATH="${RL_CKPT_ROOT}/iter_${ITER_PADDED}_hf"
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] missing converted checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi

export MODEL_NAME="Qwen3-4B-tau2-single-call-v1-raw-user-maxsteps120-${ARM}-iter${ITERATION}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${ARM}/iter${ITERATION}/seed${SEED}_summary.json"
export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-Instruct-2507}"
export DOMAINS="airline,retail,telecom"
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0

exec bash "${BASE_EVAL}" candidate "${SEED}"
