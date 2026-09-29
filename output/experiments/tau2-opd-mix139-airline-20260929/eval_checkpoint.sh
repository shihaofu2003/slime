#!/usr/bin/env bash
# Convert and evaluate one checkpoint from the corrected mix139 OPD run.
# Usage: ITER=9 bash eval_checkpoint.sh [--dry-run]
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2}"
export SERVICE_AGENT_ROOT="$(dirname "${PROJECT_ROOT}")"
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-opd-mix139-airline-20260929"
checkpoint_root="${EXPERIMENT_DIR}/arms/async/20260929_mix139-airline-opd/checkpoints"
export ITER="${ITER:?Set ITER to the OPD checkpoint iteration}"
printf -v ITER_PAD '%07d' "${ITER}"
export ITER_DIR="${checkpoint_root}/iter_${ITER_PAD}"
export OUTPUT_DIR="${checkpoint_root}/iter_${ITER_PAD}_hf"
export ORIGIN_HF_DIR="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"

export MODEL_PATH="${OUTPUT_DIR}"
export MODEL_NAME="Qwen3-4B-opd-mix139-airline-iter${ITER}"
export EVAL_LABEL="opd-mix139-airline-iter${ITER}-four-domain"
export RUN_STAMP="20260929_opd_mix139_airline_iter${ITER}_seed300"
export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/trajectories/seed300_${RUN_STAMP}"
export SUMMARY_OUTPUT="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed300_${RUN_STAMP}_summary.json"
export AGENT_EVAL_MODE=official-native
export AGENT_TOOL_CALL_PARSER=qwen
export TP=1
export AGENT_REPLICA_CUDA_GROUPS='0;1'
export USER_SGLANG=1
export USER_REPLICA_CUDA_GROUPS='2,3;4,5;6,7'
export DOMAINS=airline,retail,telecom,banking_knowledge
export DOMAIN_CONCURRENCY=airline:1,retail:2,telecom:2,banking_knowledge:4
export GLOBAL_CONCURRENCY=9 BORROW_COMPLETED_DOMAIN_SLOTS=1
export RETRIEVAL_CONFIG=bm25
export NUM_TASKS="" AGENT_MAX_TOKENS=1200 NUM_TRIALS=4 SEED=300

if [[ "${1:-}" == --dry-run ]]; then
  python3 - <<'PY'
import json
import os
keys = ["ITER", "ITER_DIR", "MODEL_PATH", "SUMMARY_OUTPUT", "DOMAINS",
        "AGENT_REPLICA_CUDA_GROUPS", "USER_REPLICA_CUDA_GROUPS",
        "DOMAIN_CONCURRENCY", "GLOBAL_CONCURRENCY", "RETRIEVAL_CONFIG",
        "AGENT_EVAL_MODE", "AGENT_MAX_TOKENS", "NUM_TRIALS", "SEED"]
print(json.dumps({key: os.environ[key] for key in keys}, indent=2))
PY
  exit 0
fi

[[ -d "${ITER_DIR}" ]] || { echo "[ERROR] Missing training checkpoint: ${ITER_DIR}" >&2; exit 1; }
if [[ ! -f "${OUTPUT_DIR}/model.safetensors.index.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh"
fi

exec bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh" full
