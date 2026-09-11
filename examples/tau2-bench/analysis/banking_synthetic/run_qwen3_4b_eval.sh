#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
PIPELINE="${PROJECT_ROOT}/examples/tau2-bench/analysis/banking_synthetic"
OFFICIAL_DIR="${PROJECT_ROOT}/examples/tau2-bench/eval/official"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-expert-sft}"
SYNTHETIC_EXPERIMENT_DIR="${SYNTHETIC_EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/tau2-banking-independent-synthetic-v1}"
EVAL_SOURCE_DIR="${EVAL_SOURCE_DIR:-${SYNTHETIC_EXPERIMENT_DIR}/final-minimal/eval-set}"
ALLOWLIST_PATH="${ALLOWLIST_PATH:-${SYNTHETIC_EXPERIMENT_DIR}/allowed_documents.json}"
BANKING_DATA="${BANKING_DATA:-${SERVICE_AGENT_ROOT}/tau2-bench/data/tau2/domains/banking_knowledge}"

MODE="${1:-scale}"
RETRIEVAL_VARIANT="${2:-bm25}"
EVAL_SEED="${3:-300}"
TASK_LIMIT="${4:-}"
EVAL_SPLIT="${5:-${EVAL_SPLIT:-}}"

case "${MODE}" in
  smoke|scale) ;;
  *)
    echo "Expected smoke or scale, got: ${MODE}" >&2
    exit 2
    ;;
esac
if [[ "${RETRIEVAL_VARIANT}" != "bm25" && "${RETRIEVAL_VARIANT}" != "golden_retrieval" ]]; then
  echo "Expected bm25 or golden_retrieval, got: ${RETRIEVAL_VARIANT}" >&2
  exit 2
fi
if [[ -n "${EVAL_SPLIT}" && "${EVAL_SPLIT}" != "train" && "${EVAL_SPLIT}" != "dev" && "${EVAL_SPLIT}" != "challenge" ]]; then
  echo "Expected split train, dev, challenge, or empty, got: ${EVAL_SPLIT}" >&2
  exit 2
fi
if [[ "${MODE}" == "smoke" && -z "${TASK_LIMIT}" ]]; then
  TASK_LIMIT="3"
fi

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the Agent HF checkpoint}"
 # Keep the Banking SFT comparison on the established Tau2 protocol: the
 # trainable Agent is Qwen3-4B, while the fixed text-only User simulator is
 # local Qwen3.6-27B.  Callers can still override these variables explicitly.
USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/models/Qwen3.6-27B}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")-banking-agent}"
USER_MODEL="${USER_MODEL:-Qwen3.6-27B-tau2-user-nonthinking}"
NUM_TRIALS="${NUM_TRIALS:-1}"
EVAL_CONCURRENCY="${MAX_CONCURRENCY:-1}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1200}"
AGENT_CONTEXT_LENGTH="${AGENT_CONTEXT_LENGTH:-131072}"
USER_CONTEXT_LENGTH="${USER_CONTEXT_LENGTH:-65536}"

for path in \
  "${EVAL_SOURCE_DIR}/eval_tasks.json" \
  "${EVAL_SOURCE_DIR}/eval_contracts.jsonl" \
  "${ALLOWLIST_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required input is missing: ${path}" >&2
    exit 1
  fi
done
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Agent HF checkpoint is missing config.json: ${MODEL_PATH}" >&2
  exit 1
fi

RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
MODEL_LABEL="$(basename "${MODEL_PATH}")"
EVAL_LABEL="${MODEL_LABEL}-${MODE}-${RETRIEVAL_VARIANT}"
if [[ -n "${EVAL_SPLIT}" ]]; then
  EVAL_LABEL="${MODEL_LABEL}-${MODE}-${EVAL_SPLIT}-${RETRIEVAL_VARIANT}"
fi
RUN_DIR="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed${EVAL_SEED}_${RUN_STAMP}"
export TAU2_DATA_DIR="${RUN_DIR}/tau2_data"
mkdir -p "${RUN_DIR}"

PREPARE_ARGS=(
  --tasks "${EVAL_SOURCE_DIR}/eval_tasks.json"
  --contracts "${EVAL_SOURCE_DIR}/eval_contracts.jsonl"
  --allowlist "${ALLOWLIST_PATH}"
  --banking-data "${BANKING_DATA}"
  --data-root "${TAU2_DATA_DIR}"
  --retrieval-variant "${RETRIEVAL_VARIANT}"
)
if [[ -n "${EVAL_SPLIT}" ]]; then
  PREPARE_ARGS+=(--split "${EVAL_SPLIT}")
fi
if [[ -n "${TASK_LIMIT}" ]]; then
  PREPARE_ARGS+=(--limit "${TASK_LIMIT}")
fi
python3 "${PIPELINE}/prepare_eval.py" "${PREPARE_ARGS[@]}" | tee "${RUN_DIR}/prepare.log"

export MODEL_PATH MODEL_NAME
export AGENT_SERVED_MODEL_NAME="${MODEL_NAME}"
export AGENT_EVAL_MODE="official-native"
export TP="${TP:-1}"
export MEM_FRACTION="${MEM_FRACTION:-0.85}"
export AGENT_CUDA_VISIBLE_DEVICES="${AGENT_CUDA_VISIBLE_DEVICES:-0}"
export AGENT_TOOL_CALL_PARSER="${AGENT_TOOL_CALL_PARSER:-qwen}"
export AGENT_SGLANG_EXTRA_ARGS="${AGENT_SGLANG_EXTRA_ARGS:---dtype bfloat16 --context-length ${AGENT_CONTEXT_LENGTH} --max-running-requests 2 --tool-call-parser ${AGENT_TOOL_CALL_PARSER}}"
export AGENT_EXTRA_BODY_JSON="${AGENT_EXTRA_BODY_JSON:-{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
export AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.6}"
export AGENT_TOP_P="${AGENT_TOP_P:-1.0}"
export AGENT_MAX_TOKENS

export USER_MODEL_PATH USER_MODEL
export USER_SGLANG="1"
export USER_TP="${USER_TP:-2}"
export USER_MEM_FRACTION="${USER_MEM_FRACTION:-0.90}"
export USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-1,2}"
export USER_SGLANG_EXTRA_ARGS="${USER_SGLANG_EXTRA_ARGS:---dtype bfloat16 --context-length ${USER_CONTEXT_LENGTH} --language-only --max-running-requests 2 --reasoning-parser qwen3 --tool-call-parser qwen3_coder}"
export USER_EXTRA_BODY_JSON="${USER_EXTRA_BODY_JSON:-{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
export USER_TEMPERATURE="${USER_TEMPERATURE:-0.0}"
export USER_TOP_P="${USER_TOP_P:-1.0}"
export USER_MAX_TOKENS="${USER_MAX_TOKENS:-512}"

export AGENT_ROUTER_POLICY="${AGENT_ROUTER_POLICY:-round_robin}"
export USER_ROUTER_POLICY="${USER_ROUTER_POLICY:-round_robin}"
export DOMAINS="banking_knowledge"
export TASK_SPLIT="generated"
export NUM_TASKS=""
export NUM_TRIALS
export MAX_STEPS="${MAX_STEPS:-200}"
export MAX_ERRORS="${MAX_ERRORS:-10}"
export MAX_RETRIES="${MAX_RETRIES:-0}"
export MAX_CONCURRENCY="${EVAL_CONCURRENCY}"
export DOMAIN_CONCURRENCY="banking_knowledge:${EVAL_CONCURRENCY}"
export GLOBAL_CONCURRENCY="${EVAL_CONCURRENCY}"
export PARALLEL_DOMAINS="0"
export RETRIEVAL_CONFIG="${RETRIEVAL_VARIANT}"
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE="${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-0}"
export SEED="${EVAL_SEED}"
export RUN_STAMP
export EVAL_SPLIT
export SAVE_PREFIX="tau2_${MODEL_LABEL}_banking_${MODE}_${EVAL_SPLIT:-all}_${RETRIEVAL_VARIANT}_seed${EVAL_SEED}_${RUN_STAMP}"
export SUMMARY_OUTPUT="${RUN_DIR}/official_summary.json"
export NAMESPACE_PROBE_OUTPUT=""

python3 - "${RUN_DIR}/runtime_models.json" <<'PY'
import json
import os
import sys

json.dump(
    {
        "agent_model_path": os.environ["MODEL_PATH"],
        "user_model_path": os.environ["USER_MODEL_PATH"],
        "agent_served_model": os.environ["AGENT_SERVED_MODEL_NAME"],
        "user_served_model": os.environ["USER_MODEL"],
        "num_trials": int(os.environ["NUM_TRIALS"]),
        "retrieval_variant": os.environ["RETRIEVAL_CONFIG"],
        "split": os.environ.get("EVAL_SPLIT") or None,
    },
    open(sys.argv[1], "w", encoding="utf-8"),
    indent=2,
)
PY

echo "[banking-synthetic-qwen3-4b] model=${MODEL_PATH} user=${USER_MODEL_PATH} mode=${MODE} split=${EVAL_SPLIT:-all} retrieval=${RETRIEVAL_VARIANT} seed=${EVAL_SEED} tasks=${TASK_LIMIT:-all}"
set +e
bash "${OFFICIAL_DIR}/run_eval.sh"
EVAL_STATUS=$?
set -e

if [[ "${EVAL_STATUS}" -ne 0 ]]; then
  echo "BANKING_SYNTHETIC_QWEN3_4B_EVAL_FAILED run_dir=${RUN_DIR} exit=${EVAL_STATUS}" >&2
  exit "${EVAL_STATUS}"
fi

python3 - "${RUN_DIR}/official_summary.json" "${RUN_DIR}/metrics.txt" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
overall = summary.get("overall", {})
diagnostics = overall.get("diagnostics", {})
lines = [
    f"tasks={overall.get('tasks')}",
    f"simulations={overall.get('simulations')}",
    f"pass_at_1={overall.get('pass_at_1')}",
    f"pass_at_4_any={overall.get('pass_at_4_any')}",
    f"pass_power_4={overall.get('pass_power_4')}",
    f"action_accuracy={diagnostics.get('action_accuracy')}",
    f"db_accuracy={diagnostics.get('db_accuracy')}",
    f"max_steps={diagnostics.get('max_steps')}",
    f"infrastructure_errors={diagnostics.get('infrastructure_errors')}",
]
open(sys.argv[2], "w", encoding="utf-8").write(" ".join(lines) + "\n")
print(" ".join(lines))
PY

echo "BANKING_SYNTHETIC_QWEN3_4B_EVAL_OK run_dir=${RUN_DIR}"
