#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 || ! "$1" =~ ^(sync|async)$ || ! "$2" =~ ^(300|301)$ ]]; then
  echo "Usage: $0 <sync|async> <300|301>" >&2
  exit 2
fi
ARM="$1"
ITERATION="${TAU2_ITERATION:-99}"
SEED_VALUE="$2"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-areal-async-rl"
RUN_ROOT="${EXPERIMENT_DIR}/arms/${ARM}/${TAU2_RUN_NAME:-ready-train100}/checkpoints"
printf -v iteration_padded '%07d' "${ITERATION}"
MODEL_PATH="${RUN_ROOT}/iter_${iteration_padded}_hf${TAU2_HF_SUFFIX:-}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Hugging Face checkpoint is not ready: ${MODEL_PATH}" >&2
  exit 1
fi

export TAU2_SRC="${EXPERIMENT_DIR}/dependencies/tau2/src"
export TAU2_DATA_DIR="$(dirname "${PROJECT_ROOT}")/tau2-bench/data"
export EXPERIMENT_DIR
export EVAL_LABEL="${ARM}-iter${ITERATION}${TAU2_EVAL_SUFFIX:-}"
export SAVE_PREFIX="${EXPERIMENT_DIR}/eval/${EVAL_LABEL}/seed${SEED_VALUE}/trajectories"
export MODEL_PATH
export MODEL_NAME="Qwen3-4B-Instruct-2507-${ARM}-vanilla-grpo-iter${ITERATION}"
export AGENT_SERVED_MODEL_NAME="${MODEL_NAME}"
export AGENT_EVAL_MODE=official-native
export AGENT_TOOL_CALL_PARSER=qwen
export DOMAINS="${TAU2_EVAL_DOMAINS:-airline,retail,telecom}"
export DOMAIN_CONCURRENCY="${TAU2_EVAL_DOMAIN_CONCURRENCY:-airline:2,retail:2,telecom:5}"
export TASK_SPLIT=test
export NUM_TASKS=""
export NUM_TRIALS=4
export SEED="${SEED_VALUE}"
export MAX_STEPS=200
export AGENT_TEMPERATURE=0.6
export AGENT_TOP_P=1.0
export AGENT_MAX_TOKENS=1200

exec python3 - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv

root = Path(os.environ["PROJECT_ROOT"])
# The isolated Tau2 source cannot discover the shared official Judge credentials.
load_dotenv(root.parent / "tau2-bench" / ".env", override=False)
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Official NL-assertion Judge requires OPENAI_API_KEY")
script = root / "examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
os.execvp("bash", ["bash", str(script)])
PY
