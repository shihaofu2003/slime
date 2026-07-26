#!/usr/bin/env bash
#
# Full tau2-bench eval (Pass@4) for the base model:
#   models/Qwen3-4B-Instruct-2507   (the GRPO checkpoint's starting point)
#
# All 3 domains, full test split, num-samples=4, user sim = gemini-2.5-flash
# (the proxy default set in run_eval.sh). Thin wrapper: it only fixes the model
# path + per-model output dir + the full-run EVAL_ARGS, then execs run_eval.sh.
# (submit.sh forwards no env to the container, so the config must live here.)
#
#   bash scripts/submit.sh --experiment tau2-eval --gpus 1 \
#       examples/tau2-bench/eval/legacy/run_full_qwen3-4b-instruct-2507.sh

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
EXAMPLE_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/legacy"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507"
export OUTPUT="${EXAMPLE_DIR}/outputs/eval/Qwen3-4B-Instruct-2507/pass4.json"
# Full run: drop the smoke --max-tasks-per-domain cap; num-samples=4 (Pass@4).
# This is an inherently NON-thinking model (no <think> in its chat template), so
# no --disable-thinking is needed — thinking is off regardless. Sampling params
# (temp/top_p/top_k/max_steps) fall back to eval.py's defaults
# (0.6 / 1.0 / 20 / 100); agent temp 0.6, user-sim temp 0.0.
export EVAL_ARGS="--num-samples 4"

exec bash "${EXAMPLE_DIR}/run_eval.sh"
