#!/usr/bin/env bash
#
# DIAGNOSTIC smoke for Qwen3.5-4B (multimodal, hybrid arch, thinking) on tau2-bench.
# 1 task/domain x 1 sample (3 tasks), thinking ON (the default), with room to
# think (max-new-tokens 4096) + presence_penalty. Purpose: confirm (a) the
# container's sglang can load qwen3_5, (b) the model emits a parseable
# <tool_call> block after <think> (JSON or qwen3_coder <function>/<parameter>),
# (c) parsing succeeds. Inspect the run log's
# "Continuing with action: AssistantMessage(content=...)" lines for the real
# output format.
#
#   bash scripts/submit.sh --experiment tau2-qwen35 --gpus 1 \
#       examples/tau2-bench/eval/legacy/smoke_qwen3.5-4b.sh

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
EXAMPLE_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/legacy"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B"
export OUTPUT="${EXAMPLE_DIR}/outputs/eval/Qwen3.5-4B/smoke.json"
# 1 task/domain x 1 sample; thinking ON (default). max-new-tokens 4096 gives the
# <think> room; presence_penalty 1.5 keeps it convergent. Agent temp 0.6 /
# user temp 0.0 come from eval.py's defaults.
export EVAL_ARGS="--max-tasks-per-domain 1 --num-samples 1 --max-new-tokens 4096 --top-p 0.95 --presence-penalty 1.5"

exec bash "${EXAMPLE_DIR}/run_eval.sh"
