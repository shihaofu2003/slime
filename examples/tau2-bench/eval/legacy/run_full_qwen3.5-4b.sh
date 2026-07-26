#!/usr/bin/env bash
#
# Full tau2-bench eval (Pass@4) for Qwen3.5-4B — a multimodal (vision+text),
# hybrid-arch, THINKING model (Qwen3_5ForConditionalGeneration).
#
# Text-only: tau2-bench sends no images, so sglang serves the model with the
# vision encoder idle (verified: loads as Qwen3_5ForConditionalGeneration,
# 8.62 GB weights, text /generate works). No special flag needed (the container
# sglang already supports qwen3_5 + linear-attn/mamba). The eval drives the raw
# /generate endpoint with a client-side chat template + client-side action
# parsing, so sglang does NOT need --reasoning-parser/--tool-call-parser here
# (those only affect the /v1/chat/completions endpoint).
#
# Thinking is ON (the default — Qwen3.5 thinks, per its README). The earlier run
# disabled it because at max_new_tokens=1200 the <think> ran out of tokens
# before emitting a tool call (~40% parse_error). The fix is room to think, not
# disabling it: --max-new-tokens 8192 + --presence-penalty 1.5 (README thinking
# best practice, suppresses non-convergent repetition). parse_action
# (actions.py) ignores <think>/prose around the single <tool_call> block and
# accepts both the Qwen3 JSON form and the Qwen3.5 qwen3_coder
# <function>/<parameter> form.
#
# Agent temperature defaults to 0.6 (eval.py); user-sim temperature 0.0.
#
#   bash scripts/submit.sh --experiment tau2-eval --gpus 1 \
#       examples/tau2-bench/eval/legacy/run_full_qwen3.5-4b.sh

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
EXAMPLE_DIR="${SERVICE_AGENT_ROOT}/slime/examples/tau2-bench/eval/legacy"

export MODEL_PATH="${SERVICE_AGENT_ROOT}/models/Qwen3.5-4B"
export OUTPUT="${EXAMPLE_DIR}/outputs/eval/Qwen3.5-4B/pass4.json"
# Pass@4; thinking ON (default). Room to think (8192) + presence_penalty to keep
# <think> convergent; top_p 0.95 per the README's thinking recommendation.
# Agent temp 0.6 / user temp 0.0 come from eval.py's defaults.
export EVAL_ARGS="--num-samples 4 --max-new-tokens 8192 --top-p 0.95 --presence-penalty 1.5"

exec bash "${EXAMPLE_DIR}/run_eval.sh"
