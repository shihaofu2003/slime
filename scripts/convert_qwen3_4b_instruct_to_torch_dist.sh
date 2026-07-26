#!/usr/bin/env bash
#
# Convert Qwen3-4B-Instruct-2507 (HF safetensors) -> Megatron-LM torch_dist
# checkpoint, the format slime loads for training.
#
# Reuses scripts/convert_qwen3_4b_to_torch_dist.sh via env-var override: same
# architecture as base Qwen3-4B, so the qwen3-4B.sh MODEL_ARGS (layers/heads/GQA
# ...) map the weights correctly. The only difference between the two models is
# the rotary base (5e6 vs 1e6), which is a runtime RoPE config -- not part of the
# stored weights -- so it has no effect on weight conversion. The training run
# sources qwen3-4B-Instruct-2507.sh (rotary base 5e6) at load time.

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507}"
export SAVE_DIR="${SAVE_DIR:-${SERVICE_AGENT_ROOT}/models/Qwen3-4B-Instruct-2507_torch_dist}"

# submit.sh copies only THIS submitted script into the log dir, so the sibling
# base convert script is NOT next to us at runtime -- resolve it via PROJECT_ROOT
# (exported by submit.sh) and fall back to BASH_SOURCE for direct runs.
if [[ -n "${PROJECT_ROOT:-}" ]]; then
    BASE_CONV="${PROJECT_ROOT}/scripts/convert_qwen3_4b_to_torch_dist.sh"
else
    BASE_CONV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/convert_qwen3_4b_to_torch_dist.sh"
fi

if [[ ! -f "${BASE_CONV}" ]]; then
    echo "[ERROR] Base convert script not found: ${BASE_CONV}" >&2
    exit 1
fi

echo "[INFO] Converting ${HF_CHECKPOINT} -> ${SAVE_DIR}"
exec bash "${BASE_CONV}" "$@"
