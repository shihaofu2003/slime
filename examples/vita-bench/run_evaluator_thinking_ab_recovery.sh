#!/usr/bin/env bash

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"

export VITA_REPLAY_RUN_MODE="recovery"
export VITA_REPLAY_INSTANCE_COUNT="1"
export VITA_REPLAY_MAX_RUNNING_REQUESTS="1"
export VITA_REPLAY_MAX_SEMANTIC_ATTEMPTS="4"
export VITA_REPLAY_BASE_ARTIFACT_DIR="${PROJECT_ROOT}/output/experiments/vitabench-evaluator-thinking-ab/artifacts"
export VITA_REPLAY_ARTIFACT_DIR="${PROJECT_ROOT}/output/experiments/vitabench-evaluator-thinking-ab/artifacts-recovery-v2"
export VITA_REPLAY_RECOVERY_REASON="Primary job pt-38ktzko8 exhausted three semantic attempts at instore:10826120:0:4. Recovery job pt-h25ofbrb reproduced three length-truncated attempts and was stopped before context overflow. This supplement uses stateless compact corrections and leaves windows 4-5 as an explicitly non-primary sensitivity result."

exec bash "${PROJECT_ROOT}/examples/vita-bench/run_evaluator_thinking_ab.sh"
