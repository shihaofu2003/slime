#!/usr/bin/env bash
#
# Four-card smoke for tau2-bench GRPO: two cards for slime actor/rollout, one
# card for the local user simulator, and one spare card for memory headroom.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime}"
TAU2_RL_SMOKE_RUN_ID="${TAU2_RL_SMOKE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export TOTAL_GPUS="${TOTAL_GPUS:-4}"
export RAY_GPUS="${RAY_GPUS:-2}"
export AGENT_CUDA_VISIBLE_DEVICES="${AGENT_CUDA_VISIBLE_DEVICES:-0,1}"
export USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-3}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-2}"
export TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-30}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1024}"
export PREPARED_RL_DATA="${PREPARED_RL_DATA:-${PROJECT_ROOT}/output/datasets/tau2-bench-rl/areal_tau2_rl_train_smoke.jsonl}"
export TAU2_RL_TRAJECTORY_DUMP_PATH="${TAU2_RL_TRAJECTORY_DUMP_PATH:-${PROJECT_ROOT}/output/tau2-rl-trajectories/smoke_${TAU2_RL_SMOKE_RUN_ID}.jsonl}"
export SAVE_DIR="${SAVE_DIR:-/mnt/afs/users/fush/projects/ServiceAgent/checkpoints_smoke/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_smoke_${TAU2_RL_SMOKE_RUN_ID}}"
export USE_WANDB="${USE_WANDB:-0}"
export PREPARE_RL_LIMIT="${PREPARE_RL_LIMIT:-${TAU2_RL_SMOKE_DATA_LIMIT:-12}}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
