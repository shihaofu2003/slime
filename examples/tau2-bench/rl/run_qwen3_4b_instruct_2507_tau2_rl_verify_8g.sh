#!/usr/bin/env bash
#
# Eight-card tau2-bench GRPO *verification* run: full AReaL RL data (so the
# long-trajectory tail is sampled) but a small number of rollouts, to confirm
# the memory fix and the reward/train pipeline end-to-end without paying for a
# full 330-rollout run. Reaches the first training step (the previous OOM site)
# in one rollout batch.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime}"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
TAU2_RL_RUN_ID="${TAU2_RL_RUN_ID:-verify8g_$(date +%Y%m%d_%H%M%S)}"

export TOTAL_GPUS="${TOTAL_GPUS:-8}"
export RAY_GPUS="${RAY_GPUS:-6}"
export AGENT_CUDA_VISIBLE_DEVICES="${AGENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-7}"

# Smaller per-rollout batch (still divisible by DP=3) for a faster first train
# step, but full data so long-tail prompts are still drawn.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-3}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
export USE_DYNAMIC_FILTER="${USE_DYNAMIC_FILTER:-1}"

export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.65}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export TAU2_USER_TEMPERATURE="${TAU2_USER_TEMPERATURE:-0.0}"
export TAU2_USER_MAX_TOKENS="${TAU2_USER_MAX_TOKENS:-512}"
export TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-60}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1024}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-8}"
export TAU2_RL_MAX_TRAIN_TOKENS="${TAU2_RL_MAX_TRAIN_TOKENS:-16384}"

export LR="${LR:-1e-6}"
export KL_COEF="${KL_COEF:-0}"

export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"

export SAVE_DIR="${SAVE_DIR:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_verify8g_${TAU2_RL_RUN_ID}}"
export TAU2_RL_TRAJECTORY_DUMP_PATH="${TAU2_RL_TRAJECTORY_DUMP_PATH:-${PROJECT_ROOT}/output/tau2-rl-trajectories/${TAU2_RL_RUN_ID}.jsonl}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
export WANDB_GROUP="${WANDB_GROUP:-tau2-agent-rl-grpo-verify8g-${TAU2_RL_RUN_ID}}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
