#!/usr/bin/env bash
#
# Eight-card tau2-bench GRPO run for the AReaL RL split.
# Uses six cards for slime actor/rollout, one card for the local trained user
# simulator, and one spare card for memory headroom.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime}"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
TAU2_RL_RUN_ID="${TAU2_RL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export TOTAL_GPUS="${TOTAL_GPUS:-8}"
export RAY_GPUS="${RAY_GPUS:-6}"
export AGENT_CUDA_VISIBLE_DEVICES="${AGENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export USER_CUDA_VISIBLE_DEVICES="${USER_CUDA_VISIBLE_DEVICES:-7}"

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-6}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-48}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-330}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-33}"
export USE_DYNAMIC_FILTER="${USE_DYNAMIC_FILTER:-1}"

export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export TAU2_USER_TEMPERATURE="${TAU2_USER_TEMPERATURE:-0.0}"
export TAU2_USER_MAX_TOKENS="${TAU2_USER_MAX_TOKENS:-512}"
export TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-60}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1024}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-8}"
# Per-trajectory token cap; also drives max_tokens_per_gpu (see base script).
# 16384 keeps ~99% of tau2 trajectories fully untruncated (no leading-message
# truncation) while guaranteeing the log-prob forward fits 80GB at TP=2.
export TAU2_RL_MAX_TRAIN_TOKENS="${TAU2_RL_MAX_TRAIN_TOKENS:-16384}"

# Hyperparameters aligned to the AReaL tau2 reference recipe
# (examples/tau2/config_1.7b_airline.yaml). The prior defaults (lr=1e-6,
# temperature=0.65, eps_clip=0.2) produced a FLAT reward curve: lr was 17x lower
# than AReaL's 1.7e-5 and the rollout temperature was too low for exploration.
export LR="${LR:-1e-5}"
export EPS_CLIP="${EPS_CLIP:-0.4}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.4}"
export KL_COEF="${KL_COEF:-0}"

# Keep the KL anchor to the SFT policy. The entropy bonus is disabled because
# entropy_coef=0.001 dominated the policy-gradient signal in the first shaped-
# reward airline run: entropy rose from 0.34 to 3.03 and outputs degenerated.
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
export ENTROPY_COEF="${ENTROPY_COEF:-0}"

# Dense reward shaping is on by default in the base script
# (USE_REWARD_SHAPING=1, reward_postprocess.tau2_reward_post_process). The
# shaping weights/alpha default to the cookbook values inside the module; override
# via TAU2_REWARD_ALPHA / TAU2_PARTIAL_*_WEIGHT if needed (export them and add to
# the Ray runtime env to reach workers).

export USER_MODEL_PATH="${USER_MODEL_PATH:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf}"
export USER_MODEL="${USER_MODEL:-Qwen3-4B-tau2-user-sft-stop-iter0006899}"
export USER_SGLANG_LOG="${USER_SGLANG_LOG:-${PROJECT_ROOT}/output/tau2-rl-user-sglang-${TAU2_RL_RUN_ID}.log}"

export SAVE_DIR="${SAVE_DIR:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_${TAU2_RL_RUN_ID}}"
export TAU2_RL_TRAJECTORY_DUMP_PATH="${TAU2_RL_TRAJECTORY_DUMP_PATH:-${PROJECT_ROOT}/output/tau2-rl-trajectories/real8g_${TAU2_RL_RUN_ID}.jsonl}"

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-dev}"
export WANDB_GROUP="${WANDB_GROUP:-tau2-agent-rl-grpo-real8g-${TAU2_RL_RUN_ID}}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
