#!/bin/bash
#
# Real (non-smoke) tau-bench GRPO run to verify the training code trains the
# model across multiple steps. Real-ish config: global-batch 32 (8 prompts x
# 4 samples -> 8 GRPO groups), kl_coef=0 (rollout_log_probs fix), num-rollout 3
# (three gradient steps so the rollout->advantage->backward->checkpoint cycle is
# exercised repeatedly, not just once like the smoke). Eval skipped to focus on
# the training loop.
#
# Fresh CHECKPOINT_ROOT so it NEVER resumes from a stale smoke iter-0 and does
# NOT touch the canonical checkpoints/ dir.

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/checkpoints_real}"

export ROLLOUT_BATCH_SIZE=8        # 8 prompts (8 GRPO groups)
export N_SAMPLES_PER_PROMPT=4     # 4 samples each -> global-batch 32
export GLOBAL_BATCH_SIZE=32
export NUM_ROLLOUT=3              # 3 gradient steps
# Reward-variance filter is OFF by default in the main script now (cold model ->
# ~90% group rejection -> very slow). Leave it off; set USE_DYNAMIC_FILTER=1
# here once the model is warm enough to produce per-group reward variance.
export SKIP_EVAL=1                # focus on the training loop
export KL_COEF=0                  # rollout_log_probs fix (no reward-shaping KL)

exec bash "${PROJECT_ROOT}/scripts/run-qwen3-4B-serviceagent-tau.sh"
