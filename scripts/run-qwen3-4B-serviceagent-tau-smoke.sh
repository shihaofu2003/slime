#!/bin/bash
#
# Minimal-cost smoke to verify the rollout_log_probs fix reaches a GRPO gradient
# step with kl_coef=0 (no reward-shaping KL workaround). 4 prompts x 2 samples =
# 8 trajectories, 1 rollout step, no reward-variance filter, no eval, on 8 GPUs.
# The point is to confirm compute_advantages_and_returns no longer crashes
# (loss.py:700-703) now that the custom tau rollout returns Sample.rollout_log_probs
# (trainable_agents.asolve -> generate_with_tau.res_to_sample) -- NOT to train
# meaningfully.
#
# Sets env vars consumed by run-qwen3-4B-serviceagent-tau.sh, then execs it.

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/slime}"

# Use a throwaway checkpoint root so the smoke NEVER resumes from a prior run's
# saved iteration (which would skip the rollout entirely and make the smoke
# meaningless). --ref-load (the torch_dist model) is absolute, so weights still
# load correctly; only --load/--save move under this fresh dir.
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent/checkpoints_smoke}"

export ROLLOUT_BATCH_SIZE=4        # 4 prompts (shrunk from reference 32)
export N_SAMPLES_PER_PROMPT=2      # 2 samples each -> 8 trajectories
export GLOBAL_BATCH_SIZE=8         # 8 GPUs / TP2 = 4 DP -> 2 samples/rank (safe)
export NUM_ROLLOUT=1               # one gradient step is enough to verify
export USE_DYNAMIC_FILTER=0        # keep all trajectories regardless of reward
export SKIP_EVAL=1                 # skip the ~13-min eval
export KL_COEF=0                   # the fix: rollout now returns per-token logprobs,
                                   # so loss.py:702 finds a non-None rollout_log_probs
                                   # and kl_coef can stay 0 (no reward-shaping KL).

exec bash "${PROJECT_ROOT}/scripts/run-qwen3-4B-serviceagent-tau.sh"
