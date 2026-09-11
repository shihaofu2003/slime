#!/usr/bin/env bash
#
# Eight-card tau2-bench GRPO on AIRLINE-only data.
#
# Why airline-only: the full 3-domain mix (airline+retail+telecom) is dominated
# by hard-domain failures (~77% of trajectories get reward 0, mostly user_stop),
# which leaves GRPO with too thin a learning signal and the reward curve stays
# flat regardless of LR. AReaL's own 1.7B tau2 example trains airline-only
# (`config_1.7b_airline.yaml`). Filtering to airline (1,148 tasks, higher success
# rate) gives a cleaner, rising reward signal to verify the pipeline learns.
#
# Inherits the corrected hyperparameters (LR=1e-5, temperature=1.0,
# eps_clip=0.4) and the v1 user simulator from real_8g; only the domain filter
# and step count differ.

set -euo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"

export TAU2_RL_RUN_ID="${TAU2_RL_RUN_ID:-real8g_airline_$(date +%Y%m%d_%H%M%S)}"
export TAU2_RL_DOMAIN="airline"

# ~1 epoch of the 1,148 airline tasks at rollout_batch_size=6.
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

exec bash "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh"
