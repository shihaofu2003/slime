# tau2-rl-usercmp

Purpose: A/B/C comparison of user simulators for tau2-bench GRPO. Three
identical real-8g RL runs (same agent SFT init, same hyperparameters, same
data) varying ONLY the local user simulator, to measure how user-model quality
drives agent RL.

Name: tau2-rl-usercmp.

The agent init, data, reward, length handling, and all hyperparameters are the
real-8g defaults (`examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh`).
Each run only overrides `USER_MODEL_PATH` / `USER_MODEL`.

Tasks:

- `pt-rj02uuoc` / v2 user — `Qwen3-4B-Instruct-2507_tau2_user_sft_stop_v2/iter_0006311_hf`
  (STOP-trained, aligned system prompt, drops content_and_tool mixed targets).
  Run log: [jobs/fsh-tau2-rl-real8g-user-v2-*/run_*.log](jobs/).
- `pt-tkq0yuw3` / v1 user — `Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf`
  (STOP-trained, terminates + runs telecom tools; the prior default). Run log:
  [jobs/fsh-tau2-rl-real8g-user-v1-*/run_*.log](jobs/).
- `pt-97zs608e` / base instruct — original `models/Qwen3-4B-Instruct-2507`
  (no user-SFT; baseline). Expected to behave more poorly as a user (no
  STOP/tool-call training); watch for elevated zero-std group drops. (Resubmitted
  2026-07-26 17:14 — the original `pt-hxwqfzb1` was deleted at the user's request.)
  Run log: [jobs/fsh-tau2-rl-real8g-user-base-*/run_*.log](jobs/).

Comparison metrics (per run, from W&B / run logs): `rollout/raw_reward` curve,
`rollout/dynamic_filter/drop_zero_std_*` (user-quality proxy), pass rate, and
final tau2-bench eval of the saved checkpoint.
