# tau2-rl

Purpose: tau2-bench GRPO training for the agent SFT checkpoint using the
trained local user simulator and each AReaL RL task's database snapshot.

- `fsh-tau2-rl-smoke-0720-231009` — 4-card smoke stopped after detecting
  `<|im_end|>` prevented parsed tool calls from reaching tau2 env. Run log:
  [jobs/fsh-tau2-rl-smoke-0720-231009/run_20260720_231009.log](jobs/fsh-tau2-rl-smoke-0720-231009/run_20260720_231009.log).
- `fsh-tau2-rl-smoke-fixed-0720-232044` — 4-card smoke after stripping chat
  special tokens before tool-call parsing; stopped to switch smoke dumps to
  per-run files. Run log:
  [jobs/fsh-tau2-rl-smoke-fixed-0720-232044/run_20260720_232044.log](jobs/fsh-tau2-rl-smoke-fixed-0720-232044/run_20260720_232044.log).
- `fsh-tau2-rl-smoke-isolated-0720-232656` — 4-card smoke with isolated
  trajectory dump; stopped to also isolate smoke checkpoint directory. Run log:
  [jobs/fsh-tau2-rl-smoke-isolated-0720-232656/run_20260720_232656.log](jobs/fsh-tau2-rl-smoke-isolated-0720-232656/run_20260720_232656.log).
- `fsh-tau2-rl-smoke-clean-0720-233517` — 4-card smoke with isolated trajectory
  dump and checkpoint directory. Run log:
  [jobs/fsh-tau2-rl-smoke-clean-0720-233517/run_20260720_233517.log](jobs/fsh-tau2-rl-smoke-clean-0720-233517/run_20260720_233517.log).
- `pt-zb620oqp` / `fsh-tau2-rl-real8g-hparam250-0721-012152` — real 8-card
  GRPO attempt on the full AReaL RL split; failed before rollout because empty
  `WANDB_MODE` was passed to W&B settings. Run log:
  [jobs/fsh-tau2-rl-real8g-hparam250-0721-012152/run_20260721_012152.log](jobs/fsh-tau2-rl-real8g-hparam250-0721-012152/run_20260721_012152.log).
- `pt-hcxc2h2k` / `fsh-tau2-rl-real8g-hparam250-retry1-0721-012741` — real
  8-card GRPO retry after omitting empty `WANDB_MODE`; failed before rollout
  because `GLOBAL_BATCH_SIZE=64` was not divisible by actor DP=3. W&B initialized
  successfully. Run log:
  [jobs/fsh-tau2-rl-real8g-hparam250-retry1-0721-012741/run_20260721_012741.log](jobs/fsh-tau2-rl-real8g-hparam250-retry1-0721-012741/run_20260721_012741.log).
- `pt-fbuxh8xv` / `fsh-tau2-rl-real8g-hparam330-retry2-0721-014019` — real
  8-card GRPO retry with DP-aligned batch. Failed at step 12 during ref
  log-prob forward with CUDA OOM: logits allocation requested 25.96 GiB while
  GPU 4 had 23.04 GiB free. Run log:
  [jobs/fsh-tau2-rl-real8g-hparam330-retry2-0721-014019/run_20260721_014019.log](jobs/fsh-tau2-rl-real8g-hparam330-retry2-0721-014019/run_20260721_014019.log).
- `pt-ufjqemwt` / `fsh-tau2-rl-real8g-hparam330-oomfix1-0721-025526` — real
  8-card GRPO retry after OOM fix. Failed before rollout because
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was propagated to SGLang
  actors, where `TorchMemorySaver` does not support expandable segments. Run log:
  [jobs/fsh-tau2-rl-real8g-hparam330-oomfix1-0721-025526/run_20260721_025526.log](jobs/fsh-tau2-rl-real8g-hparam330-oomfix1-0721-025526/run_20260721_025526.log).
- `pt-6vvzjvhr` / `fsh-tau2-rl-real8g-hparam330-oomfix2-0721-032623` — real
  8-card GRPO retry after removing the allocator env propagation that broke
  SGLang startup. Failed at rollout 13 during ref log-prob forward with CUDA
  OOM; 1,411 trajectories were dumped, and no checkpoint was saved. Run log:
  [jobs/fsh-tau2-rl-real8g-hparam330-oomfix2-0721-032623/run_20260721_032623.log](jobs/fsh-tau2-rl-real8g-hparam330-oomfix2-0721-032623/run_20260721_032623.log).
- `pt-6cb1xj0j` / `fsh-tau2-rl-real8g-oomfix3-0721-141643` — real 8-card GRPO
  retry with 1,024-token log-prob chunks and temperature scaling moved inside
  each chunk. Submitted; verification targets are the first train step and
  successful completion of rollout 13. Run log:
  [jobs/fsh-tau2-rl-real8g-oomfix3-0721-141643/run_20260721_141643.log](jobs/fsh-tau2-rl-real8g-oomfix3-0721-141643/run_20260721_141643.log).
- `pt-8kwmsovk` / `fsh-tau2-rl-verify8g-oomfix-0726-153458` — 8-card
  verification run after re-diagnosing the OOM root cause. The OOM is not
  fixable by lowering `max_tokens_per_gpu` or chunking log-probs: a single
  trajectory longer than the bin cap forms its own oversized microbatch whose
  full `[T, V]` logits OOM the forward. Fix: tokenize the FULL conversation
  on-policy (removed the leading-message truncation that broke on-policy), drop
  whole groups over `TAU2_RL_MAX_TRAIN_TOKENS=16384` via a combined dynamic
  filter (`filters.drop_zero_std_or_too_long`), and set
  `max_tokens_per_gpu == cap` so no oversized bin can be emitted. A preflight
  test (`test_rollout_logic.py`) gates the rollout. Verification targets are
  the preflight passing, the first train step completing without OOM, and the
  `rollout/dynamic_filter/drop_too_long_*` metric appearing. Run log:
  [jobs/fsh-tau2-rl-verify8g-oomfix-0726-153458/run_20260726_153458.log](jobs/fsh-tau2-rl-verify8g-oomfix-0726-153458/run_20260726_153458.log).
- `pt-vyehdd5w` / `fsh-...-airline-0728-142716` — airline-only real-8g GRPO
  with dense reward shaping, `kl_loss_coef=0.01`, and `entropy_coef=0.001`.
  Stopped at step 21 after entropy rose 0.34→3.03, output degenerated, and raw
  reward failed to improve. Run log:
  [jobs/fsh-examples-tau2-bench-rl-run_qwen3_4b_instruct_2507_tau2_rl_real_8g_airline-0728-142716/run_20260728_142716.log](jobs/fsh-examples-tau2-bench-rl-run_qwen3_4b_instruct_2507_tau2_rl_real_8g_airline-0728-142716/run_20260728_142716.log).
- `pt-ssu2b355` / `fsh-...-airline-0728-212318` — retry of the dense-shaped
  airline run with `entropy_coef=0`; `kl_loss_coef=0.01` unchanged. Failed
  during step 71 backward with a NaN grad norm after policy collapse: reward
  reached zero, truncated/repetitive malformed tool calls dominated, and the
  last trajectories all hit `max_steps`. Disabling entropy fixed only the
  earlier fast high-entropy runaway, not the underlying failure. Run log:
  [jobs/fsh-examples-tau2-bench-rl-run_qwen3_4b_instruct_2507_tau2_rl_real_8g_airline-0728-212318/run_20260728_212318.log](jobs/fsh-examples-tau2-bench-rl-run_qwen3_4b_instruct_2507_tau2_rl_real_8g_airline-0728-212318/run_20260728_212318.log).
