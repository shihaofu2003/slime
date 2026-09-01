# tau2-rl-stability-k2-fieldreward-lr-sweep

Purpose: test whether stable `k2` KL, field-level reward credit, explicit
behavior penalties, shaped-zero-variance dynamic replacement, and smaller
learning rates prevent the airline RL collapse seen near steps 49–71.

## Configuration

- Common: Qwen3-4B-Instruct-2507 multitool SFT Agent, v1 STOP-trained User,
  airline split, 8 GPUs, 100 updates, GRPO group size 8.
- KL/entropy: `kl_loss_type=k2`, `kl_loss_coef=0.01`, `entropy_coef=0`.
- LR scan: `2e-6`, `3e-6`, `5e-6`.
- Reward: official task reward plus tool-name, per-argument, DB, env, and
  communication credit; malformed JSON, nonexistent tools, repetition, and
  `max_steps` penalties.

## Acceptance checks

- No NaN/Inf loss or gradient through step 100.
- No abrupt rise in truncation, malformed tool calls, repetition, or
  `max_steps`.
- Shaped reward has within-group variance after filtering; zero-variance groups
  are replaced and excluded from training.
- Raw reward and task-completion behavior do not collapse relative to the SFT
  initialization.

## Jobs

- `2e-6`: `pt-wcd6eguo` —
  [run log](jobs/fsh-tau2-rl-k2-fieldreward-lr2e6-0729-144733/run_20260729_144733.log)
- `3e-6`: `pt-g5oxa4ib` —
  [run log](jobs/fsh-tau2-rl-k2-fieldreward-lr3e6-0729-144733/run_20260729_144733.log)
- `5e-6`: `pt-i4h2m1rm` —
  [run log](jobs/fsh-tau2-rl-k2-fieldreward-lr5e6-0729-144733/run_20260729_144733.log)

## Initial monitoring

- All three container preflights passed.
- Step 0: `2e-6` raw reward `0.1875`, grad norm `0.900`; `3e-6` raw reward
  `0.1042`, grad norm `0.896`; `5e-6` raw reward `0.2083`, grad norm `1.213`.
- Step 1 `k2` KL loss is finite: `0.000589` at `3e-6`, `0.000638` at
  `5e-6`; grad norms are `0.884` and `1.040`.
- Each completed step-0 batch had six nonzero-shaped-variance groups.
- Runtime batches retain raw-binary-zero-std groups when their field-shaped
  rewards have variance (`rollout/zero_std/count_0.0` is nonzero); no
  `drop_shaped_zero_std_*` has appeared through step 5. This confirms the
  filter is testing shaped rather than binary reward.
- Step-0 behavior counts (`3e-6` / `5e-6`): malformed JSON `5/9`,
  nonexistent tool `1/1`, repeated calls `9/8`, `max_steps` `0/0`.
- Through step 5, `2e-6` and `3e-6` remain finite with KL loss below
  `0.0013`. `5e-6` is the high-drift arm: KL loss rose from `0.00166`
  (step 2) to `0.00730` (step 4) and `0.01526` (step 5), without NaN/Inf.
- At the 2026-07-29 15:49 CST check, all jobs were running without
  NaN/Inf/OOM. Latest `step / raw reward / k2 KL / grad norm` was
  `13 / 0.2083 / 0.00201 / 0.856` at `2e-6`,
  `14 / 0.2292 / 0.00884 / 0.897` at `3e-6`, and
  `13 / 0.2500 / 0.10331 / 1.015` at `5e-6`.
- The `5e-6` arm has therefore entered clear policy drift and has also required
  repeated permanently-too-long group replacement. It remains running as the
  high-drift control. The `2e-6` arm is the stability leader.
- Real training has exercised shaped-zero-variance replacement in all three
  arms. At the 15:49 check, cumulative complete trajectory groups showed
  `45/44/45` raw-binary-zero-variance groups rescued by field shaping at
  `2e-6/3e-6/5e-6`; remaining shaped-zero-variance groups were replaced and
  excluded. One `3e-6` batch replaced two such groups plus one permanently
  overlength group and still filled the requested batch.
- At the 16:19 CST check, latest `step / five-step raw mean / k2 KL` was
  `22 / 0.254 / 0.00511`, `23 / 0.300 / 0.03368`, and
  `22 / 0.371 / 0.14589` for `2e-6/3e-6/5e-6`. All remained finite, but the
  `5e-6` five-step truncation rate was `2.92%`, versus `0.42%` and `0%`.
- At the 16:49 CST check, latest `step / five-step raw mean / k2 KL` was
  `28 / 0.413 / 0.00424`, `30 / 0.463 / 0.03419`, and
  `28 / 0.413 / 0.14553`. The first two arms had zero five-step truncation;
  `5e-6` reached `4.17%` over five steps and `8.33%` in its latest batch.
- At the 17:04 CST check, `5e-6` showed a joint collapse precursor:
  five-step raw success fell to `0.242`, five-step truncation rose to `7.92%`,
  and k2 KL reached `0.17996` at step 31. The `2e-6/3e-6` five-step raw means
  were `0.321/0.363` with zero truncation and KL `0.00694/0.03919`.
- Near the old step-40 failure point, `2e-6` step 39 retained raw success
  `0.146`, KL `0.01359`, and batch truncation `2.08%`; `3e-6` step 40 retained
  raw success `0.229`, KL `0.08436`, and zero batch truncation. The old
  `1e-5` run had raw success `0.0208` and truncation `29.17%` at step 40.
  In contrast, `5e-6` fell to raw success `0.0417` at step 38 after its
  earlier behavioral-error escalation.
- At the old core collapse point, `2e-6` step 49 retained raw success `0.458`,
  five-step success `0.408`, KL `0.01721`, and zero five-step truncation.
  `3e-6` step 50 retained raw success `0.438` and five-step success `0.350`,
  but KL/grad spiked to `0.14169/1.712`. The old `1e-5` run had raw success
  `0.0208 → 0`, truncation `54.2% → 29.2%`, and KL `0.417 → 0.538` at
  steps 49–50.
- At step 59–60, `2e-6/3e-6` five-step success was `0.454/0.513` with zero
  truncation, versus old-run step-60 raw success `0` and truncation `60.4%`.
  The stability difference remained substantial: five-step KL was
  `0.03599/0.16121`, and latest-384 repeated calls per trajectory were
  `0.141/0.682`.
- `2e-6` completed the old failure point at step 71 without NaN: raw success
  `0.229`, five-step success `0.483`, cumulative success `0.345`, five-step KL
  `0.04773`, grad norm `1.023`, and zero five-step truncation. The old run
  failed during step-71 backward after success collapsed to zero and malformed,
  repetitive `max_steps` trajectories dominated.
- All three k2 arms subsequently crossed step 71 without NaN, including the
  behaviorally unstable `5e-6` arm. This isolates two effects: k2 fixes the
  numerical failure path, while smaller LR is still required for behavioral
  stability. At the 20:21 CST check, five-step
  `success / KL / truncation` was `0.421 / 0.0394 / 0%` at `2e-6`,
  `0.517 / 0.1491 / 0.83%` at `3e-6`, and
  `0.442 / 0.3122 / 4.17%` at `5e-6`.
- W&B: [`2e-6`](https://wandb.ai/fshihao900/slime-dev/runs/yj25bbqh),
  [`3e-6`](https://wandb.ai/fshihao900/slime-dev/runs/2xfidvvf),
  [`5e-6`](https://wandb.ai/fshihao900/slime-dev/runs/lwfjhll0).

## Final outcome

- All three jobs completed 100 updates and are `SUCCEEDED`. Each saved
  `iteration 99`; no training NaN/Inf occurred. The post-save W&B
  `ConnectionResetError` messages are ignored `atexit` cleanup errors and did
  not affect checkpoints or platform status.
- Last-10-update `raw success / k2 KL / truncation` was
  `0.379 / 0.0593 / 0.42%` at `2e-6`,
  `0.396 / 0.1604 / 0.63%` at `3e-6`, and
  `0.452 / 0.3443 / 0.21%` at `5e-6`. Training raw success alone therefore
  gives the wrong LR ordering: higher LR buys much larger policy drift.
- Earliest-to-latest 384-trajectory behavior at `2e-6` improved or stayed
  controlled: raw success `0.245 → 0.385`, malformed JSON
  `0.115 → 0.031`, nonexistent tools `0.023 → 0.003`, repeated calls per
  trajectory `0.185 → 0.211`, and `max_steps` `0.26% → 0.26%`.
- At `3e-6`, latest-384 raw success was `0.414`, but repeated calls rose
  `0.107 → 0.247` and `max_steps` rose `0% → 1.30%`. It is viable only as a
  secondary evaluation candidate.
- At `5e-6`, latest-384 raw success was only `0.333` despite volatile high
  per-batch values; malformed JSON rose `0.063 → 0.177` and repeated calls
  exploded `0.135 → 2.557` per trajectory. It is rejected as behaviorally
  unstable.
- Dynamic replacement was exercised throughout training: logs contain
  `34/33/39` permanently-overlength replacement events and
  `17/21/16` shaped-zero-variance replacement events for
  `2e-6/3e-6/5e-6`, while every optimizer batch remained full.
- Recommended checkpoint for held-out evaluation:
  `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_stability_k2_fieldreward_lr2e6_20260729_065304`.
- This experiment proves the RL optimization path is repaired and identifies
  `2e-6` as the stable candidate. It does **not** by itself prove RL is better
  than the SFT model: that claim requires paired held-out official evaluation
  of SFT versus selected RL checkpoints under identical User, tasks, seeds,
  decoding parameters, and trial counts.

## Interim interpretation

- The prior run used `lr=1e-5` and `low_var_kl`. Its KL loss was already
  `0.1256` at step 10, reached `0.4168` at step 49, and then raw reward and
  truncation collapsed. The same v1 User is used here, while lowering LR and
  using polynomial `k2` produces a strong LR-dependent stability ordering.
- The earlier `entropy_coef=0.001` run had a separate fast failure: entropy
  rose from `0.340` at step 0 to `3.317` at step 20 while the same User was
  fixed. Removing entropy delayed collapse but did not fix the high-LR/KL and
  sparse-credit problems.
- The v1 User is therefore not the primary cause of the optimization collapse.
  User quality still affects the attainable score and interaction
  distribution: official evaluation shows local v1 below Gemini overall, but
  v1 is the strongest tested local User on the common-Agent macro average.
- Current trajectories average roughly `0.72–0.74` tool-name credit and
  `0.62–0.64` argument credit, versus only `0.26–0.29` exact DB success.
  This confirms that binary task reward discarded substantial partial progress
  and made group-relative learning unnecessarily sparse.
- An early/middle/late-third trajectory check near step 19 detects behavior
  drift before binary reward collapse. At `5e-6`, repeated calls per trajectory
  rose `0.138 → 0.326 → 0.583`, malformed JSON rose
  `0.063 → 0.111 → 0.154`, and `max_steps` rose
  `0.24% → 0.72% → 1.69%`. The corresponding `2e-6` malformed rate fell
  `0.115 → 0.068 → 0.057` without a sustained repetition increase.
- By step 33, `5e-6` was a confirmed behavioral failure despite occasional
  high binary-reward batches. Comparing its latest 384 trajectories with its
  earliest 384, repeated calls per trajectory rose `0.138 → 1.469`, malformed
  JSON `0.063 → 0.203`, nonexistent tools `0.016 → 0.109`, and `max_steps`
  `0.26% → 4.95%`; k2 KL reached `0.24517`.
