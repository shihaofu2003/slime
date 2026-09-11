# tau2-rl-usercmp monitor

Purpose: status log for the 3 user-model comparison real-8g GRPO runs. Checked
roughly every 30 min. Metrics of interest per run: state, latest train step +
grad_norm + raw_reward, checkpoint iterations, retry/permanently-too-long stats
(from the trajectory dump), and any OOM/Traceback/failure. The base-instruct
user run is the highest risk (degenerate user → elevated zero-std drops or a
drop-loop hang).

Jobs: v2=`pt-rj02uuoc`, v1=`pt-tkq0yuw3`, base=`pt-97zs608e`.

- 2026-07-26 17:15 Asia/Shanghai — all three RUNNING (early; preflight passed
  for v2, no train step yet on any). v2 user server booted on
  `iter_0006311_hf`. base run is the resubmitted `pt-97zs608e` (original
  `pt-hxwqfzb1` deleted at user request). Next check: first train step on each
  + watch base for zero-std drop-loop.

- 2026-07-26 17:44 CST — all RUNNING, no OOM/hang.
  - v2 pt-rj02uuoc: step 3, grad_norm 1.12, raw_reward 0.3125, zero_std drops
    4-10/rollout, no permanently_too_long.
  - v1 pt-tkq0yuw3: step 3, grad_norm 1.007, raw_reward 0.229, zero_std drops
    3-7/rollout, no permanently_too_long.
  - base pt-97zs608e: step 1 (resubmitted later), grad_norm 1.16, raw_reward
    0.375, 1 drop_permanently_too_long (1 trajectory couldn't resample short →
    group dropped, escalation path working), zero_std drops 5-6/rollout. NOT in
    a drop-loop — progressing normally.

- 2026-07-26 18:15 CST — all RUNNING, no OOM/hang. Pace ~8-10 train steps/h
  → ~330 steps in ~1.5 days.
  - v2 pt-rj02uuoc: step 8, grad_norm 1.12, raw_reward 0.354; attempts
    {1:1090,2:20,3:13}, perm=10.
  - v1 pt-tkq0yuw3: step 9, grad_norm 1.15, raw_reward 0.5625 (leading); attempts
    {1:1117,2:30,3:11}, perm=9.
  - base pt-97zs608e: step 7, grad_norm 0.97, raw_reward 0.417; attempts
    {1:898,2:34,3:12}, perm=5. Base user produces slightly more over-long
    trajectories (more retries) but retry handles most — NOT degenerate, running
    between v1 and v2 on reward.

- 2026-07-26 18:45 CST — all RUNNING, no OOM/hang. Rewards noisy rollout-to-
  rollout, all in ~0.37-0.48 band.
  - v2 pt-rj02uuoc: step 12, grad_norm 0.91, raw_reward 0.479; perm=29.
  - v1 pt-tkq0yuw3: step 14, grad_norm 1.34, raw_reward 0.375; perm=29.
  - base pt-97zs608e: step 14, grad_norm 1.00, raw_reward 0.417; perm=8 (fewest).

- 2026-07-26 19:14 CST — all RUNNING, no OOM/hang. Rewards converged to ~0.38
  (indistinguishable across runs at this stage).
  - v2 pt-rj02uuoc: step 16, grad_norm 1.37, raw_reward 0.396; perm=31.
  - v1 pt-tkq0yuw3: step 19, grad_norm 1.10, raw_reward 0.396; perm=32.
  - base pt-97zs608e: step 19, grad_norm 1.06, raw_reward 0.375; perm=9 (fewest).

- 2026-07-26 19:38 CST — all RUNNING, no OOM/hang. **First clear divergence:**
  trained users (v1/v2) pulling ahead of base.
  - v2 pt-rj02uuoc: step 21, grad_norm 1.21, raw_reward 0.5625 (rising); perm=37.
  - v1 pt-tkq0yuw3: step 24, grad_norm 1.29, raw_reward 0.6458 (leading); perm=32.
  - base pt-97zs608e: step 23, grad_norm 1.08, raw_reward 0.396 (flat ~0.38-0.42
    since start); perm=9. Base user → agent plateaus lower, as expected.

- 2026-07-26 20:24 CST — all RUNNING (steps advancing), no OOM/hang. Rewards
  noisy; v1 still leads on average.
  - v2 pt-rj02uuoc: step 29, grad_norm 1.19, raw_reward 0.4375; perm=37.
  - v1 pt-tkq0yuw3: step 33, grad_norm 1.06, raw_reward 0.5417 (avg ~0.58 over
    last 3 checks, leading); perm=32.
  - base pt-97zs608e: step 32, grad_norm 1.16, raw_reward 0.479 (rebounded from
    0.396); perm=10.

- 2026-07-26 20:44 CST — all RUNNING (log_age 0-5s), no OOM/hang. Rewards still
  noisy single-rollout (0.29-0.56 band), no clean separation at step 32-37
  (<12% of 330).
  - v2 pt-rj02uuoc: step 32, grad_norm 1.40, raw_reward 0.5625; perm=43.
  - v1 pt-tkq0yuw3: step 37, grad_norm 1.11, raw_reward 0.2917 (noisy dip); perm=37.
  - base pt-97zs608e: step 36, grad_norm 1.04, raw_reward 0.375; perm=11.

- 2026-07-26 21:49 CST — all RUNNING (log_age 1-5s), no OOM/hang. Rewards still
  noisy 0.23-0.42, no clean trend at step 42-49 (~14% of 330).
  - v2 pt-rj02uuoc: step 43, grad_norm 1.33, raw_reward 0.333; perm=59 (1.15%).
  - v1 pt-tkq0yuw3: step 47, grad_norm 1.05, raw_reward 0.4167; perm=54 (0.98%).
  - base pt-97zs608e: step 49, grad_norm 1.34, raw_reward 0.229 (dip); perm=13
    (0.25%). Real comparison needs smoothed W&B curve / final eval.

- 2026-07-26 21:51 CST — all RUNNING (log_age 1-2s), no OOM/hang. Smoothed
  reward (last 8 rollouts) shows runs tracking close: v1≈v2>base, all ~0.47-0.52.
  - v2 pt-rj02uuoc: step 44, grad_norm 1.06, raw_reward 0.521 (smoothed 0.490); perm 1.14%.
  - v1 pt-tkq0yuw3: step 48, grad_norm 1.29, raw_reward 0.417 (smoothed 0.523); perm 0.97%.
  - base pt-97zs608e: step 49, grad_norm 1.34, raw_reward 0.229 (smoothed 0.466); perm 0.25%.

- 2026-07-26 22:14 CST — all RUNNING (log_age 1-3s), no OOM/hang. Smoothed(8)
  rewards converged together at ~0.43 (no separation).
  - v2 pt-rj02uuoc: step 48, grad_norm 1.18, smoothed 0.424; perm 1.25%.
  - v1 pt-tkq0yuw3: step 51, grad_norm 1.04, smoothed 0.448; perm 0.91%.
  - base pt-97zs608e: step 54, grad_norm 1.05, smoothed 0.424; perm 0.23%.

- 2026-07-26 23:01 CST — all RUNNING (log_age 3-5s), no OOM/hang. Smoothed(8):
  v1>v2>base, gaps ~0.05 (small but consistent over last 2 checks).
  - v2 pt-rj02uuoc: step 55, grad_norm 1.42, smoothed 0.461; perm 1.12%.
  - v1 pt-tkq0yuw3: step 61, grad_norm 1.13, smoothed 0.492 (leading); perm 0.85%.
  - base pt-97zs608e: step 62, grad_norm 1.17, smoothed 0.435; perm 0.24%.

- 2026-07-26 23:14 CST — all RUNNING (log_age 2-4s), no OOM/hang. Base smoothed
  dipping (0.398) while v1/v2 hold ~0.47 — first sustained sign of expected
  divergence.
  - v2 pt-rj02uuoc: step 58, grad_norm 1.15, smoothed 0.461; perm 1.07%.
  - v1 pt-tkq0yuw3: step 64, grad_norm 1.08, smoothed 0.482 (leading); perm 0.82%.
  - base pt-97zs608e: step 65, grad_norm 1.28, smoothed 0.398 (slipping); perm 0.25%.

- 2026-07-26 23:44 CST — all RUNNING (log_age 3-4s), no OOM/hang. v1 holds a
  small lead; v2/base tied ~0.41.
  - v2 pt-rj02uuoc: step 64, grad_norm 0.89, smoothed 0.414; perm 1.02%.
  - v1 pt-tkq0yuw3: step 69, grad_norm 1.36, smoothed 0.471 (leading); perm 0.85%.
  - base pt-97zs608e: step 70, grad_norm 1.03, smoothed 0.414; perm 0.28%.

- 2026-07-27 00:14 CST — all RUNNING (log_age 2-4s), no OOM/hang. **Clearest
  split yet: v1 pulling ahead (0.534), v2 regressing to base level (~0.39).**
  - v2 pt-rj02uuoc: step 68, grad_norm 1.06, smoothed 0.383; perm 1.08%.
  - v1 pt-tkq0yuw3: step 76, grad_norm 1.01, smoothed 0.534 (clear leader); perm 0.88%.
  - base pt-97zs608e: step 75, grad_norm 1.11, smoothed 0.393; perm 0.36%.
  Note: v2 ("improved" user) tracking the untrained base, not v1 — v2's SFT
  changes haven't helped RL signal so far.

- 2026-07-27 00:44 CST — all RUNNING (log_age 4-13s), no OOM/hang. Last check's
  "v2 regressed" was mostly noise — v2 bounced back; v1/v2 neck-and-neck ~0.48,
  base ~0.44. No robust separation at ~25% of training.
  - v2 pt-rj02uuoc: step 73, grad_norm 1.25, smoothed 0.477; perm 1.01%.
  - v1 pt-tkq0yuw3: step 82, grad_norm 0.99, smoothed 0.487; perm 0.91%.
  - base pt-97zs608e: step 82, grad_norm 1.22, smoothed 0.440; perm 0.36%.

- 2026-07-27 01:14 CST — all RUNNING (log_age 1-23s), no OOM/hang. Smoothed(8)
  all converged ~0.44-0.47, no separation.
  - v2 pt-rj02uuoc: step 79, grad_norm 0.90, smoothed 0.466; perm 1.01%.
  - v1 pt-tkq0yuw3: step 87, grad_norm 1.04, smoothed 0.448; perm 0.88%.
  - base pt-97zs608e: step 87, grad_norm 1.04, smoothed 0.443; perm 0.34%.

- 2026-07-27 01:44 CST — all RUNNING (log_age 3-5s), no OOM/hang. Over last ~5
  checks v1 trendline ~0.50, v2/base ~0.43 (small persistent ~0.07 gap, still
  near noise band).
  - v2 pt-rj02uuoc: step 83, grad_norm 1.11, smoothed 0.438; perm 0.99%.
  - v1 pt-tkq0yuw3: step 92, grad_norm 1.00, smoothed 0.531; perm 0.84%.
  - base pt-97zs608e: step 93, grad_norm 1.20, smoothed 0.417; perm 0.35%.

- 2026-07-27 02:14 CST — all RUNNING (log_age 2-4s), no OOM/hang. Recent
  trendlines: v1 ~0.50, v2 ~0.44, base ~0.43. v1 holds ~0.06 lead; v2 ≈ base.
  - v2 pt-rj02uuoc: step 88, grad_norm 1.13, smoothed 0.443; perm 0.93%.
  - v1 pt-tkq0yuw3: step 98, grad_norm 1.15, smoothed 0.513; perm 0.86%.
  - base pt-97zs608e: step 98, grad_norm 1.20, smoothed 0.464; perm 0.34%.

- 2026-07-27 02:44 CST — all RUNNING (log_age 1-2s), no OOM/hang. base spiked
  to smoothed 0.607 (lead), but it's the noisiest run — likely partial reversion,
  don't over-read.
  - v2 pt-rj02uuoc: step 94, grad_norm 1.08, smoothed 0.482; perm 0.97%.
  - v1 pt-tkq0yuw3: step 103, grad_norm 1.10, smoothed 0.513; perm 0.85%.
  - base pt-97zs608e: step 104, grad_norm 1.08, smoothed 0.607 (spike); perm 0.41%.

- 2026-07-27 03:14 CST — all RUNNING (log_age 5-6s), no OOM/hang. Smoothed:
  base 0.526, v1/v2 tied 0.445. Leader keeps rotating run-to-run; no robust order.
  - v2 pt-rj02uuoc: step 98, grad_norm 1.14, smoothed 0.445; perm 0.95%.
  - v1 pt-tkq0yuw3: step 107, grad_norm 1.29, smoothed 0.445 (v1 had a 0.208 rollout); perm 0.89%.
  - base pt-97zs608e: step 110, grad_norm 1.09, smoothed 0.526; perm 0.42%.

- 2026-07-27 03:44 CST — all RUNNING (log_age 4-6s), no OOM/hang. base now
  leading 3 checks straight (0.607/0.526/0.513); v1 faded to 0.417. CAVEAT:
  training reward depends on the user model itself — an "easier" user inflates
  reward without the agent being better. Final eval (fixed reference user) is the
  real test.
  - v2 pt-rj02uuoc: step 103, grad_norm 1.27, smoothed 0.474; perm 1.03%.
  - v1 pt-tkq0yuw3: step 113, grad_norm 1.02, smoothed 0.417; perm 0.88%.
  - base pt-97zs608e: step 114, grad_norm 1.07, smoothed 0.513; perm 0.50%.

- 2026-07-27 04:14 CST — all RUNNING (log_age 3-8s), no OOM/hang. base still
  leading (4th straight check, 0.482, narrowing from peak); v1 0.438, v2 0.409.
  Same caveat: training reward is user-dependent.
  - v2 pt-rj02uuoc: step 108, grad_norm 1.11, smoothed 0.409; perm 1.01%.
  - v1 pt-tkq0yuw3: step 118, grad_norm 1.10, smoothed 0.438; perm 0.85%.
  - base pt-97zs608e: step 121, grad_norm 1.13, smoothed 0.482; perm 0.48%.

- 2026-07-27 04:44 CST — all RUNNING (log_age 3-7s), no OOM/hang. Lead rotated
  back to v1 (0.513); base's 4-check run ended (0.474). All still in 0.44-0.51
  band, no robust separation at ~38% of training.
  - v2 pt-rj02uuoc: step 113, grad_norm 1.23, smoothed 0.448; perm 0.97%.
  - v1 pt-tkq0yuw3: step 124, grad_norm 1.06, smoothed 0.513; perm 0.89%.
  - base pt-97zs608e: step 126, grad_norm 1.15, smoothed 0.474; perm 0.46%.

- 2026-07-27 05:14 CST — all RUNNING (log_age 2-4s), no OOM/hang. Lead rotated
  to base (0.482), v1 0.432, v2 0.448. Same ~0.43-0.48 band; ~40% done.
  - v2 pt-rj02uuoc: step 119, grad_norm 0.98, smoothed 0.448; perm 0.96%.
  - v1 pt-tkq0yuw3: step 129, grad_norm 0.91, smoothed 0.432; perm 0.87%.
  - base pt-97zs608e: step 132, grad_norm 0.99, smoothed 0.482; perm 0.46%.

- 2026-07-27 05:44 CST — all RUNNING (log_age 1-4s), no OOM/hang. Tight cluster
  ~0.44-0.47 (base 0.469, v2 0.448, v1 0.443); ~42% done.
  - v2 pt-rj02uuoc: step 123, grad_norm 1.05, smoothed 0.448; perm 1.01%.
  - v1 pt-tkq0yuw3: step 135, grad_norm 1.08, smoothed 0.443; perm 0.83%.
  - base pt-97zs608e: step 137, grad_norm 1.36, smoothed 0.469; perm 0.52%.

- 2026-07-27 06:14 CST — all RUNNING (log_age 3-6s), no OOM/hang. v1 back in
  lead (0.518); v2 0.458, base 0.440. v2 ~13 steps behind (more retries).
  - v2 pt-rj02uuoc: step 128, grad_norm 0.99, smoothed 0.458; perm 0.98%.
  - v1 pt-tkq0yuw3: step 141, grad_norm 1.05, smoothed 0.518; perm 0.81%.
  - base pt-97zs608e: step 142, grad_norm 1.00, smoothed 0.440; perm 0.51%.

- 2026-07-27 06:44 CST — all RUNNING (log_age 2-6s), no OOM/hang. v1 leading
  (0.497); base 0.443, v2 0.424. v1 top in 2 of last 3 checks. ~45% done.
  - v2 pt-rj02uuoc: step 135, grad_norm 0.99, smoothed 0.424; perm 0.94%.
  - v1 pt-tkq0yuw3: step 146, grad_norm 1.34, smoothed 0.497; perm 0.82%.
  - base pt-97zs608e: step 149, grad_norm 0.98, smoothed 0.443; perm 0.55%.

- 2026-07-27 07:14 CST — all RUNNING (log_age 4-21s), no OOM/hang. base leads
  again (0.536); v2 0.451, v1 0.440. Each run has now taken multiple turns
  leading — cumulative picture: equivalent on training reward. ~47% done.
  - v2 pt-rj02uuoc: step 142, grad_norm 1.29, smoothed 0.451; perm 0.94%.
  - v1 pt-tkq0yuw3: step 151, grad_norm 1.01, smoothed 0.440; perm 0.83%.
  - base pt-97zs608e: step 156, grad_norm 1.08, smoothed 0.536; perm 0.54%.

- 2026-07-27 07:44 CST — all RUNNING (log_age 2-9s), no OOM/hang. base 0.510 /
  v2 0.490 / v1 0.438 — top keeps rotating. ~48% done.
  - v2 pt-rj02uuoc: step 147, grad_norm 1.03, smoothed 0.490; perm 0.91%.
  - v1 pt-tkq0yuw3: step 157, grad_norm 0.95, smoothed 0.438; perm 0.82%.
  - base pt-97zs608e: step 161, grad_norm 1.04, smoothed 0.510; perm 0.55%.

- 2026-07-27 08:14 CST — all RUNNING (log_age 1-3s), no OOM/hang. **Halfway
  (~50%).** Tightest cluster yet: base 0.510 / v1 0.490 / v2 0.487.
  - v2 pt-rj02uuoc: step 151, grad_norm 1.22, smoothed 0.487; perm 0.91%.
  - v1 pt-tkq0yuw3: step 161, grad_norm 1.04, smoothed 0.490; perm 0.83%.
  - base pt-97zs608e: step 166, grad_norm 1.03, smoothed 0.510; perm 0.54%.

- 2026-07-27 10:27 CST — all RUNNING (log_age 1-5s), no OOM/hang. Smoothed
  0.41-0.43 band, no separation. Steps: v2 174, v1 186, base 192.
  - v2 pt-rj02uuoc: step 174, grad_norm 1.13, smoothed 0.411; perm 0.89%.
  - v1 pt-tkq0yuw3: step 186, grad_norm 1.12, smoothed 0.427; perm 0.81%.
  - base pt-97zs608e: step 192, grad_norm 1.08, smoothed 0.427; perm 0.57%.

- 2026-07-27 10:46 CST — all RUNNING (log_age 4-7s), no OOM/hang. Smoothed
  0.39-0.43 (v1 0.430, v2 0.417, base 0.391). Steps 177/188/195 (~57-59%).
  - v2 pt-rj02uuoc: step 177, grad_norm 1.11, smoothed 0.417; perm 0.89%.
  - v1 pt-tkq0yuw3: step 188, grad_norm 1.09, smoothed 0.430; perm 0.83%.
  - base pt-97zs608e: step 195, grad_norm 1.19, smoothed 0.391; perm 0.56%.

- 2026-07-27 11:15 CST — all RUNNING (log_age 3-4s), no OOM/hang. Tightest
  cluster: v1 0.458 / v2 0.453 / base 0.440. Steps 182/194/201 (~58%).
  - v2 pt-rj02uuoc: step 182, grad_norm 1.21, smoothed 0.453; perm 0.89%.
  - v1 pt-tkq0yuw3: step 194, grad_norm 1.11, smoothed 0.458; perm 0.81%.
  - base pt-97zs608e: step 201, grad_norm 1.03, smoothed 0.440; perm 0.58%.

- 2026-07-27 12:22 CST — all RUNNING (log_age 2-7s), no OOM/hang. Smoothed
  0.43-0.46 band (base 0.461, v2 0.438, v1 0.432). Steps 192/207/214 (~62% avg).
  - v2 pt-rj02uuoc: step 192, grad_norm 1.23, smoothed 0.438; perm 0.86%.
  - v1 pt-tkq0yuw3: step 207, grad_norm 1.03, smoothed 0.432; perm 0.79%.
  - base pt-97zs608e: step 214, grad_norm 1.17, smoothed 0.461; perm 0.55%.

---

## Corrected hyperparameters (2026-07-27)

The prior 3 runs (pt-rj02uuoc / pt-tkq0yuw3 / pt-97zs608e) used `lr=1e-6`,
`temperature=0.65`, `eps_clip=0.2` and produced a **flat reward curve** (v1:
q1=0.469 → q4=0.456 over 209 steps, no upward trend). Root cause: lr was 17×
lower than the AReaL tau2 recipe (`1.7e-5`), plus low exploration temperature and
tight eps_clip. Full diagnosis + fix in `output/doc/EXP_QA.md`.

Corrected config (aligned to AReaL `examples/tau2/config_1.7b_airline.yaml`):
`LR=1e-5`, `ROLLOUT_TEMPERATURE=1.0`, `EPS_CLIP=EPS_CLIP_HIGH=0.4`, rest
unchanged. The 3 flat runs were killed. Per user request, **only v1 is running
for now**: `pt-ctn45oy5` (fsh-tau2-rl-real8g-user-v1-0727-124152).

Verification target: smoothed reward (last 8 rollouts) trending UP, not flat.

- 2026-07-27 13:15 CST — v1 pt-ctn45oy5 (corrected): RUNNING, step 2, grad_norm
  0.88 (healthy, no divergence), trajectory [0.188, 0.646, 0.521], smoothed 0.451,
  perm=0, no OOM/NaN. Too early (3 rollouts) to judge rising; need ~20-30 steps.

- 2026-07-27 13:45 CST — v1 pt-ctn45oy5: step 8, grad_norm 1.01 (healthy, LR
  1e-5 stable — no divergence), smoothed 0.445 (1st half 0.406 → 2nd half 0.425,
  marginal), perm=7, no OOM/NaN. Still flat at step 8 but too early; need ~step 30.

- 2026-07-27 14:19 CST — v1 pt-ctn45oy5: step 13, grad_norm 0.98 (healthy),
  smoothed 0.409 (DOWN from 0.445 at step 8; 1st half 0.426 → 2nd half 0.399),
  perm=11, no OOM/NaN. Still NO upward trend at step 13 — below the ~30-step
  judgment threshold, but not encouraging. If still flat by ~step 30, cause is
  likely NOT lr (already 10x) — more likely small batch (48 vs AReaL 128) /
  reward sparsity / capability ceiling.

- 2026-07-27 14:45 CST — v1 pt-ctn45oy5: step 17, grad_norm 1.27 (healthy),
  smoothed 0.404 (1st half 0.417 ≈ 2nd half 0.414 — dead flat), abs_diff stable
  ~0.1-0.17, perm=13, no OOM/NaN. Still NO rise through 17 steps. Policy IS
  moving (not an LR problem). NOTE: our per-step learning is ~4.5x slower than
  AReaL (batch 48 vs 128) so AReaL's step-20 rise ≈ our step ~80; genuinely
  early, but abs reward level (~0.40) tracks the old flat run exactly.

- 2026-07-27 15:15 CST — v1 pt-ctn45oy5: step 22, grad_norm 1.57 (creeping up
  0.98→1.27→1.57 — watch for divergence), smoothed 0.448 (oscillating 0.40-0.45,
  no trend; 1st half 0.434 ≈ 2nd half 0.420), perm=13 (stable), no OOM/NaN. Last
  reward 0.667 (single high point) but no sustained rise. Approaching 30-step
  threshold still flat.

- 2026-07-27 15:46 CST — v1 pt-ctn45oy5: step 27, **FIRST ENCOURAGING SIGNAL.**
  smoothed 0.497 (up from 0.404@17 → 0.448@22 → 0.497@27); 1st half 0.412 → 2nd
  half 0.460 (+0.048, first real separation). grad_norm plateaued ~1.6 (stable,
  NOT diverging: 1.57/1.64/1.59/1.61/1.60/1.56). perm=14, no OOM/NaN. Curve may
  be starting to rise — consistent with slower per-step learning (batch 48);
  needs 1-2 more checks to confirm trend holds.

- 2026-07-27 16:16 CST — v1 pt-ctn45oy5: step 31. Smoothed 0.471 (held above
  baseline; quartile trend @8:0.445 @16:0.398 @24:0.466 @31:0.471 — dip-then-rise,
  recent level above early). 1st half 0.410 → 2nd half 0.470 (+0.06, sustained
  separation). Reward no longer dead-flat — weakly rising but noisy. grad_norm
  band creeping to ~1.6-1.94 (watch, not exploding). perm=14, no OOM/NaN.

- 2026-07-27 16:45 CST — v1 pt-ctn45oy5: step 35. Smoothed 0.461 (quartiles
  @9:0.427 @18:0.378 @27:0.497 @35:0.461 — recent ~0.46-0.50 band above early
  ~0.38-0.43). 1st half 0.416 → 2nd half 0.468 (+0.05, sustained). Weak noisy
  rise continues (~+0.05-0.07 above early plateau). grad_norm stable 1.45-1.87
  (no divergence). perm=14, no OOM/NaN.

- 2026-07-27 17:16 CST — v1 pt-ctn45oy5: step 39. Smoothed DROPPED to 0.411
  (peak was 0.497@27 / 0.492@30, now reverted). Quartiles @10:0.427 @20:0.406
  @30:0.492 @39:0.411 — oscillating 0.41-0.50, NO sustained rise; the step-27
  signal looks like a noisy peak. 2nd half 0.457 still > 1st half 0.411 (+0.046,
  only weak positive). grad_norm back to ~1.4 (stable, no divergence). perm=18.
  Through 39 steps: effectively still flat (~0.42) — prior "rise" reverted.

- 2026-07-27 17:46 CST — v1 pt-ctn45oy5: step 42, smoothed 0.357 (DOWN from
  0.497 peak @27/30; quartiles @10:0.427 @20:0.406 @30:0.492 @42:0.357).
  Oscillating ~0.36-0.50, NO sustained rise; step-27 peak was noise. grad_norm
  stable ~1.1-1.7, perm=25, no OOM/NaN. **Both old (1e-6) and new (1e-5) LR are
  flat ~0.43 → bottleneck is NOT lr.**
  ROOT-CAUSE DIAGNOSTIC: reward distribution is 77% zero / 23% one / 0% partial.
  Zero-reward terminations: user_stop 75% (user gave up before task done),
  max_steps 19%, too_many_errors 6%. The agent fails most tasks; GRPO learns
  from only the ~23% mixed-success groups → thin signal → flat. KEY GAP vs
  AReaL: their 1.7b example trains AIRLINE-ONLY; we train all 3 domains
  (incl. hard telecom) → failures dominate. Recommend: filter to airline-only
  first (match AReaL) to confirm a rising curve.

- 2026-07-27 18:15 CST — v1 pt-ctn45oy5: step 46, smoothed 0.367 (quartiles
  @11:0.430 @22:0.448 @33:0.464 @46:0.367 — peaked ~step 33, now below
  baseline). 1st half 0.427 ≈ 2nd half 0.422 (earlier +0.05 signal GONE).
  grad_norm stable 1.2-1.5, perm=27, no OOM/NaN. **Confirmed: corrected config
  (1e-5) did NOT produce learning through 46 steps — flat-to-declining ~0.37-0.46.**
  Both LRs flat → structural cause (77% failure / sparse binary reward / hard
  multi-domain). Run is a dead end; recommend kill + pivot to airline-only.

- 2026-07-27 18:45 CST — v1 pt-ctn45oy5: step 48, smoothed 0.367 (overall mean
  0.422), grad_norm 1.30, perm=28, no errors. No change — still flat/declining.
  Confirmed dead-end (both LRs flat). Awaiting user OK to kill + pivot to
  airline-only.
- 2026-07-27 19:16 CST — v1 pt-ctn45oy5: 'train/step': 51, 'train/grad_norm': 1.442279995383354, smoothed 0.440. Still flat (confirmed dead-end). Keeping run per "only v1 needed"; will not re-ask — standing by for user direction (kill+airline-only, or let finish).

---

## Airline-only pivot (2026-07-27 19:30)

The corrected-hyperparam v1 run (`pt-ctn45oy5`, LR 1e-5/temp 1.0/clip 0.4) was
**flat through ~50 steps** (smoothed ~0.37-0.46, no trend) — same as the old
1e-6 run. Both LRs flat ⇒ bottleneck is structural, not LR. Reward distribution
on the full 3-domain mix: 77% zero / 23% one (75% of zero-reward ends via
`user_stop`); GRPO learns from a thin mixed-success slice. AReaL's 1.7B tau2
example trains **airline-only**.

Killed `pt-ctn45oy5`. Added `--domain` filter to `prepare_rl_data.py` and
`TAU2_RL_DOMAIN` to the base run script; new wrapper
`run_qwen3_4b_instruct_2507_tau2_rl_real_8g_airline.sh`. Submitted airline-only
run **`pt-i8ycxht7`** (1,148 airline tasks, NUM_ROLLOUT=200 ~1 epoch, same
corrected hyperparams + v1 user). Target: higher reward baseline + a rising
trend (vs the flat ~0.43 on full-domain).

- 2026-07-27 19:47 CST — airline pt-i8ycxht7: RUNNING, still in rollout 0 (no
  train step yet; ~25 min in, setup + first rollout). `--domain airline`
  confirmed (airline tools active). BUT trajectory dump (n=124) already shows
  **81% zero / 19% one, mean 0.185** — WORSE than full-domain, and same failure
  mode: **user_stop 98/99 failures**. The "airline is easier" hypothesis is NOT
  holding — the user_stop-dominated failure is domain-independent. May indicate
  the 4B user simulator gives up prematurely (AReaL uses a 72B user). Awaiting
  first train step + reward to confirm, but early signal is concerning.

- 2026-07-27 20:17 CST — airline pt-i8ycxht7: step 2, grad_norm 0.98, trajectory
  [0.271, 0.333, 0.521], smoothed 0.375. Dump (n=432): zero 71% / one 29%,
  mean 0.292; zero-term user_stop 302/306 (still dominant), perm=0, no OOM/NaN.
  **Airline baseline (~0.29-0.375) is NOT higher than full-domain (~0.43)** —
  the domain pivot did not change the failure structure. user_stop dominates
  regardless of domain ⇒ bottleneck is the user simulator (premature give-up),
  not LR or domain. Too early (3 rollouts) to judge rising, but the hypothesis
  (airline easier) isn't confirmed.

- 2026-07-27 23:31 CST — airline pt-i8ycxht7: step 19, grad_norm 1.02, smoothed
  0.479 (mean 0.439). Trajectory 2nd half (~0.47) > 1st half (~0.41) by ~0.06 —
  mild upward drift, weakly above the full-domain flat baseline (~0.43). Dump
  still 71% zero / 29% one (mean 0.291), perm=58, user_stop still dominates
  failures. No OOM/NaN. Cautiously positive but inconclusive at step 19; need
  ~step 30-40 to confirm a real rise.

- 2026-07-27 23:47 CST — airline pt-i8ycxht7: step 20, grad_norm 1.20, smoothed
  0.464 (quartiles @5:0.438 @10:0.419 @15:0.438 @20:0.464). 2nd half 0.468 > 1st
  0.408 (+0.06). Mild upward drift, recent ~0.46-0.48 above full-domain baseline
  ~0.43. Dump still 71% zero / 29% one (mean 0.288); zero-term user_stop 1698/1804
  (94%). perm=68, no OOM/NaN. Weakly positive but success rate still capped at
  ~29% by user_stop; inconclusive at step 20.

- 2026-07-28 00:17 CST — airline pt-i8ycxht7: step 21, grad_norm 1.10, smoothed
  0.427 (DOWN from 0.464; last rewards 0.333/0.458/0.354). **CRITICAL: true
  success rate (dump one%) is DECLINING: third1 31% → third2 30% → third3 21%.**
  The kept-batch reward (~0.43-0.48) was masking this — the dynamic filter drops
  all-fail groups, so kept-batch reward stays ~0.45 regardless of true success.
  So the agent is NOT improving; if anything degrading. Airline pivot also
  failing. perm=97, no OOM/NaN. user_stop still 94% of failures.

- 2026-07-28 ~00:45 CST — airline pt-i8ycxht7 KILLED (user dir). Stale monitor
  cron removed. Diagnosis (flat RL real cause = sparse binary reward + small
  batch + agent param-imprecision, not a bug) recorded in output/doc/EXP_QA.md.
  No active run; awaiting user direction on next step (eval SFT ceiling / KL
  anchor / denser reward / bigger batch).
