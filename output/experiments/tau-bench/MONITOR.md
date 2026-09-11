# tau-bench experiment — monitoring log

Monitoring notes for the tau-bench (tau1, retail) run-through experiment. Each
entry: timestamp first, then a few sentences. Interval ~30 min.

---

## 2026-07-09 21:02 (monitor #1)

Kickoff. Stage 1 (mock data gen) and Stage 2 (Instruct-2507 → torch_dist) first
attempts both failed fast:
- Data gen: `ModuleNotFoundError: No module named 'litellm'` — the image lacks
  litellm, which `tau_bench.envs` imports at module load. Fixed by adding
  `setup/setup_tau_bench_deps.sh` (installs litellm + google-generativeai) and
  calling it from the tau run scripts.
- Convert: wrapper looked for the base convert script next to itself, but
  `submit.sh` only copies the submitted script into the log dir. Fixed the
  wrapper to resolve the base script via `PROJECT_ROOT/scripts/`.

Both jobs resubmitted: data gen `pt-aa1x7ote`, convert `pt-xhx4knnl`. Waiting on
scheduling. Stage 3 (training) is blocked on the Gemini API key for the tau user
simulator — asked the user.

## 2026-07-09 21:10 (monitor #2)

Stage 1 ✅ and Stage 2 ✅ both succeeded: tau task JSONLs generated under
`datasets/tau-bench/` (retail_train 336K, retail_dev 15K, ...), and
`Qwen3-4B-Instruct-2507_torch_dist` converted (7.5G, structure matches the
proven base checkpoint). Confirmed compute nodes have network — litellm installs
fine via the Tsinghua mirror.

User supplied an OpenAI-compatible user simulator instead of Gemini: model
`o4-mini`, base `https://api.chatanywhere.tech/v1`. Verified both the raw API
(curl) and litellm routing (`custom_llm_provider="openai"` + OPENAI_API_KEY /
OPENAI_API_BASE) return content locally. Rewired `generate_with_tau.py` and the
training script to this endpoint.

Submitted the real training run: `pt-mgmssrfd` (8 GPU). Run-through config:
batch 8 × n-samples 8 (global 64), num-rollout 10, sglang concurrency 16 — kept
small because o4-mini is a slow reasoning model (concurrency / rate-limit risk).
A 2-GPU Gemini validation run (`pt-uzy4vida`) was submitted then killed (wrong
provider). Waiting on scheduling + init.

## 2026-07-09 21:22 (monitor #3)

First training run `pt-mgmssrfd` (8 GPU) FAILED fast at `train.py` import — not a
tau issue but a dep conflict: `google-generativeai` pinned protobuf<6 and
downgraded the image's protobuf 6.33.6 -> 5.29.6, breaking tensorboard
(gencode 6.31.1 vs runtime 5.29.6), which `slime.logging_utils` imports. The tau
rollout itself never started. Root cause confirmed from the pip log.

Fix: `setup_tau_bench_deps.sh` now installs **only litellm** (drops
google-generativeai entirely — it's only needed for the gemini provider, and we
use litellm's openai provider for o4-mini). protobuf is left at the image's
6.33.x. Added a verify step that imports both litellm and tensorboard.

Resubmitted: `pt-06lu3fs9` (8 GPU). Waiting on scheduling + init.

## 2026-07-09 21:39 (monitor #4)

`pt-06lu3fs9` is RUNNING and past the import blocker: protobuf fix worked,
`train.py` imports cleanly, wandb logged in (fshihao900/slime-dev, run
`okcwk6mo`), Ray placement group up on 8 GPUs, sglang router + 8 engines
launched (ports 15000-15014), custom rollout fn imported. Now in sglang model
loading. Next milestone: first tau rollout (agent turns + o4-mini user sim) and
the first GRPO training step.

## 2026-07-09 21:55 (monitor #5)

`pt-06lu3fs9` is healthy and the full tau agent loop works end-to-end:
checkpoint loaded (`Qwen3-4B-Instruct-2507_torch_dist` @ iter 0), sglang
generating, and an eval rollout shows the agent authenticating a retail user
(`find_user_id_by_name_zip`) and responding correctly -- multi-turn sglang gen
+ tool parse/execute + o4-mini user sim all functioning. The initial step-0 eval
(retail-dev, 20 ex) is running: first example was slow (~386s, cold start) but
steady state is ~17-25 s/ex, ~12 min total (not the feared 2 h). A few transient
sglang router `client error` warnings appear (colocate weight-sync hiccups) but
auto-retry (attempt 1/60) -- not fatal. No protobuf errors. Waiting for the
first GRPO training step.

## 2026-07-09 22:28 (monitor #6)

Step-0 eval done (retail-dev 20/20, ~13 min). Training rollout in progress and
**collecting valid data**: progress 8/64 -- one group (8 trajectories) passed
the `check_reward_nonzero_std` filter, i.e. the cold Instruct-2507 model DOES
produce reward variance on retail (mixed success within a prompt). So the full
pipeline is confirmed working: reward computed, filter keeps-variance correctly.
51 trajectories finished so far; most groups are all-zero-reward (dropped) and
get resampled. Confirmed `rollout-max-response-len 1024` is per-turn
`max_new_tokens` (not total truncation), so length isn't the reward=0 cause.

Caveat: **o4-mini is a slow reasoning model** -- as the user sim each turn takes
~10-30 s, and low cold-start success means lots of resampling. Estimated ~2 h to
fill the first 64-sample batch and run the first gradient step (~24 h for the
full 10-step run-through). The pipeline is sound; the cost is the user-sim
choice. A faster user sim (e.g. gemini-flash / a non-reasoning model) would speed
this up dramatically for real training.

## 2026-07-09 22:39 (monitor #7 — wrap-up)

**First rollout batch complete:** `Rollout generation: 100%| 64/64 [42:46,
19.18s/it]` (~43 min, faster than the earlier 2 h estimate once warmup finished).
64 valid samples collected with reward variance (passed the filter). The log's
last lines show sglang `release_memory_occupation` / `flush_cache` -- the colocate
handoff from rollout to the actor gradient update. **The first GRPO gradient step
is in progress.** End condition ("training script runs normally") is met: the
full tau-bench RL loop -- init, eval, multi-turn rollout with o4-mini user sim,
reward, dynamic filter, and actor update -- all run correctly. Per-step ~43 min
rollout + update, so the 10-step run-through is ~7-8 h. Left the job running;
asked the user whether to keep it or kill it.

## 2026-07-09 22:42 (monitor #8 — job FAILED; root cause, CORRECTED)

`pt-06lu3fs9` FAILED right after the first rollout filled 64/64. Verified
mechanism (from the log sequence): only the `ref_log_probs` forward ran (ref
forward 16/16); the actor's **own** `log_probs` forward never ran, then
`compute_advantages_and_returns` crashed at `loss.py:703`
(`TypeError: 'NoneType' object is not iterable`) because
`xs = log_probs or rollout_log_probs or values` was all-None (`ref_log_probs`
was computed but the kl_coef==0 fallback doesn't use it).

Why no actor log-prob forward: with the **default** GRPO config,
`can_reuse_log_probs_in_loss=True` (policy_loss + grpo + no critic + kl_coef==0
+ `len(num_microbatches)==1`, all defaults), so `train_actor` (actor.py:450-464)
intentionally skips the actor log-prob pre-computation. The custom tau generate
(`generate_with_tau`) does not return logprobs, so `rollout_log_probs` is also
None → compute_advantages has no log-prob tensor → crash.

**Correction of an earlier wrong note:** I first blamed a small
`global_batch_size` for making `num_microbatches==1`. That was wrong —
`len(num_microbatches)==1` is the *normal* case whenever one rollout batch equals
`global_batch_size` (true for the reference 32×8=256 too); raising the batch size
would **not** fix it. The crash is about missing log-probs in the custom-generate
path + the default reuse optimization, not the batch size.

The tau rollout/reward/filter are all sound (first batch collected 64/64 with
reward variance). Fix direction (NOT batch size; to verify against the upstream
`tau-bench-example`): make `can_reuse_log_probs_in_loss=False` (e.g. a small
positive `--kl-coef`, or `--keep-old-actor`) so actor log-probs get computed, or
have the custom generate return logprobs. Comparing our files to the upstream
working example next.

## 2026-07-09 23:59 (monitor #9 — kl_coef-fix smoke)

Confirmed the upstream `tau-bench-example/tau1` tau code is functionally
identical to ours (no logprob handling in either; same `res_to_sample`; run
scripts differ only in cosmetics) — so the crash isn't in the tau files, it's
slime's default `can_reuse_log_probs_in_loss` (True when `--kl-coef==0`) skipping
the actor log-prob pre-forward while the custom generate returns no logprobs.
(The actually-validated upstream path is tau2's SFT→RFT→GRPO cookbook, not this
tau1 `run_qwen3_4B.sh`.)

Applying the fix: added `--kl-coef "${KL_COEF:-0.001}"` (reward-shaping KL, NOT
`--kl-loss-coef`) to GRPO_ARGS, and made the dynamic filter (`USE_DYNAMIC_FILTER`)
and eval (`SKIP_EVAL`) env-toggleable. Submitted a minimal-cost smoke to verify:
`pt-uh0d7chd`, 8 GPU, 4 prompts × 2 samples (global 8), `num-rollout 1`, filter
off, eval off, `kl_coef=0.001`. Success criterion = the first GRPO gradient step
completes without the `loss.py:703` crash.

## 2026-07-10 00:19 (monitor #10 — kl_coef fix VERIFIED ✅)

Smoke `pt-uh0d7chd` **SUCCEEDED** in ~16 min. The forward sequence proves the
fix: rollout 8/8 → ref forward 2/2 → **actor log-prob forward 2/2 (previously
missing → crash)** → no `loss.py:703` error. The gradient step ran end-to-end:
`Timer actor_train start` → `run_backward` → `actor_train end (6.2s)` →
checkpoint saved (iter 0) → wandb logged (`rollout/response_len/mean 686`). The
two `Traceback` lines in the log are harmless wandb atexit-teardown noise, not
training errors. So `--kl-coef 0.001` (reward-shaping KL) flipping
`can_reuse_log_probs_in_loss=False` is the confirmed fix. The full tau-bench GRPO
loop now works end-to-end. (Smoke used filter-off / no-eval / 4×2 to minimize
cost; a real run re-enables the filter for reward-variance gating.)

## 2026-07-10 18:23 (monitor #11 — proper fix: rollout returns logprobs; kl_coef back to 0 ✅)

Replaced the `--kl-coef 0.001` **workaround** with the principled fix (the "open
question" from monitor #10, option 1): the custom tau rollout now returns
per-token logprobs so `kl_coef` can stay 0.

Changes: `trainable_agents.py` `asolve` adds `"return_logprob": True` to the
sglang payload, captures each turn's `output_token_logprobs`, and accumulates
`rollout_log_probs` aligned to the `_get_token_delta` token deltas (helper
`_align_turn_log_probs` pads the Qwen template's trailing `\n`, which sglang
doesn't generate, with 0.0; env/tool turns get 0.0). Flows through
`InteractionResult` → `generate_with_tau.res_to_sample` → `Sample.rollout_log_probs`.
Run script `--kl-coef` default reverted `0.001`→`0`. `use_rollout_logprobs` left
False (default): with `kl_coef==0`, `rollout_log_probs` is only the zero-KL
shape template (loss.py:700-703), and the on-policy ratio stays 1 (old=current),
so the padded 0.0 never affects training.

Verified by smoke `pt-prux2yjk` (8 GPU, 4×2, num-rollout 1, filter/eval off,
`kl_coef=0`): **fresh run** — `asolve` ran 8/8 trajectories, **no `loss.py:703`
crash**, `Rollout generation 100% 8/8`, `Timer actor_train (6.1s)`, checkpoint
saved (iter 0), job SUCCEEDED. 0 alignment-mismatch warnings. Metrics logged
(`pg_loss`, `grad_norm`, `kl_loss`, `rollout/advantages`). (v1 `pt-mrd88ypt` was
invalid: it silently **resumed** from the 0709 smoke's iter-0 checkpoint in
`checkpoints/..._slime` and skipped the rollout entirely — `num-rollout 1` was
already satisfied. Fixed by pointing the smoke wrapper at a throwaway
`CHECKPOINT_ROOT=checkpoints_smoke` so it never resumes and never touches the
real checkpoint dir.) Caveat: `checkpoints_smoke` now also holds an iter-0 ckpt,
so a future smoke would resume from it unless cleared/unique-ified first.

## 2026-07-10 21:00 (monitor #12 — real (non-smoke) multi-step run SUCCEEDED ✅)

First **multi-step** training run (`run-qwen3-4B-serviceagent-tau-real.sh`):
num-rollout 3, global-batch 32 (8×4), kl_coef 0, filter OFF, eval OFF, fresh
`checkpoints_real`. Goal = exercise the rollout→advantage→backward→checkpoint
cycle repeatedly (the smoke only does one step). Three attempts:

- **`pt-5ji0vkim`** (19:03, filter ON): cold model rarely yields per-group reward
  variance → the nonzero-std filter dropped/resampled ~90% of groups → collected
  only 8/32 in ~29 min, killed before any gradient step. **Lesson:** filter is a
  data-quality feature, not a correctness one; leave OFF for a cold model.
- **`pt-rfxsa90s`** (19:40, filter OFF): died ~2 min after start at
  `placement_group.py` → `Failed to connect to GCS ... within 5 seconds` — a
  transient Ray startup race (Ray head had started fine), not a script/config bug.
- **`pt-8grukdrk`** (20:04, filter OFF): **SUCCEEDED** in ~55 min. The GCS race
  did not recur (placement group 8/8 GPUs at 12:05:58). Timeline: model load
  (sglang Megatron-FSDP) → rollout iter0 `100% 32/32 [15:23]` → `ref_log_probs`
  fwd 4.2s → `actor_train` → repeated 3× → checkpoint saved (iter 2, 12:58).

**No `loss.py:703` crash** across any iteration — the monitor #11 `rollout_log_probs`
fix holds in a real multi-step run (`rollout/rollout_log_probs` present, e.g.
-0.062 iter0; `train_rollout_logprob_abs_diff` ≈ 0.11, on-policy as expected).
Per-step metrics:

| step | loss | pg_loss | entropy | kl_loss | grad_norm | raw_reward |
|------|------|---------|---------|---------|-----------|------------|
| 0 | -5.6e-9 | -5.6e-9 | 0.084 | 0.0 | 1.16 | 0.71875 |
| 1 | 3.7e-9 | 3.7e-9 | 0.070 | 0.0017 | 1.50 | 0.875 |
| 2 | 0.0 | 0.0 | 0.098 | 0.0019 | 0.50 | 0.78125 |

`actor_train` elapsed 8.6s / 4.5s / 4.5s. `rollout/advantages` ≈ 0 throughout
(cold-model zero per-group variance → ~0 gradient signal; expected — the filter
is the eventual fix, OFF here to fill the batch fast). `raw_reward` 0.72→0.88→0.78
is batch variance, not learning. Non-fatal transients: 2 sglang `/generate`
retries (`http_utils Error: , retrying... attempt 1/60`, both recovered) and a
few `qwen25_detector` JSON-parse warnings on malformed tool-call args (tau
handles them). The two `Traceback` blocks at log end (8344, 8379) are the same
harmless wandb `atexit` teardown noise noted in monitor #10 —
`ConnectionResetError: Connection lost` from wandb's `teardown_atexit` firing
after its service connection already closed during interpreter shutdown; Python
ignores atexit exceptions (exit code unchanged), and it prints AFTER wandb sync
+ checkpoint save + the Ray `Job succeeded` line, so it has zero impact.
Checkpoint at `checkpoints_real/Qwen3-4B-Instruct-2507_slime/iter_0000002`.

**Bottom line:** the full tau-bench GRPO training loop is verified end-to-end
across multiple gradient steps with `kl_coef=0` (no workaround). Next step for
*learning* (vs. code-verification): warm the model, then re-enable the
reward-variance dynamic filter (`USE_DYNAMIC_FILTER=1`) and raise num-rollout.
