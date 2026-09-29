# tau2-mix-rl-sft4505-credit-signal-v1-20260927

Purpose: test progress-db-count-v1 group selection using the actual combined credit, preserving the original implementation and experiment directories. On 2026-09-27 the user requested a fresh 4-GPU persistent User first, followed by 8-GPU mix RL from SFT4505.

2026-09-27 [Telecom diagnosis](telecom_diagnosis/README.md): 205/271 training tasks default to DB scoring; 136 of those reference states are inconsistent because golden tool calls omit Agent/User synchronization. Recorded affected trajectories receive zero reward in this and the historical runs. A replay also reproduces a `2` versus `2.0` bill-description mismatch. Training was not changed during this analysis; low Telecom training reward must be interpreted with this scoring issue in mind.

## Behavior

[New producer/filter](../../../examples/tau2-bench/rl/credit_signal_v1.py) first runs the existing sample-usability checks, then computes the unchanged outcome + progress + format token advantages. A complete K=8 group is retained when any training token has a nonzero advantage. This includes all-failed and all-successful groups with effective credit; identical progress with zero normalized advantage is discarded. Removed/over-cap groups remain excluded. Unsupported progress reward bases retain the original outcome/format fallback.

Credit is computed once when the group completes. The new reward postprocessor preserves binary reward reporting and group outcome normalization while reusing prepared token advantages. The original training-side credit function, raw behavior tokens, masks, TIS and segment expansion remain in use. `credit.jsonl` now describes all usable candidate groups, including zero-signal rejects and unconsumed ready groups; it is not restricted to trained groups. `group_has_signal` distinguishes credit retention, and `tau2_pool_batch` logs identify groups actually trained.

[Separate training entry](../../../train_tau2_credit_signal.py) selects the new producer/filter/postprocessor before argument parsing so runtime configuration logs show the selected paths. The collector override preserves the original producer's admission, completion order, replacement accounting and error propagation. Legacy source files, launchers and CI files are unchanged.

## Configuration and monitoring

[Launcher](run_mix.sh) uses the same SFT4505 initialization, fresh RL optimizer, 2524-task pool, random draws with replacement, User endpoint, LR 2e-6, K=8, batch 128, progress/format weights 1/1, gamma 0.98, KL/entropy 0, seeds 1234/42, pool 32, pending 64, unlimited lag and four environment workers. The unchanged underlying launcher retains Trainer2 + three TP2 Agent generators, rollout limits and its existing preflights.

Default budget: 140 updates / 17,920 accepted trajectories, matching the controls' optimizer budget. Total generated trajectories can change when filtering changes; use `groups.csv` and producer draw counts to report that cost. This preset does not claim an identical total generation budget. Official task success and the existing held-out evaluation protocol remain the selection criteria for this experimental recipe.

All outputs use this new experiment directory. Run logs append on resume. The diagnostic path is resolved inside the base launcher after its CPU preflight, keeping test records out of `credit.jsonl`.

[New monitor](../../../scripts/plot_tau2_credit_signal_rewards.py) reads compact `tau2_credit_group` log records instead of repeatedly parsing full conversations. It writes:

- `groups.csv`: every completed group's domain, official reward, outcome class, keep/drop decision and reason.
- `reward_prefilter_by_domain.csv`: per-domain reward/counts, fixed task-pool-weighted reward, group-based standard error, all-zero/all-one/zero-signal/unusable rates and retention.
- `reward_accepted.csv`: reward and domain group counts for completed optimizer updates.
- `reward_diagnostics.png`: prefilter weighted/per-domain curves, accepted reward and outcome/retention curves. Default smoothing uses 400 complete groups; bands are approximate group-based 95% intervals, not held-out confidence intervals.

## Use and validation

Preview the configuration without starting services or training:

```bash
bash output/experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/run_mix.sh --dry-run
```

Training launcher, when an experiment run is requested:

```bash
bash scripts/submit.sh --experiment tau2-mix-rl-sft4505-credit-signal-v1-20260927 \
  --gpus 8 output/experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/run_mix.sh
```

For this submission, [run_user.sh](run_user.sh) launches two TP2 Qwen3.6-27B nonthinking User replicas and publishes this experiment's `user_endpoint.env`. Training receives that path through `TAU2_USER_ENDPOINT_FILE`, overriding the launcher's historical default. Training outputs go to `arms/async/20260927_credit-signal-v1-train/`, and plots to `plots/20260927_credit-signal-v1-train/`. No RL checkpoint existed in that output directory before submission; initialization is the selected SFT `iter_0004505_hf` with reference step 4505 and a fresh optimizer.

[CPU preflight](../../../examples/tau2-bench/rl/test_credit_signal_v1.py): 23 tests passed locally with Python 3.12.14 and CPU Torch 2.6.0, and again inside the training container. Coverage includes uniform/mixed outcomes, zero effective progress, configured weights, format-only credit, unavailable progress, unusable groups, exact legacy credit equivalence through segment expansion, single computation/dump, ready-pool accounting, exception propagation, entry-point routing, launcher parameters and plot generation. Shell syntax and configuration dry-run passed. GPU/Ray integration completed its first optimizer update in job 24446.

```bash
python3 examples/tau2-bench/rl/test_credit_signal_v1.py
```

This test is deliberately a standalone experiment preflight, executed by the new launcher. It is not registered in the fixed CI matrix because the requested change preserves all existing files.

## Jobs

| Job | GPUs | Purpose / status | Log |
|---|---:|---|---|
| 24445 (`pt-gjgd530g`) | 4 | Persistent Qwen3.6-27B User; submitted at 2026-09-27 11:18 CST; RUNNING and ready | [Run log](jobs/24445-tau2-credit-signal-user-persistent-0927-111849148/run_0_20260927_111849148.log) |
| 24446 (`pt-zjmjvx61`) | 8 | Credit-signal-v1 mix RL from SFT4505; SUCCEEDED, 140 updates complete and iter139 saved | [Run log](jobs/24446-tau2-credit-signal-mix-sft4505-0927-112507511/run_0_20260927_112507511.log) |

Both jobs use normal priority and normal admission. The new User endpoint `http://10.119.96.197:30000/v1` returned HTTP 200 with the expected model before training submission. [Service logs](../tau2-external-user-pool/service/20260927_031934/). Training uses 140 updates, LR 2e-6 and batch 128 from SFT4505, with the new credit-signal filtering path.

Runtime verification: all 23 new CPU tests passed in the training container. Resolved arguments select `credit_signal_v1.CreditSignalProducer`, `credit_signal_v1.filter_group` and `credit_signal_v1.post_process_rewards`; `finetune=True`, `no_load_optim=True`, `no_load_rng=True`, `start_rollout_id=0`, loading SFT checkpoint step 4505. GPU allocation is Trainer2 + three TP2 Agent generators. [Training log](arms/async/20260927_credit-signal-v1-train/run.log).

Optimizer update 0 completed at 2026-09-27 11:37:10 CST: loss −0.0733078, gradient norm 0.565317, LR 2e-6, batch 128; all finite. Live group logs confirm all-zero and all-one groups with credit are retained, zero-credit groups replaced, and unusable groups rejected. Initial Agent HTTP read errors entered the existing fresh-connection retry path and did not stop the update. User and training jobs remain RUNNING.

Status check at 2026-09-27 13:25 CST: both User replicas return HTTP 200 on `/health`, and the router returns the expected model. Training completed updates 0–59, with checkpoints iter9/19/29/39/49/59 saved. Latest loss −0.129449, gradient norm 0.633198, reference K2 KL 0.009008; latest-ten means are gradient norm 0.594125 and K2 KL 0.006229. No non-finite loss/gradient or OOM/RayTaskError was found. Recorded HTTP read retries and over-cap resampling remain in the existing recovery paths.

Of 1,073 completed groups, 977 were kept; 70 had zero credit, 13 contained removed samples and 13 were permanently over-cap. The 960 trained groups comprise 578 mixed-outcome, 202 all-one and 180 all-zero groups. Binary zero-variance groups therefore account for 39.79% of training groups by design; their credit is nonzero. Checked 8,296 complete credit records from 1,037 groups: no malformed/duplicate records, non-finite advantages or signal-flag mismatches; maximum outcome-normalization error 1.14e−7 and credit-sum error 8.05e−16.

Fixed-domain prefilter reward rises from 52.58% in the first 400 completed groups to 55.98% in the latest 400; Airline is approximately flat while Retail/Telecom/Banking rise. This is a training diagnostic, not an official evaluation result. The [new plots](plots/20260927_credit-signal-v1-train/reward_diagnostics.png) and CSVs refresh every five minutes. Watch asynchronous lag: mean earliest-turn lag 3.05 updates, maximum 22 (batch45, whose longest group took 2,475 seconds); latest-ten mean lag 3.33. TIS weights remain near 1 with very low clipping, so these lag metrics alone do not establish instability. No training parameters were changed during this check.


## Final checkpoint official evaluation

[Launcher](eval_iter139.sh) evaluates training job 24446's final `iter_0000139` (140 updates), converting it to adjacent `iter_0000139_hf` first. It copies the previous vanilla iter139 four-domain launcher, changing only checkpoint root and model/output labels. Shell syntax and conversion input dry-run passed.

Unchanged official protocol: all 197 test tasks across Airline/Retail/Telecom/Banking Knowledge, 4 trials each (788 simulations), seed 300, BM25, max_steps 200, max_errors 10, Agent temperature/top_p/max_tokens 0.6/1/1200. Eight GPUs: two TP1 Agent replicas and three TP2 Qwen3.6-27B nonthinking User replicas; User temperature 0 and max_tokens 512. Domain concurrency 1/2/2/4, global 9, completed-domain slot borrowing enabled.

| Job | GPUs | Purpose / status | Log |
|---|---:|---|---|
| 24720 (`pt-6rgnruwc`) | 8 | Final iter139 four-domain full official eval; SUBMITTED on 2026-09-27 21:12 CST, normal priority | [Run log](jobs/24720-tau2-credit-signal-v1-iter139-four-domain-eval-0927-211230250/run_0_20260927_211230250.log) |

Cluster accepted the evaluation with elastic admission. Outputs: `eval/credit-signal-v1-iter139-four-domain/`. The run log appears after the container starts.
