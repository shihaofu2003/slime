# tau2-opd-airline-retail-pilot

Purpose: validate a pure OPD Airline/Retail training path from SFT4505 before
any teacher-combination or OPD hyperparameter sweep. The task reward remains a
diagnostic; training uses one sample per prompt and the OPD KL signal.

2026-09-22 code/experiment review: [findings and reproduced evidence](REVIEW_20260922.md).
The review corrects the v3 "unbiased fix" explanation below, identifies completed-queue
backlog, and recomputes evaluation uncertainty and the full-run policy lag.

2026-09-22 implementation update: [current recipe and controls](../../../examples/tau2-bench/opd/README.md).
Defaults now use current-student logprobs + TIS, a total32-group prefetch budget,
LR2e-6/60 updates and the separate `pilot-current-bounded-lr2e6/` root.
Added domain diagnostics and a post-update logprob forward every10 updates.
CPU validation passed155 tests (OPD21, continuous119, argument validation15);
no new cluster run was submitted for this implementation.

## Inputs and topology

- Student HF: `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920/iter_0004505_hf`
- Student/reference torch root: the same checkpoint root, reference step 4505
- Airline teacher: `arms/async/20260920_sft4505-airline-train/checkpoints/iter_0000029_hf`
- Retail teacher: `arms/async/20260920_sft4505-retail-train/checkpoints/iter_0000009_hf`
- Data: 1,148 Airline + 563 Retail rows, merged without domain quotas
- Allocation: 16 GPUs total while running: independent four-GPU User service,
  four-GPU teacher task, and eight-GPU student task. Teachers publish endpoints
  through the mode-specific AFS directory; User publishes
  `user_endpoint.env`.

## Run records

The smoke and pilot are submitted sequentially. Each mode has a teacher and a
student job. The exact job-manager records are:

- User service: job `22555` (running), [run log](jobs/22555-tau2-opd-airline-retail-user-0921-211346094/run_0_20260921_211346094.log), endpoint `user_endpoint.env`.
- Superseded smoke teacher: job `22556` (succeeded), [run log](jobs/22556-tau2-opd-airline-retail-smoke-teacher-v3-0921-211650240/run_0_20260921_211650240.log); its preflight passed before the student test fix.
- Superseded smoke student: job `22558` (failed), [run log](jobs/22558-tau2-opd-airline-retail-smoke-student-v2-0921-212111626/run_0_20260921_212111626.log); two stale assertions failed.
- Smoke teacher: job `22570` (queued for cluster capacity), [run log](jobs/22570-tau2-opd-airline-retail-smoke-teacher-v4-0921-212519880/run_*_20260921_212519880.log).
- Smoke student: job `22572` (queued behind teacher), [run log](jobs/22572-tau2-opd-airline-retail-smoke-student-v3-0921-212555100/run_*_20260921_212555100.log); expected two updates and `iter_0000001`.
- Replacement smoke teacher: job `22608` (queued for cluster capacity), [run log](jobs/22608-tau2-opd-airline-retail-smoke-teacher-v5-0921-230545922/run_*_20260921_230545922.log).
- Replacement smoke student: job `22610` (queued behind teacher), [run log](jobs/22610-tau2-opd-airline-retail-smoke-student-v4-0921-230559781/run_*_20260921_230559781.log); this snapshot disables pure OPD only for the generic continuous preflight.
- Pilot teacher: pending smoke acceptance.
- Pilot student: pending smoke acceptance; expected 20 updates and `iter_0000019`.

## Pilot v2 (LR 1e-5, 60 updates)

Pilot v1 showed the mechanism is correct but undertrained: the teachers are
30–60-update RL fine-tunes of the same SFT init, so the per-token reverse KL
is ~0.002 and 20 updates at LR 2e-6 barely move the student (Airline gap to
teacher 11.25pp vs Retail 1.9pp at init). `opd_kl_coef` is a near no-op under
Adam (a global advantage scale cancels in `m_hat/sqrt(v_hat)`), so v2 keeps
`opd_kl_coef=1` and changes LR to `1e-5` and the budget to 60 updates. Artifacts
live under `pilot-v2-lr1e5/` via `TAU2_OPD_RUN_TAG`; fresh checkpoint root,
same teachers (airline iter29, retail iter9), same data, everything else
unchanged.

- Teacher: reuses the persistent teacher job `22758` (running since 2026-09-22
  09:33); both endpoints probed healthy before submission.
- User: reuses job `22555`.
- Student: job `22822`, [run log](jobs/22822-tau2-opd-airline-retail-pilot-v2-student-lr1e5-n60-0922-143948669/run_0_20260922_143948669.log);
  confirmed `--lr 1e-5 --num-rollout 60 --opd-kl-coef 1.0`, fresh load from
  SFT4505 iteration 4505. Expected 60 updates and `iter_0000059`, saves every
  10.
- Planned evaluation: official three-domain protocol at iter19/39/59, same
  wrapper as the iter19 evaluation below.

Pilot v2 training completed 2026-09-22 (60/60 finite updates, exit 0, mean
policy lag8.62 rising to13.28 in the last third, full-run review figures). raw_reward stayed flat
(0.44/0.46/0.44 by third) and `opd_reverse_kl` turned systematically negative
(-0.0022 in the last third). This unweighted diagnostic has conditional
expectation KL(behavior‖teacher) − KL(behavior‖current); the actual loss also
applies TIS, so its sign does not diagnose the gradient direction. Mean
grad_norm was0.59, versus0.23 for v1.

- HF conversion: jobs `22826`/`22827`/`22828` (iter19/39/59, all exit 0).
- Evaluations (identical wrapper/env as job `22800`, only MODEL_PATH/label
  changed): iter19 job `22829` ([log](jobs/22829-tau2-opd-v2-iter19-three-domain-eval-0922-152801228/run_0_20260922_152801228.log)),
  iter39 job `22830` ([log](jobs/22830-tau2-opd-v2-iter39-three-domain-eval-0922-152802091/run_0_20260922_152802091.log)),
  iter59 job `22831` ([log](jobs/22831-tau2-opd-v2-iter59-three-domain-eval-0922-152804811/run_0_20260922_152804811.log)).
  Each 400/400 sims, zero infrastructure errors, wall ~1500-1580 s.

**Pilot v2 evaluation results (2026-09-22, seed 300, three-domain).**
pass@1 / pass@4(any) / pass^4 (%). Teacher and SFT4505 rows come from the
seed300 four-domain expert evaluation with identical per-domain task counts
(airline 20, retail 40, telecom 40 tasks x 4 trials).

| Model | Airline | Retail | Telecom | Overall pass@1 |
|---|---|---|---|---:|
| SFT4505 (no OPD) | 42.50 / 65.00 / 30.00 | 58.75 / 77.50 / 35.00 | 47.50 / 82.50 / 12.50 | - |
| Airline teacher iter29 | 53.75 / 75.00 / 35.00 | 59.38 / 90.00 / 30.00 | 49.38 / 82.50 / 17.50 | - |
| Retail teacher iter9 | 42.50 / 65.00 / 30.00 | 60.63 / 87.50 / 30.00 | 47.50 / 80.00 / 15.00 | - |
| v1 iter19 (LR 2e-6, 20 upd) | 41.25 / 70.00 / 20.00 | 63.75 / 87.50 / 35.00 | 46.88 / 80.00 / 15.00 | 52.50 |
| v2 iter19 (LR 1e-5) | 43.75 / 70.00 / 15.00 | 50.62 / 65.00 / 30.00 | 54.37 / 80.00 / 20.00 | 50.75 |
| v2 iter39 (LR 1e-5) | 48.75 / 65.00 / 25.00 | 44.38 / 60.00 / 27.50 | 48.12 / 77.50 / 10.00 | 46.75 |
| v2 iter59 (LR 1e-5) | 46.25 / 70.00 / 20.00 | 39.38 / 60.00 / 12.50 | 51.25 / 80.00 / 20.00 | 45.50 |

Observations:

- Airline finally moves above SFT at every v2 checkpoint (43.75 / 48.75 /
  46.25 vs SFT 42.50; v1 was 41.25). Best is iter39 at 48.75 (+6.25pp over
  SFT), still 5.0pp below the teacher (53.75). The 5x LR does deliver the
  larger policy movement that v1 lacked.
- Retail is destroyed monotonically: 50.62 -> 44.38 -> 39.38 (iter59 is
  -19.4pp vs SFT and -21.3pp vs the retail teacher; v1 was 63.75, above
  teacher). The airline gain is bought with a retail collapse.
- Telecom (held-out) fluctuates 48.1-54.4 with no trend, consistent with
  single-seed noise at 40 tasks.
- Interpretation after review: high LR and a growing completed queue coexist
  in v2. These runs do not isolate their effects. TIS was enabled, so a negative
  unweighted sampled log-ratio does not establish an incorrect training gradient.
- Historical decision: v3 tested the behavior-logprob surrogate at the same
  LR1e-5. The later review found that this changes the objective under lag;
  it should not be interpreted as a debiased counterpart to v2.

**Historical OPD v3 behavior-logprob experiment; superseded default.**

`--opd-use-behavior-logprobs` substitutes behavior-policy logprobs in the
advantage while retaining current-policy PPO/TIS weighting. Under policy lag
this is a different surrogate, not an unbiased current-student reverse-KL fix.
The flag remains available for reproducing v3 and is now disabled by default.
The corrected recipe retains four async environment processes and applies
backpressure to the total admitted group count, including completed groups.

- v3 student: job `22858`, [run log](jobs/22858-tau2-opd-airline-retail-pilot-v3-student-behkl-lr1e5-n60-0922-161857549/run_0_20260922_161857549.log);
  confirmed `--opd-use-behavior-logprobs --lr 1e-5 --num-rollout 60`, fresh
  root `pilot-v3-behkl-lr1e5/`, reusing teacher `22758` and User `22555`.
  Direct A/B against v2 at equal LR; planned eval at iter19/39/59.

## Evaluation

After pilot acceptance, evaluate SFT4505 and OPD iter19 with the native
Airline/Retail official protocol (seed 300, test split, four trials,
temperature 0.6, max steps 200). Record domain and overall pass@1,
pass@4(any), pass^4, plus action/DB diagnostics and the training log metrics.

## OPD iter19 evaluation (2026-09-22)

Job `22800` (8 GPUs, normal priority) ran the fixed three-domain official
evaluation against the OPD iter19 HF checkpoint
(`pilot/checkpoints/iter_0000019_hf`). Earlier attempts `22767` (4 GPUs,
external-User wrapper) and `22768` (8 GPUs, external-User wrapper) failed at
startup because `USER_REPLICA_CUDA_GROUPS` was incorrectly injected when
`USER_SGLANG=0`; job `22800` switched to the in-job Agent+User topology used
by the matched SFT baseline and succeeded end-to-end (wall 1374 s, zero
infrastructure errors).

Evaluation protocol is unchanged from the SFT baseline: test split, seed 300,
four trials, max_steps 200, Agent temperature 0.6, `max_concurrency_per_domain
3`, `global_concurrency 9`. Telecom is added on top of the trained
Airline/Retail pair as a held-out transfer check; Banking is excluded from
this run.

### Domain metrics

| Domain | Tasks | Sims | pass@1 | pass@4(any) | pass^4 | action_acc | db_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Airline | 20 | 80 | 41.25% | 70.00% | 20.00% | 0.500 | 0.438 |
| Retail | 40 | 160 | 63.75% | 87.50% | 35.00% | 0.896 | 0.675 |
| Telecom | 40 | 160 | 46.88% | 80.00% | 15.00% | 0.676 | 0.230 |

### Overall (100 tasks × 4 trials = 400 sims)

| Metric | Value |
|---|---:|
| pass@1 | 52.50% |
| pass@4(any) | 81.00% |
| pass^4 | 24.00% |
| action_accuracy | 0.750 |
| db_accuracy | 0.451 |
| infrastructure_errors | 0 |

### Comparison vs SFT4505 baseline (same protocol, four-domain run)

| Domain | SFT pass@1 | OPD iter19 pass@1 | Δ | SFT pass@4 | OPD iter19 pass@4 | Δ | SFT pass^4 | OPD iter19 pass^4 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Airline | 42.50% | 41.25% | -1.25 | 65.00% | 70.00% | +5.00 | 30.00% | 20.00% | **-10.00** |
| Retail | 58.75% | 63.75% | **+5.00** | 77.50% | 87.50% | **+10.00** | 35.00% | 35.00% | 0.00 |
| Telecom | 47.50% | 46.88% | -0.62 | 82.50% | 80.00% | -2.50 | 12.50% | 15.00% | +2.50 |

Termination sanity: Airline 80/80 user_stop; Retail 154/160 user_stop, 6
error; Telecom 152/160 user_stop, 8 max_steps. No infrastructure failures.

Observations:

- Retail improves clearly under OPD: +5.00 pass@1, +10.00 pass@4(any), pass^4
  unchanged. The Retail teacher signal transfers into the student.
- Airline pass^4 drops by 10 points (30.00% → 20.00%) despite pass@4(any)
  rising by 5. This is a consistency regression: the policy becomes less
  stable across the four trials even though at least one success becomes more
  likely. Airline write-action correctness is also low (7/92 ≈ 7.6%).
- Telecom (held-out, no OPD supervision) is roughly flat, suggesting the
  update does not catastrophically forget an unseen domain but also does not
  transfer.
- Airline consistency is the main target before further OPD scaling;
  consider reweighting Airline rows or adding a consistency regularizer.

### Artifacts

- Run log: [jobs/22800-tau2-opd-three-domain-eval-iter19-fixed-8gpu-0922-124717598/run_0_20260922_124717598.log](jobs/22800-tau2-opd-three-domain-eval-iter19-fixed-8gpu-0922-124717598/run_0_20260922_124717598.log)
- Summary JSON: [`../tau2-eval-qwen36-user-async-timed/eval/full/seed300_0922_044907_summary.json`](../tau2-eval-qwen36-user-async-timed/eval/full/seed300_0922_044907_summary.json)
- Evaluated checkpoint: `pilot/checkpoints/iter_0000019_hf`
