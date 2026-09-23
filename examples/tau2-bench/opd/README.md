# tau2-bench domain-expert pure OPD

Train SFT4505 from student-generated trajectories scored by the corresponding
domain expert. Task success is a diagnostic; the training reward is
zero and the policy signal is the per-token OPD advantage.

## Recipe and loss

Student: `Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920/iter_0004505_hf`,
Megatron reference step4505. Teachers: Airline iter29 and Retail iter9 under
`output/experiments/tau2-domain-experts-sft4505-b128/arms/async/`.
Data: 1,148 Airline + 563 Retail tasks, shuffled with seed42, without domain quotas.

Defaults: LR2e-6, 60 updates, batch16, K1, save every10, temperature1/top_p1,
max_steps200, token cap16384, two over-cap retries. Task reward, reference-KL
loss coefficient and entropy coefficient are zero; turn credit, advantage
whitening and zero-variance group replacement are disabled.
`TRAIN_SEED` and `ROLLOUT_SEED` are overridable; defaults are1234 and42.

The default domains remain Airline/Retail. Set
`TAU2_OPD_DOMAINS=airline,retail,telecom,banking` for four-domain training and
teacher serving. This adds Telecom iter9 and Banking iter9 from the same expert
experiment,with teacher URLs `TAU2_OPD_TELECOM_URL` and `TAU2_OPD_BANKING_URL`.
Banking metadata aliases `banking` and `banking_knowledge` use the same expert.
The data merger accepts `--telecom` and `--banking` processed JSONL inputs;
all four sources contain2524 tasks. The
[full four-domain run](../../../output/experiments/tau2-opd-four-domain-20260923/README.md)
uses `TAU2_OPD_NUM_UPDATES=160` for approximately one source-sized trajectory budget.

The advantage is `log(teacher) - log(current student)`. The PPO ratio uses
train-side logprobs; detached TIS weights `clamp(current/behavior, 0, 2)` correct
the sampled token distribution. `TAU2_OPD_USE_BEHAVIOR_LOGPROBS=1` reproduces the
historical v3 surrogate, which changes the objective under policy lag. It is
off by default. Unweighted sampled log-ratios may be negative; they are not
fresh-current-policy KL estimates.

Teacher requests use each segment's original token context and align the input
logprobs to its response suffix. Only generated Assistant tokens enter the loss.
Multiple segments retain one whole-trajectory denominator. `OPD_KL_COEF` defaults
to1; Adam makes global loss scaling a poor substitute for changing LR, with
departures from scale invariance due to epsilon and gradient clipping.

## Async sampling and diagnostics

The student job uses Trainer2 + Generators6 GPUs and four environment processes.
A separate four-GPU teacher job serves two TP2 experts; a four-GPU User service
serves Qwen3.6-27B. Total:16 GPUs. Teacher endpoints are shared through
`teacher_endpoints.env` in the experiment root. The teacher service is persistent
by default; `TAU2_OPD_TEACHER_PERSISTENT=0` enables the student stop marker.
Four-domain serving uses an8-GPU teacher job with one TP2 expert per domain;
User4 + teachers8 + student8 use20 GPUs in total.

`TAU2_POOL_CAPACITY=32` limits concurrent dialogues, while
`TAU2_MAX_BUFFERED_GROUPS=32` counts generating, ready and currently training
groups together. Admission pauses when that total is full, including on resume.
This controls queue-induced lag while retaining async overlap. It does not bound
the policy-version span of a long conversation. The four-process path retains
`TAU2_MAX_POLICY_LAG=-1`; a finite lag limit requires the existing one-process path.
Pool size, total buffer size, environment workers, sampling mode and lag are
overridable. `TAU2_MAX_BUFFERED_GROUPS=-1` restores the historical prefetch behavior.

`rollout/opd/<domain>/` records raw task success, earliest/latest/token-weighted
lag, sampled current/teacher and behavior/teacher log-ratios, TIS-weighted
current/teacher log-ratio, TIS clipping and train/rollout logprob difference.
Token metrics use whole-trajectory loss weights and correctly combine split
segments across DP/CP ranks. Check `rollout/trained_groups_<domain>` for coverage;
a domain absent from the batch has zero-count training metrics.
`rollout/ready_groups` and `rollout/buffered_groups` expose queue occupancy.

Every10 updates, `OPD_POST_UPDATE_LOG_INTERVAL=10` recomputes the batch's logprobs
after the optimizer step and reports `opd/post_update_logprob_abs_diff` and
`opd/post_update_tis_weighted_k2`. This costs one additional logprob forward at
those intervals. K2 here is a policy-change diagnostic on collected contexts,
not an exact KL on fresh post-update trajectories. Set the interval to0 to
disable the extra forward. These diagnostics are not termination gates.

## Launch and controlled comparisons

The default run tag is `current-bounded-lr2e6`, giving fresh roots
`smoke-current-bounded-lr2e6/` and `pilot-current-bounded-lr2e6/` under the experiment.
Reusing a tag resumes that run. Use a distinct `TAU2_OPD_RUN_TAG` for each new
recipe; an explicitly empty tag selects the historical untagged root.

Submit the independent User service once, then teacher and smoke student:

```bash
bash scripts/submit.sh --experiment tau2-opd-airline-retail-pilot --gpus 4 \
  -e USER_ENDPOINT_FILE=/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2/output/experiments/tau2-opd-airline-retail-pilot/user_endpoint.env \
  scripts/serve_tau2_user_pool.sh
bash scripts/submit.sh --experiment tau2-opd-airline-retail-pilot --gpus 4 \
  examples/tau2-bench/opd/run_tau2_opd_airline_retail.sh -- smoke teacher
bash scripts/submit.sh --experiment tau2-opd-airline-retail-pilot --gpus 8 \
  examples/tau2-bench/opd/run_tau2_opd_airline_retail.sh -- smoke student
```

Smoke runs two updates. After validating the cluster path, use `pilot student`
with the same persistent teachers. The following arms share SFT, teachers, User,
data settings, K1, batch16, current-logprob advantage and60 updates:

| Arm | TAU2_OPD_RUN_TAG | TAU2_MAX_BUFFERED_GROUPS | TAU2_OPD_LR |
|---|---|---:|---:|
| A | current-bounded-lr2e6 | 32 | 2e-6 |
| B | current-unbounded-lr2e6 | -1 | 2e-6 |
| C, after A/B | current-bounded-lr5e6 | 32 | 5e-6 |

Pass overrides with `scripts/submit.sh -e KEY=value`. Keep domain sampling and
evaluation settings fixed and report the actually consumed domain counts.
Record every submitted job and its run-log link in the experiment README and
maintain `output/doc/INDEX.md`.

## Evaluation and tests

After HF conversion, `run_official_eval_external_user.sh opd 300` selects
`pilot-current-bounded-lr2e6/checkpoints/iter_0000059_hf` by default.
`TAU2_OPD_RUN_TAG` and `TAU2_OPD_EVAL_ITERATION` select another arm/checkpoint;
`MODEL_PATH` overrides the path. Use seeds300/301 and identical task, Agent/User
and concurrency settings for SFT, teachers and OPD. Report pass@1, pass@4(any),
pass^4, action/DB accuracy and paired task uncertainty.

CPU validation on2026-09-22 used Python3.11 with CPU PyTorch:

```bash
PYTHONPATH=. python tests/test_tau2_opd_training.py             # 21 passed
PYTHONPATH=. python tests/test_tau2_continuous.py               # 119 passed
PYTHONPATH=. python tests/test_megatron_argument_validation.py  # 15 passed
```

The OPD regressions are registered in the `cpu-unittest` workflow matrix.
They cover the exact-gradient condition,
buffer backpressure/resume, domain metrics across DP/CP, post-update measurement,
segment alignment and launcher settings. The cluster launcher also retains the
existing Tau2 OPD and rollout preflights.
