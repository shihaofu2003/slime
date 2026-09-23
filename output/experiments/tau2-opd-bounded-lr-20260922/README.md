# tau2-opd-bounded-lr-20260922

Purpose: compare complete Airline/Retail OPD runs at LR2e-6 and5e-6 with bounded
prefetch,using the same new train/rollout seeds1235/43. The2e-6 arm also replicates
the prior [buffer32 result](../tau2-opd-current-buffer-ab/README.md) with a new seed.
User requested submission on2026-09-22 and explicitly deferred the scoring probe.

Completed: both arms finished60 updates and all1,600 official simulations.
The current choice is2e-6: overall pass@1 is55.125%,versus49.75% for5e-6.
[Final results,uncertainty and next experiments](RESULTS.md).

2026-09-23 service update:the user requested replacing the two-domain experts.
Job23045 was stopped and replaced by four-domain8-GPU job23191 in the
[four-domain experiment](../tau2-opd-four-domain-20260923/README.md). User23043 remains running.

## Fixed recipe and services

Fresh SFT4505 initialization and optimizer per arm; Airline teacher iter29,
Retail teacher iter9; User Qwen3.6-27B. Current-student logprob advantage + TIS[0,2],
buffer32,batch16,K1,60 updates,save every10,post-update diagnostics every10,
train seed1235,rollout seed43,temperature1/top_p1,max_steps200,max_tokens1200,
training token cap16384,two over-cap retries. Same1711-task shuffled source
(1148 Airline/563 Retail),without task-reward training or zero-outcome filtering.

User service4 GPUs(two TP2 replicas) and expert service4 GPUs(two TP2 experts)
are persistent by explicit user instruction. Keep both running after training
and evaluation; do not stop them to free evaluation quota. Endpoint files are
`user_endpoint.env` and `teacher_endpoints.env` in this experiment directory.
Each complete distillation job uses8 GPUs; all four jobs together request24 GPUs,
normal priority. Student jobs may run concurrently and share the persistent
services; report observed lag and service contention when comparing throughput.

## Arms and jobs

| Arm | LR | Artifact root | Pipeline |
|---|---:|---|---|
| lr2e6 | 2e-6 | pilot-lr2e6-seed1235-43 | 60 updates → iter59 HF → official seeds300/301 |
| lr5e6 | 5e-6 | pilot-lr5e6-seed1235-43 | 60 updates → iter59 HF → official seeds300/301 |

| Job | Task | Final/current status | Run log |
|---|---|---|---|
|23043|Persistent User service,4 GPUs|RUNNING;keep running|[Log](jobs/23043-tau2-opd-lr-user-persistent-0922-215220536/run_0_20260922_215220536.log)|
|23045|Persistent Airline/Retail expert service,4 GPUs|STOPPED2026-09-23;user-requested replacement23191|[Log](jobs/23045-tau2-opd-lr-experts-persistent-0922-215258775/run_0_20260922_215258775.log)|
|23048|Initial LR2e-6 submission,8 GPUs|FAILED before training;replaced by23062|[Log](jobs/23048-tau2-opd-bounded-lr2e6-full-0922-215647527/run_0_20260922_215647527.log)|
|23049|LR5e-6 training,HF export and seeds300/301 evaluation,8 GPUs|SUCCEEDED|[Log](jobs/23049-tau2-opd-bounded-lr5e6-full-0922-215716142/run_0_20260922_215716142.log)|
|23062|LR2e-6 retry after preflight-test correction;same fresh arm and full pipeline|SUCCEEDED|[Log](jobs/23062-tau2-opd-bounded-lr2e6-full-retry-0922-220541637/run_0_20260922_220541637.log)|

## Evaluation

Each student job evaluates its iter59 HF checkpoint using the matched native
three-domain protocol:Airline20/Retail40/Telecom40 test tasks ×4 trials,seed300
then301,Agent temperature0.6/top_p1/max_tokens1200,max_steps200,max_errors10,
domain concurrency2/2/5,global9,slot borrowing. Evaluation deploys its own
Agent/User services inside the8-GPU job. The persistent training services remain
running. Reuse the previous experiment's matched SFT/teacher controls.

Report pass@1/pass@4(any)/pass^4,action/DB accuracy,domain counts,lag and OPD
diagnostics. The fixed-context scoring probe is skipped for this round.

## Status

Both arms completed60 updates and iter59 HF export; losses and gradient norms
are finite throughout. Each completed400 simulations at seed300 and400 at301,
with zero infrastructure errors. LR5e-6 evaluation ended at23:26:08 CST and
LR2e-6 at23:36:23 CST on2026-09-22. User23043 and experts23045 remain RUNNING.
[Complete training/conversion launcher](run_distillation.sh),[evaluation launcher](eval_arm.sh).

Startup history:LR2e-6 job23048 failed before training because a domain-metric
unit test assumed its first completion-ordered batch contained Retail. Limited
that test's admission to one batch so domain coverage is deterministic; production
behavior and training parameters are unchanged. The corrected test passed20
separate local pytest runs. Retry23062 used the still-fresh arm root; both final
jobs passed14 OPD +119 continuous tests. Runtime arguments confirmed the intended
learning rates,seeds1235/43,buffer32,current-logprob and SFT4505 initialization.

| Arm | Training | HF conversion | Eval300 | Eval301 |
|---|---|---|---|---|
|lr2e6|[Log](pilot-lr2e6-seed1235-43/run.log)|[Log](pilot-lr2e6-seed1235-43/convert_iter59.log)|[Summary](eval/lr2e6/seed300_summary.json),[log](eval/lr2e6/seed300.log)|[Summary](eval/lr2e6/seed301_summary.json),[log](eval/lr2e6/seed301.log)|
|lr5e6|[Log](pilot-lr5e6-seed1235-43/run.log)|[Log](pilot-lr5e6-seed1235-43/convert_iter59.log)|[Summary](eval/lr5e6/seed300_summary.json),[log](eval/lr5e6/seed300.log)|[Summary](eval/lr5e6/seed301_summary.json),[log](eval/lr5e6/seed301.log)|

The [final analysis](RESULTS.md) favors2e-6 for pass@1,replicates Retail pass^4
improvement,and retains Airline consistency and training-randomness limitations.
[Recomputed metrics and paired intervals](comparison.json),[analysis script](summarize.py).
