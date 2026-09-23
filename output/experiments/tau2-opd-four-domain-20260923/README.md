# tau2-opd-four-domain-20260923

Purpose: run full four-domain pure OPD with the selected2e-6/buffer32 recipe,
and replace the two-domain teacher service with an8-GPU persistent four-expert service.
Requested2026-09-23. Fixed-context scoring remains deferred.

## Teachers and services

Teacher checkpoints follow the selected entries in
[SFT4505 domain-expert results](../tau2-domain-experts-sft4505-b128/README.md).
Each expert has TP2 on two GPUs,context32768,max-total-tokens32768,
mem-fraction0.75,max-running-requests1,matching the two-domain service.

| Teacher | Checkpoint | GPUs | Port |
|---|---|---|---:|
|Airline|airline-train/checkpoints/iter_0000029_hf|0,1|31001|
|Retail|retail-train/checkpoints/iter_0000009_hf|2,3|31002|
|Telecom|telecom-train/checkpoints/iter_0000009_hf|4,5|31003|
|Banking|banking-train/checkpoints/iter_0000009_hf|6,7|31004|

All checkpoint paths begin with
`../tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-`.
Training metadata calls the fourth domain `banking`; official evaluation calls
it `banking_knowledge`. Both names route to the Banking expert.

Old expert job23045 was STOPPED by explicit user request. User job23043 was
STOPPED on2026-09-23 after student training; its endpoint file was reused from
the bounded LR study. The replacement experts remain running after student
training/evaluation.

| Job | Task | Status | Run log |
|---|---|---|---|
|23045|Previous two-domain experts,4 GPUs|STOPPED2026-09-23|[Log](../tau2-opd-bounded-lr-20260922/jobs/23045-tau2-opd-lr-experts-persistent-0922-215258775/run_0_20260922_215258775.log)|
|23043|Existing persistent User,4 GPUs|STOPPED2026-09-23|[Log](../tau2-opd-bounded-lr-20260922/jobs/23043-tau2-opd-lr-user-persistent-0922-215220536/run_0_20260922_215220536.log)|
|23191|Four-domain persistent experts,8 GPUs|RUNNING;all four ready,normal priority|[Log](jobs/23191-tau2-opd-four-domain-experts-persistent-0923-004734161/run_0_20260923_004734161.log)|
|23193|Four-domain160-update OPD,iter159 HF and seeds300/301 eval,8 GPUs|SUCCEEDED;normal priority|[Log](jobs/23193-tau2-opd-four-domain-lr2e6-full-0923-005900453/run_0_20260923_005900453.log)|
|23244|Replicated four-domain full eval,seed303,8 GPUs|STOPPED2026-09-23|[Log](jobs/23244-tau2-opd-four-domain-eval-seed303-0923-082742581/run_*.log)|
|23245|Replicated four-domain full eval,seed302,8 GPUs|STOPPED2026-09-23|[Log](jobs/23245-tau2-opd-four-domain-eval-seed302-0923-082742977/run_*.log)|
|23246|Replicated four-domain full eval,seed304,8 GPUs|STOPPED2026-09-23|[Log](jobs/23246-tau2-opd-four-domain-eval-seed304-0923-082743351/run_*.log)|
|23260|Update40 (`iter_0000039`) four-domain eval,seed300,8 GPUs|SUCCEEDED2026-09-23|[Log](jobs/23260-tau2-opd-four-domain-update40-seed300-eval-0923-091407905/run_*_20260923_091407905.log)|
|23261|Update80 (`iter_0000079`) four-domain eval,seed300,8 GPUs|SUCCEEDED2026-09-23|[Log](jobs/23261-tau2-opd-four-domain-update80-seed300-eval-0923-091408252/run_*_20260923_091408252.log)|
|23262|Update120 (`iter_0000119`) four-domain eval,seed300,8 GPUs|SUCCEEDED2026-09-23|[Log](jobs/23262-tau2-opd-four-domain-update120-seed300-eval-0923-091408598/run_*_20260923_091408598.log)|
|23272|Update90 (`iter_0000089`) four-domain eval,seed300,8 GPUs|SUCCEEDED2026-09-23|[Log](jobs/23272-tau2-opd-four-domain-update90-seed300-eval-0923-103109617/run_*_20260923_103109617.log)|
|23273|Update100 (`iter_0000099`) four-domain eval,seed300,8 GPUs|SUCCEEDED2026-09-23|[Log](jobs/23273-tau2-opd-four-domain-update100-seed300-eval-0923-103109981/run_*_20260923_103109981.log)|
|23274|Update110 (`iter_0000109`) four-domain eval,seed300,8 GPUs|SUCCEEDED2026-09-23|[Log](jobs/23274-tau2-opd-four-domain-update110-seed300-eval-0923-103110348/run_*_20260923_103110348.log)|

## Fixed student recipe

Fresh SFT4505 initialization,not continuation of the two-domain OPD checkpoint.
LR2e-6,batch16,K1,buffer32,pool32,pending64,4 environment workers,unlimited
policy-lag admission,train/rollout seeds1235/43,160 updates,save every10,
post-update diagnostics every10,current-student advantage with TIS[0,2].
Pure OPD:task reward0,no reward shaping,no turn credit,no uniform-outcome filtering.
Agent temperature1/top_p1,max_tokens1200,max_steps200,max_errors10,training token
cap16384,two over-cap retries. User Qwen3.6-27B,temp0/top_p1,max_tokens512,
enable_thinking=false. Student uses8 GPUs:Trainer2 plus three TP2 generators.
Services plus one student request20 GPUs,normal priority.

Use the same processed expert-training sources:Airline1148,Retail563,Telecom271,
Banking542,total2524 rows. Preserve all task/DB metadata; shuffle the combined
source without adding domain quotas. One source-sized trajectory budget needs
ceil(2524/16)=158 updates; use160 updates (2560 consumed trajectories) to align
with the10-update save interval. This corrects the initially proposed60-update
pilot budget following the user's clarification. Asynchronous ordering and
over-cap resampling mean this is approximately one source pass,not a guarantee
that every task is consumed exactly once. Report actual per-domain counts,
unique-task coverage and lag. This is a full-data expansion run,not a controlled
domain-only comparison against the60-update pilot.

## Complete pipeline and evaluation

Submitted pipeline:160 updates → iter159 HF conversion → official four-domain
evaluation at seeds300 and301. Each seed uses197 tasks ×4 trials:
Airline80,Retail160,Telecom160,Banking388 simulations,total788.
Use the domain-expert four-domain protocol:BM25,Agent temperature0.6/top_p1,
max_tokens1200,max_steps200,max_errors10,concurrency1/2/2/4,global9,slot borrowing.
Evaluation starts local Agent/User services inside the student's8-GPU job;
the four-domain experts remain available,while the old persistent User is stopped.

The historical four-domain seed300 SFT/teacher controls are available in the
expert experiment. Four-domain seed301 controls have not been evaluated; the
previous three-domain controls have different scheduling/retrieval settings and
must not be silently used as matched four-domain controls.

## Launch and validation

[Persistent expert launcher](serve_experts.sh),[complete student pipeline](run_distillation.sh),
[official four-domain evaluation](eval_student.sh),and [replicated evaluation](eval_replicate.sh).
Student job23193 (`pt-wogwpz9f`) succeeded after explicit user authorization.
Submitted command:

```bash
bash scripts/submit.sh --experiment tau2-opd-four-domain-20260923 --gpus 8 \
  --name tau2-opd-four-domain-lr2e6-full \
  output/experiments/tau2-opd-four-domain-20260923/run_distillation.sh
```

Validation completed2026-09-23:cluster preflight passed20 OPD tests,including
Telecom/Banking routing and four-domain data merging. Local
`PYTHONPATH=. /tmp/opd-launch-venv/bin/python tests/test_tau2_opd_training.py`
passed22 tests,including two-domain compatibility and four-domain160-update
launcher overrides. Existing CI registration is unchanged; no new test file.
Shell syntax checks and a pipeline execution check with GPU operations replaced
by capture scripts verified160 updates → iter159 HF → seeds300/301 with the
four-domain official wrapper. `scripts/submit.sh --dry-run` succeeded.

The merged [training data](data/four_domain_train.jsonl) contains2524 validated
rows with counts1148/563/271/542. User23043 returns HTTP200 and the expected
Qwen3.6-27B model. All four expert `/health` endpoints return200,
`/get_model_info` reports the selected checkpoint paths,and the service passed
input-token-logprob checks before publishing `teacher_endpoints.env`.

At student submission,the User and all four teacher endpoints again returned
HTTP200. The student output root had no training log or checkpoint tracker,
so this is a fresh SFT4505 run. Persistent service job23043 was later stopped;
job23191 was also stopped after the run.

01:02 CST startup:job23193 was RUNNING. Cluster checks passed20 OPD tests and119
continuous-rollout tests;the rollout-logic preflight also completed and Ray
started `train_async.py` with Trainer2/Generators6. Runtime exports confirm
160 updates,LR2e-6,seeds1235/43 and remote Telecom/Banking expert endpoints.
Training model initialization is in progress; no optimizer update is claimed yet.

Four-domain training and iter159 HF export completed. [Training log](pilot-lr2e6-seed1235-43/run.log),[conversion log](pilot-lr2e6-seed1235-43/convert_iter159.log).
[Reward curve](pilot-lr2e6-seed1235-43/reward_curve.png),[reward data](pilot-lr2e6-seed1235-43/reward_curve.csv).

Four-domain seed300 official evaluation completed. [Summary](eval/student/seed300_summary.json),[log](eval/student/seed300.log).

Four-domain seed301 official evaluation completed. [Summary](eval/student/seed301_summary.json),[log](eval/student/seed301.log).

The existing seed300/301 results are the retained evaluations for iter159.
The later seed302/303/304 submissions (jobs23245/23244/23246) were stopped.

Update40/80/90/100/110/120 checkpoint evaluations use seed300 and the same
official four-domain protocol. Saved checkpoint directories are zero-based.

## Checkpoint results

All entries use the matched seed300 protocol,197 tasks and788 simulations.
Update80 (`iter_0000079`) is the selected balanced checkpoint:it ties the best
pass@1,has the best pass^4,and has fewer `max_steps` trajectories than the
nearby checkpoints. Update100 has the highest pass@4(any) but lower pass^4.

|Update|pass@1|pass@4(any)|pass^4|Action acc.|DB acc.|`max_steps`|
|---:|---:|---:|---:|---:|---:|---:|
|SFT4505|27.92%|43.15%|13.20%|35.38%|23.67%|14|
|40|26.78%|41.12%|12.18%|37.28%|23.85%|20|
|**80**|**28.55%**|46.70%|**14.72%**|36.19%|24.77%|**11**|
|90|27.03%|41.12%|12.18%|35.33%|25.43%|18|
|100|28.43%|**47.21%**|12.18%|36.36%|24.34%|25|
|110|**28.55%**|44.16%|14.21%|**36.91%**|**25.46%**|16|
|120|27.03%|43.15%|9.64%|36.42%|24.51%|19|
|160|27.41%|46.70%|11.68%|36.69%|24.24%|22|

Update80 target-domain results and retention relative to the actual selected
teacher for each domain are below. Ratios are student metric divided by teacher
metric under the same seed300 protocol.

|Domain|Student pass@1/pass@4/pass^4|Teacher checkpoint|Teacher pass@1/pass@4/pass^4|Retention pass@1/pass@4/pass^4|
|---|---:|---|---:|---:|
|Airline|50.00/80.00/30.00%|Airline iter29|53.75/75.00/35.00%|93.0/106.7/85.7%|
|Retail|58.13/85.00/37.50%|Retail iter9|60.63/87.50/30.00%|95.9/97.1/125.0%|
|Telecom|48.13/87.50/15.00%|Telecom iter9|51.88/90.00/15.00%|92.8/97.2/100.0%|
|Banking|3.87/7.22/2.06%|Banking iter9|4.64/11.34/0.00%|83.3/63.6/undefined|

The task-count-weighted student/target-teacher ratios are93.4% for pass@1
(`225/241` successes),94.8% for pass@4(any) (`92/97` tasks),and116.0% for
pass^4 (`29/25` tasks). Banking teacher pass^4 is zero,so its per-domain ratio
is undefined. Ratios above100% are single-seed observations,not evidence of a
stable student advantage. [Update80 summary](eval/checkpoint-update80-seed300/summary.json).
