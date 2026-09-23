# tau2-opd-current-buffer-ab

Purpose: validate the corrected asynchronous OPD implementation and isolate the
effect of completed-queue backpressure with matched Airline/Retail training.
Started2026-09-22. Historical results and review:
[pilot](../tau2-opd-airline-retail-pilot/README.md).

## Fixed recipe

Fresh SFT4505 initialization and optimizer for every arm. Airline teacher iter29,
Retail teacher iter9; reuse persistent teacher22758 and User22555 after probing
the served model paths and User endpoint. Current-student advantage + TIS[0,2],
LR2e-6,60 updates,batch16,K1,temperature1/top_p1,max_steps200,token cap16384,
two over-cap retries,train seed1234,task shuffle seed42. Source tasks1148/563;
no domain quota or outcome filter. Raw task success is diagnostic only.

| Arm | Run tag | Total buffered groups | Purpose |
|---|---|---:|---|
| A | current-bounded-lr2e6 | 32 | Backpressure includes generating, ready and training groups |
| B | current-unbounded-lr2e6 | -1 | Historical prefetch control with identical loss/LR/budget |

Run A then B sequentially to avoid competition for the shared teachers/User.
Each student uses8 GPUs:Trainer2 + Generators6, with4 environment processes.
Shared services use8 GPUs; total during one training run16, normal priority.
Formal post-update diagnostics every10 updates; smoke overrides this to every
update to exercise the new GPU forward. C/LR5e-6 is deferred until A/B results.

## Jobs

| Job | Task | Run log |
|---|---|---|
|22914|Two-update corrected GPU smoke;buffer32,post-update diagnostics every step|[Log](jobs/22914-tau2-opd-current-buffer-smoke-0922-173556270/run_0_20260922_173556270.log)|
|22923|Matched SFT4505 official reference,seed300 then301|[Log](jobs/22923-tau2-opd-buffer-sft-seeds300301-0922-174312038/run_0_20260922_174312038.log)|
|22931|A60 → B60 → HF conversion → A/B official seed300/301 evaluations|[Log](jobs/22931-tau2-opd-buffer-ab-train-eval-0922-175303850/run_0_20260922_175303850.log)|
|22932|Airline/Retail teacher references,each seed300/301|[Log](jobs/22932-tau2-opd-buffer-teachers-seeds300301-0922-175348054/run_0_20260922_175348054.log)|
|22970|Resume B from iter49 after cluster crash;finish updates50–59,convert and evaluate seed300/301|[Log](jobs/22970-tau2-opd-buffer-b-resume49-eval-0922-193629067/run_0_20260922_193629067.log)|
|22972|Independent A iter59 evaluation,seed300/301;decoupled from B recovery|[Log](jobs/22972-tau2-opd-buffer-a-seeds300301-0922-193710025/run_0_20260922_193710025.log)|
|22979|Automatic requeue of elastic A evaluation22972;shares its log directory|[Log](jobs/22972-tau2-opd-buffer-a-seeds300301-0922-193710025/run_0_20260922_193710025.log)|

Reused services:teacher22758 [log](../tau2-opd-airline-retail-pilot/jobs/22758-tau2-opd-airline-retail-teacher-persistent-0922-093341169/run_0_20260922_093341169.log),
User22555 [log](../tau2-opd-airline-retail-pilot/jobs/22555-tau2-opd-airline-retail-user-0921-211346094/run_0_20260921_211346094.log).

## Validation and evaluation

Smoke: inspect two optimizer updates/checkpoints, finite gradients and losses,
Teacher token alignment, observed total buffer≤32, per-domain diagnostics and
the post-update forward. Reward/termination rates and log-ratio signs are
diagnostics, not pass/fail thresholds.

Then train A/B from fresh SFT roots and convert iter59 to HF. Official evaluation
uses the same three-domain native wrapper for SFT4505, both selected experts,
and A/B:Airline20/Retail40/Telecom40 tasks ×4 trials, seeds300/301,
temperature0.6/top_p1,max_tokens1200,max_steps200,max_errors10,Qwen3.6-27B User,
domain concurrency2/2/5,global9,slot borrowing. Each evaluation has its own
Agent/User deployment on8 GPUs. Record pass@1/pass@4(any)/pass^4,action/DB
accuracy and task-paired uncertainty; report actual consumed domain counts and
policy lag for A/B. Telecom is the transfer check. Historical four-domain
controls remain context, not replacements for the matched three-domain controls.

## Status

Completed2026-09-22:SFT22923,teacher references22932,B recovery/evaluation22970,
and A evaluation22979 all SUCCEEDED. Five models ×two seeds ×400 simulations =
4,000/4,000,zero infrastructure errors. Original task objects,trial seeds,
Agent/User configurations and environment information match across models;
raw-result pass metrics agree with the summaries.

[Full results and paired intervals](RESULTS.md),[interpretation and next experiments](ANALYSIS_AND_NEXT.md),
[training/action/DB details](final_analysis.json). A and B each consumed646 Airline
and314 Retail trajectories. A mean lag1.03,max4,total admitted≤32; B final effective
60 updates mean lag8.29,max20,peak ready325,total351. Both effective60-update
training histories have finite loss/gradient. These confirm backpressure behavior,
not a statistically established task-score win for A over B.

A matches the target teachers' pass@1 point estimates within+0.625pp in each
trained domain. A Retail pass^4 improves22.50→33.75% over SFT (paired task95% CI
for the difference[+1.25,+22.50]pp); A/B differences remain uncertain. Airline
teacher pass@1 is45.625%,versus SFT44.375% and A46.25% under this matched protocol.
The next recommendation is a fixed-context scoring probe,an independent A
training-seed replication,and the bounded LR5e-6 C comparison; none is submitted.

Recovery qualifications:22931 completed A,then failed during B after51 optimizer
updates in the user-confirmed cluster crash. Iter49 and sampler state were saved;
22970 restored the optimizer/RNG/sampler and replaced updates50–59. Pending
trajectories were regenerated,so B's last10 updates (mean lag2.69) are not an
uninterrupted unbounded-prefetch control. A evaluation22972 was elastically
reclaimed before any simulations and requeued as22979. Evaluation coverage is complete.

Smoke22914 passed two finite updates,iter1 save,domain diagnostics,post-update
forward and total-buffer≤32 checks. W&B shutdown connection resets did not change
its successful exit. Training-only teacher22758 and User22555 were stopped once
both training arms finished; future training needs those services restarted.

Arm artifacts:[A training](pilot-current-bounded-lr2e6/run.log),
[A conversion](pilot-current-bounded-lr2e6/convert_iter59.log),
[B training including resume](pilot-current-unbounded-lr2e6/run.log),
[B conversion](pilot-current-unbounded-lr2e6/convert_iter59.log).
The original local [report waiter](report_wait.log) referenced failed job22931;
cluster evaluation scripts subsequently generated the final report through
[summarize.py](summarize.py). No further experiment is running for this study.
