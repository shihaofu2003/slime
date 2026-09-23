# tau2-domain-experts-sft4505-b128

Purpose: restart four independent domain RL experts from corrected four-domain SFT iter4505, with a refreshed shared four-GPU User service and two concurrent eight-GPU training jobs. Previous SFT4673 training21629 and User20873 were stopped at user request on2026-09-20; old results are historical controls, not continuations of this run.

## Initialization and recipe

Megatron root: `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920`, explicit reference step4505. HF: same root `/iter_0004505_hf`. Each smoke and formal domain run has a separate fresh checkpoint directory, optimizer and RNG initialization, starting at RL update0. No old RL checkpoints or SFT optimizer state loaded. Existing Megatron initialization sets finetune/no-load-optim/no-load-rng when the new output has no checkpoint.

Retain LR2e-6,batch128 (16 groups ×8),progress-db-count-v1,progress1,format1,gamma0.98,KL0,entropy0,train seed1234,rollout seed42,shuffled/retry sampling,uniform-outcome group filtering,pool32,pending64,unlimited policy lag,4 environment workers,Agent concurrency48/step160,max_steps200,max_tokens1200,context cap16384,two over-cap retries. Agent request timeout60s,600s for Banking. Save every10 updates and at completion;reward plots every300s. No retrieval or filter changes in this restart.

## Data and scheduling

RL rows copied unchanged from `../tau2-domain-experts-fast-b128/data/`;all rows parsed and domain checked. The corrected SFT changes initialization;RL tasks remain Airline1148,Retail563,Telecom271,Banking542. Each job performs two-update smoke checks for its assigned domains before formal training. Each domain starts independently from SFT4505.

| Job | GPUs | Domain sequence / formal updates | Run log |
|---|---:|---|---|
|22033|8|Airline60 → Retail30 → Telecom20|[Log](jobs/22033-sft4505-experts-airline-retail-telecom-0920-231152338/run_0_20260920_231152338.log)|
|22034|8|Banking30|[Log](jobs/22034-sft4505-expert-banking-0920-231152765/run_0_20260920_231152765.log)|

Each eight-GPU job uses Trainer2 + three TP2 Generators6. Shared User22031 uses4 GPUs,two TP2 replicas. Total requested20 GPUs,normal priority. [Launcher](run_experts.sh) reads the replacement User compute-IP endpoint from its readiness log;both clients wait until it is reachable. Model Qwen3.6-27B,bfloat16,context65536,32 running requests/replica;User requests temp0/top_p1/max_tokens512/enable_thinking=false,unchanged. [User run log](../tau2-external-user-pool/jobs/22031-tau2-user-pool-refresh-sft4505-0920-230958466/run_0_20260920_230958466.log).

As of2026-09-20 23:39 CST:all three jobs RUNNING. User22031 ready at `http://10.119.97.164:30000/v1`,both training clients connected. Airline smoke completed2 optimizer updates and saved iter1;final loss-0.0421863,grad_norm0.526099,cleanup in progress. Banking smoke passed2 updates and saved iter1;formal Banking30 startup in progress,0 formal updates. Retail/Telecom smoke pending in22033. All completed optimizer metrics finite;no CUDA OOM or ActorDiedError observed. Airline had a transport read retry and subsequently completed both updates. Logs confirm reference step4505,finetune=True,no_load_optim=True. Smoke updates are separate from the140 formal-update budget.

## Matched SFT baseline

Existing official seed300 evaluation from [SFT experiment](../tau2-sft-areal3-banking-simplified-full-20260920/README.md),197 tasks ×4 trials,unchanged full four-domain protocol. Future RL comparison must use this new un-RL-trained SFT baseline.

| Domain | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
|Airline|42.50%|65.00%|30.00%|
|Retail|58.75%|77.50%|35.00%|
|Telecom|47.50%|82.50%|12.50%|
|Banking|4.12%|8.25%|1.03%|

Training outputs: `arms/async/20260920_sft4505-<domain>-<smoke|train>/`;reward curves: `plots/20260920_sft4505-<domain>-train/reward_curves.png`. Formal checkpoints receive full four-domain evaluation under the matched SFT protocol; smoke checkpoints excluded.

banking smoke: 2 updates completed; finite loss/gradient; iter1 saved. [Run log](arms/async/20260920_sft4505-banking-smoke/run.log).

airline smoke: 2 updates completed; finite loss/gradient; iter1 saved. [Run log](arms/async/20260920_sft4505-airline-smoke/run.log).

retail smoke: 2 updates completed; finite loss/gradient; iter1 saved. [Run log](arms/async/20260920_sft4505-retail-smoke/run.log).

banking train: 30 updates completed; finite loss/gradient; iter29 saved. [Run log](arms/async/20260920_sft4505-banking-train/run.log).

telecom smoke: 2 updates completed; finite loss/gradient; iter1 saved. [Run log](arms/async/20260920_sft4505-telecom-smoke/run.log).

## Checkpoint evaluations and monitoring

User requested supervision every30 minutes,with at most two active8-GPU evaluation jobs. Reuse the SFT baseline conversion and full evaluation wrapper without hyperparameter changes;Agent/User are deployed within each8-GPU evaluation job. [Launcher](eval_checkpoint.sh). Banking iter29 is pending the next free slot;later formal checkpoints queue likewise.

| Job | Expert checkpoint | Run log |
|---|---|---|
|22076|Banking iter9|[Log](jobs/22076-sft4505-banking-iter9-four-domain-0921-012443632/run_0_20260921_012443632.log)|
|22077|Banking iter19|[Log](jobs/22077-sft4505-banking-iter19-four-domain-0921-012450287/run_0_20260921_012450287.log)|

Submitted2026-09-21 01:24 CST. Next scheduled supervision01:55 CST. Check completed Banking trajectory counts every30 minutes;two consecutive intervals without completions trigger inspection of live logs for a stall. If confirmed stalled,stop that evaluation and omit Banking in subsequent evaluations,retaining completed outputs and labeling the incomplete coverage. Normal ongoing trajectory progress is not itself a stall.

2026-09-21 01:36 CST startup check:22076 RUNNING,full four-domain evaluation started01:32:46;completed Airline11/160,Retail30/160,Telecom8/80,Banking31/388. Banking result updated01:36:36,so no stall.22077 remains queued. Protocol confirmed full test,four trials,max_steps200,BM25,domain concurrency1/2/2/4,global9,slot borrowing;all four domains started. Next full supervision01:55.
Supervision 2026-09-21 01:41:23 CST.

Training: airline 13/60, retail 0/30, telecom 0/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: RUNNING; airline 25/160, retail 59/160, telecom 24/80, banking_knowledge 54/388.

Eval 22077 banking iter19: QUEUED; airline 0/160, retail 0/160, telecom 0/80, banking_knowledge 0/388.


[Supervisor](supervise.py) checks training/checkpoints and saved evaluation counts every1800s,including queued evaluations in the two-job limit,and submits pending formal checkpoints in save-time order. It pauses automatic submissions and returns for live diagnosis if a job fails or Banking has no saved result for60 minutes. New submissions append their run-log links here. Started after an initial successful check01:41,next scheduled check01:55 CST.
Supervision 2026-09-21 01:55:00 CST.

Training: airline 16/60, retail 0/30, telecom 0/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: RUNNING; airline 69/160, retail 160/160, telecom 67/80, banking_knowledge 144/388.

Eval 22077 banking iter19: QUEUED; airline 0/160, retail 0/160, telecom 0/80, banking_knowledge 0/388.

Supervision 2026-09-21 02:25:01 CST.

Training: airline 24/60, retail 0/30, telecom 0/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: RUNNING; airline 80/160, retail 160/160, telecom 160/80, banking_knowledge 380/388.

Eval 22077 banking iter19: QUEUED; airline 0/160, retail 0/160, telecom 0/80, banking_knowledge 0/388.

Supervision 2026-09-21 02:55:03 CST.

Training: airline 32/60, retail 0/30, telecom 0/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: SUCCEEDED; airline 80/160, retail 160/160, telecom 160/80, banking_knowledge 388/388.

Eval 22077 banking iter19: QUEUED; airline 0/160, retail 0/160, telecom 0/80, banking_knowledge 0/388.

Attention required: Evaluation 22076 exited successfully but domain counts are incomplete.


airline train: 60 updates completed; finite loss/gradient; iter59 saved. [Run log](arms/async/20260920_sft4505-airline-train/run.log).

retail train: 30 updates completed; finite loss/gradient; iter29 saved. [Run log](arms/async/20260920_sft4505-retail-train/run.log).

2026-09-21 07:57 CST recovery:supervisor exited02:55 because Airline/Telecom denominators were mistakenly reversed. Correct official counts are80/160/160/388;earlier displayed denominators were wrong,evaluation parameters and results were unchanged. Fixed totals. Airline60/60,Retail30/30,Banking30/30 complete;Telecom starting.

Job22077 failed after producing results:Banking task_075 trial1 exceeded262144 context tokens (261888 input +1200 output),after4 attempts. Other787 simulations have no infrastructure failure. Banking completion gaps never exceeded7.92 minutes,so this was not a stalled domain;no domain disabled. User clarified that this context-overflow trajectory counts as task failure:retain all388 trials in the denominator and use the resulting Banking score. Supervisor may continue other checkpoints after this reviewed failure.

Job22221: banking-iter29,8 GPUs,unchanged full four-domain evaluation. [Run log](jobs/22221-sft4505-banking-iter29-four-domain-0921-075726956/run_0_20260921_075726956.log).

Job22222: airline-iter9,8 GPUs,unchanged full four-domain evaluation. [Run log](jobs/22222-sft4505-airline-iter9-four-domain-0921-075745665/run_0_20260921_075745665.log).

Supervisor correction:if a running Banking evaluation has no saved result for60 minutes,it stops that job,retains partial results,and stops new submissions until Banking is excluded from subsequent evaluations. This does not change any evaluation hyperparameters.
Supervision 2026-09-21 07:58:39 CST.

Training: airline 60/60, retail 30/30, telecom 0/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22077 banking iter19: FAILED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22221 banking iter29: RUNNING; airline 0/80, retail 0/160, telecom 0/160, banking_knowledge 0/388.

Eval 22222 airline iter9: RUNNING; airline 0/80, retail 0/160, telecom 0/160, banking_knowledge 0/388.


## RL checkpoint results

pass@1,official seed300,unchanged protocol;compare corrected SFT4505 without RL.

|Domain|SFT4505|Banking iter9|Banking iter19|
|---|---:|---:|---:|
|airline|42.50%|43.75%|43.75%|
|retail|58.75%|61.25%|50.00%|
|telecom|47.50%|46.25%|43.12%|
|banking_knowledge|4.12%|4.64%|3.61% (context overflow counted as failure)|

User scoring clarification:Banking iter19 context-overflow trajectory counts as failure,not an invalid domain. Raw results remain unchanged. Banking has14 successes/388 trials:pass@1 3.61%,pass@4(any)8.25%,pass^4 0.00%. The existing summary already includes this trajectory as zero in pass_metrics;only the report interpretation changes. Job22077 remains FAILED because the original evaluator rejects infrastructure errors at exit.

Comparison verified2026-09-21 08:09 CST. Cells:pass@1 / pass@4(any) / pass^4 (%). Raw means original Qwen3-4B-Instruct-2507 without project SFT/RL. All use seed300,4 trials,max_steps200,max_errors10,BM25 and the same domain concurrency1/2/2/4.

|Model|Airline|Retail|Telecom|Banking|
|---|---:|---:|---:|---:|
|Raw Instruct (no project SFT/RL)|33.75 / 65.00 / 10.00|52.50 / 72.50 / 30.00|17.50 / 42.50 / 5.00|2.58 / 4.12 / 1.03|
|Raw Qwen3.5-4B thinking-on|82.50 / 90.00 / 75.00|83.75 / 95.00 / 67.50|69.38 / 85.00 / 52.50|4.38 / 9.28 / 1.03|
|Raw Qwen3.5-4B thinking-off|80.00 / 90.00 / 65.00|81.88 / 92.50 / 70.00|78.13 / 87.50 / 55.00|3.87 / 11.34 / 0.00|
|Current SFT4505|42.50 / 65.00 / 30.00|58.75 / 77.50 / 35.00|47.50 / 82.50 / 12.50|4.12 / 8.25 / 1.03|
|**Banking expert iter9**|43.75 / 55.00 / 30.00|61.25 / 85.00 / 32.50|46.25 / 82.50 / 15.00|**4.64 / 11.34 / 0.00**|
|Banking expert iter19|43.75 / 65.00 / 25.00|50.00 / 70.00 / 32.50|43.12 / 75.00 / 12.50|3.61 / 8.25 / 0.00|
|Airline expert iter9|42.50 / 50.00 / 30.00|55.62 / 77.50 / 30.00|55.62 / 80.00 / 30.00|3.35 / 7.22 / 1.03|
|Airline expert iter19|40.00 / 60.00 / 30.00|58.13 / 80.00 / 32.50|49.38 / 90.00 / 12.50|4.38 / 10.31 / 0.00|
|**Airline expert iter29**|**53.75 / 75.00 / 35.00**|59.38 / 90.00 / 30.00|49.38 / 82.50 / 17.50|4.12 / 9.28 / 0.00|
|Airline expert iter39|50.00 / 65.00 / 35.00|60.63 / 87.50 / 27.50|58.13 / 87.50 / 20.00|4.12 / 9.28 / 1.03|
|Airline expert iter49|46.25 / 60.00 / 25.00|56.88 / 77.50 / 30.00|51.25 / 82.50 / 15.00|4.38 / 10.31 / 0.00|
|Airline expert iter59|43.75 / 60.00 / 30.00|53.75 / 80.00 / 32.50|50.00 / 87.50 / 12.50|5.67 / 8.25 / 3.09|
|**Retail expert iter9**|42.50 / 65.00 / 30.00|**60.63 / 87.50 / 30.00**|47.50 / 80.00 / 15.00|4.90 / 8.25 / 1.03|
|Retail expert iter19|42.50 / 65.00 / 20.00|54.38 / 82.50 / 27.50|41.88 / 77.50 / 7.50|4.12 / 8.25 / 2.06|
|Retail expert iter29|41.25 / 60.00 / 25.00|60.63 / 85.00 / 32.50|43.75 / 77.50 / 7.50|4.12 / 9.28 / 1.03|
|**Telecom expert iter9**|41.25 / 60.00 / 25.00|56.25 / 85.00 / 25.00|**51.88 / 90.00 / 15.00**|3.35 / 7.22 / 1.03|
|Telecom expert iter19|40.00 / 60.00 / 25.00|58.13 / 85.00 / 30.00|43.75 / 75.00 / 20.00|5.15 / 10.31 / 1.03|

Raw Qwen3.5-4B rows are imported from `../tau2-eval-qwen35-official-native-four-domain/` (seed300, BM25, full four-domain test, four trials). The thinking-on and thinking-off arms differ only in the Agent `enable_thinking` request setting; they are external raw-model baselines and were not trained in this experiment.

Banking iter19 includes its one context-overflow trial as failure per user instruction. Banking iter29 remains in progress;its partial samples are excluded from the completed-domain comparison. Airline iter9 has completed and is included.

Raw source: /mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-banking-expert-sft/eval/raw-full-qwen36-bm25-fixed/seed300_0910_raw_fixed_summary.json. Current SFT source: ../tau2-sft-areal3-banking-simplified-full-20260920/eval/four-domain-sft-iter4505/seed300_20260920_sft_iter4505_four_domain_summary.json.
Supervision 2026-09-21 08:28:43 CST.

Training: airline 60/60, retail 30/30, telecom 2/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22077 banking iter19: FAILED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22221 banking iter29: RUNNING; airline 15/80, retail 160/160, telecom 105/160, banking_knowledge 36/388.

Eval 22222 airline iter9: RUNNING; airline 63/80, retail 160/160, telecom 69/160, banking_knowledge 152/388.

Supervision 2026-09-21 08:58:49 CST.

Training: airline 60/60, retail 30/30, telecom 4/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22077 banking iter19: FAILED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22221 banking iter29: RUNNING; airline 37/80, retail 160/160, telecom 160/160, banking_knowledge 55/388.

Eval 22222 airline iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Evaluation job22226: airline iter19,8 GPUs,unchanged full four-domain protocol. [Run log](jobs/22226-sft4505-airline-iter19-four-domain-0921-085855931/run_0_20260921_085855931.log).


Airline iter9 job22222 SUCCEEDED:788/788 simulations,zero infrastructure errors,0.9653h. [Summary](eval/airline-iter9-four-domain/seed300_20260921_sft4505_airline_iter9_summary.json). Overall pass@1/pass@4(any)/pass^4:28.55/40.61/15.74%,versus SFT4505 27.92/43.15/13.20%. Same seed300 four-domain protocol.

Against current SFT,Airline target-domain pass@1 is unchanged42.50%,pass@4(any) drops65→50%,pass^4 remains30%. Cross-domain pass@1 deltas:Retail−3.13pp,Telecom+8.13pp,Banking−0.77pp. Telecom pass^4 rises12.5→30%. This checkpoint does not establish target-domain Airline improvement;cross-domain changes are single-seed observations.
Supervision 2026-09-21 09:28:56 CST.

Training: airline 60/60, retail 30/30, telecom 5/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22077 banking iter19: FAILED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22221 banking iter29: RUNNING; airline 73/80, retail 160/160, telecom 160/160, banking_knowledge 64/388.

Eval 22222 airline iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22226 airline iter19: RUNNING; airline 63/80, retail 155/160, telecom 61/160, banking_knowledge 115/388.

Supervision 2026-09-21 09:59:03 CST.

Training: airline 60/60, retail 30/30, telecom 7/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22077 banking iter19: FAILED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22221 banking iter29: RUNNING; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 94/388.

Eval 22222 airline iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22226 airline iter19: RUNNING; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 341/388.


Airline iter19 job22226 SUCCEEDED:788/788 simulations,zero infrastructure errors. [Summary](eval/airline-iter19-four-domain/seed300_20260921_sft4505_airline_iter19_summary.json). Overall pass@1/pass@4(any)/pass^4:28.05/45.69/12.18%. Compared with SFT4505,target Airline pass@1 drops42.50→40.00%,pass@4(any)65→60%,pass^4 remains30%. Airline pass@1 also trails iter9 by2.50pp;target-domain improvement is not established at iter19. Compared with iter9,Telecom pass@1 falls55.62→49.38% and pass^4 falls30→12.5%,while pass@4(any) increases80→90%. Single-seed descriptive comparison.
Supervision 2026-09-21 10:29:12 CST.

Training: airline 60/60, retail 30/30, telecom 8/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22077 banking iter19: FAILED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22221 banking iter29: STOPPED at user request after 109/388 Banking trials; checkpoint excluded from teacher selection.

Job22231: retail-iter29,8 GPUs,unchanged full four-domain evaluation. [Run log](jobs/22231-fush-sft4505-retail-iter29-four-domain-0921-retailteacher-0921-104829180/run_*_20260921_104829180.log).

Job22286: airline-iter39,8 GPUs,unchanged full four-domain evaluation. [Run log](jobs/22286-fush-fush-sft4505-airline-iter39-four-domain-0921-airlineexpert-0921-131436879/run_*_20260921_131436879.log).

Job22288: airline-iter49,8 GPUs,unchanged full four-domain evaluation. [Run log](jobs/22288-fush-fush-sft4505-airline-iter49-four-domain-0921-airlineexpert-0921-133038386/run_*_20260921_133038386.log).

Job22336: airline-iter49 retry,8 GPUs,unchanged full four-domain evaluation. [Run log](jobs/22336-fush-fush-sft4505-airline-iter49-four-domain-0921-retry-0921-144013303/run_*_20260921_144013303.log). The earlier job22288 was reclaimed for quota/resource pressure and then cancelled before evaluation started.

Job22355: airline-iter59,8 GPUs,unchanged full four-domain evaluation. [Run log](jobs/22355-fush-sft4505-airline-iter59-four-domain-eval-0921-150852947/run_*_20260921_150852947.log).

Job22361: retail-iter9,8 GPUs,unchanged full four-domain evaluation. [Run log](jobs/22361-fush-sft4505-retail-iter9-four-domain-eval-0921-151919789/run_*_20260921_151919789.log).

2026-09-21 10:48 CST: stopped Banking iter29 evaluation job22221 and submitted Retail iter29 job22231. Reused the same full-evaluation protocol: Airline80,Retail160,Telecom160,Banking388 trials;four trials per task,seed300,BM25,max agent tokens1200,domain concurrency1/2/2/4,global concurrency9,slot borrowing enabled.

Eval 22222 airline iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22226 airline iter19: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Evaluation job22230: airline iter29,8 GPUs,unchanged full four-domain protocol. [Run log](jobs/22230-sft4505-airline-iter29-four-domain-0921-102922285/run_0_20260921_102922285.log).
Supervision 2026-09-21 10:59:23 CST.

Training: airline 60/60, retail 30/30, telecom 10/20, banking 30/30. Jobs: 22033 RUNNING, 22034 SUCCEEDED.

Eval 22076 banking iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22077 banking iter19: FAILED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22221 banking iter29: STOPPED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 112/388.

Eval 22222 airline iter9: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22226 airline iter19: SUCCEEDED; airline 80/80, retail 160/160, telecom 160/160, banking_knowledge 388/388.

Eval 22230 airline iter29: RUNNING; airline 67/80, retail 155/160, telecom 55/160, banking_knowledge 131/388.

Attention required: Evaluation 22221 is STOPPED; inspect before retry.

Evaluation 22230 airline iter29 completed successfully: 788/788 simulations, zero infrastructure errors, elapsed 3629 seconds (about 1.01 hours). [Summary](eval/airline-iter29-four-domain/seed300_20260921_sft4505_airline_iter29_summary.json). Full counts: Airline80/80,Retail160/160,Telecom160/160,Banking388/388. Results pass@1/pass@4(any)/pass^4: Airline53.75/75.00/35.00%, Retail59.38/90.00/30.00%, Telecom49.38/82.50/17.50%, Banking4.12/9.28/0.00%. The target Airline checkpoint's target-domain score is 53.75/75.00/35.00% under the unchanged seed300 four-domain protocol.

telecom train: 20 updates completed; finite loss/gradient; iter19 saved. [Run log](arms/async/20260920_sft4505-telecom-train/run.log).

Job22361 Retail iter9 failed during SGLang User worker0 startup,before official evaluation began. Fatal error:torch.distributed.DistNetworkError,TCPStore port39259 EADDRINUSE. Also observed encoder bootstrap port8997 binding conflicts. No CUDA OOM in worker logs and no evaluation results were created. This is a service port collision,not a checkpoint-quality or scoring failure;the logs do not identify the process owning39259. [Run log](jobs/22361-sft4505-retail-iter9-four-domain-eval-0921-151919789/run_0_20260921_151919789.log).

Job22377:Retail iter9 four-domain evaluation retry of22361,submitted2026-09-21 15:37 CST at explicit user request.8 GPUs,normal priority,reused22361 launcher snapshot and identical parameters;no port/config changes. [Run log](jobs/22377-sft4505-retail-iter9-four-domain-eval-retry-0921-153730124/run_0_20260921_153730124.log).

Airline iter49 job22350 SUCCEEDED:788/788 simulations,zero infrastructure errors. [Run log](jobs/22350-sft4505-airline-iter49-reconvert-eval-0921-150537674/run_0_20260921_150537674.log);[summary](eval/airline-iter49-four-domain/seed300_20260921_sft4505_airline_iter49_summary.json). Same full four-domain seed300 protocol. Overall pass@1/pass@4(any)/pass^4:28.81/43.65/11.68%,versus current SFT4505 27.92/43.15/13.20%. Target Airline pass@1 improves42.50→46.25%,but pass@4(any) drops65→60% and pass^4 drops30→25%. Airline pass@1 trails iter29(53.75%) and iter39(50.00%);the later checkpoint does not improve the best observed target-domain result.

Airline iter59 job22355 SUCCEEDED:788/788 simulations,zero infrastructure errors. [Run log](jobs/22355-sft4505-airline-iter59-four-domain-eval-0921-150852947/run_0_20260921_150852947.log);[summary](eval/airline-iter59-four-domain/seed300_20260921_sft4505_airline_iter59_summary.json). Overall pass@1/pass@4(any)/pass^4:28.30/44.16/13.71%. Compared with current SFT4505,pass@1 deltas are Airline+1.25pp,Retail−5.00pp,Telecom+2.50pp,Banking+1.55pp. Banking pass^4 rises1.03→3.09%. Target Airline pass@1(43.75%) trails iter29(53.75%),iter39(50.00%) and iter49(46.25%);iter29 remains the best observed Airline checkpoint in this seed300 evaluation.

2026-09-21 16:34 CST:user authorized two additional8-GPU Retail checkpoint evaluations;iter9 job22377 was already running. New jobs retain full four-domain seed300 evaluation and all generation/scoring parameters.

|Job|Checkpoint|Run log|
|---|---|---|
|22448|Retail iter19,convert then evaluate|[Log](jobs/22448-sft4505-retail-iter19-four-domain-eval-0921-163352884/run_0_20260921_163352884.log)|
|22450|Retail iter29,resume saved evaluations|[Log](jobs/22450-sft4505-retail-iter29-four-domain-resume-0921-163406508/run_0_20260921_163406508.log)|

Iter29 [resume launcher](eval_retail_iter29_resume.sh) sets AUTO_RESUME=1 and uses the same checkpoint,evaluation paths and parameters. Prior saved trajectory counts at submission:Airline35,Retail93,Telecom40,Banking77;official runner reuses completed trials and retries any infrastructure failures. No evaluation hyperparameters changed.

Retail iter9 job22377 SUCCEEDED:788/788 simulations,zero infrastructure errors. [Run log](jobs/22377-sft4505-retail-iter9-four-domain-eval-retry-0921-153730124/run_0_20260921_153730124.log);[summary](eval/retail-iter9-four-domain/seed300_20260921_sft4505_retail_iter9_summary.json). Overall pass@1/pass@4(any)/pass^4:28.68/44.67/12.69%. Compared with SFT4505,target Retail pass@1 rises58.75→60.63%(+1.875pp),pass@4(any)77.5→87.5%(+10pp),but pass^4 falls35→30%(−5pp). Airline and Telecom pass@1 remain42.5% and47.5%;Banking pass@1 rises4.12→4.90%. Single-seed results show improved Retail success and task coverage,but not improved four-trial consistency.

Retail iter19 job22448 SUCCEEDED:788/788 simulations,zero infrastructure errors. [Run log](jobs/22448-sft4505-retail-iter19-four-domain-eval-0921-163352884/run_0_20260921_163352884.log);[summary](eval/retail-iter19-four-domain/seed300_20260921_sft4505_retail_iter19_summary.json). Overall pass@1/pass@4(any)/pass^4:25.89/43.15/10.15%. Target Retail:54.38/82.50/27.50%;compared with SFT4505,pass@1 −4.38pp,pass@4(any)+5.00pp,pass^4 −7.50pp.

Retail iter29 job22450 SUCCEEDED:788/788 simulations,zero infrastructure errors. [Run log](jobs/22450-sft4505-retail-iter29-four-domain-resume-0921-163406508/run_0_20260921_163406508.log);[summary](eval/retail-iter29-four-domain/seed300_20260921_sft4505_retail_iter29_summary.json). Overall pass@1/pass@4(any)/pass^4:27.41/43.65/11.17%. Target Retail:60.63/85.00/32.50%;compared with SFT4505,pass@1 +1.88pp,pass@4(any)+7.50pp,pass^4 −2.50pp. Under this seed300 protocol,iter29 is the stronger Retail checkpoint than iter19.

Telecom iter9 job22494 SUCCEEDED:788/788 simulations,zero infrastructure errors. [Run log](jobs/22494-telecom-iter9-four-domain-eval-0921-0921-191251431/run_0_20260921_191251431.log);[summary](eval/telecom-iter9-four-domain/seed300_20260921_sft4505_telecom_iter9_summary.json). Overall pass@1/pass@4(any)/pass^4:27.79/45.18/11.17%. Target Telecom:51.88/90.00/15.00%;compared with SFT4505,pass@1 +4.38pp,pass@4(any)+7.50pp,pass^4 +2.50pp.

Telecom iter19 job22495 SUCCEEDED:788/788 simulations,zero infrastructure errors. [Run log](jobs/22495-telecom-iter19-four-domain-eval-0921-0921-191252262/run_0_20260921_191252262.log);[summary](eval/telecom-iter19-four-domain/seed300_20260921_sft4505_telecom_iter19_summary.json). Overall pass@1/pass@4(any)/pass^4:27.28/43.65/13.20%. Target Telecom:43.75/75.00/20.00%;compared with SFT4505,pass@1 −3.75pp,pass@4(any) −7.50pp,pass^4 +7.50pp. Iter9 remains stronger on single-shot Telecom success;iter19 has higher four-trial consistency.

### OPD student retention

The four-domain OPD student uses the selected target-domain teachers:Airline
iter29,Retail iter9,Telecom iter9,and Banking iter9. Under the same seed300
four-domain protocol,Update80 (`iter_0000079`) is the selected balanced student
checkpoint. Retention is the student target-domain metric divided by its teacher
metric.

|Domain|Student pass@1/pass@4/pass^4|Teacher pass@1/pass@4/pass^4|Retention pass@1/pass@4/pass^4|
|---|---:|---:|---:|
|Airline|50.00/80.00/30.00%|53.75/75.00/35.00%|93.0/106.7/85.7%|
|Retail|58.13/85.00/37.50%|60.63/87.50/30.00%|95.9/97.1/125.0%|
|Telecom|48.13/87.50/15.00%|51.88/90.00/15.00%|92.8/97.2/100.0%|
|Banking|3.87/7.22/2.06%|4.64/11.34/0.00%|83.3/63.6/undefined|

Across the stitched target teachers,the student retains93.4% pass@1
(`225/241` successes) and94.8% pass@4(any) (`92/97` tasks). Student pass^4 is
`29/197` versus teacher-oracle `25/197` (116.0%);Banking teacher pass^4 is zero.
Values above100% are single-seed observations. Full student results and the
checkpoint curve are in the
[four-domain OPD report](../tau2-opd-four-domain-20260923/README.md).
