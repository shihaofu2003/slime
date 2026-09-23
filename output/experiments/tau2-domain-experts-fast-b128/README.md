# tau2-domain-experts-fast-b128

Purpose: train four separate domain experts from SFT4673 with the selected fast shuffled-task / delayed-all-zero-retry recipe. First complete two real optimizer updates in each domain; only after all four pass, run the four full trainings serially in one 8-GPU allocation.

Current status (2026-09-20 23:09 CST): job21629 STOPPED at user request after Airline60/60 (iter59), Retail30/30 (iter29), Telecom20/20 (iter19), Banking14/30 (iter9 last saved). User20873 also stopped. The user replaced the SFT initialization because its training corpus was incorrect; new independent runs are documented in [SFT4505 experts](../tau2-domain-experts-sft4505-b128/README.md).

## Fixed configuration

Recipe reference: fast Airline job21168, whose iter19 official Airline pass@1/pass@4(any)/pass^4 is53.75/80/30%; see [the comparison](../tau2-airline-db-count-tuning/reports/comparison.md). These experts start afresh from SFT, using that training configuration.

LR 2e-6; batch128 = 16 task groups × 8 trajectories; progress/format weights 1/1, gamma 0.98; KL/entropy 0. Shuffled task deck, full-K8 uniform-outcome filtering, all-zero retry after two policy updates. Four environment workers, pool32, at most64 pending groups, unlimited policy lag. Agent HTTP timeout60s, with Banking increased to600s after its first smoke timed out. Seeds1234/42. Agent temperature1/top_p1/max_tokens1200, max_steps200, training token cap16384 and two over-cap retries. User Qwen3.6-27B non-thinking, temperature0/top_p1/max_tokens512. The base launcher supplies the unchanged model, credit and rollout arguments.

One 8-GPU training job: trainer2 + three TP2 generators6. Reuse existing4-GPU User job20873 (`http://10.119.96.116:30000/v1`, two TP2 Qwen3.6-27B replicas). Each domain and smoke has an independent optimizer/checkpoint directory. All initialize from `Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909/iter_0004673_hf`; the existing weights-only Megatron conversion is reused. Full training restarts from SFT after the smoke, rather than continuing its optimizer. Save every10 updates; refresh reward curves every300s.

## Data and budgets

Rows are copied unchanged from [the prepared four-domain pool](../tau2-async-db-count-four-domain/data/train.jsonl), selecting only `metadata.domain`. Task DB paths and Banking initialization/retrieval remain unchanged.

| Serial order | Domain | Tasks | Smoke updates | Full updates | Final checkpoint |
|---|---|---:|---:|---:|---|
| 1 | Airline | 1148 | 2 | 60 | iter59 |
| 2 | Retail | 563 | 2 | 30 | iter29 |
| 3 | Telecom | 271 | 2 | 20 | iter19 |
| 4 | Banking | 542 | 2 | 30 | iter29 |

Budgets round the Airline-based sampling-coverage estimates55/27/13/26 up to multiples of10. They are fixed update budgets, not a guarantee that every task contributes a gradient or completes sampling. The smoke checks successful updates, finite loss/gradients and checkpoint saving; behavior rates are diagnostics.

## Jobs and results

Launcher: [run_tau2_domain_experts_serial.sh](../../../scripts/run_tau2_domain_experts_serial.sh). Initial run stamp `20260919_224800`; timeout-only Banking retry `20260920_005400`; current Banking smoke and formal training stamp `20260920_011100`.

- Job21605 (`pt-un2lof2p`), original internal-User allocation: stopped before execution after the user clarified8 training GPUs plus the existing4-GPU User service. No training ran and no run log was created. [Submission log](jobs/21605-tau2-four-experts-fast-b128-serial-0919-224557706/submit_20260919_224557706.log).

- Job21606, submitted2026-09-19 22:46 CST:8 training GPUs, normal priority, external User20873. Airline/Retail/Telecom smokes passed; Banking failed before its first update on an Agent read timeout. [Run log](jobs/21606-tau2-four-experts-fast-b128-external-serial-0919-224651447/run_0_20260919_224651447.log).

- Job21624, submitted2026-09-20 00:48 CST:8 training GPUs, normal priority, external User20873. Banking timeout-only retry; stopped before its first update to apply early over-cap resampling. [Run log](jobs/21624-tau2-four-experts-banking-retry-serial-0920-004829816/run_0_20260920_004829816.log).

- Job21629 (`pt-r1uitj32`), submitted2026-09-20 01:10 CST:8 training GPUs, normal priority, external User20873. Banking two-update smoke passed; formal Airline60 → Retail30 → Telecom20 → Banking30 started. [Run log](jobs/21629-tau2-four-experts-context-cap-serial-0920-011022988/run_0_20260920_011022988.log).

- CPU regression job21611, submitted2026-09-19 23:27 CST:0 GPUs /8 CPUs /32 GB. Passed109 tests, including the blocked-remote-dialogue shutdown regression; exit0. [Run log](jobs/21611-tau2-expert-shutdown-tests-0919-232727759/run_0_20260919_232727759.log).

- CPU regression job21628, submitted2026-09-20 01:07 CST:0 GPUs /8 CPUs /32 GB. Passed114 tests, including early rejection of over-cap inference prompts, fresh-DB retries and removal of interrupted trajectories; exit0. [Run log](jobs/21628-tau2-expert-context-cap-tests-0920-010729142/run_0_20260920_010729142.log).

Per-domain artifacts: `arms/async/<run-stamp>-<domain>-<smoke|train>/`; formal reward curves: `plots/<run-stamp>-<domain>-train/reward_curves.png` and `.csv`.

airline smoke: 2 updates completed; finite loss/gradient; iter1 saved. [Run log](arms/async/20260919_224800-airline-smoke/run.log).

Airline's final weight synchronization finished23:07:03 CST; producer disposal returned23:13:50, a406s wait. The shutdown path joined dialogue threads before killing their remote workers, so unused in-flight User calls could hold domain switching. Changed the order to close/terminate remote environment workers before joining local collectors. Saved weights, optimizer state and sampling/credit rules are unchanged. Retail already imported the previous code; subsequent domains load the fix and rerun the preflight suite.

retail smoke: 2 updates completed; finite loss/gradient; iter1 saved. [Run log](arms/async/20260919_224800-retail-smoke/run.log).

telecom smoke: 2 updates completed; finite loss/gradient; iter1 saved. [Run log](arms/async/20260919_224800-telecom-smoke/run.log).

Shutdown fix verified by the Telecom real-training smoke: final weight synchronization to producer disposal completion was0.102s; both updates and iter1 passed. Earlier unmodified Airline/Retail exit waits were406.102s/2.192s. This is an exit-path observation, not a cross-domain training-throughput comparison.

Job21606 ended00:41 CST on Banking before its first update. An Agent request hit the60s read timeout twice; the second exception propagated through the producer. The SGLang logs still showed successful requests and long-context batches, including16 requests with337k–510k cached/context tokens in aggregate. Increased only Banking's HTTP timeout to600s for job21624; retained the finite retry and error propagation. Training hyperparameters, token cap, group filtering, data and User settings are unchanged. Airline/Retail/Telecom each already passed two updates in21606.

Job21624 was stopped at01:10 CST with0 updates and1 retained group, to apply the over-cap sampling fix. Before restart it had no read retries, but continued spending inference on contexts already longer than the16384-token training cap. The Agent now rejects a prompt once even one more output token would exceed that cap. The same two fresh-DB retries and whole-K8 group filter apply. A context exception always forces a retry, even if the retained partial episode is short; after retry exhaustion its loss is zero and the group is removed. Valid-length inference requests and training/evaluation hyperparameters are unchanged. This changes when unusable work ends, so asynchronous completion order can change.

banking smoke: 2 updates completed; finite loss/gradient; iter1 saved. [Run log](arms/async/20260920_011100-banking-smoke/run.log).

airline train: 60 updates completed; finite loss/gradient; iter59 saved. [Run log](arms/async/20260920_011100-airline-train/run.log).

retail train: 30 updates completed; finite loss/gradient; iter29 saved. [Run log](arms/async/20260920_011100-retail-train/run.log).

## Three-domain expert evaluation

Submitted2026-09-20 09:20 CST: Airline iter59 and Retail iter29 each evaluate all test tasks in Airline/Retail/Telecom. Both jobs SUCCEEDED,400/400 simulations each,zero infrastructure errors;8 GPUs each,normal priority. Parameters exactly reuse three-domain reference job21530: official-native Agent,4 trials,seed300,temperature0.6,top_p1,max_tokens1200,max_steps200,max_errors10; Qwen3.6-27B non-thinking User,temperature0,max_tokens512; domain concurrency2/2/5,global9,slot borrowing enabled. Each job converts its specified Megatron checkpoint to HF before evaluation.

- Airline iter59: job21716 (`pt-86hu6wch`). [Launcher](eval_airline_iter59_three_domain.sh); [run log](jobs/21716-expert-airline-iter59-three-domain-0920-091955134/run_0_20260920_091955134.log). Results: `eval/three-domain-airline-expert-iter59/`.
- Retail iter29: job21717 (`pt-kxvbcvl2`). [Launcher](eval_retail_iter29_three_domain.sh); [run log](jobs/21717-expert-retail-iter29-three-domain-0920-092002697/run_0_20260920_092002697.log). Results: `eval/three-domain-retail-expert-iter29/`.

Final results (2026-09-20): cells are pass@1 / pass@4(any) / pass^4 (%). Overall weights the100 tasks as Airline20/Retail40/Telecom40. Raw is Qwen3-4B-Instruct-2507 without project SFT/RL; SFT is the full-domain iter4673 initialization. Historical baselines use the same scoring/generation settings but their three domains were extracted from four-domain runs, so execution concurrency differs; these are single-seed descriptive comparisons.

| Model | Airline | Retail | Telecom | Overall |
|---|---|---|---|---|
| Raw |33.75 /65 /10|52.50 /72.50 /30|17.50 /42.50 /5|34.75 /59 /16|
| SFT4673 |47.50 /70 /25|44.375 /65 /22.50|48.125 /75 /15|46.50 /70 /20|
| Airline expert iter59 |46.25 /65 /30|53.125 /75 /27.50|58.125 /85 /20|53.75 /77 /25|
| Retail expert iter29 |43.75 /65 /20|56.875 /77.50 /30|50.625 /80 /17.50|51.75 /76 /23|

Airline expert vs SFT overall:+7.25/+7/+5pp, but Airline itself:-1.25/-5/+5pp. Retail expert vs SFT on Retail:+12.50/+12.50/+7.50pp; overall:+5.25/+6/+3pp. Single-seed results do not establish statistical significance.

Sources: [Airline expert summary](eval/three-domain-airline-expert-iter59/seed300_20260920_airline_expert_iter59_three_domain_summary.json), [Retail expert summary](eval/three-domain-retail-expert-iter29/seed300_20260920_retail_expert_iter29_three_domain_summary.json), [SFT trajectories](../tau2-airline-db-count-tuning/eval/four-domain-sft-iter4673/trajectories/), [Raw summary](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-banking-expert-sft/eval/raw-full-qwen36-bm25-fixed/seed300_0910_raw_fixed_summary.json).

## Intermediate checkpoint evaluations

Submitted2026-09-20 10:06 CST to compare every currently saved formal Airline/Retail checkpoint, including intermediate checkpoints that may outperform the final weights. Reuse completed iter59/iter29 results above. Each new job serially converts and evaluates its checkpoints on all three test domains,400 simulations per checkpoint,with exactly the same evaluation settings as21716/21717. Each allocation uses8 GPUs,normal priority.

- Airline iter9/19/29/39/49: job21725 (`pt-60ehxz9i`),SUCCEEDED. [Launcher](eval_airline_intermediate_three_domain.sh); [run log](jobs/21725-expert-airline-intermediate-three-domain-0920-100621692/run_0_20260920_100621692.log).
- Retail iter9/19: job21726 (`pt-kx76fbxd`),SUCCEEDED. [Launcher](eval_retail_intermediate_three_domain.sh); [run log](jobs/21726-expert-retail-intermediate-three-domain-0920-100628603/run_0_20260920_100628603.log).

Results use `eval/three-domain-<domain>-expert-iter<iteration>/`. Telecom/Banking have no saved formal checkpoints at submission time. Smoke checkpoints are excluded.

Final sweep results (2026-09-20 12:34 CST): all9 checkpoints completed400/400 each,zero infrastructure errors. Both intermediate jobs SUCCEEDED. Cells are pass@1 / pass@4(any) / pass^4 (%).

| Expert checkpoint | Airline | Retail | Telecom | Overall |
|---|---|---|---|---|
| airline iter9 | 46.25 / 75.00 / 30.00 | 48.75 / 75.00 / 25.00 | 45.62 / 75.00 / 10.00 | 47.00 / 75.00 / 20.00 |
| airline iter19 | 48.75 / 75.00 / 20.00 | 49.38 / 75.00 / 27.50 | 49.38 / 75.00 / 12.50 | 49.25 / 75.00 / 20.00 |
| airline iter29 | 48.75 / 75.00 / 20.00 | 59.38 / 77.50 / 45.00 | 44.38 / 77.50 / 15.00 | 51.25 / 77.00 / 28.00 |
| airline iter39 | 50.00 / 80.00 / 15.00 | 50.62 / 72.50 / 25.00 | 48.12 / 80.00 / 12.50 | 49.50 / 77.00 / 18.00 |
| airline iter49 | 60.00 / 80.00 / 25.00 | 54.37 / 72.50 / 25.00 | 45.00 / 75.00 / 7.50 | 51.75 / 75.00 / 18.00 |
| airline iter59 | 46.25 / 65.00 / 30.00 | 53.12 / 75.00 / 27.50 | 58.13 / 85.00 / 20.00 | 53.75 / 77.00 / 25.00 |
| retail iter9 | 46.25 / 75.00 / 30.00 | 50.00 / 75.00 / 27.50 | 44.38 / 80.00 / 10.00 | 47.00 / 77.00 / 21.00 |
| retail iter19 | 45.00 / 70.00 / 20.00 | 53.12 / 80.00 / 30.00 | 50.00 / 80.00 / 15.00 | 50.25 / 78.00 / 22.00 |
| retail iter29 | 43.75 / 65.00 / 20.00 | 56.88 / 77.50 / 30.00 | 50.62 / 80.00 / 17.50 | 51.75 / 76.00 / 23.00 |

Airline iter49 has the highest Airline pass@1 (60%),+13.75pp over iter59; pass@4(any) improves15pp but pass^4 drops5pp. Overall pass@1 peaks at Airline iter59 (53.75%); overall pass^4 peaks at Airline iter29 (28%). These are seed300 checkpoint comparisons,not significance claims.

Telecom iter9 three-domain evaluation submitted2026-09-20 12:44 CST: job21748 (`pt-c06nh925`),SUCCEEDED,8 GPUs,normal priority. Uses `arms/async/20260920_011100-telecom-train/checkpoints/iter_0000009`; converts to HF in the job. All evaluation parameters exactly match the preceding expert evaluations (400 simulations,seed300,4 trials). [Launcher](eval_telecom_iter9_three_domain.sh); [run log](jobs/21748-expert-telecom-iter9-three-domain-0920-124408549/run_0_20260920_124408549.log). Results: `eval/three-domain-telecom-expert-iter9/`.

Telecom iter9 completed400/400,zero infrastructure errors (checked2026-09-20 13:20 CST). pass@1/pass@4(any)/pass^4: Airline42.50/65/30%,Retail47.50/72.50/22.50%,Telecom41.875/70/10%,overall44.25/70/19%. Versus SFT,Telecom changes-6.25/-5/-5pp and overall-2.25/0/-1pp. [Summary](eval/three-domain-telecom-expert-iter9/seed300_20260920_telecom_expert_iter9_three_domain_summary.json). All10 submitted expert checkpoints are now complete.

## Seed301 evaluation

Submitted2026-09-20 13:43 CST. Only evaluation seed changes300→301; retain all three domains,4 trials,400 simulations/checkpoint and all prior generation/User/concurrency settings. Reuse HF weights. Trajectories and summaries use separate `seed301_` filenames under the same per-checkpoint evaluation directories.

- Airline iter29→49→59,serial: job21761 (`pt-h002d9v6`),SUCCEEDED,8 GPUs,normal priority. [Launcher](eval_airline_selected_seed301_three_domain.sh); [run log](jobs/21761-expert-airline-selected-seed301-0920-134305388/run_0_20260920_134305388.log).
- Retail iter29: job21762 (`pt-z6a4kq9p`),SUCCEEDED,8 GPUs,normal priority. [Launcher](eval_retail_iter29_seed301_three_domain.sh); [run log](jobs/21762-expert-retail-iter29-seed301-0920-134312906/run_0_20260920_134312906.log).

Seed301 complete (checked2026-09-20 15:21 CST):1600/1600 simulations,zero infrastructure errors. Both jobs SUCCEEDED. Each cell below is pass@1 / pass@4(any) / pass^4 (%); mean rows average seed300/301 metrics separately.

| Model | Seed | Airline | Retail | Telecom | Overall |
|---|---|---|---|---|---|
| airline iter29 | 300 | 48.75 / 75.00 / 20.00 | 59.38 / 77.50 / 45.00 | 44.38 / 77.50 / 15.00 | 51.25 / 77.00 / 28.00 |
| airline iter29 | 301 | 43.75 / 70.00 / 25.00 | 56.88 / 82.50 / 35.00 | 45.62 / 80.00 / 10.00 | 49.75 / 79.00 / 23.00 |
| airline iter29 | mean | 46.25 / 72.50 / 22.50 | 58.12 / 80.00 / 40.00 | 45.00 / 78.75 / 12.50 | 50.50 / 78.00 / 25.50 |
| airline iter49 | 300 | 60.00 / 80.00 / 25.00 | 54.37 / 72.50 / 25.00 | 45.00 / 75.00 / 7.50 | 51.75 / 75.00 / 18.00 |
| airline iter49 | 301 | 50.00 / 75.00 / 20.00 | 56.88 / 85.00 / 30.00 | 55.62 / 87.50 / 17.50 | 55.00 / 84.00 / 23.00 |
| airline iter49 | mean | 55.00 / 77.50 / 22.50 | 55.62 / 78.75 / 27.50 | 50.31 / 81.25 / 12.50 | 53.38 / 79.50 / 20.50 |
| airline iter59 | 300 | 46.25 / 65.00 / 30.00 | 53.12 / 75.00 / 27.50 | 58.13 / 85.00 / 20.00 | 53.75 / 77.00 / 25.00 |
| airline iter59 | 301 | 51.25 / 80.00 / 15.00 | 53.75 / 77.50 / 32.50 | 52.50 / 82.50 / 17.50 | 52.75 / 80.00 / 23.00 |
| airline iter59 | mean | 48.75 / 72.50 / 22.50 | 53.44 / 76.25 / 30.00 | 55.31 / 83.75 / 18.75 | 53.25 / 78.50 / 24.00 |
| retail iter29 | 300 | 43.75 / 65.00 / 20.00 | 56.88 / 77.50 / 30.00 | 50.62 / 80.00 / 17.50 | 51.75 / 76.00 / 23.00 |
| retail iter29 | 301 | 50.00 / 80.00 / 20.00 | 56.88 / 85.00 / 27.50 | 41.88 / 85.00 / 12.50 | 49.50 / 84.00 / 20.00 |
| retail iter29 | mean | 46.88 / 72.50 / 20.00 | 56.88 / 81.25 / 28.75 | 46.25 / 82.50 / 15.00 | 50.62 / 80.00 / 21.50 |

Overall mean pass@1:Airline49 53.375%,Airline59 53.25% (only0.125pp difference). Overall mean pass^4:Airline29 25.5%,Airline59 24%. Airline29 Retail pass^4 exceeds Retail29 in both seeds (45/35% vs30/27.5%). No significance claim; Raw/SFT baselines remain seed300 only.

## Banking evaluation

Airline iter49 Banking-only evaluation submitted2026-09-20 15:29 CST: job21769 (`pt-74i6c60u`),STOPPED,8 GPUs,normal priority. Domain `banking_knowledge`,BM25 retrieval,full test split,4 trials,seed300. Same Agent/User generation parameters as expert evaluations;single-domain concurrency9,slot borrowing disabled. Reuses existing iter49 HF weights. [Launcher](eval_airline_iter49_banking.sh); [run log](jobs/21769-expert-airline-iter49-banking-0920-152916889/run_0_20260920_152916889.log). Results: `eval/banking-airline-expert-iter49/`.

2026-09-20T16:27:09.672242+08:00: user requested stopping Banking evaluation;job21769 confirmed STOPPED,existing artifacts retained. SFT4673 used the old7016-row Banking expert export,not the later simplified5672-row export;all7016 Banking rows in the mixed input match the old source line-for-line. See [slowdown diagnosis](banking-slow-diagnosis.md). Training jobs were not stopped.

telecom train: 20 updates completed; finite loss/gradient; iter19 saved. [Run log](arms/async/20260920_011100-telecom-train/run.log).
