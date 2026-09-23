# Airline final-checkpoint comparison

Both final iter29 models and SFT4673 evaluated on the same official Airline test split: 20 tasks × 4 trials = 80 simulations/model, seed300, Agent temperature 0.6, top-p 1, max_tokens 1200, max_steps 200, Qwen3.6-27B User. All three completed with zero infrastructure errors.

| Model | pass@1 | pass@4(any) | pass^4 | Evidence |
|---|---:|---:|---:|---|
| SFT4673 | 46.25% | 70.00% | 30.00% | [summary](../eval/sft-iter4673/seed300_20260916_125855_summary.json) |
| DB-count LR 2e-6 iter29 | 46.25% | 70.00% | 25.00% | [summary](../eval/20260916_125855-progress-db-count-v1-lr2e-6-iter29/seed300_20260916_125855_summary.json) |
| DB-count LR 5e-6 iter29 | 46.25% | 75.00% | 10.00% | [summary](../eval/20260916_125855-progress-db-count-v1-lr5e-6-iter29/seed300_20260916_125855_summary.json) |

Jobs 19999 / 20000 both succeeded: 30 updates, final iter29 saved/exported, then automatic Airline evaluation. All models succeeded on 37/80 simulations. At least one success occurred on 14/20, 14/20 and 15/20 tasks; four successes occurred on 6/20, 5/20 and 2/20 tasks, respectively. Thus 5e-6 gains 5 percentage points in pass@4(any) versus SFT but loses 20 points in pass^4; single-shot success is unchanged. These one-seed descriptive results do not establish an overall improvement over SFT or statistical significance.

## Completed domains from four-domain evaluations

Updated 2026-09-18. All cells below are pass@1 / pass@4(any) / pass^4 (%). Training is Airline-only; official test domains contain 20/40/40 tasks ×4 trials. Seed300, Qwen3.6-27B User, temperature0.6, top-p1, max_tokens1200/max_steps200. Keep these results separate from the earlier Airline-only execution above; generation is not deterministic.

| Model | Airline | Retail | Telecom |
|---|---|---|---|
| Raw model (no project SFT/RL) | 33.75 / 65.00 / 10.00 | 52.50 / 72.50 / 30.00 | 17.50 / 42.50 / 5.00 |
| SFT4673 (no RL) | 47.50 / 70.00 / 25.00 | 44.38 / 65.00 / 22.50 | 48.12 / 75.00 / 15.00 |
| LR2e-6 batch64 iter29 | 55.00 / 75.00 / 25.00 | 48.75 / 72.50 / 27.50 | 46.88 / 77.50 / 12.50 |
| LR5e-6 batch64 iter29 | 47.50 / 75.00 / 15.00 | 55.62 / 80.00 / 22.50 | 49.38 / 77.50 / 17.50 |

LR arms each use batch64, 30 updates and 1,920 accepted trajectories. Airline favors LR2e-6 over LR5e-6: +7.50/0/+10.00pp.

## Batch comparison at LR2e-6

| Model | Updates | Accepted trajectories | Airline | Retail | Telecom |
|---|---:|---:|---|---|---|
| SFT4673 | 0 | 0 | 47.50 / 70.00 / 25.00 | 44.38 / 65.00 / 22.50 | 48.12 / 75.00 / 15.00 |
| Batch64 iter29 | 30 | 1920 | 55.00 / 75.00 / 25.00 | 48.75 / 72.50 / 27.50 | 46.88 / 77.50 / 12.50 |
| Batch128 iter19 | 20 | 2560 | 56.25 / 80.00 / 30.00 | 50.00 / 70.00 / 25.00 | 50.62 / 85.00 / 12.50 |
| Batch256 iter9 | 10 | 2560 | 48.75 / 75.00 / 25.00 | 52.50 / 75.00 / 30.00 | 44.38 / 75.00 / 12.50 |

Batch128 iter19 versus batch256 iter9 is an equal-accepted-trajectory comparison, not an equal-update comparison; Airline improves +7.50/+5.00/+5.00pp. Against SFT, batch128 iter19 improves +8.75/+10.00/+5.00pp. Against batch64 iter29 it improves +1.25/+5.00/+5.00pp, with a larger training budget. LR2e-6/batch128 is the selected working baseline. Results are single-seed observations, not significance claims. The planned 30-update batch study was stopped early by the user.

Retail for batch128 iter19 and Airline for batch256 iter9 were resumed with single-domain concurrency9 in successful jobs 20618/20620, reusing completed trajectories. Other generation/scoring settings stayed fixed. All completed three-domain results have zero infrastructure errors. Banking remains incomplete and paused; no four-domain aggregate is reported.

## External User and sampling-speed controls

Updated2026-09-19. Same official three-domain protocol: seed300,4 trials, Agent temperature0.6/top_p1/max_tokens1200/max_steps200, Qwen3.6-27B User temperature0/max_tokens512, internal8-GPU evaluation. Cells remain pass@1 / pass@4(any) / pass^4 (%).

| Model | Updates | Airline | Retail | Telecom |
|---|---:|---|---|---|
| Raw model, no SFT/RL |0|33.75 /65 /10|52.50 /72.50 /30|17.50 /42.50 /5|
| SFT4673, no RL |0|47.50 /70 /25|44.375 /65 /22.50|48.125 /75 /15|
| Original batch128 iter19 |20|56.25 /80 /30|50 /70 /25|50.625 /85 /12.50|
| External-User progress1 iter19,21033 |20|52.50 /80 /25|48.125 /67.50 /30|43.75 /82.50 /10|
| Fast shuffled/retry iter29,21168 |30|48.75 /75 /20|53.75 /82.50 /32.50|45.625 /82.50 /5|
| Fast shuffled/retry iter19,21417 eval |20|53.75 /80 /30|52.50 /77.50 /30|44.375 /85 /7.50|
| Fast uniform-source control21429,21530 eval |20|42.50 /60 /30|pending (138/160)|pending (120/160)|

Evaluation21377 completed400/400. Versus the external-User iter19 control, fast iter29 changes pass@1 by-3.75/+5.625/+1.875pp on Airline/Retail/Telecom. Airline is39/80 versus42/80; the different training budgets and single seed do not establish a causal regression. The speed change also altered task sampling; [quality diagnosis](quality-diagnosis/README.md) documents the measured distribution change, paired failures, and matched20-update controls. At matched20 updates,21417 Airline is53.75/80/30 (43/80 successes), versus old external-User52.50/80/25 and fast iter2948.75/75/20. This does not show a speed-induced decline; fast iter19 is the better observed Airline checkpoint. 21417 completed400/400; Retail52.50/77.50/30 and Telecom44.375/85/7.50. Both fast iter19 and iter29 have weighted three-domain pass@1=49.50%; the earlier checkpoint specifically improves Airline. All engineering speed fixes are retained in21429 while restoring the old uniform task draws as a separate control, whose quality result is not yet known.

Sources: [external iter19](../eval/three-domain-b128-progress1-external-iter19/seed300_20260919_progress1_external_iter19_three_domain_summary.json), [fast iter29](../eval/three-domain-b128-progress1-sampling-fix-iter29/seed300_20260919_sampling_fix_iter29_three_domain_summary.json).

## Evidence

- [four-domain-sft-iter4673](../eval/four-domain-sft-iter4673/trajectories/).
- [four-domain-lr2e-6-iter29](../eval/four-domain-lr2e-6-iter29/trajectories/).
- [four-domain-lr5e-6-iter29](../eval/four-domain-lr5e-6-iter29/trajectories/).
- [four-domain-lr2e-6-b128-iter19](../eval/four-domain-lr2e-6-b128-iter19/trajectories/).
- [four-domain-lr2e-6-b256-iter9](../eval/four-domain-lr2e-6-b256-iter9/trajectories/).
- [Raw-model summary](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-banking-expert-sft/eval/raw-full-qwen36-bm25-fixed/seed300_0910_raw_fixed_summary.json).

2026-09-19 20:13 CST: uniform-source control21429 SUCCEEDED20/20; matched training time71.86min versus80.46min fast shuffled/retry and370.22min old external uniform. Evaluation21530 RUNNING with unchanged protocol; no domain complete yet. Quality impact of restoring uniform task selection remains unresolved.

2026-09-19 20:28 CST:21530 Airline completed80/80:42.50/60/30% (34 successes), compared with fast shuffled/retry iter19 53.75/80/30% (43 successes) and SFT4673 47.50/70/25%. The uniform-source replacement has not improved Airline; it is not selected as the new recipe. Restored the generic launcher default to the previously evaluated shuffled/retry source, retaining all engineering speed fixes. Retail138/160 and Telecom120/160 are incomplete. The single seed does not establish that delayed retries themselves are beneficial or harmful.
