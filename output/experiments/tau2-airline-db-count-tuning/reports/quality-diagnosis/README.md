# Fast asynchronous RL quality diagnosis

Purpose: identify the Airline evaluation difference while preserving the measured4.60x sampling/training speedup. Experiment: `tau2-airline-db-count-tuning`. Updated2026-09-19 20:30 CST.

Matched-budget finding: fast21168 iter19 Airline is**53.75 /80 /30%**, versus**52.50 /80 /25%** for the old external-User iter19. Thus the same20-update comparison does not show a speed-induced loss. Fast iter29 is worse than fast iter19 in this evaluation; retain iter19 as the better observed Airline checkpoint. Uniform-source control21429 has now completed Airline evaluation:42.50/60/30%, below fast shuffled/retry iter19 53.75/80/30%. The uniform replacement is not selected; the generic launcher default has been returned to ShuffledTaskDataSource. This single-seed comparison does not support blaming delayed all-zero retries for the earlier performance difference.

## Official evaluation

All rows below use the existing official test tasks,4 trials and seed300. Percentages are pass@1 / pass@4(any) / pass^4. Training is Airline-only. The first historical local-User comparison also changes deployment; the external-User comparison is closer.

| Model | Training updates | Airline | Retail | Telecom |
|---|---:|---:|---:|---:|
| Raw Instruct, no SFT/RL |0|33.75 /65 /10|52.50 /72.50 /30|17.50 /42.50 /5|
| SFT4673, no RL |0|47.50 /70 /25|44.375 /65 /22.50|48.125 /75 /15|
| Original batch128 iter19, local User |20|56.25 /80 /30|50 /70 /25|50.625 /85 /12.50|
| External-User control21033 iter19 |20|52.50 /80 /25|48.125 /67.50 /30|43.75 /82.50 /10|
| Fast21168 iter29 |30|48.75 /75 /20|53.75 /82.50 /32.50|45.625 /82.50 /5|
| Fast21168 iter19, evaluation21417 |20|53.75 /80 /30|52.50 /77.50 /30|44.375 /85 /7.50|
| Fast uniform-source control21429 iter19, evaluation21530 |20|42.50 /60 /30|138/160 evaluated|120/160 evaluated|

Sources: [historical comparison](../comparison.md), [external iter19 summary](../../eval/three-domain-b128-progress1-external-iter19/seed300_20260919_progress1_external_iter19_three_domain_summary.json), [fast iter29 summary](../../eval/three-domain-b128-progress1-sampling-fix-iter29/seed300_20260919_sampling_fix_iter29_three_domain_summary.json).

Fast iter29 has39/80 Airline successes versus42/80 for the external iter19 control. Pairing by task/trial gives25 shared successes,24 shared failures,17 lost successes and14 gained successes. This single seed and the20-versus30-update comparison do not establish a causal degradation. Retail and Telecom pass@1 improved, while Telecom pass^4 fell; this is not an across-domain collapse. No infrastructure-error termination occurred in the80 fast Airline trials. The task-paired bootstrap95% interval for the pass@1 difference is[-18.75,+11.25] percentage points (10,000 resamples of20 tasks, bootstrap seed42); this is an uncertainty diagnostic, not an evaluation-parameter change.

Of17 lost successes,16 fail the DB check and one passes DB but misses required communication; there are also14 gained successes. Concrete differences include task19/trial0 modifying a reservation instead of cancelling it, task45/trial0 cancelling when the expected DB stays unchanged, and task32/trial3 retaining an extra flight segment. These are action/state errors, not failed API requests. Full paired records: [Airline failure comparison](airline_paired_failures.json).

## Confirmed training differences

Both runs use SFT4673,LR2e-6,batch128,K8,progress1,format1,gamma0.98,KL0,entropy0,seeds1234/42, the same prepared Airline task file and external User20873. The speed work also changed the sampler from uniform random draws with replacement to a shuffled deck plus all-zero retries after two published policy updates. That is an algorithm change, independent of the CPU/GPU scheduling fixes.

Matched first20 updates,320 trained groups:

| Observed metric | Old external control | Fast shuffled/retry |
|---|---:|---:|
| Completed groups by update20 |507|577|
| Filtered groups by update20 |187 (36.88%)|251 (43.50%)|
| Unique trained tasks |276|320|
| Groups revisiting an already trained task |44|0|
| Trained groups from a previously sampled task |56|38|
| Mean success within those repeated draws |49.55%|18.09%|
| Repeated groups with exactly1/8 success |9/56|28/38|
| Accepted training success |47.93%|45.39%|
| Mean earliest-turn policy lag, updates |1.17|1.62|
| Mean training/behavior log-prob absolute difference |0.00585|0.00612|
| Mean TIS clipped fraction |0.000055%|0.000017%|
| Token-weighted progress/outcome sign disagreement |16.27%|19.38%|
| Token-weighted total advantage/outcome sign disagreement |7.90%|9.47%|

At the end of update20 the fast run also has6 retained groups awaiting later updates; both comparisons contain exactly320 trained groups. [Completion-window counts](first20_sampling_counts.json).

Full fast run:915 completed groups,282 repeated task draws,73 retained repeats, of which70 actually trained.480 trained groups correspond to480 distinct tasks. This differs from counting all repeated samples as trained. Retry groups mostly remain all-zero (208/282) and are filtered. Reward shaping was not changed and no progress payload was unavailable in either run's trained groups. Group sizes and task identities are checked before advantage computation; rollout log-prob correction is almost never clipped.

The observed changes justify separating task-selection policy from engineering speed. They do not prove delayed retries caused the3-trial Airline difference. A larger mean version lag alone is also insufficient evidence: actual log-prob divergence is almost unchanged.

The first-draw subset has49.07% accepted success over the fast first20 updates versus47.59% in the old run; the overall curve is reduced by the18.09%-success retry subset. These subsets contain different tasks and are not held-out quality comparisons. [Reward mixture plot](sampling_mix_reward.png), [five-update aggregates](sampling_mix_reward.json), [plot script](plot_sampling_mix.py).

Reproduce from existing logs with [analysis script](analyze.py); compact aggregates and trained-group IDs are in [training distribution JSON](training_distribution.json). Turn-error counts include repeated calls and argument errors, not solely format penalties.

## Repair and controlled verification

The temporary default change to `RandomTaskDataSource` was reversed after control21429 scored42.50/60/30% on Airline. `scripts/run_tau2_airline_tuning.sh` again defaults to `ShuffledTaskDataSource`, matching the better observed fast21168 iter19 recipe. The21429 launcher remains explicitly uniform so the experiment remains reproducible. Preserved four CPU environment workers,256 in-flight trajectory slots,64 partial groups, per-trajectory refill, completed-group consumption, immutable DB/reference caches and60-second Agent HTTP waits with the existing one retry. Reward, advantage, LR,batch,K,User,decoding and official evaluation parameters remain unchanged. Monitoring defaults to300 seconds.

Evaluation21417 converts and evaluates fast21168 iter19 with the same8-GPU three-domain wrapper: [launcher](../../scripts/eval_b128_sampling_fix_iter19_three_domain.sh), [run log](../../jobs/21417-airline-b128-sampling-fix-iter19-three-domain-0919-182430278/run_0_20260919_182430278.log). RUNNING at18:30 CST. Airline completed80/80 at18:50 CST:43 successes,16/20 tasks with any success,6/20 with all four successes. One max_steps termination, no infrastructure-error termination. Compared with fast iter29 this is+5/+5/+10pp; compared with old external-User iter19,+1.25/0/+5pp. The original local-User baseline remains56.25/80/30, only2/80 more successes. [Matched comparison](matched_iter19_airline.json), [Airline trajectories](../../eval/three-domain-b128-progress1-sampling-fix-iter19/trajectories/seed300_20260919_sampling_fix_iter19_three_domain_airline_test_4trials/results.json). Evaluation21417 has now SUCCEEDED400/400; Retail52.50/77.50/30 and Telecom44.375/85/7.50. Weighted three-domain pass@1 is49.50%, the same as fast iter29; iter19 is specifically better on Airline. This demonstrates that the earlier20-versus30-update comparison cannot attribute the Airline loss to engineering speed; it does not prove overfitting or any single sampling mechanism.

Control21429 starts fresh from the same SFT for20 updates, restoring the old uniform sampler under all fast engineering settings: [launcher](run_fast_uniform_control20.sh), [run log](../../jobs/21429-airline-b128-fast-uniform-control20-0919-183011389/run_0_20260919_183011389.log). RUNNING on H10080GB at18:45 CST. Container preflight passed30 progress/DB tests and108 continuous-rollout tests (138 total). Runtime arguments confirm RandomTaskDataSource,4 environment workers,pool32,max64 partial groups,unlimited lag and20 updates. Sampling began18:43:13 CST. Bash syntax passed locally. Replaying uniform task draws with seed42 reproduces the task identity for all320 trained group IDs from the old external control (zero mismatches). Its iter19 will receive the unchanged three-domain evaluation. Completed20/20 updates successfully. All20 loss/gradient values are finite; grad norms0.429–0.627. The matched20-update run took71.86min, versus370.22min old uniform (5.15x) and80.46min previous fast shuffled/retry (10.68% less time). Mean Generator utilization76.01%, Trainer37.23%; training compute81.54s/update. Full-run completion counts521 groups,328 retained,193 filtered,320 trained,8 retained-unused. Airline21530 is complete:34/80 successful simulations,12/20 tasks with any success and6/20 with all four successes. This is-11.25/-20/0pp versus fast21168 iter19, and-5/-10/+5pp versus non-RL SFT4673. There is one max_steps termination and no infrastructure-error termination. Retail and Telecom remain incomplete. The existing fast21168 iter19 already retains the measured4.60x first20 speedup and improves over the old external-User iter19 on this single seed. Its HF checkpoint is `arms/async/20260919-progress1-b128-sampling-fix-long30-timeout60/checkpoints/iter_0000019_hf`.

The [evaluation follow-up](evaluate_uniform_when_done.py) checked21429 every300 seconds and submitted evaluation21530 at19:59 CST after training succeeded.21530 is RUNNING with the unchanged8-GPU three-domain protocol; HF conversion completed and simulations are underway. [Follow-up log](evaluation_followup.log). This avoids reserving evaluation GPUs during training; no evaluation parameters change.

21429 first-update timing: sampling admission to optimizer completion30.80→8.25 minutes versus the old uniform-source control (3.73x); Generator utilization23.23→75.35%. This startup window is superseded by the completed20-update comparison above. [Live control speed report](fast-uniform-control20/README.md).

Completed control: [training summary](fast-uniform-control20/training_summary.json), [reward curve](../../plots/20260919-progress1-b128-fast-uniform-control20/reward_curves.png), [evaluation21530 log](../../jobs/21530-airline-b128-fast-uniform-iter19-three-domain-0919-195948995/run_0_20260919_195948995.log). At20:28 CST, evaluation has338/400 completed simulations (Airline80/80,Retail138/160,Telecom120/160); only Airline has a final score. The remaining domains continue under unchanged parameters.

The uniform control has higher accepted training reward (51.02% versus45.39% for the fast shuffled/retry first20), yet lower Airline evaluation. Higher training reward is therefore not a selection criterion here. Its320 trained groups cover279 unique tasks; the old uniform run covers276. Although all draw IDs still map to the same seed42 task choices, only244/320 group IDs and219 task identities are common between the actual trained sets. Matching a task RNG does not reproduce the accepted training data under changed completion order, policy trajectory and filtering. This observation does not identify which mechanism caused the score difference. Log-prob divergence remains small (0.00619 average; TIS clipped fraction0.000167%). [Uniform control analysis](uniform_completed_airline_analysis.json).
