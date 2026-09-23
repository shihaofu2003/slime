# tau2-async-db-count-four-domain

Purpose: compare four-domain asynchronous vanilla GRPO and `progress-db-count-v1` from full-domain SFT iter4673. Implementation and smoke are verified; controlled training/evaluation results are pending.

## Fixed protocol

Each arm: 556 updates (final iter555), 5 tasks × K=8 per update, 2,524-task pool sampled with replacement, pool=20, unlimited lag, LR=2e-6, KL/entropy=0, PPO clip=0.2, TIS clip=2, seeds 1234/42. User: Qwen3.6-27B. Training max_steps=200, cap=16,384 tokens, two over-cap retries. Updated 2026-09-15: both new arms reject K=8 groups whose official terminal rewards are all 0 or all 1, even when auxiliary credit is nonzero. Tasks remain in the with-replacement pool. Historical runs retained uniform groups and are separate controls. Independent optimizer/checkpoint roots.

Evaluation: SFT, vanilla iter555, DB-count iter555; seed300, four trials on 20/40/40/97 Airline/Retail/Telecom/Banking tasks, 788 trajectories/model. Agent temperature=0.6; BM25 Banking. Each training/evaluation job uses 8 GPUs, 64 CPUs, 1024 GB, normal priority. Updated user instruction: run the two training arms concurrently in separate jobs; the user will stop the original serial job after vanilla training completes.

Training data: Airline 1,148 / Retail 563 / Telecom 271 / Banking 542. All original three-domain task DB paths are preserved (4 / 4 / 1 distinct DBs); fresh environments reload the task DB and apply its initial state. Banking uses the synthetic empty transaction DB plus each task’s initialization, 480 allowed documents and BM25. The same constructor serves rollout, reference state and terminal reward; 338 Banking DB tasks have progress, while 204 ACTION/COMMUNICATE tasks have outcome/format only.

## Implementation review

Fixed missing raw-token credit mapping, per-segment metadata, Banking table.data record counting on both sides, unavailable Tau2 source path, disabled reward postprocessing, smoke update override, SFT evaluation label and RL conversion. Cluster preflight v2 passed 94 pytest cases and the rollout logic preflight.

Entry points: `scripts/run_tau2_async_four_domain.sh <vanilla-grpo|progress-db-count-v1> <smoke|train>` and `scripts/eval_tau2_async_four_domain.sh <sft|vanilla-grpo|progress-db-count-v1> [iteration]`. Set `TAU2_RUN_NAME` to the same explicit training run name for RL evaluation. Resume uses `LOAD_DIR`, `CKPT_STEP` and a new `TAU2_RUN_NAME`; `SMOKE_UPDATES` is the total update endpoint.

## Jobs

- CPU preflight on cluster, job 19318 (`pt-jdigsxgi`): [run log](jobs/19318-four-domain-db-count-preflight-0915-001501937/run_0_20260915_001501937.log). Failed: 86 passed, 5 test expectation/mock failures; fixed before v2. Real four-domain environments and CP loss/gradient checks passed.

- Preflight v2, job 19324: [run log](jobs/19324-four-domain-db-count-preflight-v2-0915-002019034/run_0_20260915_002019034.log). Passed: 94 pytest cases and rollout logic preflight, including all 9 original training DB paths.

- Five-update DB-count smoke, job 19330: [run log](jobs/19330-four-domain-db-count-smoke-0915-002354385/run_0_20260915_002354385.log). Run: `20260915-progress-db-count-v1-smoke`; startup reruns preflight. Stopped at preflight: DB-count 24 passed; continuous tests 72 passed / 8 failed because default-launcher tests inherited NUM_ROLLOUT=5. Fixed subprocess environment isolation. No optimizer updates ran.

- Additional CPU regression, job 19336: [submission log](jobs/19336-four-domain-db-count-cpu-regression-0915-002904307/submit_20260915_002904307.log). 0 GPU / 8 CPU / 32 GB; cancelled before execution because per-user FIFO placed it behind smoke, whose startup already runs these tests. No run log was created.

- Five-update smoke v2, job 19338: [run log](jobs/19338-four-domain-db-count-smoke-v2-0915-003429419/run_0_20260915_003429419.log). Run: `20260915-003500-progress-db-count-v1-smoke`; Succeeded as `pt-xlmu2p4d`; iter0–4 saved. Startup passed 24 DB-count tests, 80 continuous tests and rollout logic preflight.

- Resume smoke, job 19350: [run log](jobs/19350-four-domain-db-count-resume-smoke-0915-010543997/run_0_20260915_010543997.log). `20260915-progress-db-count-v1-resume-smoke` resumes smoke v2 iter2 for updates 3–4 in an independent directory; Stopped before resume: the legacy smoke inspector hard-coded lag≤1. Inspector now reads the configured limit from the training log; regression test added.

- Resume smoke v2, job 19352: [run log](jobs/19352-four-domain-db-count-resume-smoke-v2-0915-010859219/run_0_20260915_010859219.log). Run: `20260915-011000-progress-db-count-v1-resume-smoke`; resumes smoke v2 iter2 for two updates in a separate directory. Smoke inspection passed with no violations; startup passed 24 DB-count and 81 continuous tests plus rollout preflight. Succeeded. Iter3–4 completed; all 80 pending inputs regenerated, no consumed samples repeated, five new task draws matched the restored RNG, and trained groups increased 15→25. [Recovery report](reports/resume-smoke.json).

- Formal serial workflow, job 19358: [run log](jobs/19358-four-domain-db-count-train-eval-0915-013936380/run_0_20260915_013936380.log), [submission log](jobs/19358-four-domain-db-count-train-eval-0915-013936380/submit_20260915_013936380.log). Failed after 13 optimizer updates as `pt-eaan5mgp`; last saved iter9. Banking retrieval grew one request to 261,078 input tokens + 1,200 completion tokens, exceeding the 262,144 server limit. Specific context-overflow errors now enter the existing whole-trajectory cap/retry/filter path. Resources: 8 GPUs / 64 CPUs / 1024 GB, normal priority. Run stamp: `20260915_014000`. Both training arms and all evaluations execute serially in this allocation. Vanilla startup passed rollout preflight and all 81 continuous tests. First optimizer update (iter0) completed at 2026-09-15 01:54 CST: 5 groups / 40 trajectories, finite loss, gradient norm 0.5564, TIS clip fraction 0, LR 2e-6. Iter9 and its rollout state were saved successfully at 02:14 CST after 10 updates; periodic evaluation begins at iter49.

- Formal serial workflow v2, job 19367: [run log](jobs/19367-four-domain-db-count-train-eval-v2-0915-023300093/run_0_20260915_023300093.log). Fresh SFT initialization and optimizer roots, run stamp `20260915_023500`; 8 GPU / 64 CPU / 1024 GB, normal priority. Failed as `pt-8wrr4gpp` after six completed updates, before the first checkpoint. The input alone reached 263,229 tokens; SGLang returned its second context-overflow message, which is now handled by the same retry path. Supersedes failed job 19358; its partial results are excluded from the controlled comparison.

- Formal serial workflow v3, job 19376: [run log](jobs/19376-four-domain-db-count-train-eval-v3-0915-031708260/run_0_20260915_031708260.log). Fresh SFT initialization and independent optimizer roots; run stamp `20260915_031800`. 8 GPU / 64 CPU / 1024 GB, normal priority. Running; first update (iter0) completed at 03:34:43 CST: batch=40, loss=0.00013836, gradient norm=0.43942, TIS clip fraction=0, LR=2e-6. Job 19375 passed all 112 tests and rollout preflight before submission. Supersedes both failed formal runs. By 03:59 CST, 11 updates completed and iter9 plus rollout state were saved. Actual server context overflow at sample419 attempt0 entered a fresh attempt1; training continued through iter10. Another overflow at sample393 attempt1 also entered the retry path.

| Task in job 19376 | Phase log |
|---|---|
| Vanilla GRPO, 556 updates | [training log](arms/async/20260915_031800-vanilla-grpo-train/run.log) |
| DB-count, 556 updates | [training log](arms/async/20260915_031800-progress-db-count-v1-train/run.log) |
| SFT official evaluation | [evaluation log](eval/sft-iter4673/run_20260915_031800.log) |
| Vanilla iter555 official evaluation | [evaluation log](eval/vanilla-grpo-iter555/run_20260915_031800.log) |
| DB-count iter555 official evaluation | [evaluation log](eval/progress-db-count-v1-iter555/run_20260915_031800.log) |

The full serial entry point is `scripts/run_tau2_async_four_domain_experiment.sh`: vanilla 556 updates, DB-count 556 updates, three official evaluations, then the comparison report. `RUN_STAMP` names both independent training directories and evaluation outputs. Per-phase logs are saved beside their artifacts; a failed phase stops subsequent work.

- Hourly monitor regression tests, job 19361: [run log](jobs/19361-four-domain-hourly-monitor-tests-0915-014612601/run_0_20260915_014612601.log). Succeeded: 2 tests passed. CPU-only, 8 CPU / 32 GB; checks completed-save detection, 50-update cadence, duplicate submission avoidance and evaluation arguments/documentation.

## Hourly checks and periodic evaluation

`scripts/monitor_tau2_four_domain.py 19376 20260915_031800` checks job-manager and both training logs every 3,600 seconds. It submits four-domain evaluation after completed checkpoint saves at iter49,99,…,549 for each arm (22 evaluations total), using the same seed300 × 4-trial protocol and 8 GPU / 64 CPU / 1024 GB, normal priority. Evaluations run as separate jobs while the training arms remain serial. Existing queued, running or failed evaluation jobs are reported without duplicate resubmission. Final iter555 evaluations run in job 19376. [Hourly observations](reports/hourly-monitor.log). First automatic hourly check ran at 04:17:41 CST (3,600 seconds after initial observation): job RUNNING, 18 updates completed, iter9 saved, no periodic evaluation due yet. Current monitor PID 307314 follows job 19376; PID 244798 was stopped after job 19367 failed; initial PID 181635 was stopped after job 19358 failed; it only reads files and calls job-manager, while tests, conversion and evaluation run on the compute cluster.

- Context-overflow regression preflight, job 19365: [run log](jobs/19365-four-domain-context-overflow-preflight-0915-022624472/run_0_20260915_022624472.log). CPU-only 8 CPU / 32 GB; 109 tests passed, one CPU policy-loss test failed because Torch Dynamo tried to initialize an unavailable CUDA device. HTTP overflow detection tests passed.

- Context-overflow preflight v2, job 19366: [run log](jobs/19366-four-domain-context-overflow-preflight-v2-0915-022925756/run_0_20260915_022925756.log). CPU-only 8 CPU / 32 GB, `TORCHDYNAMO_DISABLE=1` for this test job only. Passed 111 tests plus rollout logic preflight, including overflow recovery with a fresh environment/user on retry.

- Context-overflow preflight v3, job 19375: [run log](jobs/19375-four-domain-context-overflow-preflight-v3-0915-031438702/run_0_20260915_031438702.log). CPU-only 8 CPU / 32 GB, normal priority, `TORCHDYNAMO_DISABLE=1`. Adds the actual input-only context-overflow response from job 19367 to regression coverage. Succeeded: 112 tests and rollout logic preflight passed.

- Periodic vanilla-grpo iter49 (50 updates), job 19462: [run log](jobs/19462-four-db-20260915_031800-v-0049-0915-071803106/run_0_20260915_071803106.log). Four domains, seed300 × 4 trials; 8 GPU / 64 CPU / 1024 GB, normal priority.

- Standalone DB-count, job 19469: [run log](jobs/19469-four-domain-db-count-parallel-0915-082243293/run_0_20260915_082243293.log). Submitted 2026-09-15 08:22 CST; 8 GPU / 64 CPU / 1024 GB, normal priority. Queued: 8 GPUs requested, 5 remaining in quota. Run `20260915_082300-progress-db-count-v1-train`: fresh SFT iter4673 and optimizer, 556 updates, then iter555 four-domain evaluation. Startup runs monitor regression tests and existing training preflight. [Training log](arms/async/20260915_082300-progress-db-count-v1-train/run.log). Original jobs 19376 and 19462 remain running; the user plans to stop 19376 after its vanilla arm finishes. Its original serial script remains unchanged. Final comparison must use this standalone DB-count run with vanilla run `20260915_031800-vanilla-grpo-train`; arrange SFT/vanilla final evaluation separately if the original job is stopped before evaluation.

- Reward curve, CPU job 19472: [run log](jobs/19472-four-domain-reward-plot-0915-085527027/run_0_20260915_085527027.log). Succeeded: 70 completed updates (iter0–69), 2,800 accepted trajectories; overall mean 0.5168, first/last 10-update means 0.5475/0.4900. Plots completed vanilla optimizer batches only; 40 accepted trajectories per update, official terminal reward and trailing 10-update mean. [PNG](plots/reward_curves.png), [CSV](plots/reward_curves.csv).

- Uniform-outcome filter preflight, CPU job 19473: [run log](jobs/19473-uniform-outcome-filter-preflight-0915-091221027/run_0_20260915_091221027.log). Succeeded: 112 tests and rollout preflight passed. Tests both recipes on all-zero, all-one and mixed groups, including nonzero auxiliary credit; runs four-domain environment and rollout preflight.

- Mixed-outcome restart vanilla-grpo, job 19474: [run log](jobs/19474-async-grpo-mixed-outcomes-0915-091514585/run_0_20260915_091514585.log). Run `20260915_091500-vanilla-grpo-train`; fresh SFT iter4673, fresh optimizer, 556 updates, 8 GPU / 64 CPU / 1024 GB, normal priority. Reject official all-zero/all-one groups before reward postprocessing, including groups with auxiliary credit. Runs concurrently with the other arm; final iter555 evaluation follows training. Hourly monitor PID 794382, [observations](reports/hourly-monitor-19474.log), evaluates iter49,99,…,549. Submission state: SUBMITTED.

- Mixed-outcome restart progress-db-count-v1, job 19475: [run log](jobs/19475-async-credit-mixed-outcomes-0915-091515391/run_0_20260915_091515391.log). Run `20260915_091500-progress-db-count-v1-train`; fresh SFT iter4673, fresh optimizer, 556 updates, 8 GPU / 64 CPU / 1024 GB, normal priority. Reject official all-zero/all-one groups before reward postprocessing, including groups with auxiliary credit. Runs concurrently with the other arm; final iter555 evaluation follows training. Hourly monitor PID 794403, [observations](reports/hourly-monitor-19475.log), evaluates iter49,99,…,549. Submission state: SUBMITTED.

- Offline W&B replacement vanilla-grpo, job 19476: [run log](jobs/19476-async-grpo-mixed-offline-0915-092659354/run_0_20260915_092659354.log). Run `20260915_092700-vanilla-grpo-train`; fresh SFT/optimizer, 556 updates, uniform outcome groups excluded; 8 GPU / 64 CPU / 1024 GB, normal priority. Submitted concurrently. Hourly monitor PID 809480: [observations](reports/hourly-monitor-19476.log).

- Offline W&B replacement progress-db-count-v1, job 19477: [run log](jobs/19477-async-credit-mixed-offline-0915-092700127/run_0_20260915_092700127.log). Run `20260915_092700-progress-db-count-v1-train`; fresh SFT/optimizer, 556 updates, uniform outcome groups excluded; 8 GPU / 64 CPU / 1024 GB, normal priority. Submitted concurrently. Hourly monitor PID 809525: [observations](reports/hourly-monitor-19477.log).

- Mixed-outcome training reward plots, CPU job 19514: [run log](jobs/19514-mixed-reward-curves-0915-130022365/run_0_20260915_130022365.log). Both current arms, official terminal reward for accepted training trajectories; PNG/CSV in `plots/20260915_092700/<recipe>/`. Re-run the plotting entry after 556 updates to produce final curves.

- Prefilter reward curves and watcher verification, CPU job 19532: [run log](jobs/19532-prefilter-reward-curves-0915-144726621/run_0_20260915_144726621.log). Runs both arms via `watch_tau2_four_domain_rewards.py --once`; 0 GPU / 8 CPU / 32 GB, normal priority. New `reward_prefilter.png` / `.csv` include all dumped final attempts before group filtering, including rejected groups; intermediate retries are unavailable. The 60-second watcher now refreshes on new trajectory data as well as optimizer updates. Restart an already-running watcher to load this change.

- vanilla-grpo iter49 (50 completed updates), job 19743: [run log](jobs/19743-four-db-20260915_092700-v-0049-0915-212215384/run_0_20260915_212215384.log). Current mixed-outcome run `20260915_092700-vanilla-grpo-train`; four-domain official evaluation, 197 tasks × 4 trials = 788 trajectories, seed300, fixed Qwen3.6-27B User, Banking BM25. 8 GPU / 64 CPU / 1024 GB, normal priority.

- progress-db-count-v1 iter49 (50 completed updates), job 19745: [run log](jobs/19745-four-db-20260915_092700-db-0049-0915-212218870/run_0_20260915_212218870.log). Current mixed-outcome run `20260915_092700-progress-db-count-v1-train`; four-domain official evaluation, 197 tasks × 4 trials = 788 trajectories, seed300, fixed Qwen3.6-27B User, Banking BM25. 8 GPU / 64 CPU / 1024 GB, normal priority.

- DB-count iter49 evaluation resume, job 19789: [run log](jobs/19789-db-iter49-eval-resume-0916-000340811/run_0_20260916_000340811.log). Previous elastic chain 19745 → 19761 → 19765 → 19768 → 19775 ended with EOFError at the interactive resume prompt. Existing 399/788 results retained; explicit AUTO_RESUME=1, same checkpoint/run stamp/protocol, 8 GPU / 64 CPU / 1024 GB, normal priority.

- Reward curves after user pause, CPU job 19945: [run log](jobs/19945-paused-reward-curves-0916-100039620/run_0_20260916_100039620.log). Refresh both arms from saved trajectories; prefilter and accepted-batch PNG/CSV in `plots/20260915_092700/<recipe>/`. 0 GPU / 8 CPU / 32 GB, normal priority; training/evaluation remain stopped.

- Credit GRPO (progress-db-count-v1) iter119, 120 completed updates, four-domain evaluation job 19947: [run log](jobs/19947-credit-iter119-four-domain-eval-0916-102036337/run_0_20260916_102036337.log). Training run `20260915_092700-progress-db-count-v1-train`; convert checkpoint then evaluate 197 tasks × 4 trials = 788 trajectories, seed300, fixed Qwen3.6-27B User, Banking BM25. AUTO_RESUME=1; 8 GPU / 64 CPU / 1024 GB, normal priority. User requested this evaluation after pausing prior runs; training and prior evals remain stopped.

## Reporting

`examples/tau2-bench/analysis/summarize_tau2_areal_async.py --four-domain` accepts `--experiment-dir`, each model's `--<label>-eval` summary, and each RL arm's `--<label>-logs` and `--<label>-run-dir`. Outputs `reports/comparison.md`, `reports/comparison.json`, `reports/efficiency.json`, and reward PNG/CSV under `plots/`. Comparisons include all three model pairs and paired task bootstrap intervals. Training diagnostics do not define success thresholds.

The real-environment tests under `examples/tau2-bench/rl/` are run by cluster preflight because they require mounted task DBs, Banking documents and the selected tokenizer. `tests/test_tau2_continuous.py` retains its CPU CI registration.

## Results

Smoke v2 succeeded: 25 groups / 200 accepted trajectories (Retail 6, Airline 14, Banking 4, Telecom 1 groups). Four final-zero-signal groups were retained. All turn advantages and five gradient norms were finite (0.456–0.796). The first batch includes an all-wrong Retail group and two all-correct Airline groups with nonzero turn advantages; a Banking DB group also entered training with nonzero signal. Final iter4 is complete. Resume validation also passed. Formal jobs 19358 and 19367 failed on two forms of inference context overflow; both partial runs are excluded. Replacement serial job 19376 is running vanilla training before the second arm, with both arms scheduled for 556 updates and official evaluations; results remain pending. Some regenerated Banking trajectories exceeded the fixed token cap because of long knowledge retrieval content; whole unsampleable groups were replaced while tasks remained in the pool.

Standalone DB-count hourly monitor PID 726027, command `python3 scripts/monitor_tau2_four_domain.py 19469 20260915_082300 --recipe progress-db-count-v1`; evaluates iter49,99,…,549. [Observations](reports/hourly-monitor-db-count-20260915_082300.log).

User stopped all previous live experiment jobs on 2026-09-15: 19376, 19469 and 19462 are STOPPED; monitors 307314 and 726027 terminated. New experiments restart from SFT with fresh optimizers and distinct run names; retain 556 updates and existing evaluation cadence.

Jobs 19474/19475 failed before any optimizer update: secondary trainer W&B initialization retried HTTP 500 from api.wandb.ai/graphql and timed out at 90 seconds. Preflight passed in both jobs. Their monitors were stopped. Replacement runs use WANDB_MODE=offline, preserving local metrics and the mixed-outcome filtering protocol.

2026-09-16 user requested pausing all RUNNING cluster jobs. Stopped 19477 (DB-count training), 19743 (GRPO eval), and 19910 (DB-count eval); GRPO training 19476 had already failed. Existing checkpoints and results retained. vanilla-grpo: latest saved iter69; eval trajectories {'airline': 80, 'retail': 160, 'telecom': 160, 'banking_knowledge': 180}; progress-db-count-v1: latest saved iter139; eval trajectories {'airline': 80, 'retail': 160, 'telecom': 160, 'banking_knowledge': 189}.

Post-pause curve refresh 19945 completed successfully: GRPO 73 updates / 6,628 prefilter trajectories; DB-count 144 updates / 12,968 prefilter trajectories. Both prefilter and accepted-batch PNG/CSV regenerated.
