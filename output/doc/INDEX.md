# Experiments index

One entry per experiment: name, purpose, link to its README.

## Selected lineage comparison

Official four-domain seed300 protocol (197 tasks × 4 trials, BM25), pass@1 / pass@4(any) / pass^4 (%). Raw → SFT → RL → OPD lineage as of 2026-09-29.

| Domain | Raw Instruct | SFT4505 | Vanilla GRPO iter139 | Credit mix iter139 (student) | Airline-only OPD iter79 | Two-expert OPD iter159 | Airline expert (teacher) | **Selected OPD iter59** |
|---|---|---|---|---|---|---|---|---|
| Airline | 33.75 / 65.00 / 10.00 | 42.50 / 65.00 / 30.00 | 42.50 / 60.00 / 25.00 | 40.00 / 70.00 / 15.00 | **53.75 / 65.00 / 35.00** | 47.50 / 75.00 / 15.00 | 53.75 / 75.00 / 35.00 | **51.25 / 75.00 / 20.00** |
| Retail | 52.50 / 72.50 / 30.00 | 58.75 / 77.50 / 35.00 | 61.25 / 87.50 / 37.50 | 63.75 / 85.00 / 40.00 | 61.25 / 82.50 / 37.50 | 62.50 / 90.00 / 35.00 | 59.38 / 90.00 / 30.00 | **65.00 / 85.00 / 47.50** |
| Telecom | 17.50 / 42.50 / 5.00 | 47.50 / 82.50 / 12.50 | 51.88 / 85.00 / 15.00 | 56.25 / 87.50 / 20.00 | 53.75 / 80.00 / 17.50 | 48.75 / 82.50 / 15.00 | 49.38 / 82.50 / 17.50 | **54.38 / 87.50 / 12.50** |
| banking_knowledge | 2.58 / 4.12 / 1.03 | 4.12 / 8.25 / 1.03 | 5.41 / 10.31 / 0.00 | 5.67 / 13.40 / 0.00 | 4.12 / 8.25 / 1.03 | 5.15 / 8.25 / 3.09 | 4.12 / 9.28 / 0.00 | **5.15 / 11.34 / 0.00** |
| Overall | 18.91 / 31.98 / 8.63 | 27.92 / 43.15 / 13.20 | 29.95 / 46.19 / 13.20 | 31.22 / 48.73 / 13.71 | 30.84 / 43.65 / 15.23 | 29.95 / 46.70 / 13.20 | 29.57 / 47.21 / 13.20 | **31.98 / 48.22 / 14.21** |

Measured point estimates: SFT improves overall pass@1 by9.0pp over Raw, and credit mix139 adds3.3pp. Historical Airline-only OPD iter79 and two-expert iter39 reach the Airline teacher's pass@1/pass^4 point estimates. Their overall gains or forgetting relative to their own initialization remain unknown because that checkpoint was not evaluated. Sources: `tau2-mix-rl-sft4505-b128-20260925/eval/{vanilla-grpo-iter139,mix-iter139}-four-domain/` and the experiment READMEs below.

Correction 2026-09-29: the student baseline is the 0925 progress-db-count run's iter139, but the historical OPD jobs loaded the 0927 credit-signal-v1 run's iter129 (unevaluated). Their scores above are valid measurements of those models; their deltas against mix139 are **not matched before/after OPD gains**, and the earlier forgetting/retention interpretation is unestablished. The [corrected mix139 rerun](../experiments/tau2-opd-mix139-airline-20260929/README.md) uses the same iter139 for student initialization and non-Airline teacher.

## Entries

- [tau2-opd-mix139-airline-20260929](../experiments/tau2-opd-mix139-airline-20260929/README.md) — 用户最终选定 **OPD iter59**；四域 seed300 总体 pass@1/pass@4(any)/pass^4 为 **31.98/48.22/14.21%**，相对实际 mix139 初始化为 +0.76/−0.51/+0.51pp。Airline +11.25/+5.00/+5.00pp，非 Airline 三域加权 pass@1 保留98.6%。README 保留 Raw/SFT/vanilla GRPO/学生/教师参照、各 OPD 检查点完整对比及未完成评测的快照。训练25056完成160更新；User25054/experts25055已停。配方为 mix139 学生及非 Airline 教师 + Airline29、LR2e-6、current-student logprob + TIS[0,2]、buffer32/no fixed lag limit、K1/batch16。各评测197任务×4 trials、BM25、统一并发、normal优先级。

- [tau2-opd-two-expert-fulldomain-20260929](../experiments/tau2-opd-two-expert-fulldomain-20260929/README.md) — Two-teacher pure OPD follow-up to the airline-into-mix arm: Airline expert (iter29) teaches airline, mix RL iter129 anchors retail/telecom/banking against forgetting, same student init, four-domain data (2524 tasks), 160 updates (~1 pass), LR 2e-6, K=1, reward pinned to zero. Jobs: User 24959 (4G), two-expert service 24960 (4G, converts iter129 to HF in-job), training 24961 (8G). Training complete 2026-09-29 11:05 CST with all-finite 160 updates (iter_0000159 saved); iter159 official four-domain seed300 eval 24983 complete 12:33 CST; iter39/89/119 curve evals complete ~15:15 CST (jobs 25019/25020/25021); persistent User 24959 and experts 24960 stopped after eval submission. Curve (pass@1/pass@4(any)/pass^4): **iter39 is the arm's peak and selected checkpoint** — Airline `53.75/75.00/35.00` (teacher level on all three) with overall `30.20/44.16/15.74` (within noise of the student baseline, best pass^4 of the run); later checkpoints oscillate Airline (45.0→47.5, pass^4 35→15) and erode overall pass^4 (15.74→11.68→13.20), the same mid-training-peak pattern as the parent RL runs. iter159: overall `29.95/46.70/13.20`, Airline `47.50/75.00/15.00`. Single seed; iter39 needs a second-seed confirmation.

- [tau2-opd-airline-into-mix-20260929](../experiments/tau2-opd-airline-into-mix-20260929/README.md) — Pure OPD distilling the Airline expert (iter29, 53.75% Airline pass@1) into the progress-db-count-v1 mix RL student (baseline iter139, fresh optimizer). Airline-only 1148 tasks, 80 updates, LR 2e-6, K=1, reward pinned to zero so no trajectory-level advantage remains. Training 24947 SUCCEEDED 2026-09-29 01:48 CST with all-finite updates; iter79 official four-domain seed300 eval 24954 SUCCEEDED 02:33 CST; persistent User 24945 and expert 24946 stopped after eval submission. Result (pass@1/pass@4(any)/pass^4): Airline `53.75/65.00/35.00` (+13.75pp pass@1 vs student iter139's 40.00, reaches teacher 53.75; pass^4 15→35 = teacher level), overall `30.84/43.65/15.23` vs student `31.22/48.73/13.71` (single seed, overall delta within noise). Checkpoint-curve evals iter49/59/69 complete 2026-09-29 ~13:40 CST (jobs 25000/25001/25002): the Airline gain is late-training (pass@1 dips to 42.50 at iter59, climbs 47.50→53.75 by iter79; pass^4 20→20→25→35), overall flat across the curve (30.0-31.0 pass@1), non-airline domains stable; iter79 remains selected.

- [tau2-retail-expert-sft4505-goal-v4-20260928](../experiments/tau2-retail-expert-sft4505-goal-v4-20260928/README.md) — Retail-only v4 expert from fresh SFT4505: 563 tasks, goal-progress credit, shuffled task deck, batch128/LR2e-6/30 updates. Training24828 (`pt-8h1psh0m`) SUCCEEDED at iter29; three-domain iter9/19/29 evaluations24894/24895/24887 SUCCEEDED with overall pass@1/pass@4(any)/pass^4=`48.00/76.00/25.00%`, `53.50/83.00/25.00%`, `51.00/83.00/22.00%` respectively.

- [tau2-telecom-expert-sft4505-goal-v4-20260928](../experiments/tau2-telecom-expert-sft4505-goal-v4-20260928/README.md) — Telecom-only v4 expert from SFT4505: 271 tasks, shuffled task deck, current credit-aware filter, batch128/LR2e-6/20 updates. 8-GPU job24882 SUCCEEDED; three-domain iter9/iter19 evaluations24938/24939 SUCCEEDED with overall pass@1/pass@4(any)/pass^4=`50.50/77.00/23.00%` and `47.50/74.00/21.00%` respectively.

- [tau2-mix-rl-sft4505-goal-v4-20260928](../experiments/tau2-mix-rl-sft4505-goal-v4-20260928/README.md) — Prepared independent task consistency and goal-progress candidate: 522 Retail requests grounded in DB/targets; 41 handoff/read-only/selection tasks retained verbatim; restored declared evidence mode for244 Banking tasks while retaining298 BM25 tasks; DB/ACTION/COMM progress and per-trajectory stage normalization. 542 reference checks passed, actual User probes recorded, controlled task_consistent/goal_progress launchers prepared. Mixed arms are not submitted; Retail-only expert runs in its separate experiment.

- [tau2-mix-rl-sft4505-reward-v3-20260927](../experiments/tau2-mix-rl-sft4505-reward-v3-20260927/README.md) — Independent numeric tool/JSON ACTION fixes plus v2 reference/Retail fixes. All 2,524 reference replays, 906 equivalence pairs and 26 CPU regression tests passed. Training 24668 SUCCEEDED after 140 updates from SFT4505 (batch128/LR2e-6). Final iter139 8-GPU four-domain official evaluation 24757 (`pt-g4ex7d5a`) SUBMITTED on 2026-09-28 09:44 CST, unchanged seed300/4-trial protocol, normal priority.

- [tau2-mix-rl-sft4505-reward-v2-20260927](../experiments/tau2-mix-rl-sft4505-reward-v2-20260927/README.md) — Prepared independent reward/environment fixes across all four domains: Telecom reference synchronization and numeric text, Airline numeric DB equivalence, Banking application IDs, Retail per-item variants. Audited 2,524 tasks; 447 reference checks, 137 Retail modification checks and 17 CPU tests passed. No GPU job submitted for v2.

- [tau2-mix-rl-sft4505-credit-signal-v1-20260927](../experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/README.md) — Fresh SFT4505 credit-signal mix RL, User24445/training24446. Training SUCCEEDED after 140 updates (batch128, LR2e-6). Final iter139 8-GPU four-domain official evaluation 24720 (`pt-6rgnruwc`) SUBMITTED on 2026-09-27 21:12 CST; unchanged seed300/4-trial protocol, normal priority. [Telecom diagnosis](../experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/telecom_diagnosis/README.md) identifies inconsistent reference states in 136/271 training tasks and numeric-format DB mismatches; interpret low Telecom reward accordingly. The diagnostic did not change the running training.

- [tau2-mix-rl-sft4505-credit-no-replay-20260926](../experiments/tau2-mix-rl-sft4505-credit-no-replay-20260926/README.md) — Fresh SFT4505 progress-db-count-v1 mix RL, 8 GPUs, 140 updates, batch128, LR2e-6. Discard all-zero groups and randomly refill; no two-update deferred task retries. Existing mix sampler and other parameters retained. Job 24150 (`pt-im4oekgy`) SUBMITTED at 2026-09-26 11:26 CST.

- [tau2-mix-rl-instruct-b128-20260926](../experiments/tau2-mix-rl-instruct-b128-20260926/README.md) — Four-domain vanilla GRPO from original Qwen3-4B-Instruct-2507, changing only initialization relative to SFT4505 control 24088. Same 8 GPUs, 140 updates, batch128, LR2e-6, User23962, task pool, reward and sampling settings. Job 24144 (`pt-accb91g4`) STOPPED on user request on 2026-09-26; latest completed checkpoint iter59 is retained.

- [tau2-eval-qwen38-nonthinking-banking-20260925](../experiments/tau2-eval-qwen38-nonthinking-banking-20260925/README.md) — Job 23973: Qwen3.8-27B nonthinking Agent, official Banking Knowledge only, 97 tasks × 4 trials, seed 300, Agent max_tokens 1200, unchanged sampling settings and Qwen3.6-27B User. Eight GPUs split into two TP2 Agent and two TP2 User replicas; RUNNING, 12/388 simulations saved by 10:55 CST, actual request settings verified.

- [tau2-mix-rl-sft4505-b128-20260925](../experiments/tau2-mix-rl-sft4505-b128-20260925/README.md) — Controlled four-domain asynchronous mix-RL from SFT4505: credit run 23971 and outcome-only GRPO control 24088 completed 140 updates with matched User/data/sampling/LR2e-6/batch128. Credit iter89/iter129/iter139 evaluations 24037/24061/24067 and vanilla iter89 evaluation 24128 SUCCEEDED. Vanilla iter129/iter139 evaluations 24142 (`pt-tf1k2v9i`)/24143 (`pt-oq8ythfq`) were submitted on 2026-09-26 at 07:51 CST, each with 8 GPUs and unchanged job 24128 parameters; statuses are RUNNING/SUBMITTED. Earlier iter59 submission test 24125 was STOPPED on request before evaluation started.

- [tau2-opd-four-domain-20260923](../experiments/tau2-opd-four-domain-20260923/README.md) — Student23193 completed full pure OPD from SFT4505 at2e-6,buffer32,160 updates for2524 source tasks. Controlled seed300 checkpoint evaluation selects Update80:28.55/46.70/14.72% pass@1/pass@4(any)/pass^4 and93.4% target-teacher pass@1 retention. User23043 and experts23191 were stopped; seed302/303/304 submissions were stopped.

- [tau2-opd-bounded-lr-20260922](../experiments/tau2-opd-bounded-lr-20260922/README.md) — Completed both60-update arms and1,600 official simulations. Overall pass@1:2e-6 55.125%,5e-6 49.75%; retain2e-6,with replicated Retail consistency gains and unresolved Airline pass^4. User/experts remain persistent; scoring probe deferred. [Results](../experiments/tau2-opd-bounded-lr-20260922/RESULTS.md).

- [tau2-opd-current-buffer-ab](../experiments/tau2-opd-current-buffer-ab/README.md) — Completed4,000 matched official simulations:buffer32 A versus unbounded B,LR2e-6/60 updates. A controls lag and retains teacher point estimates; Retail consistency improves over SFT,no clear A/B score winner. B resumed after cluster crash. [Analysis/next experiments](../experiments/tau2-opd-current-buffer-ab/ANALYSIS_AND_NEXT.md).

- [tau2-opd-airline-retail-pilot](../experiments/tau2-opd-airline-retail-pilot/README.md) — Pure Airline/Retail OPD pilot from SFT4505 with a 4-GPU User service, domain-routed Airline/Retail teachers, an 8-GPU student, two-update smoke, then fresh 20-update pilot. v2: LR 1e-5, 60 updates; v3: behavior-logprob estimator. Current status in [OPD status](OPD_STATUS.md).

## Cross-experiment analysis

- [TAU2 async/sync speed and credit assignment](TAU2_ASYNC_SYNC_SPEED_CREDIT.md) — Full-stack speed estimate, matched async/sync performance comparison, and credit-assignment versus outcome-only GRPO evidence.
- [OPD status](OPD_STATUS.md) — Completed two-domain LR comparison and current four-domain preparation,with persistent service jobs and evaluation comparison limits.

- [tau2-domain-experts-sft4505-b128](../experiments/tau2-domain-experts-sft4505-b128/README.md) — Corrected SFT4505 four-domain experts:job22033 Airline60/Retail30/Telecom20;job22034 Banking30;each8 GPUs,shared refreshed4-GPU User22031. Fresh optimizer per domain,LR2e-6/batch128/progress1;two-update smoke checks before formal training. Full checkpoint four-domain evaluations and30-minute supervision;maximum two concurrent8-GPU eval jobs. Retail iter9 unchanged retry:22377 (2026-09-21). Retail iter19/iter29 eval:22448/22450 (2026-09-21). Telecom iter9/iter19 eval:22494/22495 (2026-09-21).
- [tau2-sft-areal3-banking-simplified-full-20260920](../experiments/tau2-sft-areal3-banking-simplified-full-20260920/README.md) — New four-domain SFT data:30376 unchanged AReaL rows +5672 full simplified Banking rows=36048. Three-domain comparison against old mixed data:zero differing lines. SFT21863 SUCCEEDED4506 updates,final iter4505;runtime2h49m,Final iter4505 eval21925 SUCCEEDED788/788,zero infra errors;Banking4.12/8.25/1.03%,four-domain27.92/43.15/13.20%. README documents SFT length selection versus RL16K overflow,including user-reported batch10 observations.
- [tau2-domain-experts-fast-b128](../experiments/tau2-domain-experts-fast-b128/README.md) — Historical SFT4673 experts;job21629 stopped at user request2026-09-20 after Airline60/Retail30/Telecom20/Banking14 updates. Superseded by corrected SFT4505 initialization.

- [tau2-airline-db-count-tuning](../experiments/tau2-airline-db-count-tuning/README.md) — Airline-only RL from SFT4673, LR2e-6/batch128; [raw/SFT/RL comparisons](../experiments/tau2-airline-db-count-tuning/reports/comparison.md). Fast21168 iter19:Airline53.75/80/30%,20 updates in80.46min (4.60x old370.22min). Uniform-source control21429 SUCCEEDED20 updates in71.86min (5.15x), but evaluation21530 Airline is42.50/60/30%; uniform replacement is not selected and the generic source default is restored to shuffled/retry. At2026-09-19 20:28 CST,21530 remains RUNNING,338/400 evaluated (Airline80/80,Retail138/160,Telecom120/160). [Quality diagnosis](../experiments/tau2-airline-db-count-tuning/reports/quality-diagnosis/README.md); full jobs/logs in experiment README.

- [tau2-async-db-count-four-domain](../experiments/tau2-async-db-count-four-domain/README.md) — Four-domain asynchronous GRPO versus DB-count from full-domain SFT. Five-update smoke and two-update resume passed. Both inference context-overflow response forms now handled; cluster job 19375 passed 112 tests and rollout preflight. Fresh serial job 19376 completed 11 updates and saved iter9 with rollout state; real context overflow entered retry and training continued; hourly / every-50-update evaluation monitor active (PID 307314); automatic 04:17 check verified 18 completed updates and iter9 saved. Periodic eval latest job: 19462 (vanilla-grpo iter49). User requested concurrent DB-count: standalone job 19469 (556 updates, fresh SFT, stamp 20260915_082300), queued for 8 GPUs; original jobs preserved. Reward curve CPU job: 19472. 2026-09-15 restart: prior jobs 19376/19469/19462 stopped; both recipes now reject uniform official outcomes. Preflight job 19473. New mixed-outcome arms: 19474 vanilla / 19475 credit, stamp 20260915_091500; preflight 112 tests passed; separate hourly monitors enabled. W&B HTTP 500 caused both startup failures (0 updates); offline replacements 19476/19477, stamp 20260915_092700, hourly monitors active. Current two-arm reward curves: CPU job 19514; final curves required after 556 updates. Prefilter reward PNG/CSV and trajectory-triggered watcher: CPU verification 19532. Current mixed-outcome iter49 four-domain evaluations: 19743 GRPO / 19745 DB-count, submitted 2026-09-15 21:22. DB-count iter49 eval resume 19789 (AUTO_RESUME=1), after 19775 EOFError; 399 results retained. 2026-09-16: all running jobs stopped at user request; artifacts retained. Post-pause reward-curve refresh: CPU job 19945. Credit GRPO iter119 (120 updates) four-domain eval: job 19947, AUTO_RESUME=1.

- [tau2-areal-async-rl](../experiments/tau2-areal-async-rl/README.md) — matched 8-GPU synchronous/asynchronous GRPO with uniform task draws, recorded behavior tokens and bounded policy lag;20-update comparison completed with training speedup1.060×;100-update job18009 completed both training arms and both sync evaluations, then was intentionally stopped during its redundant async evaluation; independent async job18161 completed both seeds; lag/pool ablation retry18434 completed three 20-update arms; evaluation18687 failed on output-directory collision, fixed single-seed evaluation18811 completed; unlimited-lag pool20 train200 job18996 and Airline-only counterpart18997 failed preflight; fixed and retried as19001; Airline iter19 eval19014 reclaimed; retry19019 hit existing conversion output; retry19033 failed on single-domain slot borrowing, now disabled; iter99 seed300 evaluations19097 (mixed three-domain) and19098 (Airline) submitted; training18996/19001 stopped at user request to release GPUs for evaluation; single-domain eval19098 stopped and replaced by19101 using unchanged three-domain evaluation, matching19097; both completed, seed300 SFT/mixed/Airline pass@1=56.25/59.25/55.00%, pass^4=30/27/21%. Mixed iter69 evaluation19245 completed: seed300 58.25/85/26%, zero infrastructure errors.

- [tau2-banking-simplified-full-sft](../experiments/tau2-banking-simplified-full-sft/README.md) —
  raw-model SFT on all 5,672 simplified Banking rows followed by Banking evaluation.

- [tau2-banking-simplified-sft-test](../experiments/tau2-banking-simplified-sft-test/README.md) —
  raw-model SFT on completed408 simplified Banking data; matched Banking length and latency evaluation.

- [tau2-banking-simplify-concurrency-v1](../experiments/tau2-banking-simplify-concurrency-v1/README.md) —
  matched 8-GPU concurrency 1/2/4 benchmark before changing the live Banking simplification job.

- [tau2-banking-simplify-qwen38-v1](../experiments/tau2-banking-simplify-qwen38-v1/README.md) —
  deletion-only Qwen3.8-27B simplification of all Banking expert trajectories,
  with independent semantic review, sequential replay, and complete non-thinking SFT export.
  Pilot complete (64/75 accepted); full job 17371 stopped for concurrency-four
  continuation 17467, preserving completed production decisions.

- [tau2-sft-official-native-expanded-turn-filtered](../experiments/tau2-sft-official-native-expanded-turn-filtered/README.md) —
  two-epoch SFT and matched checkpoint evaluation on the turn-quality-filtered
  official-native data, using the prior Processed SFT as the data-ablation
  control.

- [tau2-rl-sft-raw-vs-processed-vanilla-grpo](../experiments/tau2-rl-sft-raw-vs-processed-vanilla-grpo/README.md) —
  paired 100-update vanilla GRPO runs from raw-all and processed official-native
  SFT initializations, followed by fixed two-seed official evaluation.

- [qwen38-27b-inference-smoke](../experiments/qwen38-27b-inference-smoke/README.md) —
  one-GPU BF16 SGLang compatibility check for the local Qwen3.8-27B Hugging
  Face checkpoint, covering `xhigh` reasoning, text, parsed tool calls, and
  image input.

- [tau2-sft-turn-quality-qwen38-xhigh-v1](../experiments/tau2-sft-turn-quality-qwen38-xhigh-v1/README.md) —
  conservative turn-only quality filtering of the selected official-native
  Agent SFT rows using two local Qwen3.8-27B `xhigh` reviews, with no target
  reconstruction or repair.

- [audio-01](../experiments/audio-01/README.md) — queue-facing alias for the
  Qwen3.8 turn-quality resume jobs; the real experiment name is recorded in
  [EXPERIMENT_ALIASES.md](EXPERIMENT_ALIASES.md).

- [tau2-eval-qwen36-user-parserfix-smoke](../experiments/tau2-eval-qwen36-user-parserfix-smoke/README.md) —
  three-domain, five-task smoke validating non-thinking text-only Qwen3.6-27B
  User serving with the Qwen3-Coder tool-call parser.

- [tau2-eval-official-native-smoke](../experiments/tau2-eval-official-native-smoke/README.md) —
  cluster preflight and four-domain GPU smoke for the Tau2 official native
  `llm_agent` evaluation path.

- [tau2-eval-official-native-full](../experiments/tau2-eval-official-native-full/README.md) —
  formal four-domain, four-trial evaluation through the Tau2 official native
  `llm_agent` path.

- [tau2-eval-qwen35-official-native-four-domain](../experiments/tau2-eval-qwen35-official-native-four-domain/README.md) —
  matched four-domain, four-trial official-native evaluations of Qwen3.5-4B
  with Agent thinking enabled and disabled.

- [tau2-qwen3-qwen35-atomic-gap-analysis](../experiments/tau2-qwen3-qwen35-atomic-gap-analysis/README.md) —
  deterministic and blinded contrastive analysis of two Qwen3-4B-Instruct-2507
  evaluations against Qwen3.5-4B non-thinking, with targeted prefix replay to
  separate local action choice from upstream evidence and state accumulation.

- [tau2-banking-task-curriculum](../experiments/tau2-banking-task-curriculum/README.md) —
  97-task Banking inventory and dependency graphs, plus 564 benchmark-derived
  diagnostic probes over 94 replayable sources; all are excluded from training.

- [tau2-banking-independent-synthetic-v1](../experiments/tau2-banking-independent-synthetic-v1/README.md) —
  accepted independent 542/75/30 Banking train/dev/challenge set with an empty
  DB, a 480-document isolated runtime, reward-one reference replay, and Qwen3.8
  Agent/User evaluation.

- [audio-curriculum-v1](../experiments/audio-curriculum-v1/README.md) —
  historical queue-facing alias for earlier jobs in the independent synthetic
  curriculum; later jobs use the real experiment name recorded in
  [EXPERIMENT_ALIASES.md](EXPERIMENT_ALIASES.md).

- [tau2-banking-task-curriculum-scale-v2](../experiments/tau2-banking-task-curriculum-scale-v2/README.md) —
  node-level diagnostic scale-out to 2,740 Banking specifications: every annotated
  evidence fact, matching BM25 retrieval, and reference action across the 94
  replayable benchmark tasks; all pass reference replay and none is trainable.

- [tau2-banking-scale-v2-qwen38-eval](../experiments/tau2-banking-scale-v2-qwen38-eval/README.md) —
  one Qwen3.8-27B Agent/User trajectory and exact Tau2 score for each of the
  2,740 benchmark-derived diagnostic tasks: 2,163 exact successes (78.94%);
  trajectories are excluded from SFT, GRPO, and OPD.

- [tau2-banking-curriculum-qwen36-eval](../experiments/tau2-banking-curriculum-qwen36-eval/README.md) —
  single-trial local Qwen3.6-27B evaluation of all 564 benchmark-derived
  Banking diagnostic tasks, split by required retrieval mode.

- [tau2-banking-curriculum-qwen38-eval](../experiments/tau2-banking-curriculum-qwen38-eval/README.md) —
  single-trial Qwen3.8-27B evaluation of all 564 benchmark-derived diagnostic
  tasks, using Qwen3.8-27B for both Agent and User; all 16 initial protocol
  failures are recovered and final exact success is 329/564 (58.33%).

- [tau-bench](../experiments/tau-bench/README.md) — tau-bench (tau1, retail)
  RL run-through for Qwen3-4B-Instruct-2507 with slime: generate mock data,
  convert the Instruct checkpoint to torch_dist, then GRPO training with the
  custom tau rollout. Monitoring log: [experiments/tau-bench/MONITOR.md](../experiments/tau-bench/MONITOR.md).

- [tau2-deps-probe](../experiments/tau2-deps-probe/README.md) — probe job that
  discovers which packages the fresh cluster image needs for tau2 (build:
  hatchling, editables; runtime: toml, deepdiff, litellm; [gym]: gymnasium;
  [knowledge] BM25: rank-bm25) and runs dependency smokes.

- [tau2-eval](../experiments/tau2-eval/README.md) — tau2-bench evaluation
  (airline/retail/telecom) for a sglang-served policy. Current path uses tau2's
  official agent/runner APIs; legacy direct-`AgentGymEnv` runs are archived.

- [tau2-eval-local-user](../experiments/tau2-eval-local-user/README.md) —
  tau2-bench official eval smoke with both agent and user simulator served by
  local sglang.

- [tau2-eval-qwen36-user-async-timed](../experiments/tau2-eval-qwen36-user-async-timed/README.md) —
  eight-GPU official eval with parallel domains, elastic concurrency, and
  Agent/User/environment timing breakdowns. The current routing comparison
  uses two TP1 Qwen3-4B Agent replicas and three TP2 Qwen3.6-27B User replicas;
  the User router changes from cache-aware to round-robin.

- [tau2-eval-qwen36-user-four-domain](../experiments/tau2-eval-qwen36-user-four-domain/README.md) —
  eight-GPU asynchronous evaluation of Airline, Retail, Telecom, and
  Banking Knowledge with explicit BM25 retrieval.

- [tau2-eval-user-sft](../experiments/tau2-eval-user-sft/README.md) —
  tau2-bench official Pass@4 evaluations with the trained local user SFT model
  served by sglang.

- [tau2-hf-convert](../experiments/tau2-hf-convert/README.md) — conversion of
  tau2 user and agent SFT torch_dist checkpoints to Hugging Face directories for
  sglang evaluation.

- [tau2-qwen35](../experiments/tau2-qwen35/README.md) — diagnostic smokes that
  stood up tau2-bench eval for the multimodal/thinking Qwen3.5-4B (text-only
  serving + `--disable-thinking` + multi-format parser) before the full run.

- [tau2-qwen35-nonthinking-eval](../experiments/tau2-qwen35-nonthinking-eval/README.md) —
  final two-seed raw Qwen3.5-4B non-thinking, one-tool-per-turn baseline:
  pass@1/pass@4(any)/pass^4 `32.75/61.00/9.00%`, plus the matched seed-300
  thinking-mode comparison and single-call SFT-data conversion check.

- [tau2-agent-single-call-v1](../experiments/tau2-agent-single-call-v1/README.md) —
  independent strict-single-v1 SFT-to-GRPO retraining from the selected boundary
  data, with source-order call/result serialization and matched two-seed formal
  evaluation of the new SFT and RL checkpoints. SFT, RL iter99, and selected RL
  iter199 score `25.88/49.50/8.50%`, `27.50/53.50/10.00%`, and
  `31.13/61.50/10.50%`. Iter199 improves over iter99 by
  `+3.63/+8.00/+0.50pp`; paired two-seed pass@1 and pass@4(any) intervals are
  positive, with gains concentrated in retail and telecom. Its README also
  contains the unified parser-ON matrix separating RL training User,
  evaluation User, and training/evaluation `max_steps`; every cell now has a
  final seed300/301 mean.

- [tau2-agent-single-call-credit-core](../experiments/tau2-agent-single-call-credit-core/README.md) —
  controlled three-arm strict-single-v1 comparison of fresh turn-credit-v1,
  outcome-only GRPO, and fixed-budget turn-credit-v2 from the same SFT, with
  matched zero-signal handling and two-seed official evaluation.

- [tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl](../experiments/tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl/README.md) —
  four parallel 8-GPU strict-single-v1 runs from the same SFT: one mixed and
  three single-domain turn-credit-v2 lambda-0.1 agents, all trained with the
  raw Qwen3 User and rollout `max_steps=120` before targeted official eval.

- [tau2-rl-boundary-v2-domain-continuations-20260816](../experiments/tau2-rl-boundary-v2-domain-continuations-20260816/README.md) —
  cancelled wrong-lineage submission; boundary-v2/v1-User/max60 jobs were
  stopped before training after the intended single-call/raw-User/max120
  configuration was clarified.

- [tau2-agent-single-call-v1-raw-user-maxsteps120-domain-rl130](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-domain-rl130/README.md) —
  stopped invalid submissions that restarted from the single-call SFT instead
  of the required RAW-User maxsteps120 iter99 checkpoint.

- [tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130/README.md) —
  corrected mixed/Retail/Telecom continuations initialized from RAW-User
  maxsteps120 iter99, running updates 100–129 continuously with turn-credit-v1
  and checkpoint-only saves at iter109/119/129.

- [tau2-agent-single-call-v1-user-ablation](../experiments/tau2-agent-single-call-v1-user-ablation/README.md) —
  matched two-seed official evaluation of the strict-single SFT and selected RL
  iter199 Agents with the unmodified Qwen3-4B-Instruct-2507 User, changing only
  the User checkpoint relative to the v1 User-SFT controls. Raw-User SFT and RL
  score `28.00/51.00/9.00%` and `26.88/52.50/8.00%`; under the same raw User,
  RL-minus-SFT is `-1.13/+1.50/-1.00pp`. Its complete two-axis matrix separates
  RL training User (V1/RAW) from evaluation User (V1/RAW); the five crossed RL
  cells collected later are final seed300/301 means.

- [tau2-agent-single-call-v1-rl-crossed-parser-on](../experiments/tau2-agent-single-call-v1-rl-crossed-parser-on/README.md) —
  completed seed300/301 evaluations for the five previously missing
  strict-single RL crossed cells, all 400/400 per seed with parser ON and zero
  infrastructure errors. Every crossed two-seed mean trails its
  rollout-User-matched control on all three aggregate pass metrics.

- [tau2-raw-agent-raw-user-parser-on](../experiments/tau2-raw-agent-raw-user-parser-on/README.md) —
  completed parser-ON raw-Agent evaluations with V1 and RAW Users at seeds
  300/301. RAW-User means are Qwen3 `17.38/35.50/2.50%`, Qwen3.5 thinking
  `53.25/78.00/25.00%`, and Qwen3.5 non-thinking `25.63/53.50/7.00%`; the
  matched V1-User means remain explicit in the experiment table.

- [tau2-agent-single-call-v1-raw-user-rl100](../experiments/tau2-agent-single-call-v1-raw-user-rl100/README.md) —
  100-update strict-single-v1 GRPO ablation from the selected single-call SFT,
  changing only the rollout User from v1 User SFT to the unmodified
  Qwen3-4B-Instruct-2507 model. Iter99 raw-User evaluation scores
  `29.25/54.00/6.50%`, or `+1.25/+3.00/-2.50pp` over the matched raw-User SFT;
  all overall paired intervals include zero and the gain is Telecom-led.

- [tau2-agent-single-call-v1-raw-user-rl200](../experiments/tau2-agent-single-call-v1-raw-user-rl200/README.md) —
  continuation of the raw-User strict-single-v1 RL lineage from iter99 to
  iter199, preserving the 16,384-token training cap and all other recipe
  settings; 8-GPU job `pt-x6p7gg51` reached step199 and saved the complete
  `iter_0000199` checkpoint; HF conversion and both 400/400 raw-User
  evaluations succeeded. Iter199 scores `31.50/58.00/8.00%`, versus raw-User
  RL99 `29.25/54.00/6.50%` (`+2.25/+4.00/+1.50pp`, overall intervals include
  zero). Against v1-User-trained RL199 under the same raw-User evaluation,
  pass@1 improves by `+4.62pp` with interval `[+0.75,+8.62]pp`.

- [tau2-agent-single-call-v1-raw-user-maxsteps120-rl100](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl100/README.md) —
  controlled repeat of raw-User strict-single RL100 with rollout
  `max_steps=120` instead of 60, testing whether the longer interaction budget
  reduces artificial truncation and improves learned behavior; job `pt-e4a6ryoo`
  completed step99 and saved `iter_0000099`; HF conversion and both 400/400
  raw-User evaluations succeeded. It scores `28.38/56.50/9.50%` versus max60
  `29.25/54.00/6.50%`; the pass@1 delta is `−0.88pp` and its overall interval
  includes zero.

- [tau2-agent-single-call-v1-raw-user-maxsteps120-rl200](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl200/README.md) —
  completed 100-update continuation of the max120 raw-User RL checkpoint from
  iter99 to iter199; retry job `1110`, HF conversion `1232`, and both raw-User
  evaluations (`1234`/`1235`, 400/400 each) succeeded. The iter199 mean is
  `30.13/58.50/7.00%`; relative to max120 iter99 it is
  `+1.75/+2.00/-2.50pp`, while relative to max60 iter199 it is
  `-1.38/+0.50/-1.00pp`.

- [tau2-rl](../experiments/tau2-rl/README.md) — tau2-bench GRPO training for
  the agent SFT checkpoint using the trained local user simulator and per-task
  AReaL RL databases. Monitoring log:
  [experiments/tau2-rl/MONITOR.md](../experiments/tau2-rl/MONITOR.md).

- [tau2-rl-usercmp](../experiments/tau2-rl-usercmp/README.md) — A/B/C
  comparison of three user simulators (v2 SFT, v1 SFT, original instruct) for
  tau2-bench GRPO; identical real-8g runs varying only the user model.

- [tau2-rl-stability-k2-fieldreward-lr-sweep](../experiments/tau2-rl-stability-k2-fieldreward-lr-sweep/README.md) —
  airline GRPO stability scan at `2e-6`, `3e-6`, and `5e-6` with `k2` KL,
  field-level reward credit, explicit failure penalties, and shaped-zero-std
  dynamic replacement.

- [tau2-rl-stability-k2-fieldreward-final-eval](../experiments/tau2-rl-stability-k2-fieldreward-final-eval/README.md) —
  Hugging Face conversion and paired tau2 official held-out evaluation of the
  final `2e-6/3e-6/5e-6` stability-sweep checkpoints using the training-time
  v1 STOP-trained User.

- [tau2-rl-local-first-dependency-safe-k2-fieldreward](../experiments/tau2-rl-local-first-dependency-safe-k2-fieldreward/README.md) —
  gated airline GRPO from the local-first relaxed SFT under the shared
  dependency-safe protocol: evaluate at iter99, resume optimizer/RNG state to
  iter199 only after promotion, then select the final eligible checkpoint.

- [tau2-traj-pattern](../experiments/tau2-traj-pattern/README.md) —
  paired deterministic + Gemini trajectory-pattern analysis across raw/SFT/RL.
  Deterministic results show RL gains only on no-Agent-write workflows and that
  SFT/RL multitool behavior violates the one-tool-per-turn domain policies;
  calibrated semantic review covers all 1,200 trajectories with exact rule facts
  taking precedence over contradictory judge labels.

- [tau2-sft-quality-audit](../experiments/tau2-sft-quality-audit/README.md) —
  source-dialog reconstruction, deterministic quality gates, stratified success
  analysis, and calibrated local-first LLM review of the exact tau2 SFT training
  data, with API escalation and a traceable filtered-data manifest.

- [tau2-sft-multitool-quality-audit](../experiments/tau2-sft-multitool-quality-audit/README.md) —
  protocol-neutral dependency-safe-multi follow-up that separates label failure risk,
  matched successful-trajectory quality, and causal failure attribution, with a final
  turn-level filtering specification and provenance-only decisions.

- [tau2-sft-local-first-relaxed](../experiments/tau2-sft-local-first-relaxed/README.md) —
  local Qwen3.6-27B full-corpus, target-prefix filtering under the calibrated
  dependency-safe-multi standard, followed by raw-init Agent SFT and paired official
  tau2 comparison against the raw and previous SFT checkpoints; pass@1/pass@4(any)
  improve significantly, while the pass^4 regression keeps the recipe in broad-tier
  candidate status pending a consistency-anchor ablation.

- [tau2-sft-agent-user-boundary-v2](../experiments/tau2-sft-agent-user-boundary-v2/README.md) —
  signed Agent-owned tool contract, strict User-event provenance, native tool
  tokenization, and matched-budget contract-only versus boundary-anchor SFT.

- [tau2-sft-official-native-expanded](../experiments/tau2-sft-official-native-expanded/README.md) —
  two-pass direct expansion of successful raw AReaL Airline, Retail, and Telecom
  rows into current Tau2 official-native target-only SFT data, followed by a
  two-epoch raw-Instruct SFT checkpoint curve and matched official-native evals.

- [tau2-sft-full-domain](../experiments/tau2-sft-full-domain/README.md) —
  full-domain Agent SFT on the concatenated official-native AReaL and Banking
  expert datasets, using the prior three-domain SFT recipe.

- [tau2-sft-full-domain-eval](../experiments/tau2-sft-full-domain-eval/README.md) —
  official four-domain evaluation curve for the later full-domain SFT checkpoints.

- [tau2-sft-raw-all-max16384](../experiments/tau2-sft-raw-all-max16384/README.md) —
  raw AReaL Tau2 SFT control retaining successful and failed rows, with only
  over-16,384-token examples removed before final-only two-seed evaluation.

- [tau2-rl-agent-user-boundary-v2](../experiments/tau2-rl-agent-user-boundary-v2/README.md) —
  turn-aware three-domain GRPO from the v3-selected Contract + boundary SFT,
  with absolute Assistant-turn penalties, strict span/token alignment,
  same-domain replacement, and iter9/iter19/iter99 promotion gates; Pilot A
  improved namespace and held-out capability but stopped at iter9 on four
  strict behavior/truncation checks.

- [tau2-rl-agent-user-boundary-v2-long100](../experiments/tau2-rl-agent-user-boundary-v2-long100/README.md) —
  fresh 100-update rerun of the same turn-aware recipe, with health-only
  iter9/iter19 continuation gates; iter99 passed both seed evaluations and the
  final comparison against selected SFT and historical iter9 classified `win`.
  Two-seed mean pass@1/pass@4(any)/pass^4 is `29.50/56.50/10.50%`, improving
  over selected SFT by `+5.88/+13.00/+2.00pp` and historical iter9 by
  `+4.25/+6.50/+1.50pp`. The cross-experiment problem/solution chain is in
  [EXP_QA.md](EXP_QA.md).

- [tau2-rl-agent-user-boundary-v2-domain-experts](../experiments/tau2-rl-agent-user-boundary-v2-domain-experts/README.md) —
  three isolated 30-update single-domain continuations from selected long100
  iter99, compared with mixed iter109/119/129 prefixes as future OPD teacher
  candidates without launching distillation. Retail iter119 and Telecom iter129
  show small replicated advantages and remain teacher candidates. Airline
  iter129 trails mixed RL on the target domain and is not selected, providing
  evidence of useful cross-domain transfer rather than an invalid checkpoint.

- [tau2-rl-agent-user-boundary-v2-checkpoint-curve](../experiments/tau2-rl-agent-user-boundary-v2-checkpoint-curve/README.md) —
  seed-300 same-lineage curve at iter9/39/69/89/99 under the fixed signed
  boundary-v2 protocol, with independent HF conversions and paired task-level
  comparisons. The curve is non-monotonic (`25.00/26.25/24.00/29.75/27.50%`
  pass@1); iter89 is the diagnostic single-seed peak, while historical iter9
  and raw Instruct remain background controls only.

- [tau2-rl-agent-user-boundary-v2-long200](../experiments/tau2-rl-agent-user-boundary-v2-long200/README.md) —
  exact cross-root continuation of the selected long100 iter99 optimizer/RNG and
  per-domain sampling state. It was stopped after update176 when 10-step K2 KL
  reached `0.14287` and updates 175–176 were both `>=0.20`; this produced the
  original early-stop label. A diagnostic continuation reached iter199 and
  completed two 400-simulation evals: mean pass@1/pass@4(any)/pass^4 was
  `32.38/58.00/11.50%`, versus iter99 `29.50/56.50/10.50%`. Telecom improved,
  but seed301 airline fell `7.5pp`, overall deltas versus iter99 were uncertain,
  and namespace raw attempts increased. Iter199 is a valid high-KL checkpoint
  and Telecom-oriented candidate; long100 iter99 remains the better-balanced
  default.

- [tau2-sft](../experiments/tau2-sft/README.md) — tau2-bench SFT data
  conversion and Qwen3-4B-Instruct-2507 supervised fine-tuning scripts.

- [tau2-banking-expert-sft](../experiments/tau2-banking-expert-sft/README.md) —
  Banking synthetic expert-trajectory conversion to native Qwen3 SFT rows and
  the Banking-only supervised fine-tuning experiment.

- [tau2-user-sft](../experiments/tau2-user-sft/README.md) — tau2-bench user
  simulator SFT data conversion and Qwen3-4B-Instruct-2507 user-model
  supervised fine-tuning.

- [tau2-eval-user-stop-parser](../experiments/tau2-eval-user-stop-parser/README.md) —
  tau2-bench official Pass@4 with the STOP-trained user model and the user
  sglang Qwen tool-call parser enabled; telecom user tools now execute (0 →
  ~1000) and telecom metrics rise sharply.

- [tau2-eval-user-stop-v2-parser](../experiments/tau2-eval-user-stop-v2-parser/README.md) —
  completed tau2-bench comparison of Qwen3.5-4B, Qwen3-4B-Instruct-2507,
  and the multitool SFT Agent under Gemini, original Qwen, v1, and v2 User
  simulators, including official `pass^k` and at-least-one-success `pass@k`.

- [tau2-eval-qwen36-user-timing-tp2](../experiments/tau2-eval-qwen36-user-timing-tp2/README.md) —
  four-GPU official tau2 evaluation of raw Qwen3-4B-Instruct-2507 with a
  non-thinking Qwen3.6-27B User on TP2, retaining request and stage timing
  evidence for the final wall-time breakdown.

- [vitabench-qwen35-smoke](../experiments/vitabench-qwen35-smoke/README.md) —
  install and compatibility validation for VitaBench's 29 dependencies, plus a
  one-task Qwen3.5-4B smoke using the existing remote user/evaluator API.

- [vitabench-qwen35-full-eval](../experiments/vitabench-qwen35-full-eval/README.md) —
  full Chinese VitaBench evaluation of Qwen3.5-4B across four 100-task suites
  with four trials per task on four GPUs, strict completeness checks, lossless
  Qwen/GPT/Gemini SFT journals, and comparison to the published four-domain
  average score of 22.0.

- [vitabench-qwen35-local-role-eval](../experiments/vitabench-qwen35-local-role-eval/README.md) —
  cost-controlled Chinese VitaBench evaluation of Qwen3.5-4B with local
  Qwen3.6-27B user-simulator and evaluator replicas on eight 80GB GPUs.

- [vitabench-evaluator-thinking-ab](../experiments/vitabench-evaluator-thinking-ab/README.md) —
  paired frozen-prompt and chained-state replay of the 6,349 persisted
  VitaBench evaluator windows; the measured speed/quality tradeoff establishes
  Qwen3.6-27B non-thinking evaluation as the default for fresh runs.

- [vitabench-qwen3-4b-instruct-2507-full-eval](../experiments/vitabench-qwen3-4b-instruct-2507-full-eval/README.md) —
  complete Chinese VitaBench evaluation of raw Qwen3-4B-Instruct-2507 with
  local non-thinking Qwen3.6-27B User and Evaluator roles on eight 80GB GPUs.

- [two-node-gpu-probe](../experiments/two-node-gpu-probe/README.md) — two-node GPU submission probe; 2 × 8 GPU normal task replacing the stopped spot task.

- [serve](../experiments/serve/README.md) — long-running single-GPU sglang
  serving of Qwen3-4B-Instruct-2507 on the cluster with an OpenAI-compatible
  endpoint reachable off-cluster.

- [cross-job-probe](../experiments/cross-job-probe/README.md) — one-GPU probe
  confirming a job container can call the sglang server deployed in a different
  job over the pod network (TCP, health, chat, tool call all pass).

- [tau2-external-user-pool](../experiments/tau2-external-user-pool/README.md) — Two TP2 Qwen3.6-27B User replicas on4 GPUs;old20873 stopped2026-09-20,replacement22031 submitted for both SFT4505 expert jobs. API uses the replacement job compute IP.
