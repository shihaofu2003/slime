# Qwen3.8 Banking Curriculum Evaluation

Experiment name: `tau2-banking-curriculum-qwen38-eval`. Purpose: measure the
single-trial accuracy of Qwen3.8-27B on all 564 validated Banking curriculum
task specifications, using Qwen3.8-27B for both Agent and User.

## Protocol

- Agent: `/mnt/afs/models/Qwen3.8-27B`, official-native Tau2 Agent, `xhigh`
  thinking with model-card sampling (`temperature=1.0`, `top_p=0.95`,
  `top_k=20`). The initial full pass uses 8,192 output tokens per turn; most
  targeted recovery and all scale-out use 16,384 after a valid recovery needed
  12,359 tokens. The final `task_071` recovery uses 32,768. Initial runs preserve
  prior-turn reasoning; compact-history recovery keeps current-turn xhigh
  reasoning but does not replay hidden reasoning from earlier turns. Scale-out
  and recovery use the model's native 262,144-token context; the initial pass
  used 163,840 and exposed one 166,622-token multi-turn request.
- User: `/mnt/afs/models/Qwen3.8-27B`, thinking disabled, temperature 0,
  top-p 1. The initial pass uses 512 output tokens per turn; scale-out uses
  1,024 after a long full-composition User response reached the old limit.
- Banking environment: real Tau2 environment, one trial per task, seed 300,
  maximum 200 steps.
- Metric: exact Tau2 task success, also grouped by curriculum variant, level,
  retrieval mode, and business category.

## Evaluation partitions

- `golden_retrieval` shards 0–3: 95, 96, 92, and 92 tasks.
- `bm25` shards 0–1: 95 and 94 tasks.
- The six partitions are disjoint and cover all 564 task ids. Variants from one
  source stay together within each retrieval-mode split.

## Artifacts

Each run directory contains its staged task manifest, official Tau2 summary,
raw `results.json`, and `curriculum_metrics.json` / `.md`. The six-partition
first-pass aggregate is [all_8k_raw.md](all_8k_raw.md); it preserves all 16
infrastructure-error rows as failures. The final
[recovered aggregate](all_recovered.md) replaces exactly those 16 rows, while
the legacy-named [diagnostic manifest](training_pool_manifest.json) retains the
selected trajectory path for every task and marks every record as ineligible
for training.

## Results

- Final exact success is `329/564` (`58.33%`), versus `327/564` (`57.98%`)
  when the 16 first-pass protocol failures are conservatively counted as zero.
  All replacements are valid trajectories: 563 terminate with `user_stop` and
  one with `max_steps`; there are no remaining infrastructure errors.
- By variant: decision-only `84/94` (`89.36%`), evidence-given `58/94`
  (`61.70%`), full-composition `32/94` (`34.04%`), retrieval-only `93/94`
  (`98.94%`), single-action `48/94` (`51.06%`), and two-skill composition
  `14/94` (`14.89%`). BM25 tasks score `126/189` (`66.67%`) and supplied-golden
  tasks score `203/375` (`54.13%`).
- Atomic checks pass at `752/1232` actions (`61.04%`), `283/383`
  communication facts (`73.89%`), and `424/563` DB states (`75.31%`). Agent
  actions pass `673/1065` (`63.19%`), while User actions pass `79/167`
  (`47.31%`).
- The conservative first-attempt score is `315/564` (`55.85%`); 116 framework
  retry events affect 50 tasks. Raising the output budget recovered two exact
  successes (`task_090` and `task_096`) as well as fourteen substantive
  failures that must remain scored zero.
- All 564 tasks are benchmark-derived diagnostics. The 329 successful
  environment-verified trajectories measure model capability but are not
  SFT/OPD candidates and cannot enter GRPO.

## Jobs

- **STOPPED** — seven-task BM25 smoke with Qwen3.8-27B serving both Agent and
  User; it verified xhigh reasoning and structured calls, then exposed a
  65,536-token Agent context overflow on a long BM25 trajectory:
  [run log](jobs/12887-dual-qwen38-banking-curriculum-bm25-smoke-0903-142649514/run_0_20260903_142649514.log).
- **SUCCEEDED** — seven-task BM25 smoke with the Agent context raised to
  163,840 tokens; exact success `2/7`, Agent retrieval actions `4/4`, DB success
  `2/3`, all seven terminated with `user_stop`:
  [run log](jobs/12891-dual-qwen38-banking-bm25-smoke-ctx160k-0903-144443059/run_0_20260903_144443059.log).
- **STOPPED BEFORE START** — full BM25 partition; replaced by 4-GPU shards
  after waiting 15 minutes for an 8-GPU node, with zero GPU time consumed:
  [run log](jobs/12934-qwen38-banking-bm25-full-0903-151434348/run_0_20260903_151434348.log).
- **STOPPED BEFORE START** — full golden-retrieval shard 0; replaced by 4-GPU
  shards after waiting 15 minutes for an 8-GPU node, with zero GPU time
  consumed:
  [run log](jobs/12935-qwen38-banking-golden-s0-full-0903-151434614/run_0_20260903_151434614.log).
- **STOPPED BEFORE START** — full golden-retrieval shard 1; replaced by 4-GPU
  shards after waiting 15 minutes for an 8-GPU node, with zero GPU time
  consumed:
  [run log](jobs/12936-qwen38-banking-golden-s1-full-0903-151434871/run_0_20260903_151434871.log).
- **STOPPED** — 95-task golden-retrieval shard 0/4; the first four trajectories
  exposed gold-label leakage and strict-format false negatives in the generated
  User scenarios:
  [run log](jobs/12964-qwen38-bank-golden-s0of4-0903-153208314/run_0_20260903_153208314.log).
- **STOPPED** — 95-task BM25 shard 0/2 before evaluation for the same task
  interface correction:
  [run log](jobs/12965-qwen38-bank-bm25-s0of2-0903-153208571/run_0_20260903_153208571.log).
- **STOPPED** — 92-task golden-retrieval shard 2/4 before evaluation for the
  same task interface correction:
  [run log](jobs/12966-qwen38-bank-golden-s2of4-0903-153208841/run_0_20260903_153208841.log).
- **STOPPED** — 96-task golden-retrieval shard 1/4 before evaluation for the
  same task interface correction:
  [run log](jobs/12967-qwen38-bank-golden-s1of4-0903-153209119/run_0_20260903_153209119.log).
- **STOPPED BEFORE START** — 94-task BM25 shard 1/2 for the same task interface
  correction, with zero GPU time consumed:
  [run log](jobs/12968-qwen38-bank-bm25-s1of2-0903-153209392/run_0_20260903_153209392.log).
- **STOPPED BEFORE START** — 92-task golden-retrieval shard 3/4 for the same
  task interface correction, with zero GPU time consumed:
  [run log](jobs/12969-qwen38-bank-golden-s3of4-0903-153209654/run_0_20260903_153209654.log).
- **FAILED BEFORE INFERENCE** — corrected three-task golden-retrieval smoke;
  its new reference-replay preflight inherited the not-yet-staged run data path:
  [run log](jobs/12979-qwen38-bank-fixed-golden-smoke-0903-155401731/run_0_20260903_155401731.log).
- **FAILED BEFORE INFERENCE** — corrected three-task BM25 smoke with the same
  preflight ordering issue:
  [run log](jobs/12980-qwen38-bank-fixed-bm25-smoke-0903-155402027/run_0_20260903_155402027.log).
- **STOPPED BEFORE INFERENCE** — corrected three-task golden-retrieval smoke v2;
  its full replay preflight duplicated the already-passed environment invariant
  and held GPUs idle, so it was removed:
  [run log](jobs/12981-qwen38-bank-fixed-golden-smoke-v2-0903-155705699/run_0_20260903_155705699.log).
- **SUCCEEDED** — corrected three-task BM25 smoke v2, exact success `3/3`;
  all action, DB, and communication checks passed. The original full task needed
  one framework retry after a reasoning-only Assistant response, then passed:
  [run log](jobs/12982-qwen38-bank-fixed-bm25-smoke-v2-0903-155705997/run_0_20260903_155705997.log),
  [metrics](eval/smoke-bm25-shard0of1/seed300_0903_075906/curriculum_metrics.md).
- **SUCCEEDED** — corrected three-task golden-retrieval smoke v3, exact success
  `3/3`; all action, DB, and communication checks passed:
  [run log](jobs/12983-qwen38-bank-fixed-golden-smoke-v3-0903-160110017/run_0_20260903_160110017.log),
  [metrics](eval/smoke-golden-retrieval-shard0of1/seed300_0903_080225/curriculum_metrics.md).
- **STOPPED AFTER 10/95** — corrected 95-task BM25 full shard 0/2; the paired
  Golden run exposed unrelated User-tool carryover in bounded tasks, so this
  first full attempt was stopped before further compute was spent:
  [run log](jobs/12986-qwen38-bank-fixed-bm25-s0of2-0903-161402578/run_0_20260903_161402578.log).
- **STOPPED AFTER 6/95** — corrected 95-task golden-retrieval full shard 0/4;
  `task_005_single_action` executed the evaluated email change correctly, then
  an unrelated exposed User application tool changed the DB and created a
  false failure. Bounded tasks now expose only evaluated User tools:
  [run log](jobs/12987-qwen38-bank-fixed-golden-s0of4-0903-161402886/run_0_20260903_161402886.log).
- **STOPPED BEFORE INFERENCE** — seven-task bounded-User Golden smoke v4; its
  source-limit selected `task_001/002` instead of the intended diagnostic pair:
  [run log](jobs/12989-qwen38-bank-bounded-user-smoke-v4-0903-163223522/run_0_20260903_163223522.log).
- **STOPPED BEFORE INFERENCE** — seven-task bounded-User Golden smoke v5; target
  sources were correct, but review found that systematic Evidence prompts still
  asked for a subjective "most relevant" line under an exact-string scorer:
  [run log](jobs/12991-qwen38-bank-bounded-user-smoke-v5-0903-163437712/run_0_20260903_163437712.log).
- **SUCCEEDED** — final seven-task objective-interface smoke v6 for
  `task_001/005`, exact success `6/7`; every trajectory ended with `user_stop`,
  and all Evidence, Decision, and single-action tasks passed. The remaining
  two-skill failure was a substantive Agent refusal of the supplied verification
  and email-change sequence, not User-tool carryover or a scorer/interface
  failure:
  [run log](jobs/12992-qwen38-bank-objective-smoke-v6-0903-164213738/run_0_20260903_164213738.log),
  [metrics](eval/smoke-golden-retrieval-shard0of4/seed300_0903_084338/curriculum_metrics.md).
- **STOPPED AFTER 18/95** — corrected full BM25 shard 0/2; stopped when the
  paired Golden run exposed asymmetric comma normalization in Tau2's
  communicate evaluator, so these partial scores are excluded:
  [run log](jobs/12997-qwen38-bank-objective-bm25-s0of2-v3-0903-165536476/run_0_20260903_165536476.log).
- **STOPPED AFTER 40/95** — corrected full golden-retrieval shard 0/4; an
  Agent answer exactly equal to a comma-containing expected string was scored
  false because commas were removed only from the answer side. These partial
  scores are excluded and the shard will be rerun after the evaluator fix:
  [run log](jobs/12998-qwen38-bank-objective-golden-s0of4-v3-0903-165536754/run_0_20260903_165536754.log).
- **DIAGNOSTIC SUCCEEDED, EXCLUDED** — scorer-fixed 4K-token
  golden-retrieval shard 0/4, exact success `52/95` (`54.74%`), with all 95
  trajectories ending in `user_stop`. It is excluded from the final aggregate
  because later full-composition tasks showed 4,096 output tokens were
  insufficient for `xhigh` reasoning:
  [run log](jobs/13018-qwen38-bank-scorerfixed-golden-s0of4-v4-0903-172557360/run_0_20260903_172557360.log),
  [metrics](eval/full4-golden-retrieval-shard0of4/seed300_0903_093149/curriculum_metrics.md).
- **STOPPED AFTER 37/95, EXCLUDED** — scorer-fixed 4K-token BM25 shard 0/2,
  partial exact success `25/37`; several full-composition attempts exhausted
  all 4,096 tokens in reasoning without producing a final message, so all six
  partitions were restarted with an 8,192-token limit:
  [run log](jobs/13019-qwen38-bank-scorerfixed-bm25-s0of2-v4-0903-172557644/run_0_20260903_172557644.log).
- **STOPPED DURING MODEL LOAD, EXCLUDED** — 4K-token golden-retrieval shard
  1/4:
  [run log](jobs/13042-qwen38-bank-scorerfixed-golden-s1of4-v4-0903-182436198/run_0_20260903_182436198.log).
- **STOPPED DURING MODEL LOAD, EXCLUDED** — 4K-token BM25 shard 1/2:
  [run log](jobs/13043-qwen38-bank-scorerfixed-bm25-s1of2-v4-0903-182446940/run_0_20260903_182446940.log).
- **STOPPED BEFORE START, EXCLUDED** — 4K-token golden-retrieval shard 2/4:
  [run log](jobs/13044-qwen38-bank-scorerfixed-golden-s2of4-v4-0903-182447875/run_0_20260903_182447875.log).
- **STOPPED BEFORE START, EXCLUDED** — 4K-token golden-retrieval shard 3/4:
  [run log](jobs/13045-qwen38-bank-scorerfixed-golden-s3of4-v4-0903-182448985/run_0_20260903_182448985.log).
- **SUCCEEDED** — final 8K-token golden-retrieval shard 0/4, exact success
  `54/95` (`56.84%`), all 95 trajectories ended in `user_stop`:
  [run log](jobs/13048-qwen38-bank-8k-golden-s0of4-v5-0903-183347988/run_0_20260903_183347988.log),
  [metrics](eval/full4-golden-retrieval-shard0of4/seed300_0903_103555/curriculum_metrics.md).
- **FAILED OFFICIAL VALIDITY CHECK** — final 8K-token golden-retrieval shard
  1/4 produced 95 valid trajectories plus one infrastructure-error simulation,
  `task_086_evidence_given`, after four reasoning-only Agent responses. Its raw
  exact result is `50/96`; the infrastructure-error row will be replaced by the
  targeted recovery result:
  [run log](jobs/13049-qwen38-bank-8k-golden-s1of4-v5-0903-183348841/run_0_20260903_183348841.log).
- **SUCCEEDED** — final 8K-token golden-retrieval shard 2/4, exact success
  `48/92` (`52.17%`), all 92 trajectories ended in `user_stop`:
  [run log](jobs/13050-qwen38-bank-8k-golden-s2of4-v5-0903-183350187/run_0_20260903_183350187.log),
  [metrics](eval/full4-golden-retrieval-shard2of4/seed300_0903_103547/curriculum_metrics.md).
- **SUCCEEDED** — final 8K-token golden-retrieval shard 3/4, exact success
  `51/92` (`55.43%`), all 92 trajectories ended in `user_stop`, with no
  framework retry:
  [run log](jobs/13051-qwen38-bank-8k-golden-s3of4-v5-0903-183351300/run_0_20260903_183351300.log),
  [metrics](eval/full4-golden-retrieval-shard3of4/seed300_0903_103604/curriculum_metrics.md).
- **FAILED OFFICIAL VALIDITY CHECK** — final 8K-token BM25 shard 0/2 produced
  all 95 rows with raw exact success `62/95`, including nine infrastructure
  errors that are being targeted for recovery:
  [run log](jobs/13052-qwen38-bank-8k-bm25-s0of2-v5-0903-183352232/run_0_20260903_183352232.log).
- **FAILED OFFICIAL VALIDITY CHECK** — final 8K-token BM25 shard 1/2 produced
  all 94 rows with raw exact success `62/94`, including six infrastructure
  errors that are being targeted for recovery:
  [run log](jobs/13053-qwen38-bank-8k-bm25-s1of2-v5-0903-183353725/run_0_20260903_183353725.log).
- **SUCCEEDED** — targeted recovery of `task_086_evidence_given`, using
  Qwen3.8-27B for both roles, a 16K Agent output limit, and seed 301. The
  protocol completed with `user_stop` after 12,359 output tokens; exact success
  was `0/1` because the final answer missed the required communication fact.
  This replaces an infrastructure failure with a substantive scored failure:
  [run log](jobs/13119-qwen38-bank-recover-task086-16k-v6-0903-191837496/run_0_20260903_191837496.log).
- **SUCCEEDED** — targeted 16K, seed-301 recovery of the 8K protocol failure
  `task_022_full_composition`. It completed a valid 92-message trajectory with
  `user_stop`; exact success remained `0/1`, so this is a substantive failure:
  [run log](jobs/13146-qwen38-bank-recover-task022-16k-v6-0903-193918483/run_0_20260903_193918483.log).
- **STOPPED AFTER ATTEMPT 1** — targeted seed-301 recovery of
  `task_058_full_composition`. Although logical context was 262K, the old 0.90
  memory setting supplied only 170,070 physical cache tokens; its 172,558-token
  input was rejected, so the remaining retries were stopped:
  [run log](jobs/13149-qwen38-bank-recover-task058-16k-ctx262k-v7-0903-195148938/run_0_20260903_195148938.log).
- **STOPPED AFTER ATTEMPT 1** — corrected single-GPU
  `task_058_full_composition` recovery, using seed 302, 16K output, 262K
  logical context, and 0.95 memory fraction. The trajectory reached a
  203,829-token input, exhausting the 203,904-token physical KV cache and
  leaving only 73 generated tokens, so the remaining same-topology retries
  were stopped; recovery continues with a two-GPU Agent:
  [run log](jobs/13154-qwen38-bank-recover-task058-16k-ctx262k-mem95-v8-0903-200535874/run_0_20260903_200535874.log).
- **SUCCEEDED** — targeted seed-301 recovery of
  `task_041_full_composition` after four empty-final 8K attempts, using 16K
  output, 262K context, and 0.95 Agent memory fraction. It completed 72
  messages with `user_stop`; exact success was `0/1`, with 14/25 expected
  actions matched and a final DB mismatch, so this is a substantive failure:
  [run log](jobs/13152-qwen38-bank-recover-task041-16k-ctx262k-v7-0903-195810167/run_0_20260903_195810167.log).
- **SUCCEEDED** — targeted seed-303 recovery of
  `task_058_full_composition`, job 13215. The Agent used tensor parallelism
  across two Qwen3.8-27B GPUs so its 262K context was not bounded by one GPU's
  KV cache; the User remained the same Qwen3.8-27B model on a third GPU. It
  completed with `user_stop`, no infrastructure error, and exact success
  `0/1` due to a final DB mismatch:
  [run log](jobs/13215-qwen38-bank-recover-task058-16k-tp2-v10-0903-205624475/run_0_20260903_205624475.log).
- **EVALUATION COMPLETE; OFFICIAL VALIDITY CHECK FAILED** — targeted seed-304
  recovery of the remaining 12 BM25 infrastructure-error task IDs, job 13249.
  It saved all 12 rows, including exact successes for `task_090` and
  `task_096`, valid substantive failures for eight tasks, and two repeated
  infrastructure rows (`task_063` and `task_071`). The wrapper correctly
  failed its no-infrastructure-error check; later compact-history runs replace
  those two rows:
  [run log](jobs/13249-qwen38-bank-recover-bm25-remaining12-16k-v11-0903-232116482/run_0_20260903_232116482.log).
- **FAILED BEFORE START** — targeted seed-305 TP2 recovery of
  `task_063_full_composition`, job 13275. The cluster rejected the unsupported
  3-GPU/42-CPU/594-GiB resource tuple, so no GPU time was consumed:
  [submission log](jobs/13275-qwen38-bank-recover-task063-tp2-v12-0904-000445485/submit_20260904_000445485.log).
- **FAILED OFFICIAL VALIDITY CHECK** — replacement TP2 recovery of
  `task_063_full_composition`, job 13276. It explicitly requested the supported
  4-GPU/56-CPU/792-GiB tuple; TP2 removed the physical KV-cache limit, but two
  attempts exceeded the model's native 262,144-token context and two returned
  a reasoning-only empty final:
  [run log](jobs/13276-qwen38-bank-recover-task063-tp2-v13-0904-000721222/run_0_20260904_000721222.log).
- **SUCCEEDED** — compact-history TP2 recovery of
  `task_063_full_composition`, job 13285, with the same Qwen3.8 xhigh Agent and
  Qwen3.8 User. It keeps the current turn's reasoning but omits prior-turn
  hidden reasoning from later prompts, and requested 4 GPUs, 56 CPUs, and 792
  GiB memory. The 85-message trajectory ended with `user_stop`; exact success
  was `0/1` because only 2/4 expected actions matched and the final DB differed:
  [run log](jobs/13285-qwen38-bank-recover-task063-tp2-compact-v14-0904-003511451/run_0_20260904_003511451.log).
- **SUCCEEDED** — 32K compact-history TP2 recovery of
  `task_071_full_composition`, job 13297. It explicitly requested the supported
  4-GPU/56-CPU/792-GiB tuple, started about 79 seconds after submission, and
  completed a valid 66-message `user_stop` trajectory. Exact success remained
  `0/1`, so the final aggregate records a substantive failure rather than the
  original reasoning-only protocol failure:
  [run log](jobs/13297-qwen38-bank-recover-task071-32k-tp2-compact-v15-0904-010927384/run_0_20260904_010927384.log).
