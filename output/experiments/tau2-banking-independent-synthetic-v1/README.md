# tau2-banking-independent-synthetic-v1

Purpose: build and validate an independent Banking task curriculum without using
benchmark tasks, entities, requests, gold actions, initial states, or required
document sets as generation inputs.

## Current status

The accepted minimal scope is exactly 542 train, 75 dev, and 30 challenge tasks:
twice the 271 unique Telecom tasks in the AReaL training source. All 647 tasks
have distinct scenario IDs and four valid Qwen3.8 autonomous trials. Every
train task has a successful training trajectory; 174 are GRPO frontier tasks
and 368 are behavior anchors. Reviewed job 16251 passed final acceptance after
replaying all 647 references at reward one, checking all 530 BM25 queries at
top 10, auditing the 480-document runtime with zero benchmark reuse, and
directly validating 1,914 successful trajectories for reward, source,
eligibility, unique IDs, and complete train coverage. Agent and User runtime
manifests both resolve to `/mnt/afs/models/Qwen3.8-27B`.

The 105 held-out tasks scored 82.69% pass@1, 91.30% pass@4(any), and 73.91%
pass^4 on the original four attempts, whose first protocol-failure rate was
13.33%. Recovery produced exactly 420 valid trials across all 105 tasks; the
resulting valid-trial aggregate is 82.38%/91.43%/71.43%. DB, ACTION, and
COMMUNICATE success are 79.41%, 80.00%, and 100.00% in that aggregate.

New submissions use this real experiment name directly and stay within 24
Banking GPUs. Earlier `audio-*` submissions remain historical logs only.
The larger `scale/reduced-r18` selection and all non-selected candidates remain
diagnostic artifacts; they are not part of the minimal training set.

The superseded 1,500-task pilot had 1,500/1,500 valid Tau2 schemas, 1,500/1,500
reference reward-one replays, and 1,070/1,070 BM25 top-10 hits, with zero reuse
reported by the isolation audit. Its Qwen3.8 smoke then exposed a stricter
semantic defect: category-compatible documents did not guarantee that the
initialized product, decisive policy fact, requested operation, and reference
action described the same case. Those smoke results are diagnostic only. The
r5 templates bound action tasks to exact allowlisted policy anchors and aligned
products, facts, state, and action arguments; all 1,500 reference replays passed.
The follow-up r6 removed a trivial title-copy shortcut from L1/L3 and locked the
semantic outcome against naturalizer edits; its 1,500-task rebuild passed every
schema, reference, retrieval, and isolation check. A final document-level audit
then replaced generic identity/PIN anchors with the allowlisted Gold Years
policy and aligned account state to those policies. The r7 pilot replay passed;
the explicit 75-cell category × case-kind template matrix also passed 75/75
reference replays and 120/120 BM25 checks with zero isolation findings. The new
30-task naturalization passed 30/30 with zero static or isolation failures.
The r7 seed-300 smoke showed that correct L1 paraphrases and L3 decisions could
fail exact-substring communication checks, and that refusal Users could stop
before an implied transfer. Those runs are diagnostic. Candidate r8 uses short
communication contracts and explicit transfer consent; its full replay passed
1,575/1,575 tasks and 1,190/1,190 target-document retrieval checks in job
14613. Expansion remains gated on the replacement smoke and four-seed pilot.
The strict replay checks the target document for each individual search, plus
date, amount, and foreign-key integrity. A complete L1 audit then found 13
introductory, incomplete, or table-header strings among 150 r8 facts. Candidate
r9 skips those lines; job 14656 passed all 1,575 replays and 1,190 retrieval
checks. The r8 replacement smoke then exposed two final contract issues: numeric
facts could fail after harmless currency formatting, and User-owned actions had
no private execution protocol. Candidate r11 evaluates numeric communication
anchors and tells only the User simulator when to execute its allowed tool;
job 14895 passed its full replay. A subsequent visible-solvability audit found
that randomized discoverable-tool suffixes were absent from the allowed policy
corpus. Candidate r12 adds programmatic, non-private case-note action labels and
parameters to the User's opening brief, while preserving policy retrieval and
multi-step execution as Agent work. Job 14912 passed its full replay; job 14921
naturalized all 30 smoke tasks with zero static or isolation failures. The r12
golden-retrieval smoke completed 13/13 valid trials at pass@1 76.92%. Its three
business failures exposed equivalent-path contract defects rather than protocol
failures: an Agent dynamic action did not name its wrapper, transfer compared an
irrelevant reason value, and a reasonable repeated verification added a log row
outside the reference state. Candidate r13 makes the wrapper invocation exact,
ignores transfer metadata, and initializes applicable L3/L4 write cases with an
explicitly visible completed verification. Its full replay ran as job 14973
and passed all 1,575 environment executions, all 1,190 BM25 checks, and both
isolation audits. The r12 BM25 diagnostic finished 15 valid trials at pass@1
46.67% plus two infrastructure errors; every business failure exercised one of
the r13 contract fixes. The r13 replacement naturalization completed 30/30
with zero static or isolation findings in job 14974. Its golden smoke completed
13 valid trials at pass@1 84.62% with no protocol failures; its BM25 smoke had
14 valid trials at pass@1 42.86% plus three protocol failures. These runs then
showed that a private User
instruction still cannot guarantee that the User model repeats a randomized
tool label in its opening. Candidate r14 makes the Qwen-naturalized request a
standard initial User message visible to both participants, while keeping
identity fields and User-tool timing private. Job 15013 passed all 1,575
reference replays, all 1,190 BM25 checks, and both isolation audits. The r13
BM25 smoke also showed that an L5 refusal could be semantically correct yet fail
because ACTION scoring required arbitrary filler reads. Candidate r15 scores
L4--L6 refusals by the customer-owned human-review request's final DB state;
job 15015 passed all 1,575 reference replays, all 1,190 BM25 checks, and both
isolation audits. Its final 30-task naturalization completed with 30 valid
outputs, three reported original-wording fallbacks, and zero static or
isolation findings in job 15031. The replacement Agent/User smoke runs are jobs
15040 and 15041. The golden-retrieval arm completed 12 valid trajectories at
pass@1 91.67%, with one `max_steps` protocol failure selected for a 32,768-token
recovery in job 15051; that recovery produced one valid reward-one trajectory.
The BM25 arm completed 12 valid trajectories at pass@1
66.67%; its five empty-response infrastructure failures were selected for a
32,768-token recovery in job 15055. That diagnostic recovery produced two valid
trajectories and retained three infrastructure failures; it is not retried after
r16 superseded the task definitions. Inspection found that a semantically correct
decision response could miss the literal `Yes` check and that profile-change
Users were not given the requested new email. Candidate r16 makes the answer
format explicit and includes that target value in both the opening and private
known information. Job 15057 passed all 1,575 reference replays, all 1,190 BM25
checks, and both isolation audits. Its replacement smoke naturalization is job
15060; all 30 outputs passed with no fallback or isolation finding. The r16
Agent/User golden-retrieval smoke completed 13/13 valid trajectories at pass@1
92.31% with no protocol failure in job 15073; its sole failure omitted a required
L1 numeric fact. The BM25 arm completed 14 valid trajectories at pass@1 85.71%
in job 15072, with two empty-response failures and one context-limit failure.
The three protocol-failed tasks received a 32,768-token recovery in job 15089:
two produced reward-one trajectories and one retained an empty-response failure.
That single missing valid trial is queued again in job 15135. Full-pilot
naturalization in job 15092 produced 1,500/1,500 valid tasks, with ten safe
fallbacks, no generation failure, and zero static or isolation findings. All
28 four-seed pilot evaluation shards are submitted as jobs 15099--15112,
15114--15120, 15124--15127, 15130--15131, and 15133. Those r16 evaluations are
diagnostic and the template is not frozen.
The three seed-300 golden-retrieval shards have completed: 672/675 trajectories
were valid and 639/672 valid trajectories succeeded (pass@1 95.09%); all three
missing trials were empty-response infrastructure failures selected for later
32,768-token recovery. Their business failures exposed a remaining L1 semantic
issue: a category-topical document could yield an unrelated first fact, such as
an environmental-report cadence for an account-referral task. Candidate r17
selects the first complete fact that matches the business category and keeps a
short FAQ answer with its question context. This changes 29/150 pilot L1 facts;
52 local tests pass. The 23 nonterminal r16 evaluation/recovery jobs were stopped
without deleting their diagnostic outputs. Job 15200 is the replacement full
reference replay and isolation audit; it passed all 1,575 environment replays,
1,190 BM25 checks, and both zero-finding isolation audits. Job 15202 is the
replacement 30-task naturalization; it completed 30/30 with one safe fallback,
no generation failure, and zero static or isolation findings. The replacement
golden-retrieval smoke in job 15216 completed 13/13 valid trajectories at
pass@1 100%, with zero protocol failure; the replacement BM25 smoke is job
15217. Its 14 valid trajectories reached pass@1 71.43%, while three
infrastructure failures were selected for the 32,768-token recovery in job
15236. That recovery produced two valid trajectories and retained one
empty-response infrastructure failure, selected alone for job 15275. The four
valid business failures are capability failures on executable L4--L6 contracts,
not a new category/fact mismatch. Full-pilot r17
naturalization in job 15228 completed 1,500/1,500 tasks with three safe
fallbacks, no generation failure, and zero static or isolation findings. All
28 r17 four-seed pilot evaluation shards were uniquely submitted as jobs
15243--15255, 15257--15261, and 15263--15272. Incremental results exposed one
remaining L1 ambiguity: a broad category description could match several facts
in the same document even though the program selected a later section. Those
29 nonterminal evaluation/recovery jobs were stopped with diagnostic outputs
retained. Candidate r18 adds a short protected opening-phrase locator for the
selected L1 fact without copying its complete answer; all 150 pilot locators
identify their gold fact and none equals the full fact. Its full replay is job
15292; all 1,575 executions, 1,190 BM25 checks, and both isolation audits
passed. The replacement 30-task naturalization is job 15294; all outputs passed
with one safe fallback, and all four L1 locators were preserved. Replacement
Agent/User smoke jobs are 15297 and 15298. The golden arm produced 12/12 valid
reward-one trajectories plus one empty-response infrastructure failure; only
that task received a 32,768-token recovery in job 15307 and then produced a
valid reward-one trajectory.
The BM25 arm completed with 16 valid trajectories at pass@1 75.00% and one
`max_steps` protocol failure. Its first 32,768-token recovery in job 15310
returned an empty Assistant response and remains a protocol deficit. No r18
smoke failure indicates a new contract ambiguity, so full
pilot naturalization is job 15311; all 1,500 outputs passed without fallback,
all 150 L1 locators were preserved, and the isolation audit had zero findings.
Twenty of 28 four-seed pilot shards ran as jobs 15314--15332 and 15343. They
produced 4,275/4,275 first-attempt records: 3,902 protocol-valid trajectories,
3,597 reward-one trajectories, and 373 protocol deficits. The remaining eight
first-attempt shards were submitted after the job-manager endpoint recovered.
Seed 302 golden 2/3 and all seven seed-303 shards are jobs 15399,
15400--15407, and 15514. All 28 r18 pilot shards completed and produced the
expected 6,000 records. Of these, 5,515 are protocol-valid and 5,102 succeeded:
pass@1 is 92.51%, pass@4(any) is 97.70%, and pass^4 is 87.23% among the 1,175
tasks already having four valid trials. The 485 protocol deficits comprise 447
infrastructure errors and 38 max-step terminations. Recovery round 1 selects
325 unique tasks (319 BM25 and six golden-retrieval) and runs as jobs
15545--15547 with a 32,768-token output limit. It produced 325/325 records,
250 protocol-valid trajectories, and 218 reward-one trajectories. All six
golden deficits are resolved; 151 BM25 tasks still need 235 valid replacement
trials. Recovery round 2 then produced 151/151 records, 98 valid trajectories,
and 76 successes; its 53 protocol failures comprise 50 infrastructure errors
and three max-step terminations. Recovery round 3 (job 15754) produced all 79
requested records, including 47 valid and 45 successful trajectories. The
exact remaining pilot deficit was 90 valid trials across 46 BM25 tasks; their
next one-attempt round was aliased job 15811. That round produced 46/46 records,
including 20 valid trajectories and 26 protocol failures. Across all completed
attempts, 34 BM25 tasks still lack 70 valid trials; the next exact 34-task round
is aliased job 15866. That round produced 34/34 records, including 18 valid
trajectories and 15 reward-one trajectories. The cumulative deficit is now 52
valid trials across 22 BM25 tasks, and 1,478/1,500 pilot tasks have four valid
trials. Those 22 tasks run next as aliased job 15907. No new
contract ambiguity was found in the four-seed pilot. Job 15907 produced 22/22
records, including six valid and five reward-one trajectories; 18 tasks now
lack 46 valid trials and run next as aliased job 15916. That round produced
18/18 records, including four valid reward-one trajectories; 16 tasks still
lack 42 valid trials and run next as aliased job 15927. That round produced
16/16 records, including two valid reward-one trajectories; 15 tasks still lack
40 valid trials. Job 15935 was user-stopped after 13/15 records and produced
one valid trajectory. Its partial payload is preserved for diagnosis but is
excluded from selection; full replacement job 15955 is the authoritative round
9 payload. It produced 15/15 records, including three valid reward-one
trajectories and 12 protocol failures. Aliased recovery job 16004 added 15
attempts. Across 721 recovery records, 1,486/1,500 pilot tasks now have four
valid trials; 14 BM25 tasks still lack 35 valid trials and run next in job
16018. The r18 template is frozen. Five 2,000-task
train candidate batches plus 500 dev and 500 challenge tasks were built with
full reference replay and isolation checks in jobs 15548--15554.
All seven jobs passed: 11,000/11,000 schemas and reference replays earned
reward 1, all 8,702 reference BM25 queries hit top 10, and every benchmark-reuse
count remained zero. Qwen3.8 naturalization of the seven candidate shards ran
as jobs 15565--15571. All seven completed with 11,000/11,000 outputs,
six safe wording fallbacks, no failures, and zero static or isolation findings.
The merged set has 10,000/500/500 train/dev/challenge scenarios with no
cross-split overlap; its 6,300 BM25 and 4,700 golden-retrieval tasks are split
into 26 and 19 first-trial shards. All 45 first-trial shards have been submitted;
admission remains limited to three concurrent Banking jobs under the 24-GPU cap.
Combined static validation passed all 11,000 schemas, and the merged
isolation audit found zero benchmark entity, request, document-set, or initial-
record reuse; the nearest request token Jaccard was 0.210. Further submissions
use the `audio` alias and keep total nonterminal GPU demand at or below 24.
All 19 golden-retrieval first-trial shards have now completed with 4,700/4,700
records. Of these, 4,680 are protocol-valid and 4,599 succeeded (valid pass@1
98.27%); the 20 deficits comprise 18 infrastructure errors and two max-step
terminations. Aliased recovery job 15816 produced 20/20 valid replacement
records, 18 of them successful, so every golden-retrieval task now has one
valid autonomous trial and this recovery queue is empty.
BM25 first-trial shards 0--14 have completed with 3,638/3,638 records, 3,133
protocol-valid trajectories, and 2,669 successes (valid pass@1 85.19%). Their
505 deficits comprise 465 infrastructure errors and 40 max-step terminations;
shards 15, 17, and 18 retain their 184, 179, and 109 partial records for the
combined recovery selector. Shard 16/26 completed with 211 valid trajectories
and 172 successes. Shard 19/26 completed with 242/242 unique records, 205 valid
trajectories, and 183 successes; its 37 protocol deficits are retained for the
combined recovery selector. Shard 20/26 also completed with 242/242 unique
records, 213 valid trajectories, and 175 successes; its 29 protocol deficits
are likewise retained for combined recovery. Jobs 15912 and 15923 were
user-stopped after 213/242 and 135/242 records, respectively. Their partial
payloads are preserved for diagnosis but excluded from selection; full
replacement jobs 15953 and 15954 are the authoritative shard 21/26 and 22/26
first-trial payloads. They completed with 208 and 206 valid trajectories and 179
and 177 successes, respectively. Shard 23 completed with 225 valid trajectories,
197 successes, and 17 infrastructure deficits. Shards 24--25 were stopped after
the final train target was reduced; their 61 and 48 partial records are retained
only as candidate-selection evidence. The four-trial follow-up selector now
prioritizes exactly the 1,400 retained action-bearing L3--L6 train tasks before knowledge/decision forms while
retaining category and first-outcome balance inside each form. Frontier-gap
replacement requests likewise use harder replaceable form slots first instead
of spending candidate batches on near-deterministic L1/L2/decision cells; its
62-test regression suite passes. Evaluation aggregation filters larger candidate
result files to the retained contracts and keeps compact per-simulation
statistics instead of all message payloads at once; the 761 MB pilot input that
previously exceeded local memory completes with exact single-shard metric parity.
The first node-teacher smoke reached SGLang but returned an opaque HTTP 400 in
job 15136. The teacher now exposes one tool with `tool_choice: required` and
preserves HTTP error bodies; job 15151 showed that the Qwen chat template
requires a User query. Job 15152 adds that query to the hidden teacher prompt
and succeeded: all three action nodes plus the final response completed on their
first attempts, the real environment returned reward 1, and the saved trajectory
contains no hidden guidance.

The pre-existing 3,304 benchmark-derived diagnostic specifications and 2,492
successful trajectories remain in their original experiments with
`data_origin: benchmark_derived_diagnostic` and `training_eligible: false`; this
pipeline does not copy them into any candidate pool. The current manifests
confirm 564/564 and 2,740/2,740 excluded records and 329 + 2,163 successful
diagnostic trajectories:
[564-task manifest](../tau2-banking-curriculum-qwen38-eval/training_pool_manifest.json)
and
[2,740-task manifest](../tau2-banking-scale-v2-qwen38-eval/training_pool_manifest.json).

## Artifacts

- [final-minimal](final-minimal): accepted 542/75/30 tasks, contracts, scenario
  specs, 1,914 successful train trajectories, pool manifest, metrics, and
  validation reports.
- [final metrics](final-minimal/metrics.md) and
  [machine-readable metrics](final-minimal/metrics.json): original-attempt,
  recovery, aggregate, category, difficulty, and template results.
- [final acceptance](final-minimal/acceptance.json): final counts and acceptance
  result; all companion failure lists are empty.
- `allowed_documents.json`: the exact 480-document complement of the 218
  benchmark-required Banking documents.
- `candidates/pilot_{scenario_specs,contracts,tasks}.*`: independent pilot data.
- `pilot_reference_validation.json`: real Banking environment replay summary.
- `pilot_isolation_audit.json`: sealed-benchmark comparison performed only after
  generation.
- [pilot/r18_round2_metrics.json](pilot/r18_round2_metrics.json): pilot first
  trials plus the first two completed protocol-recovery rounds.
- [scale/merged-r18](scale/merged-r18): merged 10,000/500/500 tasks, contracts,
  scenario specs, static/isolation reports, and golden first-trial metrics.
- [scale/reduced-r18](scale/reduced-r18): quota-balanced 2,000/500/500 retained
  tasks, contracts, scenario specs, static/isolation reports, and recovery
  selections.
- [reduced first-attempt metrics](scale/reduced-r18/first_attempt_metrics.md):
  retained-contract metrics from the 45 original seed-300 result files.
- [scale BM25 shards 0--13 metrics](scale/merged-r18/scale_bm25_first_shards00_13_metrics.json):
  partial first-trial report; the 26-shard aggregate remains pending.
- [scale/recovery-golden-first-r18/recovery_report.json](scale/recovery-golden-first-r18/recovery_report.json):
  exact 20-task golden protocol-recovery selection.
- [PLAN.md](PLAN.md): frozen scope, quotas, evaluation protocol, and acceptance
  gates.

## Jobs

- Admission failure, unsupported 56 CPU / 256 GB shape (job 14425):
  [jobm.log](jobs/14425-banking-synthetic-pilot-build-v1-0906-165917491/jobm.log).
- Stopped after identifying excessive Tau2 DEBUG output (job 14428):
  [run log](jobs/14428-banking-synthetic-pilot-build-v1-rerun-0906-170540586/run_0_20260906_170540586.log).
- First parallel replay; exposed 167 retrieval-query validation defects (job
  14434): [run log](jobs/14434-banking-synthetic-pilot-reference-v1-0906-171305255/run_0_20260906_171305255.log).
- Corrected pilot build, replay, and isolation audit; succeeded (job 14435):
  [run log](jobs/14435-banking-synthetic-pilot-reference-v1-r2-0906-171804986/run_0_20260906_171804986.log).
- Stopped non-representative smoke selection before inference (job 14439):
  [run log](jobs/14439-banking-synthetic-smoke-naturalize-qwen38-0906-172228591/run_0_20260906_172228591.log).
- Representative 30-task Qwen3.8 naturalization (job 14440):
  [run log](jobs/14440-banking-synthetic-smoke-naturalize-qwen38-r2-0906-172623086/run_0_20260906_172623086.log).
- Qwen3.8 Agent/User BM25 smoke on the superseded template; stopped after its
  diagnostic value was exhausted (job 14452):
  [run log](jobs/14452-banking-synthetic-smoke-qwen38-bm25-0906-173453098/run_0_20260906_173453098.log).
- Qwen3.8 Agent/User golden-retrieval smoke, seed 300 (job 14453):
  [run log](jobs/14453-banking-synthetic-smoke-qwen38-golden-0906-173453449/run_0_20260906_173453449.log).
- Revised category-grounded pilot build and full reference replay (job 14464):
  [run log](jobs/14464-banking-synthetic-pilot-reference-v1-r3-0906-174947662/run_0_20260906_174947662.log).
- Revised representative naturalization; rejected 5/30 title rewrites (job
  14467):
  [run log](jobs/14467-banking-synthetic-smoke-naturalize-qwen38-r3-0906-175200299/run_0_20260906_175200299.log).
- Pilot reference replay with the knowledge base explicitly restricted to the
  480-document allowlist (job 14468):
  [run log](jobs/14468-banking-synthetic-pilot-reference-v1-r4-allowlist-0906-175512811/run_0_20260906_175512811.log).
- Revised naturalization with literal-preserving correction retries (job
  14475):
  [run log](jobs/14475-banking-synthetic-smoke-naturalize-qwen38-r4-retry-0906-175902440/run_0_20260906_175902440.log).
- Qwen3.8 Agent/User golden-retrieval smoke stopped before inference after a
  document-category audit found remaining selector defects (job 14477):
  [run log](jobs/14477-banking-synthetic-smoke-qwen38-golden-r3-0906-180617964/run_0_20260906_180617964.log).
- Qwen3.8 Agent/User BM25 smoke stopped for the same pre-inference gate (job
  14478):
  [run log](jobs/14478-banking-synthetic-smoke-qwen38-bm25-r3-0906-180618308/run_0_20260906_180618308.log).
- Pilot rebuild stopped before execution to correct incompatible case labels
  and smoke balance after the document-selector change (job 14491):
  [run log](jobs/14491-banking-synthetic-pilot-reference-v1-r5-semantic-docs-0906-181346269/run_0_20260906_181346269.log).
- Pilot rebuild with semantic document pools, compatible case labels, and
  balanced smoke selection (job 14492):
  [run log](jobs/14492-banking-synthetic-pilot-reference-v1-r6-template-quality-0906-181656068/run_0_20260906_181656068.log).
- Final-template 30-task Qwen3.8 naturalization (job 14505):
  [run log](jobs/14505-banking-synthetic-smoke-naturalize-qwen38-r5-template-quality-0906-181947089/run_0_20260906_181947089.log).
- Superseded Qwen3.8 Agent/User BM25 smoke, seed 300; failed its required clean
  run with three infrastructure errors after completing 17 diagnostic tasks
  (job 14509):
  [run log](jobs/14509-banking-synthetic-smoke-qwen38-bm25-r4-final-0906-182810966/run_0_20260906_182810966.log).
- Superseded Qwen3.8 Agent/User golden-retrieval smoke, seed 300; 13/13 valid
  trajectories and pass@1 53.85%, retained only as diagnostic evidence (job
  14510):
  [run log](jobs/14510-banking-synthetic-smoke-qwen38-golden-r4-final-0906-182811344/run_0_20260906_182811344.log).
- Policy/product-grounded r5 pilot rebuild and full reference replay (job
  14554); 1,500/1,500 schema and reward-one replay, 1,070/1,070 BM25 top-10,
  zero isolation findings:
  [run log](jobs/14554-banking-synthetic-pilot-reference-v1-r7-policy-grounded-0906-190159587/run_0_20260906_190159587.log).
- Content-grounded r6 pilot rebuild and full reference replay; 1,500/1,500
  schema and reward-one replay, 1,070/1,070 BM25 top-10, zero isolation
  findings (job 14555):
  [run log](jobs/14555-banking-synthetic-pilot-reference-v1-r8-content-grounded-0906-190743707/run_0_20260906_190743707.log).
- Content-grounded r6 30-task Qwen3.8 naturalization; stopped before generation
  after the final policy-anchor audit superseded its inputs (job 14558):
  [run log](jobs/14558-banking-synthetic-smoke-naturalize-qwen38-r6-content-grounded-0906-191020769/run_0_20260906_191020769.log).
- Policy-audited r7 pilot rebuild and full reference replay (job 14560):
  [run log](jobs/14560-banking-synthetic-pilot-reference-v1-r9-policy-audited-0906-191433638/run_0_20260906_191433638.log).
- Policy-audited r7 pilot plus 75-cell executable template matrix (job 14571):
  [run log](jobs/14571-banking-synthetic-pilot-reference-v1-r10-template-matrix-0906-191850136/run_0_20260906_191850136.log).
- Policy-audited r7 30-task Qwen3.8 naturalization (job 14578):
  [run log](jobs/14578-banking-synthetic-smoke-naturalize-qwen38-r7-policy-audited-0906-192139334/run_0_20260906_192139334.log).
- Policy-audited r7 Qwen3.8 golden-retrieval diagnostic smoke, seed 300; 13
  valid trials, pass@1 7.69%, and no protocol failures (job 14591):
  [run log](jobs/14591-banking-synthetic-smoke-qwen38-golden-r7-policy-audited-0906-192957292/run_0_20260906_192957292.log).
- Policy-audited r7 Qwen3.8 BM25 diagnostic smoke, seed 300; 14 valid trials,
  pass@1 28.57%, and three protocol failures (job 14592):
  [run log](jobs/14592-banking-synthetic-smoke-qwen38-bm25-r7-policy-audited-0906-192957667/run_0_20260906_192957667.log).
- Strict per-query retrieval, date/amount, and foreign-key replay for the r7
  pilot and 75-cell matrix; 1,575/1,575 reward-one replays and 1,190/1,190
  target-document BM25 top-10 checks passed (job 14604):
  [run log](jobs/14604-banking-synthetic-pilot-reference-v1-r11-exact-retrieval-0906-193917877/run_0_20260906_193917877.log).
- Candidate r8 replay after shortening communication golds and making transfer
  consent explicit; all pilot and matrix checks passed (job 14613):
  [run log](jobs/14613-banking-synthetic-pilot-reference-v1-r12-communicate-contract-0906-194839308/run_0_20260906_194839308.log).
- Candidate r8 Qwen3.8 naturalization with positive write-action coverage in
  the 30-task smoke selection (job 14616):
  [run log](jobs/14616-banking-synthetic-smoke-naturalize-qwen38-r8-communicate-contract-0906-195242535/run_0_20260906_195242535.log).
- Candidate r8 Qwen3.8 golden-retrieval diagnostic smoke, seed 300; 13/13 valid
  trajectories and pass@1 38.46% (job 14638):
  [run log](jobs/14638-banking-synthetic-smoke-qwen38-golden-r8-communicate-contract-0906-200025675/run_0_20260906_200025675.log).
- Candidate r8 Qwen3.8 BM25 diagnostic smoke, seed 300; 15 valid trajectories,
  pass@1 26.67%, and two infrastructure failures (job 14639):
  [run log](jobs/14639-banking-synthetic-smoke-qwen38-bm25-r8-communicate-contract-0906-200026022/run_0_20260906_200026022.log).
- Candidate r9 full replay with concrete L1/L3 fact extraction; all replay,
  retrieval, and isolation checks passed (job 14656):
  [run log](jobs/14656-banking-synthetic-pilot-reference-v1-r13-concrete-facts-0906-200847808/run_0_20260906_200847808.log).
- Candidate r11 full replay with stable numeric communication anchors and a
  private User-owned action protocol; all replay checks passed, but it was
  superseded by the visible-action audit (job 14895):
  [run log](jobs/14895-banking-synthetic-pilot-reference-v1-r14-stable-communicate-user-protocol-0906-212928358/run_0_20260906_212928358.log).
- Candidate r12 full replay with visible synthetic action labels, exact case
  parameters, and dependency-aware User-tool timing; all checks passed (job
  14912):
  [run log](jobs/14912-banking-synthetic-pilot-reference-v1-r15-visible-action-contract-0906-213849865/run_0_20260906_213849865.log).
- Candidate r12 representative 30-task Qwen3.8 naturalization (job 14921):
  [run log](jobs/14921-banking-synthetic-smoke-naturalize-qwen38-r12-visible-action-contract-0906-214108521/run_0_20260906_214108521.log).
- Candidate r12 Qwen3.8 golden-retrieval smoke, seed 300 (job 14945):
  [run log](jobs/14945-banking-synthetic-smoke-qwen38-golden-r12-visible-action-contract-0906-214911204/run_0_20260906_214911204.log).
- Candidate r12 Qwen3.8 BM25 diagnostic smoke, seed 300; 15 valid trajectories,
  pass@1 46.67%, and two infrastructure errors (job 14946):
  [run log](jobs/14946-banking-synthetic-smoke-qwen38-bm25-r12-visible-action-contract-0906-214911622/run_0_20260906_214911622.log).
- Candidate r13 full replay with explicit dynamic-wrapper calls, equivalent
  transfer matching, and preverified L3/L4 write cases (job 14973):
  [run log](jobs/14973-banking-synthetic-pilot-reference-v1-r16-equivalent-path-contract-0906-221623553/run_0_20260906_221623553.log).
- Candidate r13 representative 30-task Qwen3.8 naturalization (job 14974):
  [run log](jobs/14974-banking-synthetic-smoke-naturalize-qwen38-r13-equivalent-path-contract-0906-221903199/run_0_20260906_221903199.log).
- Candidate r13 Qwen3.8 golden-retrieval smoke, seed 300; 13 valid trials,
  pass@1 84.62%, and no protocol failures (job 14976):
  [run log](jobs/14976-banking-synthetic-smoke-qwen38-golden-r13-equivalent-path-contract-0906-222552430/run_0_20260906_222552430.log).
- Candidate r13 Qwen3.8 BM25 diagnostic smoke, seed 300; 14 valid trials,
  pass@1 42.86%, and three protocol failures (job 14977):
  [run log](jobs/14977-banking-synthetic-smoke-qwen38-bm25-r13-equivalent-path-contract-0906-222553292/run_0_20260906_222553292.log).
- Candidate r14 full replay with a guaranteed visible initial User message (job
  15013); all 1,575 reference replays, 1,190 BM25 checks, and isolation checks
  passed:
  [run log](jobs/15013-banking-synthetic-pilot-reference-v1-r17-visible-initial-message-0906-224322945/run_0_20260906_224322945.log).
- Candidate r15 full replay with state-scored L4--L6 refusal workflows (job
  15015); all replay, retrieval, and isolation checks passed:
  [run log](jobs/15015-banking-synthetic-pilot-reference-v1-r18-state-scored-refusal-0906-224953475/run_0_20260906_224953475.log).
- Candidate r15 naturalization rejected three L1 rewrites that changed protected
  policy literals (job 15016):
  [run log](jobs/15016-banking-synthetic-smoke-naturalize-qwen38-r15-state-scored-refusal-0906-225325213/run_0_20260906_225325213.log).
- Candidate r15 replacement naturalization masks protected literals from the
  prose rewrite and restores them afterward; three L3 decisions still dropped
  a protected placeholder and were rejected (job 15030):
  [run log](jobs/15030-banking-synthetic-smoke-naturalize-qwen38-r15-masked-literals-0906-230106091/run_0_20260906_230106091.log).
- Candidate r15 naturalization with a reported original-wording fallback after
  four invalid Qwen rewrites; 30/30 outputs passed static and isolation checks,
  with three fallbacks (job 15031):
  [run log](jobs/15031-banking-synthetic-smoke-naturalize-qwen38-r15-safe-fallback-0906-230739252/run_0_20260906_230739252.log).
- Candidate r15 Qwen3.8 BM25 replacement smoke, seed 300; 12 valid trajectories
  at pass@1 66.67%, plus five empty-response infrastructure failures (job
  15040):
  [run log](jobs/15040-banking-synthetic-smoke-qwen38-bm25-r15-final-0906-231310880/run_0_20260906_231310880.log).
- Candidate r15 Qwen3.8 golden-retrieval replacement smoke, seed 300 (job
  15041); 12 valid trajectories at pass@1 91.67%, plus one `max_steps`
  protocol failure:
  [run log](jobs/15041-banking-synthetic-smoke-qwen38-golden-r15-final-0906-231311292/run_0_20260906_231311292.log).
- Candidate r15 Qwen3.8 golden-retrieval 32,768-token recovery for the sole
  protocol-failed task, seed 300; one valid reward-one trajectory (job 15051):
  [run log](jobs/15051-banking-synthetic-smoke-qwen38-golden-r15-recovery32k-s300-0906-232628167/run_0_20260906_232628167.log).
- Candidate r15 Qwen3.8 BM25 32,768-token recovery for the five
  infrastructure-failed tasks, seed 300; two valid trajectories and three
  retained infrastructure failures (job 15055):
  [run log](jobs/15055-banking-synthetic-smoke-qwen38-bm25-r15-recovery32k-s300-0906-233640688/run_0_20260906_233640688.log).
- Candidate r16 full replay with an explicit decision answer format and defined
  profile-change target email; all 1,575 replays, 1,190 BM25 checks, and both
  isolation audits passed (job 15057):
  [run log](jobs/15057-banking-synthetic-pilot-reference-v1-r19-stable-decision-intent-0906-234052246/run_0_20260906_234052246.log).
- Candidate r16 representative 30-task Qwen3.8 naturalization; 30/30 valid,
  zero fallback, and zero isolation findings (job 15060):
  [run log](jobs/15060-banking-synthetic-smoke-naturalize-qwen38-r16-stable-decision-intent-0906-234405058/run_0_20260906_234405058.log).
- Candidate r16 Qwen3.8 BM25 replacement smoke, seed 300; 14 valid trajectories,
  pass@1 85.71%, and three infrastructure failures (job 15072):
  [run log](jobs/15072-banking-synthetic-smoke-qwen38-bm25-r16-final-0906-235010788/run_0_20260906_235010788.log).
- Candidate r16 Qwen3.8 golden-retrieval replacement smoke, seed 300 (job
  15073); 13/13 valid trajectories, pass@1 92.31%, and no protocol failure:
  [run log](jobs/15073-banking-synthetic-smoke-qwen38-golden-r16-final-0906-235011625/run_0_20260906_235011625.log).
- Candidate r16 Qwen3.8 BM25 32,768-token recovery for the three
  protocol-failed tasks, seed 300; two reward-one trajectories and one retained
  infrastructure failure (job 15089):
  [run log](jobs/15089-banking-synthetic-smoke-qwen38-bm25-r16-recovery32k-s300-0907-001746658/run_0_20260907_001746658.log).
- Candidate r16 full 1,500-task Qwen3.8 naturalization; 1,500/1,500 valid, ten
  safe fallbacks, and zero static or isolation findings (job 15092):
  [run log](jobs/15092-banking-synthetic-pilot-naturalize-qwen38-r16-0907-001941915/run_0_20260907_001941915.log).
- Pilot seed 300 evaluation shards: BM25
  [0/4](jobs/15099-banking-synthetic-pilot-qwen38-bm25-s300-shard0of4-0907-003056350/run_0_20260907_003056350.log),
  [1/4](jobs/15100-banking-synthetic-pilot-qwen38-bm25-s300-shard1of4-0907-003056763/run_0_20260907_003056763.log),
  [2/4](jobs/15103-banking-synthetic-pilot-qwen38-bm25-s300-shard2of4-0907-003057885/run_0_20260907_003057885.log), and
  [3/4](jobs/15102-banking-synthetic-pilot-qwen38-bm25-s300-shard3of4-0907-003057511/run_0_20260907_003057511.log); golden retrieval
  [0/3](jobs/15104-banking-synthetic-pilot-qwen38-golden-s300-shard0of3-0907-003058264/run_0_20260907_003058264.log),
  [1/3](jobs/15101-banking-synthetic-pilot-qwen38-golden-s300-shard1of3-0907-003057135/run_0_20260907_003057135.log), and
  [2/3](jobs/15105-banking-synthetic-pilot-qwen38-golden-s300-shard2of3-0907-003103719/run_0_20260907_003103719.log).
- Pilot seed 301 evaluation shards: BM25
  [0/4](jobs/15111-banking-synthetic-pilot-qwen38-bm25-s301-shard0of4-0907-003152153/run_0_20260907_003152153.log),
  [1/4](jobs/15108-banking-synthetic-pilot-qwen38-bm25-s301-shard1of4-0907-003130901/run_0_20260907_003130901.log),
  [2/4](jobs/15106-banking-synthetic-pilot-qwen38-bm25-s301-shard2of4-0907-003123955/run_0_20260907_003123955.log), and
  [3/4](jobs/15110-banking-synthetic-pilot-qwen38-bm25-s301-shard3of4-0907-003144955/run_0_20260907_003144955.log); golden retrieval
  [0/3](jobs/15112-banking-synthetic-pilot-qwen38-golden-s301-shard0of3-0907-003208155/run_0_20260907_003208155.log),
  [1/3](jobs/15109-banking-synthetic-pilot-qwen38-golden-s301-shard1of3-0907-003137848/run_0_20260907_003137848.log), and
  [2/3](jobs/15107-banking-synthetic-pilot-qwen38-golden-s301-shard2of3-0907-003124328/run_0_20260907_003124328.log).
- Pilot seed 302 evaluation shards: BM25
  [0/4](jobs/15115-banking-synthetic-pilot-qwen38-bm25-s302-shard0of4-0907-003234417/run_0_20260907_003234417.log),
  [1/4](jobs/15114-banking-synthetic-pilot-qwen38-bm25-s302-shard1of4-0907-003234036/run_0_20260907_003234036.log),
  [2/4](jobs/15117-banking-synthetic-pilot-qwen38-bm25-s302-shard2of4-0907-003255119/run_0_20260907_003255119.log), and
  [3/4](jobs/15116-banking-synthetic-pilot-qwen38-bm25-s302-shard3of4-0907-003249688/run_0_20260907_003249688.log); golden retrieval
  [0/3](jobs/15120-banking-synthetic-pilot-qwen38-golden-s302-shard0of3-0907-003312934/run_0_20260907_003312934.log),
  [1/3](jobs/15119-banking-synthetic-pilot-qwen38-golden-s302-shard1of3-0907-003307493/run_0_20260907_003307493.log), and
  [2/3](jobs/15118-banking-synthetic-pilot-qwen38-golden-s302-shard2of3-0907-003302023/run_0_20260907_003302023.log).
- Pilot seed 303 evaluation shards: BM25
  [0/4](jobs/15126-banking-synthetic-pilot-qwen38-bm25-s303-shard0of4-0907-003343105/run_0_20260907_003343105.log),
  [1/4](jobs/15127-banking-synthetic-pilot-qwen38-bm25-s303-shard1of4-0907-003350208/run_0_20260907_003350208.log),
  [2/4](jobs/15124-banking-synthetic-pilot-qwen38-bm25-s303-shard2of4-0907-003335830/run_0_20260907_003335830.log), and
  [3/4](jobs/15125-banking-synthetic-pilot-qwen38-bm25-s303-shard3of4-0907-003336205/run_0_20260907_003336205.log); golden retrieval
  [0/3](jobs/15131-banking-synthetic-pilot-qwen38-golden-s303-shard0of3-0907-003428451/run_0_20260907_003428451.log),
  [1/3](jobs/15133-banking-synthetic-pilot-qwen38-golden-s303-shard1of3-0907-003441420/run_0_20260907_003441420.log), and
  [2/3](jobs/15130-banking-synthetic-pilot-qwen38-golden-s303-shard2of3-0907-003428049/run_0_20260907_003428049.log).
- Stopped duplicate submission for seed 301 golden shard 0 before it started
  (job 15113):
  [scheduler log](jobs/15113-banking-synthetic-pilot-qwen38-golden-s301-shard0of3-0907-003216295/jobm.log).
- Candidate r16 BM25 second 32,768-token recovery for the sole remaining smoke
  protocol failure, seed 300 (job 15135):
  [run log](jobs/15135-banking-synthetic-smoke-qwen38-bm25-r16-recovery32k-r2-s300-0907-003530872/run_0_20260907_003530872.log).
- Candidate r16 node-guided Qwen3.8 teacher smoke on one autonomous dynamic-tool
  failure; SGLang returned an opaque HTTP 400 before any environment action
  (job 15136):
  [run log](jobs/15136-banking-synthetic-teacher-smoke-qwen38-r16-dynamic-0907-003831329/run_0_20260907_003831329.log).
- Replacement node-teacher smoke using the sole exposed tool with
  `tool_choice: required`; failed before execution with `No user query found in
  messages` (job 15151):
  [run log](jobs/15151-banking-synthetic-teacher-smoke-qwen38-r16-required-r2-0907-004619494/run_0_20260907_004619494.log).
- Replacement node-teacher smoke with a minimal User query following each
  hidden node/final-response prompt; succeeded with reward 1 and no saved hidden
  guidance (job 15152):
  [run log](jobs/15152-banking-synthetic-teacher-smoke-qwen38-r16-user-query-r3-0907-005253383/run_0_20260907_005253383.log).
- Candidate r17 full pilot and 75-cell matrix replay with category-matched,
  complete L1/L3 policy facts; all checks passed (job 15200):
  [run log](jobs/15200-banking-synthetic-pilot-reference-v1-r20-category-facts-0907-013801210/run_0_20260907_013801210.log).
- Candidate r17 representative 30-task Qwen3.8 naturalization (job 15202):
  [run log](jobs/15202-banking-synthetic-smoke-naturalize-qwen38-r17-category-facts-0907-014039112/run_0_20260907_014039112.log).
- Candidate r17 Qwen3.8 golden-retrieval smoke, seed 300; 13/13 valid
  trajectories, pass@1 100%, and no protocol failure (job 15216):
  [run log](jobs/15216-banking-synthetic-smoke-qwen38-golden-r17-category-facts-s300-0907-014731716/run_0_20260907_014731716.log).
- Candidate r17 Qwen3.8 BM25 smoke, seed 300; 14 valid trajectories at pass@1
  71.43% plus three infrastructure failures (job 15217):
  [run log](jobs/15217-banking-synthetic-smoke-qwen38-bm25-r17-category-facts-s300-0907-014732127/run_0_20260907_014732127.log).
- Candidate r17 full 1,500-task Qwen3.8 naturalization; 1,500/1,500 valid,
  three safe fallbacks, and zero static or isolation findings (job 15228):
  [run log](jobs/15228-banking-synthetic-pilot-naturalize-qwen38-r17-category-facts-0907-020717477/run_0_20260907_020717477.log).
- Candidate r17 BM25 32,768-token recovery for the three smoke infrastructure
  failures; two valid trajectories and one retained infrastructure failure,
  seed 300 (job 15236):
  [run log](jobs/15236-banking-synthetic-smoke-qwen38-bm25-r17-recovery32k-s300-0907-020856923/run_0_20260907_020856923.log).
- Candidate r17 second BM25 32,768-token recovery for the sole unresolved smoke
  trial, seed 300 (job 15275):
  [run log](jobs/15275-banking-synthetic-smoke-qwen38-bm25-r17-recovery32k-r2-s300-0907-022324658/run_0_20260907_022324658.log).
- Candidate r17 pilot seed 300 shards: BM25
  [0/4](jobs/15243-banking-synthetic-pilot-qwen38-bm25-r17-s300-shard0of4-0907-021739938/run_0_20260907_021739938.log),
  [1/4](jobs/15246-banking-synthetic-pilot-qwen38-bm25-r17-s300-shard1of4-0907-021741277/run_0_20260907_021741277.log),
  [2/4](jobs/15244-banking-synthetic-pilot-qwen38-bm25-r17-s300-shard2of4-0907-021740403/run_0_20260907_021740403.log), and
  [3/4](jobs/15251-banking-synthetic-pilot-qwen38-bm25-r17-s300-shard3of4-0907-021808900/run_0_20260907_021808900.log); golden retrieval
  [0/3](jobs/15263-banking-synthetic-pilot-qwen38-golden-r17-s300-shard0of3-0907-022019417/run_0_20260907_022019417.log),
  [1/3](jobs/15245-banking-synthetic-pilot-qwen38-golden-r17-s300-shard1of3-0907-021740841/run_0_20260907_021740841.log), and
  [2/3](jobs/15264-banking-synthetic-pilot-qwen38-golden-r17-s300-shard2of3-0907-022027776/run_0_20260907_022027776.log).
- Candidate r17 pilot seed 301 shards: BM25
  [0/4](jobs/15249-banking-synthetic-pilot-qwen38-bm25-r17-s301-shard0of4-0907-021747676/run_0_20260907_021747676.log),
  [1/4](jobs/15252-banking-synthetic-pilot-qwen38-bm25-r17-s301-shard1of4-0907-021814406/run_0_20260907_021814406.log),
  [2/4](jobs/15248-banking-synthetic-pilot-qwen38-bm25-r17-s301-shard2of4-0907-021742130/run_0_20260907_021742130.log), and
  [3/4](jobs/15250-banking-synthetic-pilot-qwen38-bm25-r17-s301-shard3of4-0907-021803357/run_0_20260907_021803357.log); golden retrieval
  [0/3](jobs/15247-banking-synthetic-pilot-qwen38-golden-r17-s301-shard0of3-0907-021741714/run_0_20260907_021741714.log),
  [1/3](jobs/15265-banking-synthetic-pilot-qwen38-golden-r17-s301-shard1of3-0907-022035093/run_0_20260907_022035093.log), and
  [2/3](jobs/15266-banking-synthetic-pilot-qwen38-golden-r17-s301-shard2of3-0907-022042578/run_0_20260907_022042578.log).
- Candidate r17 pilot seed 302 shards: BM25
  [0/4](jobs/15255-banking-synthetic-pilot-qwen38-bm25-r17-s302-shard0of4-0907-021847436/run_0_20260907_021847436.log),
  [1/4](jobs/15267-banking-synthetic-pilot-qwen38-bm25-r17-s302-shard1of4-0907-022058906/run_0_20260907_022058906.log),
  [2/4](jobs/15260-banking-synthetic-pilot-qwen38-bm25-r17-s302-shard2of4-0907-021916304/run_0_20260907_021916304.log), and
  [3/4](jobs/15253-banking-synthetic-pilot-qwen38-bm25-r17-s302-shard3of4-0907-021820992/run_0_20260907_021820992.log); golden retrieval
  [0/3](jobs/15268-banking-synthetic-pilot-qwen38-golden-r17-s302-shard0of3-0907-022106142/run_0_20260907_022106142.log),
  [1/3](jobs/15269-banking-synthetic-pilot-qwen38-golden-r17-s302-shard1of3-0907-022113209/run_0_20260907_022113209.log), and
  [2/3](jobs/15257-banking-synthetic-pilot-qwen38-golden-r17-s302-shard2of3-0907-021854599/run_0_20260907_021854599.log).
- Candidate r17 pilot seed 303 shards: BM25
  [0/4](jobs/15270-banking-synthetic-pilot-qwen38-bm25-r17-s303-shard0of4-0907-022120302/run_0_20260907_022120302.log),
  [1/4](jobs/15261-banking-synthetic-pilot-qwen38-bm25-r17-s303-shard1of4-0907-021932359/run_0_20260907_021932359.log),
  [2/4](jobs/15259-banking-synthetic-pilot-qwen38-bm25-r17-s303-shard2of4-0907-021909136/run_0_20260907_021909136.log), and
  [3/4](jobs/15254-banking-synthetic-pilot-qwen38-bm25-r17-s303-shard3of4-0907-021828804/run_0_20260907_021828804.log); golden retrieval
  [0/3](jobs/15271-banking-synthetic-pilot-qwen38-golden-r17-s303-shard0of3-0907-022141854/run_0_20260907_022141854.log),
  [1/3](jobs/15258-banking-synthetic-pilot-qwen38-golden-r17-s303-shard1of3-0907-021901669/run_0_20260907_021901669.log), and
  [2/3](jobs/15272-banking-synthetic-pilot-qwen38-golden-r17-s303-shard2of3-0907-022149055/run_0_20260907_022149055.log).
- Candidate r18 full pilot and 75-cell matrix replay with an unambiguous,
  protected L1 evidence locator (job 15292):
  [run log](jobs/15292-banking-synthetic-pilot-reference-v1-r21-evidence-locator-0907-024836136/run_0_20260907_024836136.log).
- Candidate r18 representative 30-task Qwen3.8 naturalization (job 15294):
  [run log](jobs/15294-banking-synthetic-smoke-naturalize-qwen38-r18-evidence-locator-0907-025110250/run_0_20260907_025110250.log).
- Candidate r18 Qwen3.8 golden-retrieval smoke, seed 300 (job 15297):
  [run log](jobs/15297-banking-synthetic-smoke-qwen38-golden-r18-evidence-locator-s300-0907-025713340/run_0_20260907_025713340.log).
- Candidate r18 Qwen3.8 BM25 smoke, seed 300 (job 15298):
  [run log](jobs/15298-banking-synthetic-smoke-qwen38-bm25-r18-evidence-locator-s300-0907-025726324/run_0_20260907_025726324.log).
- Candidate r18 Qwen3.8 golden-retrieval 32,768-token recovery for its sole
  protocol failure, seed 300; valid reward-one result (job 15307):
  [run log](jobs/15307-banking-synthetic-smoke-qwen38-golden-r18-recovery32k-s300-0907-030811033/run_0_20260907_030811033.log).
- Candidate r18 Qwen3.8 BM25 32,768-token recovery for its sole protocol
  failure, seed 300; retained an empty-response protocol failure (job 15310):
  [run log](jobs/15310-banking-synthetic-smoke-qwen38-bm25-r18-recovery32k-s300-0907-032327999/run_0_20260907_032327999.log).
- Candidate r18 full 1,500-task Qwen3.8 naturalization (job 15311):
  [run log](jobs/15311-banking-synthetic-pilot-naturalize-qwen38-r18-evidence-locator-0907-032351292/run_0_20260907_032351292.log).
- Candidate r18 pilot seed 300 shards: BM25
  [0/4](jobs/15314-banking-synthetic-pilot-qwen38-bm25-r18-s300-shard0of4-0907-033534605/run_0_20260907_033534605.log),
  [1/4](jobs/15315-banking-synthetic-pilot-qwen38-bm25-r18-s300-shard1of4-0907-033535456/run_0_20260907_033535456.log),
  [2/4](jobs/15316-banking-synthetic-pilot-qwen38-bm25-r18-s300-shard2of4-0907-033536272/run_0_20260907_033536272.log), and
  [3/4](jobs/15317-banking-synthetic-pilot-qwen38-bm25-r18-s300-shard3of4-0907-033537121/run_0_20260907_033537121.log); golden retrieval
  [0/3](jobs/15318-banking-synthetic-pilot-qwen38-golden-r18-s300-shard0of3-0907-033537948/run_0_20260907_033537948.log),
  [1/3](jobs/15319-banking-synthetic-pilot-qwen38-golden-r18-s300-shard1of3-0907-033538832/run_0_20260907_033538832.log), and
  [2/3](jobs/15320-banking-synthetic-pilot-qwen38-golden-r18-s300-shard2of3-0907-033700850/run_0_20260907_033700850.log).
- Candidate r18 pilot seed 301 shards: BM25
  [0/4](jobs/15321-banking-synthetic-pilot-qwen38-bm25-r18-s301-shard0of4-0907-033717619/run_0_20260907_033717619.log),
  [1/4](jobs/15322-banking-synthetic-pilot-qwen38-bm25-r18-s301-shard1of4-0907-033718473/run_0_20260907_033718473.log),
  [2/4](jobs/15323-banking-synthetic-pilot-qwen38-bm25-r18-s301-shard2of4-0907-033719366/run_0_20260907_033719366.log), and
  [3/4](jobs/15324-banking-synthetic-pilot-qwen38-bm25-r18-s301-shard3of4-0907-033747703/run_0_20260907_033747703.log); golden retrieval
  [0/3](jobs/15325-banking-synthetic-pilot-qwen38-golden-r18-s301-shard0of3-0907-033830544/run_0_20260907_033830544.log),
  [1/3](jobs/15326-banking-synthetic-pilot-qwen38-golden-r18-s301-shard1of3-0907-033831431/run_0_20260907_033831431.log), and
  [2/3](jobs/15327-banking-synthetic-pilot-qwen38-golden-r18-s301-shard2of3-0907-033832284/run_0_20260907_033832284.log).
- Candidate r18 pilot seed 302 submitted shards: BM25
  [0/4](jobs/15328-banking-synthetic-pilot-qwen38-bm25-r18-s302-shard0of4-0907-033916194/run_0_20260907_033916194.log),
  [1/4](jobs/15329-banking-synthetic-pilot-qwen38-bm25-r18-s302-shard1of4-0907-033917069/run_0_20260907_033917069.log),
  [2/4](jobs/15330-banking-synthetic-pilot-qwen38-bm25-r18-s302-shard2of4-0907-034002071/run_0_20260907_034002071.log), and
  [3/4](jobs/15331-banking-synthetic-pilot-qwen38-bm25-r18-s302-shard3of4-0907-034002995/run_0_20260907_034002995.log); golden retrieval
  [0/3](jobs/15343-banking-synthetic-pilot-qwen38-golden-r18-s302-shard0of3-0907-040449595/run_0_20260907_040449595.log) and
  [1/3](jobs/15332-banking-synthetic-pilot-qwen38-golden-r18-s302-shard1of3-0907-034308054/run_0_20260907_034308054.log), and
  [2/3](jobs/15399-banking-synthetic-pilot-qwen38-golden-r18-s302-shard2of3-0907-104421076/run_0_20260907_104421076.log).
- Candidate r18 pilot seed 303 submitted shards: BM25
  [0/4](jobs/15400-banking-synthetic-pilot-qwen38-bm25-r18-s303-shard0of4-0907-104441142/run_0_20260907_104441142.log),
  [1/4](jobs/15402-banking-synthetic-pilot-qwen38-bm25-r18-s303-shard1of4-0907-104442078/run_0_20260907_104442078.log),
  [2/4](jobs/15404-banking-synthetic-pilot-qwen38-bm25-r18-s303-shard2of4-0907-104442996/run_0_20260907_104442996.log), and
  [3/4](jobs/15405-banking-synthetic-pilot-qwen38-bm25-r18-s303-shard3of4-0907-104443888/run_0_20260907_104443888.log); golden retrieval
  [0/3](jobs/15406-banking-synthetic-pilot-qwen38-golden-r18-s303-shard0of3-0907-104444771/run_0_20260907_104444771.log) and
  [1/3](jobs/15407-banking-synthetic-pilot-qwen38-golden-r18-s303-shard1of3-0907-104527890/run_0_20260907_104527890.log), and
  [2/3](jobs/15514-banking-synthetic-pilot-qwen38-golden-r18-s303-shard2of3-0907-133341315/run_0_20260907_133341315.log).
- Candidate r18 pilot protocol recovery round 1: BM25
  [0/2](jobs/15545-banking-synthetic-pilot-recovery32k-r18-bm25-round1-shard0of2-0907-145148699/run_0_20260907_145148699.log) and
  [1/2](jobs/15546-banking-synthetic-pilot-recovery32k-r18-bm25-round1-shard1of2-0907-145149607/run_0_20260907_145149607.log), plus
  [golden retrieval](jobs/15547-banking-synthetic-pilot-recovery32k-r18-golden-round1-0907-145150529/run_0_20260907_145150529.log).
- Candidate r18 pilot BM25 protocol recovery round 2:
  [0/2](jobs/15695-banking-synthetic-pilot-recovery32k-r18-bm25-round2-shard0of2-0907-191610391/run_0_20260907_191610391.log) and
  [1/2](jobs/15696-banking-synthetic-pilot-recovery32k-r18-bm25-round2-shard1of2-0907-191621022/run_0_20260907_191621022.log).
- Candidate r18 pilot BM25 protocol recovery round 3, 79 exact task IDs:
  [run](jobs/15754-banking-synthetic-pilot-recovery32k-r18-bm25-round3-0907-204742865/run_0_20260907_204742865.log).
- Frozen-r18 train candidate build, reference replay, and isolation audit:
  [0--1,999](jobs/15548-banking-synthetic-scale-train-r18-00000-01999-0907-145318627/run_0_20260907_145318627.log),
  [2,000--3,999](jobs/15549-banking-synthetic-scale-train-r18-02000-03999-0907-145319584/run_0_20260907_145319584.log),
  [4,000--5,999](jobs/15550-banking-synthetic-scale-train-r18-04000-05999-0907-145320485/run_0_20260907_145320485.log),
  [6,000--7,999](jobs/15551-banking-synthetic-scale-train-r18-06000-07999-0907-145321373/run_0_20260907_145321373.log), and
  [8,000--9,999](jobs/15552-banking-synthetic-scale-train-r18-08000-09999-0907-145322267/run_0_20260907_145322267.log).
- Frozen-r18 held-out candidate build, reference replay, and isolation audit:
  [500 dev](jobs/15553-banking-synthetic-scale-dev-r18-00000-00499-0907-145323156/run_0_20260907_145323156.log) and
  [500 challenge](jobs/15554-banking-synthetic-scale-challenge-r18-00000-00499-0907-145325477/run_0_20260907_145325477.log).
- Frozen-r18 train candidate Qwen3.8 naturalization:
  [0--1,999](jobs/15565-banking-synthetic-scale-naturalize-qwen38-train-r18-00000-01999-0907-150211354/run_0_20260907_150211354.log),
  [2,000--3,999](jobs/15566-banking-synthetic-scale-naturalize-qwen38-train-r18-02000-03999-0907-150212315/run_0_20260907_150212315.log),
  [4,000--5,999](jobs/15567-banking-synthetic-scale-naturalize-qwen38-train-r18-04000-05999-0907-150213278/run_0_20260907_150213278.log),
  [6,000--7,999](jobs/15568-banking-synthetic-scale-naturalize-qwen38-train-r18-06000-07999-0907-150214176/run_0_20260907_150214176.log), and
  [8,000--9,999](jobs/15569-banking-synthetic-scale-naturalize-qwen38-train-r18-08000-09999-0907-150215098/run_0_20260907_150215098.log).
- Frozen-r18 held-out candidate Qwen3.8 naturalization:
  [500 dev](jobs/15570-banking-synthetic-scale-naturalize-qwen38-dev-r18-00000-00499-0907-150216019/run_0_20260907_150216019.log) and
  [500 challenge](jobs/15571-banking-synthetic-scale-naturalize-qwen38-challenge-r18-00000-00499-0907-150218909/run_0_20260907_150218909.log).
- Frozen-r18 scale first-trial golden-retrieval seed-300 shards:
  [0/19](jobs/15575-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard0of19-0907-151719681/run_0_20260907_151719681.log),
  [1/19](jobs/15576-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard1of19-0907-151720608/run_0_20260907_151720608.log),
  [2/19](jobs/15577-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard2of19-0907-151721525/run_0_20260907_151721525.log),
  [3/19](jobs/15578-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard3of19-0907-151724421/run_0_20260907_151724421.log),
  [4/19](jobs/15579-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard4of19-0907-151725445/run_0_20260907_151725445.log),
  [5/19](jobs/15580-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard5of19-0907-151726386/run_0_20260907_151726386.log),
  [6/19](jobs/15581-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard6of19-0907-151846615/run_0_20260907_151846615.log),
  [7/19](jobs/15582-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard7of19-0907-151931388/run_0_20260907_151931388.log),
  [8/19](jobs/15583-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard8of19-0907-151932363/run_0_20260907_151932363.log),
  [9/19](jobs/15584-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard9of19-0907-152007563/run_0_20260907_152007563.log),
  [10/19](jobs/15588-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard10of19-0907-152455142/run_0_20260907_152455142.log),
  [11/19](jobs/15589-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard11of19-0907-152541359/run_0_20260907_152541359.log),
  [12/19](jobs/15590-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard12of19-0907-152542418/run_0_20260907_152542418.log),
  [13/19](jobs/15591-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard13of19-0907-152556704/run_0_20260907_152556704.log),
  [14/19](jobs/15592-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard14of19-0907-152557724/run_0_20260907_152557724.log),
  [15/19](jobs/15593-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard15of19-0907-152558746/run_0_20260907_152558746.log),
  [16/19](jobs/15596-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard16of19-0907-153332628/run_0_20260907_153332628.log),
  [17/19](jobs/15597-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard17of19-0907-153334580/run_0_20260907_153334580.log), and
  [18/19](jobs/15598-banking-synthetic-scale-first-qwen38-golden-r18-s300-shard18of19-0907-153335580/run_0_20260907_153335580.log).
- Frozen-r18 scale first-trial BM25 seed-300 shards:
  [0/26](jobs/15599-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard0of26-0907-153336527/run_0_20260907_153336527.log),
  [1/26](jobs/15600-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard1of26-0907-153337473/run_0_20260907_153337473.log),
  [2/26](jobs/15601-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard2of26-0907-153338495/run_0_20260907_153338495.log),
  [3/26](jobs/15602-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard3of26-0907-153532762/run_0_20260907_153532762.log),
  [4/26](jobs/15603-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard4of26-0907-153533785/run_0_20260907_153533785.log),
  [5/26](jobs/15604-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard5of26-0907-153534747/run_0_20260907_153534747.log),
  [6/26](jobs/15605-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard6of26-0907-153537730/run_0_20260907_153537730.log),
  [7/26](jobs/15606-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard7of26-0907-153538761/run_0_20260907_153538761.log),
  [8/26](jobs/15616-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard8of26-0907-154007892/run_0_20260907_154007892.log),
  [9/26](jobs/15619-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard9of26-0907-154009333/run_0_20260907_154009333.log),
  [10/26](jobs/15621-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard10of26-0907-154010280/run_0_20260907_154010280.log),
  [11/26](jobs/15623-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard11of26-0907-154011204/run_0_20260907_154011204.log),
  [12/26](jobs/15624-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard12of26-0907-154014213/run_0_20260907_154014213.log),
  [13/26](jobs/15625-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard13of26-0907-154015207/run_0_20260907_154015207.log),
  [14/26](jobs/15699-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard14of26-0907-192003646/run_0_20260907_192003646.log),
  [15/26](jobs/15716-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard15of26-0907-192618114/run_0_20260907_192618114.log),
  [16/26](jobs/15718-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard16of26-0907-193049031/run_0_20260907_193049031.log),
  [17/26](jobs/15719-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard17of26-0907-193151667/run_0_20260907_193151667.log), and
  [18/26](jobs/15753-banking-synthetic-scale-first-qwen38-bm25-r18-s300-shard18of26-0907-203842218/run_0_20260907_203842218.log).
- Aliased pilot protocol-recovery round 4 (job 15811):
  [run log](../audio-curriculum-v1/jobs/15811-audio-pilot-r18-recovery-r4-0907-222251217/run_0_20260907_222251217.log).
- Aliased scale golden-retrieval protocol-recovery round 1 (job 15816):
  [run log](../audio-curriculum-v1/jobs/15816-audio-scale-golden-recovery-r1-0907-223209428/run_0_20260907_223209428.log).
- Aliased scale BM25 first-trial shard 19/26 (job 15834):
  [run log](../audio-curriculum-v1/jobs/15834-audio-scale-r18-s300-19of26-0907-224307841/run_0_20260907_224307841.log).
- Aliased scale BM25 first-trial shard 20/26 (job 15843):
  [run log](../audio-curriculum-v1/jobs/15843-audio-scale-r18-s300-20of26-0907-225443940/run_0_20260907_225443940.log).
- Aliased pilot protocol-recovery round 5 (job 15866):
  [run log](../audio-curriculum-v1/jobs/15866-audio-pilot-r18-recovery-r5-0907-233444682/run_0_20260907_233444682.log).
- Aliased pilot protocol-recovery round 6 (job 15907):
  [run log](../audio-curriculum-v1/jobs/15907-audio-pilot-r18-recovery-r6-0908-011307493/run_0_20260908_011307493.log).
- Aliased scale BM25 first-trial shard 21/26 (job 15912):
  [run log](../audio-curriculum-v1/jobs/15912-audio-scale-r18-s300-21of26-0908-013537153/run_0_20260908_013537153.log).
- Aliased pilot protocol-recovery round 7 (job 15916):
  [run log](../audio-curriculum-v1/jobs/15916-audio-pilot-r18-recovery-r7-0908-015557036/run_0_20260908_015557036.log).
- Aliased scale BM25 first-trial shard 22/26 (job 15923):
  [run log](../audio-curriculum-v1/jobs/15923-audio-scale-r18-s300-22of26-0908-021710614/run_0_20260908_021710614.log).
- Aliased pilot protocol-recovery round 8 (job 15927):
  [run log](../audio-curriculum-v1/jobs/15927-audio-pilot-r18-recovery-r8-0908-025134817/run_0_20260908_025134817.log).
- Aliased pilot protocol-recovery round 9 (job 15935):
  [run log](../audio-curriculum-v1/jobs/15935-audio-pilot-r18-recovery-r9-0908-033536515/run_0_20260908_033536515.log).
- Aliased scale BM25 first-trial shard 21/26 retry 1 (job 15953):
  [run log](../audio-curriculum-v1/jobs/15953-audio-scale-r18-s300-21of26-retry1-0908-041612934/run_0_20260908_041612934.log).
- Aliased scale BM25 first-trial shard 22/26 retry 1 (job 15954):
  [run log](../audio-curriculum-v1/jobs/15954-audio-scale-r18-s300-22of26-retry1-0908-041624475/run_0_20260908_041624475.log).
- Aliased pilot protocol-recovery round 9 retry 1 (job 15955):
  [run log](../audio-curriculum-v1/jobs/15955-audio-pilot-r18-recovery-r9-retry1-0908-041630860/run_0_20260908_041630860.log).
- Aliased scale BM25 first-trial shard 23/26 (job 15972):
  [run log](../audio-curriculum-v1/jobs/15972-audio-scale-r18-s300-23of26-0908-045503112/run_0_20260908_045503112.log).
- Aliased scale BM25 first-trial shard 24/26, stopped after scope reduction
  (job 15997):
  [run log](../audio-curriculum-v1/jobs/15997-audio-scale-r18-s300-24of26-0908-072145808/run_0_20260908_072145808.log).
- Aliased scale BM25 first-trial shard 25/26, stopped after scope reduction
  (job 15998):
  [run log](../audio-curriculum-v1/jobs/15998-audio-scale-r18-s300-25of26-0908-072524708/run_0_20260908_072524708.log).
- Aliased pilot protocol-recovery round 10 (job 16004):
  [run log](../audio-curriculum-v1/jobs/16004-audio-pilot-r18-recovery-r10-0908-075633619/run_0_20260908_075633619.log).
- Aliased reduced-set BM25 recovery round 1, shard 1/2 (job 16013):
  [run log](../audio-curriculum-v1/jobs/16013-audio-reduced-r18-recovery-r1-1of2-0908-081840286/run_0_20260908_081840286.log).
- Aliased reduced-set BM25 recovery round 1, shard 0/2 (job 16014):
  [run log](../audio-curriculum-v1/jobs/16014-audio-reduced-r18-recovery-r1-0of2-0908-081840722/run_0_20260908_081840722.log).
- Aliased pilot protocol-recovery round 11 (job 16018):
  [run log](../audio-curriculum-v1/jobs/16018-audio-pilot-r18-recovery-r11-0908-083119880/run_0_20260908_083119880.log).
- Minimal validation with the unsupported 56-CPU combination (job 16052) was
  rejected before execution: [job-manager log](jobs/16052-validate-minimal-0908-113154020/jobm.log).
- Minimal reference replay and isolation audit (job 16053):
  [run log](jobs/16053-validate-0908-113255723/run_0_20260908_113255723.log).
- Minimal BM25 evaluations: seed
  [300](jobs/16054-eval-bm25-300-0908-113329813/run_0_20260908_113329813.log),
  [301](jobs/16055-eval-bm25-301-0908-113330733/run_0_20260908_113330733.log),
  [302](jobs/16056-eval-bm25-302-0908-113333090/run_0_20260908_113333090.log), and
  [303](jobs/16057-eval-bm25-303-0908-113334079/run_0_20260908_113334079.log).
- Minimal golden-retrieval evaluations: seeds
  [300--301](jobs/16058-eval-golden-a-0908-113334970/run_0_20260908_113334970.log) and
  [302--303](jobs/16059-eval-golden-b-0908-113335887/run_0_20260908_113335887.log).
- Golden-retrieval protocol recovery for the sole seed-303 gap (job 16076):
  [run log](jobs/16076-recover-golden-0908-120721842/run_0_20260908_120721842.log).
- Early BM25 protocol recovery for 11 observed task gaps (job 16077):
  [run log](jobs/16077-recover-bm25-0908-120833382/run_0_20260908_120833382.log).
- Early BM25 protocol recovery for seven additional deficits (job 16088):
  [run log](jobs/16088-recover-bm25-2-0908-121942425/run_0_20260908_121942425.log).
- Early BM25 protocol recovery for eight confirmed remaining deficits (job
  16092): [run log](jobs/16092-recover-bm25-3-0908-123016551/run_0_20260908_123016551.log).
- Minimal-set BM25 recovery rounds:
  [1](jobs/16105-recover-bm25-r1-0908-131029578/run_0_20260908_131029578.log),
  [2](jobs/16106-recover-bm25-r2-0908-131030097/run_0_20260908_131030097.log),
  [3](jobs/16104-recover-bm25-r3-0908-131029023/run_0_20260908_131029023.log), and
  [4](jobs/16103-recover-bm25-r4-0908-131028542/run_0_20260908_131028542.log).
- Invalid-path submissions stopped before useful work: [16122](jobs/16122-recover-bm25-r6-0908-135122621/jobm.log), [16123](jobs/16123-recover-bm25-r5-0908-135123118/jobm.log), [16124](jobs/16124-recover-bm25-r7-0908-135123600/jobm.log), [16130](jobs/16130-recover-bm25-r7-0908-135343472/jobm.log).
- Minimal-set BM25 recovery rounds [5](jobs/16128-recover-bm25-r5-0908-135245806/run_0_20260908_135245806.log), [6](jobs/16129-recover-bm25-r6-0908-135323594/run_0_20260908_135323594.log), [7](jobs/16132-recover-bm25-r7-0908-135429680/run_0_20260908_135429680.log), [8](jobs/16161-recover-bm25-r8-0908-140953210/run_0_20260908_140953210.log), [9](jobs/16183-recover-bm25-r9-0908-142803885/run_0_20260908_142803885.log), [10](jobs/16189-recover-bm25-r10-0908-142859466/run_0_20260908_142859466.log), [11](jobs/16190-recover-bm25-r11-0908-142935605/run_0_20260908_142935605.log), [12](jobs/16199-recover-bm25-r12-0908-144019303/run_0_20260908_144019303.log), [13](jobs/16200-recover-bm25-r13-0908-144019826/run_0_20260908_144019826.log), [14](jobs/16201-recover-bm25-r14-0908-144239244/run_0_20260908_144239244.log), [15](jobs/16204-recover-bm25-r15-0908-145702477/run_0_20260908_145702477.log), [16](jobs/16205-recover-bm25-r16-0908-145703081/run_0_20260908_145703081.log), [17](jobs/16209-recover-bm25-r17-0908-152023299/run_0_20260908_152023299.log), [18](jobs/16215-recover-bm25-r18-0908-153754903/run_0_20260908_153754903.log), [19](jobs/16216-recover-bm25-r19-0908-154002921/run_0_20260908_154002921.log), [20](jobs/16217-recover-bm25-r20-0908-154004036/run_0_20260908_154004036.log), [21](jobs/16218-recover-bm25-r21-0908-154005033/run_0_20260908_154005033.log), [22](jobs/16219-recover-bm25-r22-0908-154006008/run_0_20260908_154006008.log), [23](jobs/16220-recover-bm25-r23-0908-154007069/run_0_20260908_154007069.log), and [24](jobs/16224-recover-bm25-r24-0908-155231089/run_0_20260908_155231089.log).
- Final minimal-set replay, isolation audit, and acceptance (job 16229):
  [run log](jobs/16229-finalize-0908-160156386/run_0_20260908_160156386.log).
- Reviewed final acceptance with result de-duplication, paired model-path checks,
  and direct successful-trajectory validation (job 16251):
  [run log](jobs/16251-finalize-reviewed-0908-165650671/run_0_20260908_165650671.log).

## Scope boundary

This experiment does not modify the Banking RL environment, connect data to
`prepare_rl_data.py`, submit SFT/GRPO/OPD, or evaluate the sealed 97-task
benchmark.
