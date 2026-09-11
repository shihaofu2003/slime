# vitabench-qwen35-local-role-eval

Purpose: evaluate Qwen3.5-4B on Chinese VitaBench without paid user/evaluator
APIs, using local Qwen3.6-27B for both roles after strict deployment and output
validation.

Name: `vitabench-qwen35-local-role-eval`.

## Protocol

- The completed protocol-v5 `artifacts-v2` run keeps its original four-agent,
  two-user, two-evaluator topology and suite-pinned shard execution.
- Existing protocol-v6 `artifacts-v3*` runs used two Qwen3.5-4B agents, two
  non-thinking Qwen3.6-27B users, and four thinking Qwen3.6-27B evaluators, all
  BF16 and single-GPU. Fresh runs write `artifacts-v4*`, retain the same
  topology, and default the four evaluators to non-thinking mode. Four workers
  dynamically claim full shards from one AFS-safe locked queue; gate suites
  remain pinned one per worker.
- Native queue locking retries AFS `EACCES`/`EAGAIN` every 50 ms for up to 120
  seconds. Fifteen CPU contracts and a real four-process AFS claim preflight
  pass without duplicate or missing claims.
- Four suites, 400 Chinese tasks, four trials, 1,600 trajectories; the same
  task IDs, seed, limits, strict shard checks, evaluator semantic retries, and
  lossless journals as the stopped remote-API experiment.
- A one-GPU smoke must prove 80GB fit, non-empty user output, and parseable
  evaluator JSON before the eight-GPU evaluation is submitted.

## Legacy protocol-v5 full result

`pt-n7kttlor` completed the original four-agent, two-user, two-evaluator
evaluation after reusing 37 strict-valid shards and recomputing only `delivery`
shards 7–9. The recovery job used 4:06:32 of compute time; its shard stage took
12,671 seconds. The final artifact contains 40/40 shards, 400 unique tasks, and
1,600/1,600 unique task/trial pairs, with zero missing, duplicate, or invalid
simulation and validated trial seeds.

| Scope | Tasks | Trajectories | Successful trajectories | Avg@4 / pass@1 | pass@4 (any) | pass^4 (all) |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 400 | 1,600 | 369 | 23.0625% | 47.75% | 4.50% |
| Delivery | 100 | 400 | 115 | 28.75% | 63.00% | 6.00% |
| Instore | 100 | 400 | 151 | 37.75% | 63.00% | 10.00% |
| OTA | 100 | 400 | 64 | 16.00% | 36.00% | 2.00% |
| Cross-domain | 100 | 400 | 39 | 9.75% | 29.00% | 0.00% |

| Scope | 0/4 successes | 1/4 | 2/4 | 3/4 | 4/4 |
|---|---:|---:|---:|---:|---:|
| Overall | 209 | 84 | 54 | 35 | 18 |
| Delivery | 37 | 31 | 18 | 8 | 6 |
| Instore | 37 | 14 | 20 | 19 | 10 |
| OTA | 64 | 18 | 10 | 6 | 2 |
| Cross-domain | 71 | 21 | 6 | 2 | 0 |

| Scope | Agent stop | Invalid agent message | Max steps | Too many errors |
|---|---:|---:|---:|---:|
| Overall | 918 (57.38%) | 496 (31.00%) | 182 (11.38%) | 4 (0.25%) |
| Delivery | 238 (59.50%) | 101 (25.25%) | 61 (15.25%) | 0 |
| Instore | 288 (72.00%) | 94 (23.50%) | 14 (3.50%) | 4 (1.00%) |
| OTA | 207 (51.75%) | 143 (35.75%) | 50 (12.50%) | 0 |
| Cross-domain | 185 (46.25%) | 158 (39.50%) | 57 (14.25%) | 0 |

- Trajectory duration: mean 123.88 seconds, p50 106.36 seconds, p95 290.66
  seconds, and maximum 1,252.48 seconds.
- Evaluation: 6,349 sliding windows and 10,597 rubric checks; 9,171 checks met,
  for an 86.54% met rate. Rubric coverage was 918/1,600 simulations.
- Activity: 124,790 messages and 56,181 assistant tool calls. Tool responses
  contained 166 errors (0.30%) and 199 calls were unanswered (0.35%); 977
  simulations (61.06%) used multitool calls.
- Local inference recorded 1,750,552,566 prompt tokens and 42,328,231
  completion tokens. No API price was configured, so monetary cost is absent.
- The SFT export accepted 84,500/84,500 journals with zero rejection and wrote
  65,750 agent, 12,341 user, and 6,409 evaluator examples (8.19 GiB). Its
  separate training-data diagnostics flag 79 duplicate example instances, 450
  logical conflicts, and 2,687 exact-content duplicates; these do not change
  the 1,600-trajectory evaluation rectangle.
- Exact credential scanning passed. The 23.0625 score is 1.0625 points above
  the model-card reference of 22.0, but is not strictly comparable because the
  reference omits enough user/judge protocol detail and this run uses local
  Qwen3.6-27B user and evaluator models.

Artifacts: [summary JSON](artifacts-v2/summary.json),
[summary CSV](artifacts-v2/summary.csv), [SFT index](artifacts-v2/sft/index.json),
and [40 raw shards](artifacts-v2/shards/).

## Evaluator thinking decision

The completed `artifacts-v2` result above used Qwen3.6-27B Evaluator with
Thinking enabled. A full paired replay kept Agent/User trajectories fixed and
propagated non-thinking rubric state through all 6,349 evaluator windows:

| Evaluator | Avg@4 / pass@1 | pass@4 (any) | pass^4 (all) | Mean evaluator window | Estimated full-eval critical path |
|---|---:|---:|---:|---:|---:|
| Thinking on | 23.0625% | 47.75% | 4.50% | 125.50s | 276.00h |
| Thinking off | 21.7500% | 45.75% | 4.25% | 28.80s | 105.45h |
| Difference | −1.3125pp | −2.00pp | −0.25pp | 4.36× faster | 2.62× faster |

The accepted tradeoff makes non-thinking the default for fresh local runs;
historical artifacts are unchanged. Full uncertainty, agreement, token, and
recovery details are in the
[evaluator Thinking A/B report](../vitabench-evaluator-thinking-ab/README.md).

## Jobs

- `pt-71uevxim` — `fsh-qwen36-27b-bf16-download-smoke-0801-123439`, reserved
  1×N6lS-80GB model download and deployment gate:
  [run log](jobs/fsh-qwen36-27b-bf16-download-smoke-0801-123439/run_20260801_123439.log).
  Stopped after the Hugging Face endpoint remained idle without creating model
  files; no deployment was attempted.
- `pt-ry3pqn42` — `fsh-qwen36-27b-bf16-modelscope-smoke-0801-124150`, reserved
  1×N6lS-80GB retry through the official ModelScope repository:
  [run log](jobs/fsh-qwen36-27b-bf16-modelscope-smoke-0801-124150/run_20260801_124150.log).
  Passed on an NVIDIA H100 80GB HBM3: all 15 BF16 weight shards loaded with a
  32K context, the non-thinking user probe returned non-empty exact output,
  and the thinking evaluator returned the required JSON schema.
- `pt-fic28pqc` — `fsh-qwen35-agent-qwen36-27b-bf16-local-roles-0801-132500`,
  reserved 8×N6lS-80GB full evaluation:
  [run log](jobs/fsh-qwen35-agent-qwen36-27b-bf16-local-roles-0801-132500/run_20260801_132501.log).
  Failed before any trajectory after all eight servers started and all four
  agent tool-call probes passed: the evaluator probe capped a thinking model at
  64 tokens, so its HTTP-200 response ended in reasoning before final content.
  The probe now uses the same 1,024-token allowance as the successful smoke.
- `pt-o9031fzv` — `fsh-qwen35-agent-qwen36-27b-bf16-local-roles-retry-0801-134323`,
  reserved 8×N6lS-80GB retry with the corrected evaluator probe:
  [run log](jobs/fsh-qwen35-agent-qwen36-27b-bf16-local-roles-retry-0801-134323/run_20260801_134323.log).
  All eight endpoint probes and all 16 preflight trajectories passed, including
  parseable local judge decisions from both evaluator replicas. It stopped
  before formal shards because the old validator incorrectly required every
  suite to reach a 10-message judge window; all four delivery trials instead
  terminated early with `invalid_agent_message`.
- `pt-u6t6nns4` — `fsh-qwen35-agent-qwen36-27b-bf16-local-roles-v2-0801-141825`,
  reserved 8×N6lS-80GB protocol-v5 retry using fresh `artifacts-v2`:
  [run log](jobs/fsh-qwen35-agent-qwen36-27b-bf16-local-roles-v2-0801-141825/run_20260801_141826.log).
  The corrected gate passed 16/16 trajectories and validated all 49 judge
  windows, with both local evaluator replicas exercised. It stopped before
  formal shards because the SFT exporter had not yet accepted schema 5.
- `pt-ggiug8qm` — `fsh-schema5-gate-sft-preflight-0801-145434`, reserved
  1×N6lS schema-5 artifact preflight:
  [run log](jobs/fsh-schema5-gate-sft-preflight-0801-145434/run_20260801_145434.log).
  Passed: the fixed schema contract accepted 365/365 unique gate journals with
  zero rejected journals and produced 365 SFT examples.
- `pt-1o63padk` — `fsh-qwen35-agent-qwen36-27b-bf16-local-roles-resume-0801-145706`,
  reserved 8×N6lS-80GB resume after the schema-5 preflight:
  [run log](jobs/fsh-qwen35-agent-qwen36-27b-bf16-local-roles-resume-0801-145706/run_20260801_145706.log).
  Endpoint probes passed, then the static manifest lock stopped before formal
  shards because the first helper migration incorrectly included two Git metadata
  fields in `protocol_bundle_sha256`; the exact eight-field bundle is now fixed.
- `pt-1s4lhw0j` — `fsh-schema5-manifest-lock-preflight-0801-151435`, reserved
  1×N6lS manifest-lock and repeatable SFT preflight:
  [run log](jobs/fsh-schema5-manifest-lock-preflight-0801-151435/run_20260801_151435.log).
  Passed the exact eight-field protocol bundle (`1c53…1773`), current helper
  hashes, and the repeatable 365/365-journal SFT export with zero rejections.
- `pt-57v6i9p0` — `fsh-qwen35-agent-qwen36-27b-local-formal-resume-0801-151701`,
  reserved 8×N6lS-80GB formal resume after both one-GPU preflights:
  [run log](jobs/fsh-qwen35-agent-qwen36-27b-local-formal-resume-0801-151701/run_20260801_151701.log).
  Failed after 17.7 hours: `delivery` task `40711001`, trial 1 deterministically
  exhausted the agent's 262,144-token context and returned HTTP 400. Vita did not
  persist a zero-reward terminal result, so eight batch retries left the same
  single missing pair and triggered peer-worker shutdown; 1,119 formal
  simulations remain valid and reusable.
- `pt-50niwo9k` — `fsh-qwen35-agent-qwen36-27b-local-contextfix-resume-0802-155416`,
  reserved 8×N6lS-80GB resume after context-overflow terminalization and a
  protocol-compatible manifest migration:
  [run log](jobs/fsh-qwen35-agent-qwen36-27b-local-contextfix-resume-0802-155416/run_20260802_155416.log).
  Failed at 2026-08-02 21:59 CST after the shared runner source was updated
  during the live job; the protocol lock correctly stopped before mixing source
  versions. The run left 37/40 strict-valid formal shards (1,480 trajectories),
  with only `delivery` shards 7–9 absent and no corrupt shard. The schema-5
  manifest and run plan remain unchanged; the pre-migration manifest is
  `artifacts-v2/manifest.pre_context_overflow_migration.json`.
- `pt-ere11793` — `fsh-fsh-vitabench-qwen35-local-2a2u4e-gate-0802-225540`,
  reserved 8×N6lS-80GB protocol-v6 capability gate for the two-agent,
  two-user, four-evaluator topology:
  [run log](jobs/fsh-fsh-vitabench-qwen35-local-2a2u4e-gate-0802-225540/run_20260802_225540.log).
  Passed all 16 trajectories in 565 seconds with 68/68 valid sliding-window
  rubric decisions and real journals from all eight endpoint routes. The SFT
  export accepted 564/564 journals; evaluator/user queues stayed at zero,
  agent queues peaked at five for at most three consecutive samples, and no
  OOM, traceback, port conflict, or batch retry occurred.
- `pt-n7kttlor` — `fsh-vitabench-qwen35-v2-legacy-resume-0802-232149`,
  reserved 8×N6lS-80GB schema-5 recovery using the original four-agent,
  two-user, two-evaluator topology:
  [run log](jobs/fsh-vitabench-qwen35-v2-legacy-resume-0802-232149/run_20260802_232149.log).
  Succeeded after reusing 37 strict-valid shards and running only `delivery`
  shards 7–9. The complete metrics and artifact links are recorded in the
  legacy protocol-v5 result above.
- `pt-34oe9fzt` — `fsh-vitabench-qwen35-local-2a2u4e-ab-shard0-0802-232158`,
  reserved 8×N6lS-80GB protocol-v6 A/B candidate on fixed shard 0 of each suite:
  [run log](jobs/fsh-vitabench-qwen35-local-2a2u4e-ab-shard0-0802-232158/run_20260802_232158.log).
  Stopped after its 16-trajectory gate passed but three of four simultaneous
  shard claims received AFS `EAGAIN` from blocking `flock`; the only successful
  worker was terminated to avoid wasting an hour on a doomed partial run. The
  incomplete artifact remains available for diagnosis and is excluded from A/B.
- `pt-60y35nhe` —
  `fsh-vitabench-qwen35-local-2a2u4e-ab-shard0-afs-retry-0802-235330`, reserved
  8×N6lS-80GB A/B retry with an AFS blocking-lock compatibility adapter:
  [run log](jobs/fsh-vitabench-qwen35-local-2a2u4e-ab-shard0-afs-retry-0802-235330/run_20260802_235330.log).
  Succeeded with four strict-complete shard-0 files (160/160 trajectories),
  four unique dynamic claims, 9,910/9,910 accepted journals, zero rejected SFT
  records, no evaluator/user queues, and only three or fewer consecutive agent
  queue samples. The shard stage took 6,056 seconds while reusing one trajectory
  from the stopped attempt, giving a 1.313× upper-bound speedup over the
  7,952.85-second baseline; it did not meet the 1.5× target. The strict
  [A/B report](artifacts-v3-ab-shard0/ab_comparison.json) measures 1.078× because
  its artifact envelope correctly includes the inter-job idle gap. The
  [attempt analysis](artifacts-v3-ab-shard0/ab_attempt_analysis.json) preserves
  both measurements and requires a fresh 160-trajectory run.
- `pt-44kf0u1b` —
  `fsh-vitabench-qwen35-local-2a2u4e-ab-shard0-clean-native-0803-033845`,
  reserved 8×N6lS-80GB clean protocol-v6 A/B using native AFS lock retries and
  a fresh artifact directory:
  [run log](jobs/fsh-vitabench-qwen35-local-2a2u4e-ab-shard0-clean-native-0803-033845/run_20260803_033845.log).
  Succeeded with a zero-reuse 160/160-trajectory result, four unique native
  queue claims, and 9,887/9,887 accepted journals with zero rejection. The
  job used 2:18:23 of compute time and the shard stage took 6,698 seconds; the
  strict [A/B report](artifacts-v3-ab-shard0-clean-native/ab_comparison.json) measures a
  6,694.11-second candidate envelope versus the 7,952.85-second baseline:
  1.188× speedup and 15.83% wall-time reduction, below the 1.5× target.
  Evaluator and user queues stayed at zero, one agent queue sample reached one
  request, and no OOM or worker failure occurred. Independent audit confirmed
  40 unique tasks and 160 unique task/trial pairs with no missing or duplicate
  pair. The full 400-task × four-trial run was not submitted because the speed
  gate failed.
