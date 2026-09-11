# vitabench-evaluator-thinking-ab

Purpose: determine whether Qwen3.6-27B can evaluate the completed VitaBench
trajectories without thinking while preserving rubric decisions and scores and
materially reducing evaluator latency.

Name: `vitabench-evaluator-thinking-ab`.

## Design

- Source: the strict-complete schema-5 `artifacts-v2` result from
  `vitabench-qwen35-local-role-eval`; Agent and User are never rerun.
- Frozen replay: resend each persisted system/user prompt byte-for-byte with
  only `chat_template_kwargs.enable_thinking=false`. This is the paired
  window-level judge comparison.
- Chained replay: preserve each system prompt and conversation window, but
  propagate non-thinking rubric state and justification to the next window.
  This is the end-to-end counterfactual score comparison.
- Both modes use the identical Qwen3.6-27B model fingerprint, BF16, 32,768
  context, temperature 0, and 8,192 output-token cap. Eight single-GPU H100
  servers expose four request slots each.
- Every result is atomically checkpointed per window with prompts, hashes,
  response content, usage, latency, retry history, and reasoning fields.

## Source audit

The pre-submission strict audit found 40 shards, 400 tasks, 1,600 trajectories,
918 judged trajectories, 6,349 final windows, and 10,597 final rubric checks.
Contrary to the initial assumption that every window passed on its first try,
6,339 passed first try, nine passed on attempt two, and one passed on attempt
three. The 11 semantic retries make 6,360 formal evaluator requests.

The 6,360 response journals contain 27,068,780 prompt tokens and 26,208,179
completion tokens, including 19,466,258 reasoning tokens (74.28%). The 6,349
persisted final responses alone contain 26,128,234 completion and 19,398,373
reasoning tokens. The job repeats this audit and fingerprints all source shards
before serving the candidate.

## Execution and gates

The job first replays two complete tasks per suite in both modes. The full
6,349-window frozen replay and 918-trajectory chained replay start only if the
pilot has complete valid coverage, at least 99.5% first-attempt schema validity,
zero reasoning tokens/content, and only `finish_reason=stop`.

The full chained result approves the switch only when all gates pass:

| Gate | Requirement |
|---|---|
| Operational | 100% valid coverage; first-attempt validity ≥99.5%; zero reasoning; all stop normally |
| Score point estimate | absolute Avg@4/pass@1 delta ≤1 percentage point |
| Score uncertainty | task-cluster bootstrap 95% CI lies within ±2 percentage points |
| Agreement | trajectory success and final rubric agreement each ≥97% |
| Domain guardrail | absolute per-suite Avg@4/pass@1 delta ≤3 percentage points |
| Evaluator efficiency | ≥3× evaluator critical-path speedup and ≥60% completion-token reduction |
| End-to-end estimate | ≥2× total critical-path speedup, holding non-evaluator time fixed |

The latency comparison uses the prior 125.5-second mean evaluator call and
276-hour aggregate critical-path estimate as explicit assumptions; candidate
latency is measured at the HTTP caller. Frozen agreement is diagnostic, while
the chained result alone controls the score decision.

## Results

The pilot passed its operational gate. The full frozen replay completed all
6,349 windows. The primary chained job checkpointed 6,347 windows, then one
window exhausted three semantic attempts; the v2 recovery overlay requested
only that window and its successor, reusing the other 6,347 results. The first
recovery attempt again reached 8,192 tokens; a stateless compact correction
returned 589 tokens and passed on attempt two. All candidate responses had zero
reasoning tokens/content.

The chained replay is the operational comparison because each non-thinking
rubric state feeds the next window. Agent and User trajectories are held fixed;
this is an evaluator counterfactual, not a fresh 1,600-trajectory rerun.

| Evaluator setting | Successful trajectories | Avg@4 / pass@1 | pass@4 (any) | pass^4 (all) |
|---|---:|---:|---:|---:|
| Thinking on, source baseline | 369/1,600 | 23.0625% | 47.75% | 4.50% |
| Thinking off, chained replay | 348/1,600 | 21.7500% | 45.75% | 4.25% |
| Difference | −21 | −1.3125pp | −2.00pp | −0.25pp |

| Evaluator setting | Mean per window | Evaluator critical path | Estimated full-eval critical path | Completion tokens |
|---|---:|---:|---:|---:|
| Thinking on, measured/estimated baseline | 125.50s | 221.33h | 276.00h | 26,128,234 |
| Thinking off, chained replay | 28.80s | 50.79h | 105.45h | 8,037,716 |
| Improvement | 4.36× faster | 170.55h saved | 2.62× faster / 170.55h saved | 69.2% fewer |

The Avg@4/pass@1 difference has a task-cluster bootstrap 95% CI of
[−2.38, −0.25]pp. Final-rubric agreement is 96.06% and trajectory agreement is
95.56%. The −1.3125pp point estimate is a 5.7% relative reduction. Per-suite
Avg@4 differences are delivery 0pp, instore −2.75pp, OTA −1.25pp, and
cross-domain −1.25pp.

The hour totals are aggregate critical-path estimates used for the speedup
comparison, not observed eight-GPU job wall time. Actual fresh-run wall time
also depends on Agent/User generation and scheduler utilization.

The frozen-prompt diagnostic, which does not propagate changed rubric state,
measures 4.80× evaluator speedup, 2.74× estimated total speedup, 72.3% fewer
completion tokens, and a −0.19pp Avg@4/pass@1 difference with 95% CI
[−0.50, +0.13]pp. It is not used for the default-setting decision.

The generated report retains its predeclared conservative recommendation
`switch_evaluator_to_nonthinking=false`: the chained score/equivalence and
agreement gates failed, and two chained windows required a recovery overlay.
The measured quality tradeoff is now explicitly accepted for routine runs.
Operational decision: fresh local VitaBench evaluations default to
`chat_template_kwargs.enable_thinking=false` for Qwen3.6-27B Evaluator. The
historical Thinking-on artifacts and the generated gate result remain
unchanged.

## Artifacts

The primary run writes `artifacts/source_audit.json`, immutable
`artifacts/manifest.json`, `artifacts/pilot_report.json`,
`artifacts/frozen_report.json`, per-window results, replay sessions, and eight
SGLang logs. Separate `artifacts-recovery*` overlays record post-failure
recovery without mutating the primary results or erasing an operational-gate
failure. The combined final report is `artifacts-recovery-v2/report.json`;
`artifacts-recovery/failures/` preserves the reproduced truncation and
`artifacts-recovery-v2/` contains exactly two overlay results plus the resolved
failure journal. Quality-gate failure is an experimental result, not a job
failure; operational corruption or incomplete coverage fails the primary job.

## Jobs

- `pt-38ktzko8` — `fsh-evaluator-thinking-ab-full-0803-115820`, 8×N6lS-80GB;
  pilot and frozen completed, then chained stopped at 6,347/6,349 because one
  window exhausted three semantic attempts:
  [run log](jobs/fsh-evaluator-thinking-ab-full-0803-115820/run_20260803_115820.log).
- `pt-h25ofbrb` — `fsh-evaluator-thinking-ab-recovery-0803-153430`, 1×N6lS-80GB;
  stopped after reproducing three 8,192-token length truncations; no result was
  promoted from this recovery:
  [run log](jobs/fsh-evaluator-thinking-ab-recovery-0803-153430/run_20260803_153430.log).
- `pt-x2ovdg99` — `fsh-evaluator-thinking-ab-recovery-v2-0803-160225`,
  1×N6lS-80GB, succeeded; immutable-base overlay using stateless compact
  correction for the two missing chained windows:
  [run log](jobs/fsh-evaluator-thinking-ab-recovery-v2-0803-160225/run_20260803_160225.log).
