# Tau2 Banking Task Curriculum Scale V2

Experiment name: `tau2-banking-task-curriculum-scale-v2`. Purpose: expand the
validated 94-source Banking diagnostic set at the dependency-graph node level
to measure fact extraction and single-action capability. All source tasks are
part of the final benchmark, so these artifacts are excluded from training.

## Scope

- 933 evidence-given tasks: one for every source-grounded fact annotation.
- 933 BM25 retrieval tasks: one matching retrieval task for every fact.
- 874 single-action tasks: one for every reference action, with preceding
  reference actions applied as initial state.
- Total: 2,740 new candidate specifications across 94 replayable sources and
  all 15 Banking business categories.
- These are benchmark-derived diagnostic candidates, not SFT, GRPO, or OPD
  candidates.
- Decision-only tasks are not multiplied because the first Qwen3.8 pass is
  already 89.36%; two-skill tasks are not multiplied because it is 14.89%.

## Artifacts

- `scaled_tasks.json`: executable Tau2 task specifications.
- `scaled_contracts.jsonl`: source, level, capability, reference-step, and
  success contracts.
- `scaled_validation_report.json`: static and real-environment validation.

## Validation

Seventeen zero-GPU curriculum tests pass, including node-level scale-out,
initialization-prefix coverage, and result-pool assignment. Static validation
passes all 2,740 task/contract pairs. Real Tau2 environment schema validation
and reference replay run in the jobs below.

The subsequent dual-Qwen3.8 evaluation completed all 2,740 trajectories with
zero infrastructure errors and exact success `2163/2740` (`78.94%`):
[evaluation aggregate](../tau2-banking-scale-v2-qwen38-eval/all_results.md).
Together with the 564-task six-variant curriculum, the current Banking corpus
contains 3,304 executable specifications and 2,492 verified successful
Qwen3.8 trajectories.

## Jobs

- **FAILED** — first 2,740-task real-environment schema and reference replay.
  Schema and action replay passed, but 9/933 reference BM25 queries missed the
  target document because Tau2 indexes document bodies but not titles:
  [run log](jobs/13151-banking-scale-v2-env-validate-0903-195545878/run_0_20260903_195545878.log).
- **SUCCEEDED** — full validation after adding an indexed-body search anchor to
  every retrieval-only task. All 2,740 schemas and reference trajectories
  replayed successfully, all 933 BM25 queries retrieved their required
  document, and the final validation report has zero errors:
  [run log](jobs/13178-banking-scale-v2-env-validate-v2-0903-204344119/run_0_20260903_204344119.log).
