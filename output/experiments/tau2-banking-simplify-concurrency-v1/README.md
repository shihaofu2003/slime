# Banking simplification concurrency benchmark

Purpose: measure whether per-TP2 concurrency 2 or 4 improves Banking
simplification throughput without changing the deletion/review/replay protocol.
Name: `tau2-banking-simplify-concurrency-v1`.

## Matched comparison

- One eight-GPU job: four independent TP2 servers, concurrency 1, 2, 4, 4.
  The last arm repeats concurrency 4 to expose run-to-run differences.
- Every arm processes the same 16 original pilot sources: four length-spaced
  examples per retrieval-variant/tool-error stratum, including each stratum's
  shortest and longest source. No production decisions are imported.
- Preserve BF16, 262K context, 32K output, thinking/xhigh, seeds, prompts,
  one revision, sequential replay, independent review, and 16K SFT cap.
- Independent client processes own disjoint source shards and separate output
  files; no shared environment state or concurrent writes to one JSONL file.
- Outputs live in `c1/`, `c2/`, `c4/`, `c4_repeat/`; `samples.json` lists
  matched inputs. `comparison.json` reports pipeline wall time (first trajectory
  start to last completion, excluding server startup/export), trajectories/hour,
  generated tokens/second, truncations, status disagreements, and export counts.
- Compare status differences using saved Qwen reviews, not assumed exact output
  equality. Inspect server memory/preemption/errors and job metrics as well as
  throughput. Reuse production results only after deciding on the tested setting;
  do not merge benchmark decisions into production to select favorable outcomes.

## Validation and jobs

- Existing registered CPU test extended: 22 tests, 18 passed, four integration
  tests skipped locally. Tests cover matched deterministic selection and complete,
  disjoint source assignment for concurrency 1/2/4. Both launchers pass `bash -n`.
- Job `17398`, eight GPUs, normal priority, submitted 2026-09-11 13:20 CST:
  [run log](jobs/17398-banking-simplify-concurrency-0911-132036369/run_0_20260911_132036369.log).
  Initially queued for cluster resources, then remained SUBMITTED without
  starting. Stopped at the user's request before resubmission; no benchmark
  results were produced.
- Job `17408`, same eight-GPU configuration, resubmitted at the user's request
  on 2026-09-11 13:37 CST:
  [run log](jobs/17408-banking-simplify-concurrency-retry-0911-133716381/run_0_20260911_133716381.log).
- Production `17371` was kept running during the benchmark, then stopped for
  the switch below. Full final export and all-source accounting remain required.

## Initial results and switch

Concurrency 2 completed in 32.38 minutes (13 accepted / three rejected);
concurrency 4 in 24.04 minutes (12/four), repeat in 25.17 minutes (14/two).
All three exported successfully with zero processing errors; no OOM or
retraction was found in inspected server logs. Semantic decisions vary between
arms, including the repeated concurrency-four runs; counts alone do not measure
quality equivalence. The single-concurrency baseline is still finishing.

At the user's request on 2026-09-11 14:22 CST, stopped production `17371` and
submitted concurrency-four continuation `17467`, preserving 202 completed
production decisions and assigning 1,712 pending/error IDs. No benchmark
decisions were merged into production. The benchmark job continues to completion.
