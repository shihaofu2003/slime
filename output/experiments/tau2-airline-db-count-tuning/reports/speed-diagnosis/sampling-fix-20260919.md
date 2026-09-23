# Sampling blockage and prefix screening

Purpose: diagnose job21033 sampling stalls, evaluate early screening, and validate targeted fixes. Reference run: `20260919-progress-db-count-v1-lr2e-6-b128-pw1-external-control` (20 updates, batch128/K8, progress1).

## Measured bottlenecks

Excluding the first update, mean trainer step1071.88s: data wait990.24s, training81.64s, weight application1.11s (timers overlap; do not sum all fields). During18:00–23:37 UTC, Trainer GPUs0/1 average7.5% utilization, Generator GPUs2–7 average21.2–21.7%. Generator routing is balanced. Decode-log samples average1.50 running requests per Agent worker and2.15/2.18 per User worker; queues nearly zero. These are log samples, not time-weighted service occupancy.

All51,530 Agent requests have server timings: client elapsed mean17.751s (p95=89.678s), SGLang service mean0.614s (p95=1.784s), server queue mean0.0131s (p95=0.00086s). Mean client-minus-server gap17.137s includes Agent admission, client CPU/HTTP and transport; it cannot be attributed entirely to CPU without the new stage timings. This local-server measurement strongly rules out slow GPU generation as the main per-request delay. [Full latency summary](agent-client-server-latency.json).

The old pool replenishes only complete groups. Seven completed trajectories do not free a group slot while the eighth is outstanding. Group-start to simulation-end intervals imply156.7 unfinished trajectories on average across the run, against256 reserved worker slots. This includes startup/drain and is not GPU concurrency. Retained groups mean20.80min, p95=46.13min, max87.58min.

Each trajectory also reloads large Airline JSON databases from shared storage. Local12-repeat benchmarks on the6.8MB/21MB DB measured0.117/0.406s per legacy load versus0.099/0.324s from cached immutable JSON bytes with fresh Pydantic validation. These are CPU microbenchmarks, not training speedups. The original reference-state LRU permits simultaneous misses: eight concurrent same-task requests built eight distinct targets in1.545s. Serialized construction gives1miss/7hits, one shared immutable target,0.183s on the same local example.

Live diagnostic21096 applied those first-round fixes but still used one Python process. It reached554 threads while consuming120.6% CPU (about1.2 cores). A20-second GIL profile captured930 samples: database record expansion429, Pydantic serialization278, record comparison96, and field-distance comparison75; together94.4% of the sampled Python execution. A later896-sample profile again concentrated in DB validation/serialization/comparison. This confirms a client-side CPU/GIL bottleneck: hundreds of dialogue threads do not execute Python DB work on hundreds of cores. [Profile summary](single-process-cpu-profile.json); [raw profiles](../../arms/async/20260919-progress1-b128-sampling-fix-probe4-v2/profile/).

## First-four evidence

All507 groups/4056 trajectories were examined. Ordering uses recorded simulation end time; dump order yields the same extreme-prefix counts. This approximates online completed-outcome order because evaluation/tokenization follow simulation completion. Raw analysis: [JSON](prefix4-analysis.json), [compact group records](prefix4-analysis.samples.json). Reproduce with `scripts/analyze_tau2_group_prefixes.py RUN_DIR OUTPUT_JSON`.

| First four completed trajectories | Groups | Ultimately uniform8 | Ultimately retained mixed8 | Other filtered mixed8 |
|---|---:|---:|---:|---:|
| 0/4 successful |152|104 all-zero|43|5|
| 4/4 successful |130|76 all-one|54|0|
| 1–3/4 successful |225|0|223|2|

Rejecting both extremes after four completions loses97/320=30.31% of useful groups. False rejection among the0/4 candidates is43/152=28.29%; among4/4 candidates54/130=41.54%. Ordering by launch index instead still loses84/320=26.25%. Even waiting for six equal outcomes loses39 useful groups; seven loses17. The first four have47.58% mean success, the last four43.64%, demonstrating completion-order bias in this run.

The187 rejected groups comprise104 all-zero (mean29.70min),76 all-one (15.33min), and7 mixed groups containing removed samples. Some uniform groups also contain removed samples; defer/retry is therefore limited to the actual `official_outcome_all_zero` filter reason, not every zero reward caused by an invalid sample.

Of the43 retained groups with0/4 initial successes,33 eventually have exactly1/8 success. Of the54 retained groups with4/4 initial successes,29 eventually have exactly7/8 successes. These minority outcomes provide nonzero group-relative learning signal; an early hard filter would specifically remove many such groups.

## Changes

- With unlimited policy lag, refill after individual trajectory completion. Keep at most `pool_capacity × K` unfinished trajectories and an additional `max_pending_groups` bound. Airline defaults:256 trajectory slots,64 unfinished groups. Retained groups still contain8 complete trajectories from one task. Existing bounded-lag behavior and final update budget remain enforced.
- Near the end, stop admission only when completed ready groups plus the current training batch cover the remaining training budget. Pending groups no longer force the final batch to wait for every tail. The optimizer still consumes exactly the configured update count and K8 batch size; extra unfinished/unused sampling stops at producer shutdown. This final refinement was added after the live probe described below and is covered by the final CPU regression run.
- Execute environment/User orchestration and reward computation in four CPU-only Ray processes. The existing GPU layout remains Trainer2+three TP2 Generators6, with external User20873. Divide total Agent admission48 and step admission160 across workers; dispatch trajectories to the worker with the fewest unfinished calls. Weight synchronization pauses/drains Agent requests in every worker, publishes the new version to every worker, then resumes. User requests are not paused for weight synchronization. Each trajectory retains its actual per-turn policy versions, generated token IDs and behavior log probabilities. Multi-process mode currently supports async sampling with unlimited policy lag; single-process mode remains available with `TAU2_ENVIRONMENT_WORKERS=1`.
- Airline task source traverses a shuffled deck without replacement. Confirmed all-zero groups queue a new full K8 draw after two additional published policy versions; due retries alternate with fresh tasks. No old trajectories are replayed, and no first-four hard filter is enabled. Deck/RNG/retry state uses the existing data-source checkpoint metadata. This changes the sampling distribution and must be reported as such in outcome comparisons.
- Cache immutable Airline/Retail JSON bytes while constructing independent mutable DB instances. Reuse the schema environment for the first rollout attempt; retries still construct fresh environments. Telecom retains its existing TOML loader.
- Construct each cold reference target once across concurrent K8 requests; expand the read-only target cache32→64 for the larger partial-group pool.
- Record per-trajectory wall/thread-CPU time, setup, Agent/step admission waits, active request time, User time, simulation, tokenization, official evaluation, packing, progress and dumping. These include nested timings and must not be summed as disjoint stages. Accepted-batch means and cumulative filter reasons are emitted in rollout metrics; completion logs include every trajectory including rejected groups.

## Validation

CPU job21093 passed125 tests; two failures came from a test module loaded before the matching source/test edit, and one from a missing `HF_CHECKPOINT` in the validation wrapper. The wrapper was corrected and job21094 passed all128 tests in88.04s. Live speed validation is a separate four-update run with the same LR2e-6, batch128, K8, progress1, User, decoding settings and Trainer2+Generator6 topology.

Probe21095 failed only in test environment isolation before model startup; corrected the test and21096 passed30+98 tests. Stopped21096 after the CPU bottleneck was demonstrated, before any optimizer update. Probe21102 (`20260919-progress1-b128-sampling-fix-probe4-workers4`) succeeded with four CPU environment processes and four optimizer updates, four checkpoints and four weight synchronizations. Container preflight passed all130 tests (30 progress/DB tests in65.28s,100 continuous-training tests in18.90s). Gradient norms were0.523/0.484/0.415/0.492; same LR2e-6, global batch128/K8, progress1 and H10080GB topology. Two Agent read errors recovered through the existing one-attempt fresh-connection retry. Probe saved every update, costing about30–41s each, whereas the baseline saved every10 updates.

| Measurement | Original21033 | Four-process21102 |
|---|---:|---:|
| First-batch ready wait |1549.85s|326.87s|
| Next three ready waits |971.27 /892.67 /925.84s|58.47 /16.50 /106.73s|
| Ready-batch log to Trainer start, first four |215.76 /260.61 /220.53 /246.37s|0.48 /0.91 /1.10 /1.04s|
| First four optimizer updates, from initial pool admission to fourth training completion |94.26min|15.84min|
| Mean training computation per update, first four |81.77s|82.14s|
| Mean Generator GPU utilization during first-batch sampling, excluding its first minute |24.13%|79.18%|

The first-four elapsed comparison is5.95× faster in this run. It includes the new shuffled task order and variable filtering/trajectory lengths, so it is not an isolated causal estimate or a long-run throughput guarantee. Model startup and the final checkpoint are excluded from that particular interval. [Update timing](four-update-timing-comparison.json), [GPU sampling windows](first-batch-gpu-utilization.json). Process profiles measured407–420% aggregate CPU across four workers, versus about121% in the one-process probe. In79 matched initial sample IDs shared with the short one-process probe, mean trajectory wall time was238.93→143.92s; matching is limited to trajectories completed in both probes and does not control generation nondeterminism. [Trajectory/request comparison](multiprocess-live-comparison.json).

The ready-to-Trainer interval covers rollout postprocessing/serialization/delivery and uses second-resolution ready-log timestamps, so its new subsecond values are approximate. Its216–261s old delay disappearing after moving CPU work out of RolloutManager is further evidence of producer-process contention. [Boundary measurements](batch-ready-to-trainer-start.json).

All108 completed groups/864 trajectories were recorded:64 retained and44 filtered (40.7%). Thus the higher speed did not come from eliminating the uniform-group filter. First synchronization drained Agent requests in2.85s, applied weights in0.27s and resumed all workers in0.32s; subsequent version1/2/3 trajectories were recorded. The new four-process deployment is verified end to end.

The probe's last update exposed a remaining budget-drain issue: admission stopped with15 ready groups and one unfinished trajectory. The final refill change removes that compulsory tail;13 focused tests passed, including a last update that finishes while an old group remains blocked, exact optimizer-budget consumption and no admission after the budget is consumed. Full CPU validation21114 passed129 tests; one CPU loss-equivalence test failed because Torch Dynamo accessed an unallocated host GPU. The zero-GPU test wrapper now explicitly hides CUDA devices; replacement21115 succeeded with all130 tests passing in91.16s. Bash syntax and whitespace checks passed.

Checkpoint inspection verified zero retries at policy versions0 and1, five at version2, and nine at version3. All nine retry groups used fresh sample IDs; their earliest Agent versions were2 or3. Two retries became useful mixed groups (1/8 success each), seven remained all-zero; this is a sampling observation, not proof that the weight updates caused the change. Fresh-task groups retained62/99; retry groups retained2/9. The cooldown prevents immediate repetition, while interleaving preserves fresh-task supply. [Checkpoint counters](live-retry-state.json), [per-retry outcomes](live-retry-outcomes.json).

Remaining tuning should follow these timings: keep the regular checkpoint interval for full training (the probe intentionally saved every update), and increase CPU worker count only if a longer run again shows CPU saturation with idle Generators. First-four hard filtering remains disabled because of its measured loss of useful mixed groups. The final pool behavior spends some extra sampling work to avoid end-of-run waiting; record unused samples separately when comparing rollout cost.

Sustained validation21168 subsequently SUCCEEDED30/30 with Agent HTTP timeout60s: matched first20 updates370.22→80.46min (4.60x), Generator utilization21.54→76.22%, longest synchronization28.68s. Full30 updates took124.19min through the last optimizer completion;480 groups trained and10 retained groups unused. No first-four hard filter was enabled. [Full report, outcomes and curves](long30-timeout60/README.md). Unchanged-protocol three-domain iter29 evaluation21377 is RUNNING; HF conversion is complete and evaluation services are starting.
