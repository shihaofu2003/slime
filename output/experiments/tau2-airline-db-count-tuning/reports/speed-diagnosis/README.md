# External User deployment speed diagnosis

Purpose: explain why job20886 (Trainer2 + three TP2 Generators, external two-TP2 User replicas) is not substantially faster than the previous batch128 run. Read-only investigation; training recipe and running processes unchanged.

## Measured throughput

At inspection,4/20 optimizer updates complete. First three update intervals:37.35,30.97,26.09min. Old batch128 whole-run post-first average28.44min; these windows and progress weights differ, so this is not an isolated topology comparison.

First-batch comparison, old/new: ready-group wait1952.53/1913.23s; Agent request mean1.12/0.99s; User turn mean16.53/17.28s. Logged Agent admission queue p95=0s in both. First batch required16 retained groups;12/13 groups were filtered before batch0. Log deduplication and a single first batch limit precision.

## CPU bottleneck

Read-only py-spy dump on rollout PID13024: process CPU102% (approximately one core),554 threads.87 dialogue threads were waiting on Loguru handler locks; active GIL owner was in database_records. A subsequent20s,49Hz GIL profile produced947 samples: model_dump412, tree_field_distance105, DBCount step comparison line19099, database_records26, score line13424, DBCount step line1899. These DB-related leaf frames total675/947=71.3%. Logging accounted for9 inclusive CPU samples: lock waiting is visible but is not the dominant measured CPU work. Profile: [raw stacks](rollout_gil.raw).

`DBCountProgressOrchestrator.step` snapshots the entire environment DB and compares records before each Agent decision. Its superclass then calls `ProgressPotential.score`, which snapshots the same unchanged state again and compares it to the target. All dialogue workers run as threads in one RolloutManager process, sharing the Python GIL. Adding inference replicas does not parallelize these Python operations.

## Inference and admission

First-batch final20min Generator GPU pair mean utilization: old2/3≈5.8%,4/5≈40.9%; new2/3≈2.9%,4/5≈7.1%,6/7≈42.3%. Agent router uses cache_aware: imbalance is observed; cache affinity is a plausible contributor, not proven as the sole cause.

User decode-log samples over the observed service lifetime: replica0 mean1.46 running requests (max22), replica1 mean1.50 (max23), versus configured32 per replica. These are decode-log samples, not time-weighted utilization. Both replicas receive requests; they are not fully supplied with work.

Limits remain global Agent calls32, Agent HTTP connections32, orchestrator steps80, dialogue pool32×8. They do not scale automatically with replicas. However, near-zero measured Agent admission wait means raising Agent concurrency alone is not supported as the primary fix.

## Synchronization and next changes

Weight-update apply phase for updates0–3:38.78,23.89,3.69,9.77s; the first144s total should not be treated as GPU transfer time. It also includes RPC pause/drain and resume. Logs lack internal pause timestamps, so the exact cause of the long first pause remains unresolved; CPU/log contention is plausible.

Prioritize semantics-preserving CPU work reduction: reuse one DB snapshot at each decision for both DB-count and potential; reduce synchronous per-turn DEBUG logging while preserving trajectory artifacts; profile again. Then consider mutation-aware DB caching/record comparison, with reward equivalence checked. If one-core pressure remains, distribute environment/credit work across processes. Test Agent routing and concurrency only after CPU supply improves. Keep progress0.5, gamma0.98, batch128 and scoring semantics fixed when comparing deployment optimizations.


2026-09-18 code changes: StatePotential accepts an optional decision-state snapshot; DBCount passes the same snapshot to its parent, eliminating the second full serialization at each Agent decision. No cross-turn snapshot caching or reward semantics changes. Per-turn Python timing logs now use DEBUG; asynchronous launcher sets LOGURU_LEVEL=WARNING and propagates it into Ray. Trajectory JSONL and batch-level metrics remain enabled. Sampling admission counters are released before diagnostic logging; pause entry and internal gate duration are now logged separately.

User clarified continuous refill semantics. In unlimited-lag mode, pool capacity now counts only active groups. A completed retained K8 group moves to the ready queue and immediately admits a new task; it no longer occupies an active slot until training finishes. The trainer takes the oldest16 ready groups, while slow active groups continue. Total outstanding+consumed groups cannot exceed the remaining planned training budget; bounded-lag mode retains its admission constraints. Added coverage for immediate refill before any training and for slow logging not holding the sampling gate; updated pool-capacity expectations. Existing running job20886 retains its already-imported code; these changes take effect in a new process. No training restart performed.

Validation: all114 tests in `test_progress.py`, `test_db_count.py`, and `tests/test_tau2_continuous.py` passed in63.03s on the training container via a separate CPU Ray job; launcher Bash syntax passed. Includes real airline/retail/telecom snapshot-score equivalence, mutation/restoration boundaries, banking tests, refill behavior and logging/pause interaction. [Validation log](validation.log). No live throughput claim yet.


2026-09-18 15:13 CST follow-up: job20929 still had unequal Generator utilization (pairs approximately50%,7%,0% in the inspected window). All three router workers were healthy; actual policy was cache_aware. New GIL profile726 samples still concentrated in DB serialization/validation/comparison; [raw stacks](fast_pool_gil.raw). Async Agent router now defaults to round_robin. DBCount caches snapshots, DB-count and potential scores across read-only/conversation steps for environments using base no-op sync_tools; any mutating tool invalidates before execution, including partial-write failures. Environments with custom synchronization retain fresh snapshots. Local User model is registered with existing zero-cost semantics to avoid per-turn unknown-price ERROR logs. Training reward formula and pool/budget unchanged.

Validation:116 tests passed (67.24s), covering cache/full-recompute equivalence for writes, restoration, record addition/deletion, existing domain and continuous-pool tests; [test log](cache-validation.log). Separate container check confirmed local User cost lookup returns0. Bash syntax passed. Stopped20929 and submitted20934 to load these changes; live utilization and throughput remain to be verified after scheduling/startup.


2026-09-18 15:31 CST live correction:20934 RUNNING,0/20 updates, no completed trajectory artifacts. Three Generator servers healthy; initial weight broadcast finished15:28:35 in3s. Six dashboard samples over30s showed all8 GPUs at0%; this is startup, not steady-state throughput. Rollout stack is in initial pool admission, imports/LiteLLM lazy imports and path resolution ([stack](balanced-cache-startup-stack.txt)). Runtime RouterArgs unexpectedly still says cache_aware: the launcher used --sglang-router-policy, whereas installed RouterArgs accepts --router-policy; unknown option did not override the default. Corrected launcher to --router-policy and verified real container parser produces round_robin. Current20934 retains cache_aware until restart; no restart in this status inspection. Earlier statement that routing fix was deployed was premature. No speed improvement established.


2026-09-18 16:25 CST concurrency-only inspection (CPU optimization deferred at user request): Agent gate32 and shared HTTP max/keepalive32 are global, versus3 Generator engines each max_running_requests16. Global SAMPLING_STEPS80 covers full Agent/User/tool steps, including time waiting for Agent gate. Runtime stack:43 dialogue threads waiting for Agent admission,58 waiting for step admission;20 Agent threads in HTTP reads,11 at HTTP pool lock and4 User threads in HTTP reads.117 dialogue workers idle awaiting work. This is a snapshot, not time-weighted occupancy. [Stacks](concurrency-stack-1623.txt).

Eight service metric samples about4s apart: each Generator reported0–1 running requests and0 queued; request counters remained near equal (2431/2431/2432 at final sample). Thus gate wait is upstream, not evidence of saturated GPU engines. [Metrics](concurrency-metrics-1623.json). User /metrics returns404; recent decode logs instead show mean1.39/1.21 requests, maxima4/3. User HTTP pool80 already exceeds two server replicas' combined64 running slots. HTTP mutex waits do not establish connection exhaustion by themselves.

Recommended next controlled concurrency trial: keep task pool32, batch128/K8/LR/reward unchanged; Agent gate and HTTP limit32→48 (three engines×16), step limit80→160 to reduce cross-role contention. Then evaluate pool32→64 only if effective-group throughput remains supply-limited. Merely doubling task pool leaves both gates unchanged and increases waiters. No code edits, runtime changes or restart performed in this inspection; no claim of measured speedup from proposed settings.
