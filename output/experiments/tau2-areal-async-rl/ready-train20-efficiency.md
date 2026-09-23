# Tau2 20-update synchronous/asynchronous GRPO efficiency

Both jobs succeeded: sync17907 and async17908, each20 updates,100 accepted K=8 groups/800 trajectories,8 H10080GB /64 CPU /1024GB, User32 / HTTP80, binary GRPO and TIS. Iter9 and iter19 checkpoints were saved. No automatic continuation is scheduled.

| Metric | Sync | Async |
|---|---:|---:|
| End-to-end seconds | 3614.0 | 3408.0 |
| Loop seconds | 3146.7 | 2941.9 |
| Loop seconds/update | 157.3 | 147.1 |
| Trainer seconds | 509.3 | 514.1 |
| Checkpoint seconds | 59.1 | 64.3 |
| Weight-update seconds | 6.8 | 15.4 |
| Ready + lag wait seconds | 2448.1 | 2200.1 |
| Of which lag wait seconds | 1701.2 | 2039.3 |
| Sync sampling-barrier seconds | 34.1 | 0.1 |
| Agent requests overlapping training | 0 | 2590 |
| User requests overlapping training | 0 | 1533 |
| Accepted groups: airline/retail/telecom | 58/30/12 | 59/32/9 |
| Replaced groups | 3 | 6 |
| Extra attempts in accepted trajectories | 12 | 21 |
| HTTP read retries | 0 | 0 |
| Training input tokens | 8207940 | 8350804 |

The measured end-to-end speedup is **1.060×**, saving206s (3m26s), a5.70% elapsed-time reduction. Loop speedup is1.070×, a6.51% reduction. The2× target is not met. End-to-end starts before User initialization and ends after training/checkpoint completion; installation, preflight, queueing and official evaluation are excluded. All in-run retries/replacements are included. The timing components are diagnostic and are not an exhaustive additive partition of the loop.

Async sampling demonstrably overlaps training, but its trainer work remains about514s and ready/lag waiting totals2200s, about75% of its loop. The earliest-turn lag≤1 rule retains slow groups and constrains further optimizer advancement. More overlap alone does not establish a2× end-to-end improvement.

This single pair shows a modest measured advantage, not a statistically established speed gain. Actual task mix, replacements and trajectory lengths differ; async accepted fewer Telecom groups. Jobs overlapped on the cluster but started about23.4 minutes apart, with shared-storage contention not isolated. Both raw-token/probability/lag inspections report no violations. Official held-out performance was not evaluated, so this result says nothing conclusive about final policy quality.

Sources: [comparison JSON](ready-train20-comparison.json), [sync inspection](ready-train20-sync-inspection.json), [async inspection](ready-train20-async-inspection.json), [sync timing](ready-train20-sync-timing.json), [async timing](ready-train20-async-timing.json), [sync run](jobs/17907-tau2-ready-sync-train20-0912-165246921/run_0_20260912_165246921.log), [async run](jobs/17908-tau2-ready-async-train20-0912-165247940/run_0_20260912_165247940.log).
