# Sampling long-run speed comparison

Purpose: verify sustained speed with the completed sampling fixes. Run: `20260919-progress1-b128-sampling-fix-long30`.

Completed updates: **18/30**. Reference: `20260919-progress-db-count-v1-lr2e-6-b128-pw1-external-control` (20 updates).

Both use SFT4673, LR2e-6, batch128/K8, progress1, Trainer2+Generator6 and external User. Task order/retry scheduling changed. This compares the combined system, not an isolated parameter.

| Window | Updates compared | Baseline min/update | Current min/update | Speedup |
|---|---:|---:|---:|---:|
| matched | 18 | 19.03 | 4.87 | 3.90x |
| stable | 17 | 18.33 | 4.69 | 3.90x |

Stable excludes the reference's first and last two updates; intervals run between consecutive optimizer completions and include intervening sampling, checkpoint and synchronization time. Matched total starts at initial pool admission. Model startup and the final checkpoint are excluded.

| Metric over the matched window | Baseline | Current |
|---|---:|---:|
| Batch ready wait, seconds/update | 892.70 | 141.84 |
| Batch ready to trainer start, seconds | 142.17 | 0.99 |
| Training compute, seconds/update | 80.30 | 80.17 |
| Trainer GPU utilization | 6.89% | 26.66% |
| Generator GPU utilization | 21.57% | 61.75% |

GPU utilization is the mean of 5-second samples across the respective GPUs. Batch-ready log timestamps have one-second precision; phase timers overlap and must not be summed.

| Current update window | Mean min/update | Mean ready wait, seconds |
|---|---:|---:|
| 1–5 | 4.03 | 151.51 |
| 6–10 | 4.90 | 129.93 |
| 11–15 | 3.80 | 126.15 |
| 16–18 | 8.02 | 171.70 |

GPU samples, per-update times, group-filter counts and completed-but-unused groups are in [the JSON report](speed_comparison.json). Training reward curves are separate from held-out evaluation.

Status: **STOPPED after18/30 updates** to bound repeated Agent HTTP waits. This is not the completed long-run result. See the [HTTP diagnosis](http-wait.md) and [replacement30-update validation](../long30-timeout60/README.md).
