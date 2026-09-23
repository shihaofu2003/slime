# Sampling long-run speed comparison

Purpose: verify sustained speed with the completed sampling fixes. Run: `20260919-progress1-b128-fast-uniform-control20`.

Completed updates: **20/20**. Reference: `20260919-progress-db-count-v1-lr2e-6-b128-pw1-external-control` (20 updates).

Both use SFT4673, LR2e-6, batch128/K8, progress1, Trainer2+Generator6 and external User. Task selection is recorded in each run's launcher; asynchronous completion can change which groups train. This reports observed end-to-end speed.

| Window | Updates compared | Baseline min/update | Current min/update | Speedup |
|---|---:|---:|---:|---:|
| matched | 20 | 18.51 | 3.59 | 5.15x |
| stable | 17 | 18.33 | 3.33 | 5.51x |

Stable excludes the reference's first and last two updates; intervals run between consecutive optimizer completions and include intervening sampling, checkpoint and synchronization time. Matched total starts at initial pool admission. Model startup and the final checkpoint are excluded.

| Metric over the matched window | Baseline | Current |
|---|---:|---:|
| Batch ready wait, seconds/update | 868.65 | 117.70 |
| Batch ready to trainer start, seconds | 133.95 | 1.05 |
| Training compute, seconds/update | 81.85 | 81.54 |
| Trainer GPU utilization | 7.22% | 37.23% |
| Generator GPU utilization | 21.54% | 76.01% |

GPU utilization is the mean of 5-second samples across the respective GPUs. Batch-ready log timestamps have one-second precision; phase timers overlap and must not be summed.

| Current update window | Mean min/update | Mean ready wait, seconds |
|---|---:|---:|
| 1–5 | 4.63 | 184.32 |
| 6–10 | 3.49 | 115.29 |
| 11–15 | 2.97 | 68.33 |
| 16–20 | 3.28 | 102.89 |

GPU samples, per-update times, group-filter counts and completed-but-unused groups are in [the JSON report](speed_comparison.json). Training reward curves are separate from held-out evaluation.
