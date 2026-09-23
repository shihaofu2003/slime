# Sampling long-run speed comparison

Purpose: verify sustained speed with the completed sampling fixes. Run: `20260919-progress1-b128-sampling-fix-long30-timeout60`.

Completed updates: **30/30**. Reference: `20260919-progress-db-count-v1-lr2e-6-b128-pw1-external-control` (20 updates).

Both use SFT4673, LR2e-6, batch128/K8, progress1, Trainer2+Generator6 and external User. Task order/retry scheduling changed. This compares the combined system, not an isolated parameter.

| Window | Updates compared | Baseline min/update | Current min/update | Speedup |
|---|---:|---:|---:|---:|
| matched | 20 | 18.51 | 4.02 | 4.60x |
| stable | 17 | 18.33 | 3.83 | 4.79x |

Stable excludes the reference's first and last two updates; intervals run between consecutive optimizer completions and include intervening sampling, checkpoint and synchronization time. Matched total starts at initial pool admission. Model startup and the final checkpoint are excluded.

| Metric over the matched window | Baseline | Current |
|---|---:|---:|
| Batch ready wait, seconds/update | 868.65 | 142.63 |
| Batch ready to trainer start, seconds | 133.95 | 0.85 |
| Training compute, seconds/update | 81.85 | 82.57 |
| Trainer GPU utilization | 7.22% | 33.73% |
| Generator GPU utilization | 21.54% | 76.22% |

GPU utilization is the mean of 5-second samples across the respective GPUs. Batch-ready log timestamps have one-second precision; phase timers overlap and must not be summed.

| Current update window | Mean min/update | Mean ready wait, seconds |
|---|---:|---:|
| 1–5 | 4.07 | 150.03 |
| 6–10 | 3.27 | 106.01 |
| 11–15 | 4.23 | 144.27 |
| 16–20 | 4.53 | 170.19 |
| 21–25 | 4.53 | 166.84 |
| 26–30 | 4.22 | 152.95 |

GPU samples, per-update times, group-filter counts and completed-but-unused groups are in [the JSON report](speed_comparison.json). Training reward curves are separate from held-out evaluation.

Job21168 **SUCCEEDED** at2026-09-19 17:43 CST. Container preflight passed138 tests; all30 losses and gradient norms were finite (gradient norm0.406–1.113). Checkpoints iter9/19/29 each have their synchronous-save completion and all four Megatron shards. The final30 updates consumed480 K8 groups,3,840 training trajectories.

Sampling began15:34:46 CST; the last optimizer update finished17:38:57, giving124.19 minutes for30 updates. Producer shutdown finished17:41:12,126.45 minutes after sampling began. The scheduler records135.48 minutes from job start to completion, including startup and final cleanup/reporting. Updates21–30 averaged4.37 minutes each; the last five averaged4.22. This confirms sustained improvement beyond the short probe.

Across all30 updates, Generator GPU utilization averaged75.22%, and Trainer32.78%. During the measured training-compute intervals, Trainer averaged98.26%: its remaining idle time is predominantly waiting for enough retained groups. In the matched first20 window, the three Generator replicas averaged75.80/76.91/75.95%, showing balanced use. These are the8 training GPUs; the separate4-GPU User service is unchanged in both runs.

All30 weight synchronizations completed: mean15.57s, median14.87s, maximum28.68s. Agent ReadTimeout/ReadError recovery was exercised without a failed update. The underlying HTTP errors remain observable; the earlier168–500s pauses did not recur in this run. See the [HTTP diagnosis](../long30/http-wait.md).

Of915 completed groups,490 were retained and425 filtered (46.45%);480 were trained and10 retained groups were unused. Another23 partially completed groups contributed130 recorded trajectories. Total recorded trajectories:7,450. Mean retained/filtered group duration was6.80/7.73min, versus20.80/25.65min in the old20-update run; task mix and retry scheduling differ. Among282 completed repeat draws,73 were retained and208 remained all-zero. A repeat draw is a fresh sample of a previously drawn task; these observations do not isolate the effect of policy learning. [Outcome and timing details](outcomes.json).

Final curves: [accepted training batches](../../../plots/20260919-progress1-b128-sampling-fix-long30-timeout60/reward_curves.png), [before filtering](../../../plots/20260919-progress1-b128-sampling-fix-long30-timeout60/reward_prefilter.png). The accepted-batch curve contains all30 updates; its final10-update mean is0.453125. Official three-domain iter29 evaluation was submitted as21377 with the previous evaluation parameters; results are pending. See the [experiment record](../../../README.md).
