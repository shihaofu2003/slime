# Mixed GRPO timing analysis

Jobs 18815 (vanilla) and 18816 (credit), 40 updates each. Source: training perf logs and completed trajectory dumps.

| Measurement | Vanilla | Credit |
|---|---:|---:|
| Job runtime | 2.8617 h | 5.9023 h |
| Sum rollout wall time | 8437.20 s | 19256.58 s |
| Sum train time | 534.13 s | 543.56 s |
| Mean rollout per update | 210.93 s | 481.41 s |
| Completed candidate records | 1899 | 1746 |
| Extra resample attempts | 96 | 73 |
| Removed candidate records | 30 | 32 |
| Mean final simulation Agent messages | 15.42 | 14.99 |
| Mean candidate response tokens | 4495 | 4416 |
| Mean final simulation duration | 37.55 s | 112.58 s |

Rollout wall-time delta is 10819.38 s, explaining about 98.8% of the job-runtime delta.
Actual training adds only 9.42 s. Candidate records include surplus/rejected groups and are not accepted training counts;
each model trains on 1600 accepted trajectories. Dumps only retain the final attempt; retry and aborted trajectory work is not fully represented.

| Domain | Vanilla simulation mean | Credit simulation mean | Credit state-scoring mean |
|---|---:|---:|---:|
| Airline | 32.35 s | 120.55 s | 42.68 s |
| Retail | 29.34 s | 88.38 s | 17.95 s |
| Telecom | 75.61 s | 116.89 s | 1.07 s |

State scoring averages 31.76 s per completed candidate (p95 119.83 s), including target construction and boundary scores.
It serializes full Agent/User databases, recursively builds flattened path dictionaries and compares the union of fields before each Agent decision.
Example observed DB inputs are 7.08 MB Airline, 10.63 MB Retail, and 9.63 KB Telecom TOML.
The same task target is reconstructed per rollout attempt. Most read-only turns still scan the full DB.

Rollouts execute through asyncio.to_thread in the same Python process. Python dictionary traversal therefore plausibly introduces GIL contention and delays other trajectories/requests. This mechanism is inferred from code and the measured scoring cost, not established by CPU profiling.
Scoring uses perf_counter elapsed time, not CPU time; concurrent intervals overlap. The 55445.42 s sum of scoring spans is not additive job wall time.
User generation_time_seconds is absent in these dumps; remaining delays cannot be assigned precisely to User inference.
The rollout non_generation_time metric includes more than environment/credit CPU work and cannot be interpreted as pure credit cost.

Conclusion: the measured slowdown is in rollout, with repeated full-DB state scoring a substantial identified extra cost. Longer trajectories or more retries do not explain this run-level gap. RTG/token combination and GPU training are not the observed bottleneck. A CPU profile and a matched-trajectory scoring replay would quantify target construction versus flattening/distance versus thread contention before an optimization.
