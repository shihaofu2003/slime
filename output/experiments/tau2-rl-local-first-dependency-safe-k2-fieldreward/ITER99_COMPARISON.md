# tau2 official evaluation comparison

## Protocol

Profile: `dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

Warning: Legacy baselines lack the candidate's signed Agent protocol. Point estimates and paired intervals are descriptive, not a controlled single-variable comparison.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| raw | 100 | 20.50% | 43.00% | 6.00% |
| old_sft | 100 | 21.25% | 41.00% | 10.00% |
| new_sft | 100 | 27.00% | 54.00% | 4.00% |
| old_rl100 | 100 | 25.50% | 49.00% | 7.00% |
| new_rl100 | 100 | 19.00% | 38.00% | 5.00% |

## Domain results

| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| raw | airline | 20.00% | 50.00% | 5.00% |
| raw | retail | 26.25% | 47.50% | 10.00% |
| raw | telecom | 15.00% | 35.00% | 2.50% |
| old_sft | airline | 32.50% | 50.00% | 20.00% |
| old_sft | retail | 20.62% | 40.00% | 12.50% |
| old_sft | telecom | 16.25% | 37.50% | 2.50% |
| new_sft | airline | 30.00% | 45.00% | 10.00% |
| new_sft | retail | 29.38% | 57.50% | 5.00% |
| new_sft | telecom | 23.12% | 55.00% | 0.00% |
| old_rl100 | airline | 31.25% | 50.00% | 20.00% |
| old_rl100 | retail | 21.25% | 45.00% | 7.50% |
| old_rl100 | telecom | 26.88% | 52.50% | 0.00% |
| new_rl100 | airline | 28.75% | 45.00% | 10.00% |
| new_rl100 | retail | 23.12% | 45.00% | 7.50% |
| new_rl100 | telecom | 10.00% | 27.50% | 0.00% |

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---:|---:|---:|
| new_rl100-raw | pass@1 / pass^1 | -1.50 pp | [-7.00, +4.00] pp | no |
| new_rl100-raw | pass@4(any) | -5.00 pp | [-15.00, +5.00] pp | no |
| new_rl100-raw | pass^4 | -1.00 pp | [-6.00, +4.00] pp | no |
| new_rl100-old_sft | pass@1 / pass^1 | -2.25 pp | [-7.25, +2.50] pp | no |
| new_rl100-old_sft | pass@4(any) | -3.00 pp | [-13.00, +7.00] pp | no |
| new_rl100-old_sft | pass^4 | -5.00 pp | [-11.00, +1.00] pp | no |
| new_rl100-new_sft | pass@1 / pass^1 | -8.00 pp | [-13.25, -2.75] pp | yes |
| new_rl100-new_sft | pass@4(any) | -16.00 pp | [-27.00, -5.00] pp | yes |
| new_rl100-new_sft | pass^4 | +1.00 pp | [-2.00, +4.00] pp | no |
| new_rl100-old_rl100 | pass@1 / pass^1 | -6.50 pp | [-11.75, -1.25] pp | yes |
| new_rl100-old_rl100 | pass@4(any) | -11.00 pp | [-21.00, -1.00] pp | yes |
| new_rl100-old_rl100 | pass^4 | -2.00 pp | [-7.00, +3.00] pp | no |

## Trajectory tool-mode diagnostic

A trajectory is `multi_exposed` when at least one assistant turn contains more than one tool call. These success rates are observational and are not causal effects of batching.

| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |
|---|---|---:|---:|---:|---:|
| raw | multi_exposed | 5 | 20.00% | 0 | 1 |
| raw | non_multi_exposed | 395 | 20.51% | 3 | 13 |
| old_sft | multi_exposed | 271 | 23.99% | 24 | 9 |
| old_sft | non_multi_exposed | 129 | 15.50% | 12 | 8 |
| new_sft | multi_exposed | 311 | 26.37% | 15 | 9 |
| new_sft | non_multi_exposed | 89 | 29.21% | 6 | 3 |
| old_rl100 | multi_exposed | 302 | 27.15% | 19 | 14 |
| old_rl100 | non_multi_exposed | 98 | 20.41% | 5 | 7 |
| new_rl100 | multi_exposed | 339 | 18.29% | 93 | 5 |
| new_rl100 | non_multi_exposed | 61 | 22.95% | 7 | 2 |
