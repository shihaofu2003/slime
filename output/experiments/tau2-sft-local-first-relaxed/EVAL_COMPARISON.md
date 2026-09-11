# tau2 local-first SFT official evaluation comparison

## Protocol

Profile: `dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| raw | 100 | 20.50% | 43.00% | 6.00% |
| old_sft | 100 | 21.25% | 41.00% | 10.00% |
| new_sft | 100 | 27.00% | 54.00% | 4.00% |

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

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---:|---:|---:|
| new_sft-raw | pass@1 / pass^1 | +6.50 pp | [+1.00, +12.00] pp | yes |
| new_sft-raw | pass@4(any) | +11.00 pp | [+2.00, +21.00] pp | yes |
| new_sft-raw | pass^4 | -2.00 pp | [-8.00, +4.00] pp | no |
| new_sft-old_sft | pass@1 / pass^1 | +5.75 pp | [+0.50, +11.00] pp | yes |
| new_sft-old_sft | pass@4(any) | +13.00 pp | [+2.00, +24.00] pp | yes |
| new_sft-old_sft | pass^4 | -6.00 pp | [-12.00, -1.00] pp | yes |

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
