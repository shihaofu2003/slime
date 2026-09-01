# tau2 official evaluation comparison

## Protocol

Profile: `agent-owned-dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| raw_instruct | 100 | 23.75% | 48.00% | 9.00% |
| old_sft | 100 | 14.00% | 28.00% | 4.00% |
| current_sft | 100 | 28.00% | 51.00% | 8.00% |
| contract-only | 100 | 25.25% | 45.00% | 5.00% |
| contract-boundary | 100 | 23.50% | 45.00% | 9.00% |

## Domain results

| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| raw_instruct | airline | 27.50% | 50.00% | 15.00% |
| raw_instruct | retail | 28.75% | 47.50% | 15.00% |
| raw_instruct | telecom | 16.88% | 47.50% | 0.00% |
| old_sft | airline | 37.50% | 50.00% | 20.00% |
| old_sft | retail | 5.00% | 12.50% | 0.00% |
| old_sft | telecom | 11.25% | 32.50% | 0.00% |
| current_sft | airline | 32.50% | 40.00% | 15.00% |
| current_sft | retail | 30.63% | 52.50% | 10.00% |
| current_sft | telecom | 23.12% | 55.00% | 2.50% |
| contract-only | airline | 31.25% | 55.00% | 10.00% |
| contract-only | retail | 25.62% | 40.00% | 7.50% |
| contract-only | telecom | 21.88% | 45.00% | 0.00% |
| contract-boundary | airline | 30.00% | 45.00% | 20.00% |
| contract-boundary | retail | 26.88% | 45.00% | 12.50% |
| contract-boundary | telecom | 16.88% | 45.00% | 0.00% |

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---:|---:|---:|
| contract-only-raw_instruct | pass@1 / pass^1 | +1.50 pp | [-3.75, +6.75] pp | no |
| contract-only-raw_instruct | pass@4(any) | -3.00 pp | [-14.00, +8.00] pp | no |
| contract-only-raw_instruct | pass^4 | -4.00 pp | [-10.00, +2.00] pp | no |
| contract-only-old_sft | pass@1 / pass^1 | +11.25 pp | [+5.25, +17.50] pp | yes |
| contract-only-old_sft | pass@4(any) | +17.00 pp | [+7.00, +27.00] pp | yes |
| contract-only-old_sft | pass^4 | +1.00 pp | [-3.00, +5.00] pp | no |
| contract-only-current_sft | pass@1 / pass^1 | -2.75 pp | [-7.50, +2.00] pp | no |
| contract-only-current_sft | pass@4(any) | -6.00 pp | [-15.00, +3.00] pp | no |
| contract-only-current_sft | pass^4 | -3.00 pp | [-9.00, +3.00] pp | no |
| contract-only-contract-boundary | pass@1 / pass^1 | +1.75 pp | [-3.25, +7.00] pp | no |
| contract-only-contract-boundary | pass@4(any) | +0.00 pp | [-9.00, +9.00] pp | no |
| contract-only-contract-boundary | pass^4 | -4.00 pp | [-10.00, +1.00] pp | no |

## Trajectory tool-mode diagnostic

A trajectory is `multi_exposed` when at least one assistant turn contains more than one tool call. These success rates are observational and are not causal effects of batching.

| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |
|---|---|---:|---:|---:|---:|
| raw_instruct | multi_exposed | 1 | 0.00% | 0 | 0 |
| raw_instruct | non_multi_exposed | 399 | 23.81% | 6 | 7 |
| old_sft | multi_exposed | 83 | 31.33% | 0 | 1 |
| old_sft | non_multi_exposed | 317 | 9.46% | 0 | 55 |
| current_sft | multi_exposed | 227 | 29.52% | 2 | 5 |
| current_sft | non_multi_exposed | 173 | 26.01% | 4 | 9 |
| contract-only | multi_exposed | 335 | 22.39% | 50 | 7 |
| contract-only | non_multi_exposed | 65 | 40.00% | 7 | 2 |
| contract-boundary | multi_exposed | 294 | 22.45% | 9 | 11 |
| contract-boundary | non_multi_exposed | 106 | 26.42% | 1 | 23 |
