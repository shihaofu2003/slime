# tau2 official evaluation comparison

## Protocol

Profile: `agent-owned-dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| raw_instruct | 100 | 22.75% | 49.00% | 5.00% |
| old_sft | 100 | 9.50% | 21.00% | 3.00% |
| current_sft | 100 | 23.25% | 43.00% | 8.00% |
| contract-only | 100 | 26.25% | 50.00% | 11.00% |
| contract-boundary | 100 | 23.75% | 42.00% | 8.00% |

## Domain results

| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| raw_instruct | airline | 22.50% | 45.00% | 10.00% |
| raw_instruct | retail | 28.12% | 57.50% | 7.50% |
| raw_instruct | telecom | 17.50% | 42.50% | 0.00% |
| old_sft | airline | 27.50% | 45.00% | 10.00% |
| old_sft | retail | 4.38% | 10.00% | 2.50% |
| old_sft | telecom | 5.62% | 20.00% | 0.00% |
| current_sft | airline | 27.50% | 40.00% | 20.00% |
| current_sft | retail | 27.50% | 50.00% | 10.00% |
| current_sft | telecom | 16.88% | 37.50% | 0.00% |
| contract-only | airline | 30.00% | 45.00% | 15.00% |
| contract-only | retail | 32.50% | 60.00% | 17.50% |
| contract-only | telecom | 18.12% | 42.50% | 2.50% |
| contract-boundary | airline | 32.50% | 45.00% | 20.00% |
| contract-boundary | retail | 24.38% | 37.50% | 10.00% |
| contract-boundary | telecom | 18.75% | 45.00% | 0.00% |

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---:|---:|---:|
| contract-only-raw_instruct | pass@1 / pass^1 | +3.50 pp | [-1.25, +8.50] pp | no |
| contract-only-raw_instruct | pass@4(any) | +1.00 pp | [-8.00, +10.00] pp | no |
| contract-only-raw_instruct | pass^4 | +6.00 pp | [+0.00, +12.00] pp | no |
| contract-only-old_sft | pass@1 / pass^1 | +16.75 pp | [+10.50, +23.50] pp | yes |
| contract-only-old_sft | pass@4(any) | +29.00 pp | [+18.00, +40.00] pp | yes |
| contract-only-old_sft | pass^4 | +8.00 pp | [+2.00, +15.00] pp | yes |
| contract-only-current_sft | pass@1 / pass^1 | +3.00 pp | [-1.25, +7.00] pp | no |
| contract-only-current_sft | pass@4(any) | +7.00 pp | [-2.00, +16.00] pp | no |
| contract-only-current_sft | pass^4 | +3.00 pp | [-1.00, +8.00] pp | no |
| contract-only-contract-boundary | pass@1 / pass^1 | +2.50 pp | [-1.50, +6.75] pp | no |
| contract-only-contract-boundary | pass@4(any) | +8.00 pp | [-1.00, +17.00] pp | no |
| contract-only-contract-boundary | pass^4 | +3.00 pp | [-2.00, +8.00] pp | no |

## Trajectory tool-mode diagnostic

A trajectory is `multi_exposed` when at least one assistant turn contains more than one tool call. These success rates are observational and are not causal effects of batching.

| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |
|---|---|---:|---:|---:|---:|
| raw_instruct | multi_exposed | 1 | 0.00% | 0 | 0 |
| raw_instruct | non_multi_exposed | 399 | 22.81% | 6 | 11 |
| old_sft | multi_exposed | 88 | 22.73% | 0 | 1 |
| old_sft | non_multi_exposed | 312 | 5.77% | 4 | 58 |
| current_sft | multi_exposed | 242 | 22.73% | 4 | 3 |
| current_sft | non_multi_exposed | 158 | 24.05% | 1 | 7 |
| contract-only | multi_exposed | 338 | 24.26% | 46 | 9 |
| contract-only | non_multi_exposed | 62 | 37.10% | 8 | 2 |
| contract-boundary | multi_exposed | 294 | 23.13% | 3 | 24 |
| contract-boundary | non_multi_exposed | 106 | 25.47% | 3 | 17 |
