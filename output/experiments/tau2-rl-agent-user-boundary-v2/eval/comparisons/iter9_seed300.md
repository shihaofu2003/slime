# tau2 official evaluation comparison

## Protocol

Profile: `agent-owned-dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| selected_sft | 100 | 23.75% | 42.00% | 8.00% |
| rl_iter9 | 100 | 26.25% | 52.00% | 10.00% |

## Domain results

| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| selected_sft | airline | 32.50% | 45.00% | 20.00% |
| selected_sft | retail | 24.38% | 37.50% | 10.00% |
| selected_sft | telecom | 18.75% | 45.00% | 0.00% |
| rl_iter9 | airline | 33.75% | 45.00% | 30.00% |
| rl_iter9 | retail | 26.88% | 52.50% | 7.50% |
| rl_iter9 | telecom | 21.88% | 55.00% | 2.50% |

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---:|---:|---:|
| rl_iter9-selected_sft | pass@1 / pass^1 | +2.50 pp | [-2.50, +7.50] pp | no |
| rl_iter9-selected_sft | pass@4(any) | +10.00 pp | [-1.00, +21.00] pp | no |
| rl_iter9-selected_sft | pass^4 | +2.00 pp | [-2.00, +6.00] pp | no |

## Trajectory tool-mode diagnostic

A trajectory is `multi_exposed` when at least one assistant turn contains more than one tool call. These success rates are observational and are not causal effects of batching.

| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |
|---|---|---:|---:|---:|---:|
| selected_sft | multi_exposed | 294 | 23.13% | 3 | 24 |
| selected_sft | non_multi_exposed | 106 | 25.47% | 3 | 17 |
| rl_iter9 | multi_exposed | 301 | 27.24% | 2 | 19 |
| rl_iter9 | non_multi_exposed | 99 | 23.23% | 1 | 20 |
