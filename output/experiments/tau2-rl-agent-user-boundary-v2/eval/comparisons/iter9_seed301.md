# tau2 official evaluation comparison

## Protocol

Profile: `agent-owned-dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| selected_sft | 100 | 23.50% | 45.00% | 9.00% |
| rl_iter9 | 100 | 24.25% | 48.00% | 8.00% |

## Domain results

| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| selected_sft | airline | 30.00% | 45.00% | 20.00% |
| selected_sft | retail | 26.88% | 45.00% | 12.50% |
| selected_sft | telecom | 16.88% | 45.00% | 0.00% |
| rl_iter9 | airline | 31.25% | 50.00% | 10.00% |
| rl_iter9 | retail | 26.25% | 47.50% | 10.00% |
| rl_iter9 | telecom | 18.75% | 47.50% | 5.00% |

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---:|---:|---:|
| rl_iter9-selected_sft | pass@1 / pass^1 | +0.75 pp | [-3.75, +5.25] pp | no |
| rl_iter9-selected_sft | pass@4(any) | +3.00 pp | [-6.00, +12.00] pp | no |
| rl_iter9-selected_sft | pass^4 | -1.00 pp | [-7.00, +5.00] pp | no |

## Trajectory tool-mode diagnostic

A trajectory is `multi_exposed` when at least one assistant turn contains more than one tool call. These success rates are observational and are not causal effects of batching.

| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |
|---|---|---:|---:|---:|---:|
| selected_sft | multi_exposed | 294 | 22.45% | 9 | 11 |
| selected_sft | non_multi_exposed | 106 | 26.42% | 1 | 23 |
| rl_iter9 | multi_exposed | 299 | 22.74% | 3 | 26 |
| rl_iter9 | non_multi_exposed | 101 | 28.71% | 2 | 19 |
