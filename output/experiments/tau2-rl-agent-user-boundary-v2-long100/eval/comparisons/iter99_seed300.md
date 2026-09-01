# tau2 official evaluation comparison

## Protocol

Profile: `agent-owned-dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| selected_sft | 100 | 23.75% | 42.00% | 8.00% |
| historical_rl_iter9 | 100 | 26.25% | 52.00% | 10.00% |
| long100_iter99 | 100 | 27.50% | 53.00% | 9.00% |

## Domain results

| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| selected_sft | airline | 32.50% | 45.00% | 20.00% |
| selected_sft | retail | 24.38% | 37.50% | 10.00% |
| selected_sft | telecom | 18.75% | 45.00% | 0.00% |
| historical_rl_iter9 | airline | 33.75% | 45.00% | 30.00% |
| historical_rl_iter9 | retail | 26.88% | 52.50% | 7.50% |
| historical_rl_iter9 | telecom | 21.88% | 55.00% | 2.50% |
| long100_iter99 | airline | 35.00% | 65.00% | 15.00% |
| long100_iter99 | retail | 30.00% | 47.50% | 12.50% |
| long100_iter99 | telecom | 21.25% | 52.50% | 2.50% |

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Scope | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---|---:|---:|---:|
| long100_iter99-selected_sft | overall | pass@1 / pass^1 | +3.75 pp | [-1.00, +8.50] pp | no |
| long100_iter99-selected_sft | overall | pass@4(any) | +11.00 pp | [+2.00, +20.00] pp | yes |
| long100_iter99-selected_sft | overall | pass^4 | +1.00 pp | [-3.00, +5.00] pp | no |
| long100_iter99-selected_sft | airline | pass@1 / pass^1 | +2.50 pp | [-6.25, +10.00] pp | no |
| long100_iter99-selected_sft | airline | pass@4(any) | +20.00 pp | [+5.00, +40.00] pp | yes |
| long100_iter99-selected_sft | airline | pass^4 | -5.00 pp | [-20.00, +10.00] pp | no |
| long100_iter99-selected_sft | retail | pass@1 / pass^1 | +5.62 pp | [-1.88, +13.75] pp | no |
| long100_iter99-selected_sft | retail | pass@4(any) | +10.00 pp | [-2.50, +22.50] pp | no |
| long100_iter99-selected_sft | retail | pass^4 | +2.50 pp | [+0.00, +7.50] pp | no |
| long100_iter99-selected_sft | telecom | pass@1 / pass^1 | +2.50 pp | [-5.62, +10.00] pp | no |
| long100_iter99-selected_sft | telecom | pass@4(any) | +7.50 pp | [-7.50, +22.50] pp | no |
| long100_iter99-selected_sft | telecom | pass^4 | +2.50 pp | [+0.00, +7.50] pp | no |
| long100_iter99-historical_rl_iter9 | overall | pass@1 / pass^1 | +1.25 pp | [-3.50, +6.00] pp | no |
| long100_iter99-historical_rl_iter9 | overall | pass@4(any) | +1.00 pp | [-9.00, +11.00] pp | no |
| long100_iter99-historical_rl_iter9 | overall | pass^4 | -1.00 pp | [-5.00, +3.00] pp | no |
| long100_iter99-historical_rl_iter9 | airline | pass@1 / pass^1 | +1.25 pp | [-7.50, +10.00] pp | no |
| long100_iter99-historical_rl_iter9 | airline | pass@4(any) | +20.00 pp | [+5.00, +40.00] pp | yes |
| long100_iter99-historical_rl_iter9 | airline | pass^4 | -15.00 pp | [-30.00, +0.00] pp | no |
| long100_iter99-historical_rl_iter9 | retail | pass@1 / pass^1 | +3.12 pp | [-4.38, +10.62] pp | no |
| long100_iter99-historical_rl_iter9 | retail | pass@4(any) | -5.00 pp | [-20.00, +7.50] pp | no |
| long100_iter99-historical_rl_iter9 | retail | pass^4 | +5.00 pp | [+0.00, +12.50] pp | no |
| long100_iter99-historical_rl_iter9 | telecom | pass@1 / pass^1 | -0.62 pp | [-8.75, +7.50] pp | no |
| long100_iter99-historical_rl_iter9 | telecom | pass@4(any) | -2.50 pp | [-22.50, +17.50] pp | no |
| long100_iter99-historical_rl_iter9 | telecom | pass^4 | +0.00 pp | [+0.00, +0.00] pp | no |

## Trajectory tool-mode diagnostic

A trajectory is `multi_exposed` when at least one assistant turn contains more than one tool call. These success rates are observational and are not causal effects of batching.

| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |
|---|---|---:|---:|---:|---:|
| selected_sft | multi_exposed | 294 | 23.13% | 3 | 24 |
| selected_sft | non_multi_exposed | 106 | 25.47% | 3 | 17 |
| historical_rl_iter9 | multi_exposed | 301 | 27.24% | 2 | 19 |
| historical_rl_iter9 | non_multi_exposed | 99 | 23.23% | 1 | 20 |
| long100_iter99 | multi_exposed | 307 | 28.01% | 1 | 20 |
| long100_iter99 | non_multi_exposed | 93 | 25.81% | 1 | 15 |
