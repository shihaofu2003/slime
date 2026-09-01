# tau2 official evaluation comparison

## Protocol

Profile: `agent-owned-dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| selected_sft | 100 | 23.50% | 45.00% | 9.00% |
| historical_rl_iter9 | 100 | 24.25% | 48.00% | 8.00% |
| long100_iter99 | 100 | 31.50% | 60.00% | 12.00% |

## Domain results

| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| selected_sft | airline | 30.00% | 45.00% | 20.00% |
| selected_sft | retail | 26.88% | 45.00% | 12.50% |
| selected_sft | telecom | 16.88% | 45.00% | 0.00% |
| historical_rl_iter9 | airline | 31.25% | 50.00% | 10.00% |
| historical_rl_iter9 | retail | 26.25% | 47.50% | 10.00% |
| historical_rl_iter9 | telecom | 18.75% | 47.50% | 5.00% |
| long100_iter99 | airline | 41.25% | 60.00% | 25.00% |
| long100_iter99 | retail | 32.50% | 57.50% | 15.00% |
| long100_iter99 | telecom | 25.62% | 62.50% | 2.50% |

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Scope | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---|---:|---:|---:|
| long100_iter99-selected_sft | overall | pass@1 / pass^1 | +8.00 pp | [+2.50, +13.50] pp | yes |
| long100_iter99-selected_sft | overall | pass@4(any) | +15.00 pp | [+4.00, +26.00] pp | yes |
| long100_iter99-selected_sft | overall | pass^4 | +3.00 pp | [-2.00, +8.00] pp | no |
| long100_iter99-selected_sft | airline | pass@1 / pass^1 | +11.25 pp | [+2.50, +23.75] pp | yes |
| long100_iter99-selected_sft | airline | pass@4(any) | +15.00 pp | [+0.00, +30.00] pp | no |
| long100_iter99-selected_sft | airline | pass^4 | +5.00 pp | [+0.00, +15.00] pp | no |
| long100_iter99-selected_sft | retail | pass@1 / pass^1 | +5.62 pp | [-2.50, +14.37] pp | no |
| long100_iter99-selected_sft | retail | pass@4(any) | +12.50 pp | [-5.00, +30.00] pp | no |
| long100_iter99-selected_sft | retail | pass^4 | +2.50 pp | [-7.50, +12.50] pp | no |
| long100_iter99-selected_sft | telecom | pass@1 / pass^1 | +8.75 pp | [-1.25, +18.12] pp | no |
| long100_iter99-selected_sft | telecom | pass@4(any) | +17.50 pp | [-2.50, +37.50] pp | no |
| long100_iter99-selected_sft | telecom | pass^4 | +2.50 pp | [+0.00, +7.50] pp | no |
| long100_iter99-historical_rl_iter9 | overall | pass@1 / pass^1 | +7.25 pp | [+2.50, +12.00] pp | yes |
| long100_iter99-historical_rl_iter9 | overall | pass@4(any) | +12.00 pp | [+2.00, +22.00] pp | yes |
| long100_iter99-historical_rl_iter9 | overall | pass^4 | +4.00 pp | [-1.00, +10.00] pp | no |
| long100_iter99-historical_rl_iter9 | airline | pass@1 / pass^1 | +10.00 pp | [+0.00, +21.25] pp | no |
| long100_iter99-historical_rl_iter9 | airline | pass@4(any) | +10.00 pp | [+0.00, +25.00] pp | no |
| long100_iter99-historical_rl_iter9 | airline | pass^4 | +15.00 pp | [+0.00, +30.00] pp | no |
| long100_iter99-historical_rl_iter9 | retail | pass@1 / pass^1 | +6.25 pp | [+1.25, +11.88] pp | yes |
| long100_iter99-historical_rl_iter9 | retail | pass@4(any) | +10.00 pp | [-2.50, +22.50] pp | no |
| long100_iter99-historical_rl_iter9 | retail | pass^4 | +5.00 pp | [+0.00, +12.50] pp | no |
| long100_iter99-historical_rl_iter9 | telecom | pass@1 / pass^1 | +6.88 pp | [-2.50, +15.62] pp | no |
| long100_iter99-historical_rl_iter9 | telecom | pass@4(any) | +15.00 pp | [-5.00, +35.00] pp | no |
| long100_iter99-historical_rl_iter9 | telecom | pass^4 | -2.50 pp | [-10.00, +5.00] pp | no |

## Trajectory tool-mode diagnostic

A trajectory is `multi_exposed` when at least one assistant turn contains more than one tool call. These success rates are observational and are not causal effects of batching.

| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |
|---|---|---:|---:|---:|---:|
| selected_sft | multi_exposed | 294 | 22.45% | 9 | 11 |
| selected_sft | non_multi_exposed | 106 | 26.42% | 1 | 23 |
| historical_rl_iter9 | multi_exposed | 299 | 22.74% | 3 | 26 |
| historical_rl_iter9 | non_multi_exposed | 101 | 28.71% | 2 | 19 |
| long100_iter99 | multi_exposed | 323 | 32.51% | 4 | 23 |
| long100_iter99 | non_multi_exposed | 77 | 27.27% | 0 | 10 |
