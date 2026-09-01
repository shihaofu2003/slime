# tau2 official evaluation comparison

## Protocol

Profile: `agent-owned-dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| selected_sft | 100 | 23.75% | 42.00% | 8.00% |
| long100_iter99 | 100 | 27.50% | 53.00% | 9.00% |
| long200_iter199_kl_waiver | 100 | 31.00% | 55.00% | 11.00% |

## Domain results

| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| selected_sft | airline | 32.50% | 45.00% | 20.00% |
| selected_sft | retail | 24.38% | 37.50% | 10.00% |
| selected_sft | telecom | 18.75% | 45.00% | 0.00% |
| long100_iter99 | airline | 35.00% | 65.00% | 15.00% |
| long100_iter99 | retail | 30.00% | 47.50% | 12.50% |
| long100_iter99 | telecom | 21.25% | 52.50% | 2.50% |
| long200_iter199_kl_waiver | airline | 32.50% | 50.00% | 15.00% |
| long200_iter199_kl_waiver | retail | 30.00% | 45.00% | 17.50% |
| long200_iter199_kl_waiver | telecom | 31.25% | 67.50% | 2.50% |

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Scope | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---|---:|---:|---:|
| long200_iter199_kl_waiver-selected_sft | overall | pass@1 / pass^1 | +7.25 pp | [+2.25, +12.25] pp | yes |
| long200_iter199_kl_waiver-selected_sft | overall | pass@4(any) | +13.00 pp | [+3.00, +23.00] pp | yes |
| long200_iter199_kl_waiver-selected_sft | overall | pass^4 | +3.00 pp | [-3.00, +9.00] pp | no |
| long200_iter199_kl_waiver-selected_sft | airline | pass@1 / pass^1 | +0.00 pp | [-7.50, +7.50] pp | no |
| long200_iter199_kl_waiver-selected_sft | airline | pass@4(any) | +5.00 pp | [-10.00, +20.00] pp | no |
| long200_iter199_kl_waiver-selected_sft | airline | pass^4 | -5.00 pp | [-25.00, +15.00] pp | no |
| long200_iter199_kl_waiver-selected_sft | retail | pass@1 / pass^1 | +5.62 pp | [-2.50, +14.37] pp | no |
| long200_iter199_kl_waiver-selected_sft | retail | pass@4(any) | +7.50 pp | [-7.50, +22.50] pp | no |
| long200_iter199_kl_waiver-selected_sft | retail | pass^4 | +7.50 pp | [+0.00, +17.50] pp | no |
| long200_iter199_kl_waiver-selected_sft | telecom | pass@1 / pass^1 | +12.50 pp | [+4.38, +20.62] pp | yes |
| long200_iter199_kl_waiver-selected_sft | telecom | pass@4(any) | +22.50 pp | [+2.50, +42.50] pp | yes |
| long200_iter199_kl_waiver-selected_sft | telecom | pass^4 | +2.50 pp | [+0.00, +7.50] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | overall | pass@1 / pass^1 | +3.50 pp | [-1.25, +8.50] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | overall | pass@4(any) | +2.00 pp | [-8.00, +12.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | overall | pass^4 | +2.00 pp | [-3.00, +8.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | airline | pass@1 / pass^1 | -2.50 pp | [-11.25, +6.25] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | airline | pass@4(any) | -15.00 pp | [-30.00, +0.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | airline | pass^4 | +0.00 pp | [-20.00, +20.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | retail | pass@1 / pass^1 | +0.00 pp | [-6.25, +6.25] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | retail | pass@4(any) | -2.50 pp | [-15.00, +10.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | retail | pass^4 | +5.00 pp | [+0.00, +12.50] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | telecom | pass@1 / pass^1 | +10.00 pp | [+1.25, +19.38] pp | yes |
| long200_iter199_kl_waiver-long100_iter99 | telecom | pass@4(any) | +15.00 pp | [-2.50, +32.50] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | telecom | pass^4 | +0.00 pp | [-7.50, +7.50] pp | no |

## Trajectory tool-mode diagnostic

A trajectory is `multi_exposed` when at least one assistant turn contains more than one tool call. These success rates are observational and are not causal effects of batching.

| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |
|---|---|---:|---:|---:|---:|
| selected_sft | multi_exposed | 294 | 23.13% | 3 | 24 |
| selected_sft | non_multi_exposed | 106 | 25.47% | 3 | 17 |
| long100_iter99 | multi_exposed | 307 | 28.01% | 1 | 20 |
| long100_iter99 | non_multi_exposed | 93 | 25.81% | 1 | 15 |
| long200_iter199_kl_waiver | multi_exposed | 376 | 30.59% | 2 | 37 |
| long200_iter199_kl_waiver | non_multi_exposed | 24 | 37.50% | 0 | 2 |
