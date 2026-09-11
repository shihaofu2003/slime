# tau2 official evaluation comparison

## Protocol

Profile: `agent-owned-dependency-safe-multi`; paired bootstrap unit: task; samples: 100,000.

## Overall results

| Model | Tasks | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| selected_sft | 100 | 23.50% | 45.00% | 9.00% |
| long100_iter99 | 100 | 31.50% | 60.00% | 12.00% |
| long200_iter199_kl_waiver | 100 | 33.75% | 61.00% | 12.00% |

## Domain results

| Model | Domain | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| selected_sft | airline | 30.00% | 45.00% | 20.00% |
| selected_sft | retail | 26.88% | 45.00% | 12.50% |
| selected_sft | telecom | 16.88% | 45.00% | 0.00% |
| long100_iter99 | airline | 41.25% | 60.00% | 25.00% |
| long100_iter99 | retail | 32.50% | 57.50% | 15.00% |
| long100_iter99 | telecom | 25.62% | 62.50% | 2.50% |
| long200_iter199_kl_waiver | airline | 33.75% | 55.00% | 15.00% |
| long200_iter199_kl_waiver | retail | 34.38% | 60.00% | 17.50% |
| long200_iter199_kl_waiver | telecom | 33.12% | 65.00% | 5.00% |

## Paired deltas

Positive values favor the candidate. Intervals resample the 100 matched tasks.

| Comparison | Scope | Metric | Delta | 95% paired bootstrap CI | Excludes 0 |
|---|---|---|---:|---:|---:|
| long200_iter199_kl_waiver-selected_sft | overall | pass@1 / pass^1 | +10.25 pp | [+4.50, +16.25] pp | yes |
| long200_iter199_kl_waiver-selected_sft | overall | pass@4(any) | +16.00 pp | [+5.00, +27.00] pp | yes |
| long200_iter199_kl_waiver-selected_sft | overall | pass^4 | +3.00 pp | [-3.00, +10.00] pp | no |
| long200_iter199_kl_waiver-selected_sft | airline | pass@1 / pass^1 | +3.75 pp | [-5.00, +16.25] pp | no |
| long200_iter199_kl_waiver-selected_sft | airline | pass@4(any) | +10.00 pp | [-10.00, +30.00] pp | no |
| long200_iter199_kl_waiver-selected_sft | airline | pass^4 | -5.00 pp | [-20.00, +10.00] pp | no |
| long200_iter199_kl_waiver-selected_sft | retail | pass@1 / pass^1 | +7.50 pp | [+0.62, +14.37] pp | yes |
| long200_iter199_kl_waiver-selected_sft | retail | pass@4(any) | +15.00 pp | [+2.50, +27.50] pp | yes |
| long200_iter199_kl_waiver-selected_sft | retail | pass^4 | +5.00 pp | [-7.50, +17.50] pp | no |
| long200_iter199_kl_waiver-selected_sft | telecom | pass@1 / pass^1 | +16.25 pp | [+4.38, +27.50] pp | yes |
| long200_iter199_kl_waiver-selected_sft | telecom | pass@4(any) | +20.00 pp | [-2.50, +42.50] pp | no |
| long200_iter199_kl_waiver-selected_sft | telecom | pass^4 | +5.00 pp | [+0.00, +12.50] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | overall | pass@1 / pass^1 | +2.25 pp | [-2.50, +7.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | overall | pass@4(any) | +1.00 pp | [-7.00, +9.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | overall | pass^4 | +0.00 pp | [-6.00, +6.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | airline | pass@1 / pass^1 | -7.50 pp | [-18.75, +2.50] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | airline | pass@4(any) | -5.00 pp | [-15.00, +0.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | airline | pass^4 | -10.00 pp | [-30.00, +10.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | retail | pass@1 / pass^1 | +1.88 pp | [-4.38, +8.75] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | retail | pass@4(any) | +2.50 pp | [-10.00, +15.00] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | retail | pass^4 | +2.50 pp | [-7.50, +12.50] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | telecom | pass@1 / pass^1 | +7.50 pp | [-0.62, +15.62] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | telecom | pass@4(any) | +2.50 pp | [-12.50, +17.50] pp | no |
| long200_iter199_kl_waiver-long100_iter99 | telecom | pass^4 | +2.50 pp | [+0.00, +7.50] pp | no |

## Trajectory tool-mode diagnostic

A trajectory is `multi_exposed` when at least one assistant turn contains more than one tool call. These success rates are observational and are not causal effects of batching.

| Model | Exposure | Simulations | Success rate | Too many errors | Max steps |
|---|---|---:|---:|---:|---:|
| selected_sft | multi_exposed | 294 | 22.45% | 9 | 11 |
| selected_sft | non_multi_exposed | 106 | 26.42% | 1 | 23 |
| long100_iter99 | multi_exposed | 323 | 32.51% | 4 | 23 |
| long100_iter99 | non_multi_exposed | 77 | 27.27% | 0 | 10 |
| long200_iter199_kl_waiver | multi_exposed | 371 | 33.15% | 1 | 37 |
| long200_iter199_kl_waiver | non_multi_exposed | 29 | 41.38% | 0 | 1 |
