# Progress RTG iter39 official evaluation

Processed SFT init; fixed Qwen3.6 User and official binary success. Training seed 1234.

| Model / domain | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---:|---:|---:|---:|---:|
| progress / overall | 52.50% | 83.00% | 23.00% | 75.96% | 45.90% |
| progress / airline | 46.25% | 70.00% | 25.00% | 53.07% | 46.25% |
| progress / retail | 60.00% | 82.50% | 35.00% | 86.25% | 64.10% |
| progress / telecom | 48.12% | 90.00% | 10.00% | 72.62% | 27.27% |
| vanilla / overall | 57.25% | 81.00% | 29.00% | 74.05% | 46.41% |
| vanilla / airline | 48.75% | 65.00% | 25.00% | 53.51% | 48.75% |
| vanilla / retail | 63.12% | 85.00% | 37.50% | 86.16% | 66.03% |
| vanilla / telecom | 55.62% | 85.00% | 22.50% | 67.42% | 25.32% |
| sft / overall | 56.25% | 81.00% | 30.00% | 74.27% | 43.70% |
| sft / airline | 51.25% | 65.00% | 35.00% | 51.75% | 51.25% |
| sft / retail | 57.50% | 82.50% | 27.50% | 88.01% | 62.34% |
| sft / telecom | 57.50% | 87.50% | 30.00% | 66.72% | 21.29% |

## Paired task bootstrap

100,000 task resamples; evaluation seeds averaged within each task before resampling.

| Progress minus control | Domain | Metric | Delta (pp) | 95% CI (pp) |
|---|---|---|---:|---:|
| vanilla | overall | pass@1 | -4.75 | [-10.25, +0.75] |
| vanilla | overall | pass@4(any) | +2.00 | [-5.00, +9.00] |
| vanilla | overall | pass^4 | -6.00 | [-14.00, +2.00] |
| vanilla | airline | pass@1 | -2.50 | [-12.50, +7.50] |
| vanilla | airline | pass@4(any) | +5.00 | [-10.00, +20.00] |
| vanilla | airline | pass^4 | +0.00 | [+0.00, +0.00] |
| vanilla | retail | pass@1 | -3.12 | [-10.00, +4.38] |
| vanilla | retail | pass@4(any) | -2.50 | [-12.50, +5.00] |
| vanilla | retail | pass^4 | -2.50 | [-17.50, +12.50] |
| vanilla | telecom | pass@1 | -7.50 | [-18.12, +2.50] |
| vanilla | telecom | pass@4(any) | +5.00 | [-7.50, +20.00] |
| vanilla | telecom | pass^4 | -12.50 | [-25.00, +0.00] |
| sft | overall | pass@1 | -3.75 | [-10.25, +2.75] |
| sft | overall | pass@4(any) | +2.00 | [-5.00, +9.00] |
| sft | overall | pass^4 | -7.00 | [-16.00, +2.00] |
| sft | airline | pass@1 | -5.00 | [-15.00, +6.25] |
| sft | airline | pass@4(any) | +5.00 | [+0.00, +15.00] |
| sft | airline | pass^4 | -10.00 | [-25.00, +0.00] |
| sft | retail | pass@1 | +2.50 | [-8.75, +13.12] |
| sft | retail | pass@4(any) | +0.00 | [-12.50, +12.50] |
| sft | retail | pass^4 | +7.50 | [-7.50, +22.50] |
| sft | telecom | pass@1 | -9.38 | [-19.38, +1.25] |
| sft | telecom | pass@4(any) | +2.50 | [-10.00, +15.00] |
| sft | telecom | pass^4 | -20.00 | [-35.00, -7.50] |

## Decision

Stop at iter39; the predeclared trend criterion was not met.
Criterion: pass@1 > 57.25% and pass^4 ≥ 29.00%. Passing only supports continuing the experiment.
Intervals containing zero do not establish improvement. Conclusions are limited to this one training seed.
