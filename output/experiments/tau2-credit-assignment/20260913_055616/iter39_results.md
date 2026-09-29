# Progress RTG iter39 official evaluation

Processed SFT init; fixed Qwen3.6 User and official binary success. Training seed 1234.

Training job 18158 (original recipe); evaluation job 18245, seed300, 100 test tasks × 4 trials,
official-native, Agent temperature 0.6, max_steps 200. All 400 trials completed with no infrastructure
errors; 9 reached max_steps (all Telecom). Source: [official summary](eval/progress-rtg-v1-iter39/seed300/summary.json).

| Model / domain | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---:|---:|---:|---:|---:|
| progress / overall | 56.50% | 81.00% | 28.00% | 76.26% | 48.19% |
| progress / airline | 46.25% | 65.00% | 30.00% | 56.14% | 46.25% |
| progress / retail | 66.25% | 87.50% | 40.00% | 89.73% | 68.55% |
| progress / telecom | 51.88% | 82.50% | 15.00% | 67.15% | 27.21% |
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
| vanilla | overall | pass@1 | -0.75 | [-6.50, +5.00] |
| vanilla | overall | pass@4(any) | +0.00 | [-7.00, +7.00] |
| vanilla | overall | pass^4 | -1.00 | [-10.00, +7.00] |
| vanilla | airline | pass@1 | -2.50 | [-16.25, +11.25] |
| vanilla | airline | pass@4(any) | +0.00 | [-20.00, +20.00] |
| vanilla | airline | pass^4 | +5.00 | [-10.00, +20.00] |
| vanilla | retail | pass@1 | +3.12 | [-5.00, +10.62] |
| vanilla | retail | pass@4(any) | +2.50 | [-5.00, +10.00] |
| vanilla | retail | pass^4 | +2.50 | [-12.50, +17.50] |
| vanilla | telecom | pass@1 | -3.75 | [-13.12, +5.62] |
| vanilla | telecom | pass@4(any) | -2.50 | [-12.50, +7.50] |
| vanilla | telecom | pass^4 | -7.50 | [-20.00, +5.00] |
| sft | overall | pass@1 | +0.25 | [-5.50, +6.00] |
| sft | overall | pass@4(any) | +0.00 | [-7.00, +7.00] |
| sft | overall | pass^4 | -2.00 | [-11.00, +7.00] |
| sft | airline | pass@1 | -5.00 | [-16.25, +6.25] |
| sft | airline | pass@4(any) | +0.00 | [-15.00, +15.00] |
| sft | airline | pass^4 | -5.00 | [-20.00, +10.00] |
| sft | retail | pass@1 | +8.75 | [-1.25, +19.38] |
| sft | retail | pass@4(any) | +5.00 | [-5.00, +15.00] |
| sft | retail | pass^4 | +12.50 | [-5.00, +30.00] |
| sft | telecom | pass@1 | -5.62 | [-13.12, +1.88] |
| sft | telecom | pass@4(any) | -5.00 | [-17.50, +7.50] |
| sft | telecom | pass^4 | -15.00 | [-27.50, -5.00] |

## Decision

Stop at iter39; the predeclared trend criterion was not met.
Criterion: pass@1 > 57.25% and pass^4 ≥ 29.00%. Passing only supports continuing the experiment.
Intervals containing zero do not establish improvement. Conclusions are limited to this one training seed.
