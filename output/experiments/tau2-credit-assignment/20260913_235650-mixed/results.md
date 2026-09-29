# Mixed-domain GRPO comparison

40 updates; seed300; 100 tasks × 4 trials.

| Model / domain | pass@1 | pass@4(any) | pass^4 | Action | DB |
|---|---:|---:|---:|---:|---:|
| vanilla-grpo / overall | 55.00% | 77.00% | 29.00% | 75.91% | 41.42% |
| vanilla-grpo / airline | 41.25% | 60.00% | 20.00% | 59.65% | 41.25% |
| vanilla-grpo / retail | 61.25% | 82.50% | 40.00% | 89.38% | 65.81% |
| vanilla-grpo / telecom | 55.62% | 80.00% | 22.50% | 65.30% | 15.28% |
| progress-rtg-v1 / overall | 54.75% | 83.00% | 23.00% | 74.09% | 45.50% |
| progress-rtg-v1 / airline | 51.25% | 75.00% | 30.00% | 51.54% | 55.70% |
| progress-rtg-v1 / retail | 61.25% | 87.50% | 30.00% | 86.78% | 64.15% |
| progress-rtg-v1 / telecom | 50.00% | 82.50% | 12.50% | 67.25% | 20.53% |
| sft / overall | 56.25% | 81.00% | 30.00% | 74.27% | 43.70% |
| sft / airline | 51.25% | 65.00% | 35.00% | 51.75% | 51.25% |
| sft / retail | 57.50% | 82.50% | 27.50% | 88.01% | 62.34% |
| sft / telecom | 57.50% | 87.50% | 30.00% | 66.72% | 21.29% |

## Paired bootstrap

100,000 task resamples; differences in percentage points.

| Credit minus control | Domain | Metric | Delta | 95% CI |
|---|---|---|---:|---|
| vanilla-grpo | overall | pass_at_1 | -0.25 | [-6.25, +5.76] |
| vanilla-grpo | overall | pass_at_4_any | +6.00 | [-2.00, +14.00] |
| vanilla-grpo | overall | pass_power_4 | -6.00 | [-16.00, +3.00] |
| vanilla-grpo | airline | pass_at_1 | +10.00 | [+0.00, +20.00] |
| vanilla-grpo | airline | pass_at_4_any | +15.00 | [+0.00, +30.00] |
| vanilla-grpo | airline | pass_power_4 | +10.00 | [-10.00, +30.00] |
| vanilla-grpo | retail | pass_at_1 | +0.00 | [-8.12, +8.75] |
| vanilla-grpo | retail | pass_at_4_any | +5.00 | [-7.50, +17.50] |
| vanilla-grpo | retail | pass_power_4 | -10.00 | [-25.00, +5.00] |
| vanilla-grpo | telecom | pass_at_1 | -5.62 | [-16.88, +5.62] |
| vanilla-grpo | telecom | pass_at_4_any | +2.50 | [-10.00, +15.00] |
| vanilla-grpo | telecom | pass_power_4 | -10.00 | [-25.00, +5.00] |
| sft | overall | pass_at_1 | -1.50 | [-7.75, +4.75] |
| sft | overall | pass_at_4_any | +2.00 | [-5.00, +9.00] |
| sft | overall | pass_power_4 | -7.00 | [-16.00, +2.00] |
| sft | airline | pass_at_1 | +0.00 | [-11.25, +12.50] |
| sft | airline | pass_at_4_any | +10.00 | [+0.00, +25.00] |
| sft | airline | pass_power_4 | -5.00 | [-20.00, +10.00] |
| sft | retail | pass_at_1 | +3.75 | [-6.25, +14.37] |
| sft | retail | pass_at_4_any | +5.00 | [-7.50, +17.50] |
| sft | retail | pass_power_4 | +2.50 | [-12.50, +17.50] |
| sft | telecom | pass_at_1 | -7.50 | [-17.50, +1.88] |
| sft | telecom | pass_at_4_any | -5.00 | [-17.50, +7.50] |
| sft | telecom | pass_power_4 | -17.50 | [-32.50, -5.00] |
