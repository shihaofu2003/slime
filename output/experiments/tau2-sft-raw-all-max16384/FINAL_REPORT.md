# Raw-all SFT control report

## Seed-300 four-way comparison

| Agent | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---|---:|---:|---:|---:|---:|
| Raw Qwen3-4B-Instruct-2507 | 37.75% | 58.00% | 19.00% | 66.51% | 39.04% |
| Raw-all SFT final | 53.25% | 82.00% | 25.00% | 73.89% | 45.99% |
| Processed SFT final | 56.25% | 81.00% | 30.00% | 74.27% | 43.70% |
| Qwen3.5-4B thinking | 77.75% | 90.00% | 63.00% | 79.79% | 58.33% |

Raw Instruct and Qwen3.5-4B are three-domain subsets of existing four-domain official-native seed-300 runs; Qwen3.5 thinking used an 8,192-token output cap.

## Raw-all versus processed overall

| Seed | Agent | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | Raw-all | 53.25% | 82.00% | 25.00% | 73.89% | 45.99% |
| 300 | Processed | 56.25% | 81.00% | 30.00% | 74.27% | 43.70% |
| 301 | Raw-all | 53.00% | 83.00% | 21.00% | 75.07% | 46.07% |
| 301 | Processed | 55.25% | 81.00% | 26.00% | 75.25% | 44.53% |
| 300/301 | Raw-all mean | 53.12% | 82.50% | 23.00% | 74.48% | 46.03% |
| 300/301 | Processed mean | 55.75% | 81.00% | 28.00% | 74.76% | 44.12% |

## Processed minus raw-all

| Seed | Scope | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | overall | +3.00pp | -1.00pp | +5.00pp | +0.38pp | -2.29pp |
| 300 | airline | +0.00pp | +5.00pp | +0.00pp | -3.95pp | +0.00pp |
| 300 | retail | -3.12pp | -7.50pp | -5.00pp | +2.35pp | -2.44pp |
| 300 | telecom | +10.62pp | +2.50pp | +17.50pp | +1.33pp | +0.55pp |
| 301 | overall | +2.25pp | -2.00pp | +5.00pp | +0.18pp | -1.54pp |
| 301 | airline | +3.75pp | +5.00pp | +0.00pp | -7.10pp | +3.21pp |
| 301 | retail | -5.00pp | -12.50pp | +17.50pp | +1.79pp | -5.12pp |
| 301 | telecom | +8.75pp | +5.00pp | -5.00pp | +2.22pp | +3.04pp |
| mean | overall | +2.62pp | -1.50pp | +5.00pp | +0.28pp | -1.91pp |
| mean | airline | +1.87pp | +5.00pp | +0.00pp | -5.53pp | +1.61pp |
| mean | retail | -4.06pp | -10.00pp | +6.25pp | +2.07pp | -3.78pp |
| mean | telecom | +9.69pp | +3.75pp | +6.25pp | +1.77pp | +1.79pp |

## Domain metrics

| Seed | Agent | Domain | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---|---:|---:|---:|---:|---:|
| 300 | Raw-all | airline | 51.25% | 60.00% | 35.00% | 55.70% | 51.25% |
| 300 | Raw-all | retail | 60.62% | 90.00% | 32.50% | 85.65% | 64.78% |
| 300 | Raw-all | telecom | 46.88% | 85.00% | 12.50% | 65.39% | 20.74% |
| 300 | Processed | airline | 51.25% | 65.00% | 35.00% | 51.75% | 51.25% |
| 300 | Processed | retail | 57.50% | 82.50% | 27.50% | 88.01% | 62.34% |
| 300 | Processed | telecom | 57.50% | 87.50% | 30.00% | 66.72% | 21.29% |
| 301 | Raw-all | airline | 42.50% | 65.00% | 30.00% | 54.91% | 43.04% |
| 301 | Raw-all | retail | 63.75% | 92.50% | 22.50% | 85.84% | 68.99% |
| 301 | Raw-all | telecom | 47.50% | 82.50% | 15.00% | 68.83% | 20.45% |
| 301 | Processed | airline | 46.25% | 70.00% | 30.00% | 47.81% | 46.25% |
| 301 | Processed | retail | 58.75% | 80.00% | 40.00% | 87.63% | 63.87% |
| 301 | Processed | telecom | 56.25% | 87.50% | 10.00% | 71.04% | 23.49% |

## Behavior and runtime

| Seed | Agent | Scope | Terminations | nonexistent | invalid args | tool errors | retries | wall |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 300 | Raw-all | overall | max_steps:10, too_many_errors:16, user_stop:374 | 0 | 14 | 358 | 0 | 22.39 min |
| 300 | Raw-all | airline | user_stop:80 | 0 | 11 | 35 | 0 | 12.35 min |
| 300 | Raw-all | retail | too_many_errors:1, user_stop:159 | 0 | 3 | 109 | 0 | 19.46 min |
| 300 | Raw-all | telecom | max_steps:10, too_many_errors:15, user_stop:135 | 0 | 0 | 214 | 0 | 22.21 min |
| 300 | Processed | overall | max_steps:3, too_many_errors:8, user_stop:389 | 0 | 6 | 294 | 0 | 21.36 min |
| 300 | Processed | airline | user_stop:80 | 0 | 6 | 40 | 0 | 13.82 min |
| 300 | Processed | retail | too_many_errors:6, user_stop:154 | 0 | 0 | 121 | 0 | 19.70 min |
| 300 | Processed | telecom | max_steps:3, too_many_errors:2, user_stop:155 | 0 | 0 | 133 | 0 | 21.28 min |
| 301 | Raw-all | overall | max_steps:12, too_many_errors:19, user_stop:369 | 0 | 17 | 418 | 0 | 24.07 min |
| 301 | Raw-all | airline | too_many_errors:1, user_stop:79 | 0 | 16 | 39 | 0 | 12.63 min |
| 301 | Raw-all | retail | max_steps:1, too_many_errors:1, user_stop:158 | 0 | 1 | 112 | 0 | 21.39 min |
| 301 | Raw-all | telecom | max_steps:11, too_many_errors:17, user_stop:132 | 0 | 0 | 267 | 0 | 23.73 min |
| 301 | Processed | overall | max_steps:6, too_many_errors:10, user_stop:384 | 0 | 1 | 270 | 0 | 25.28 min |
| 301 | Processed | airline | user_stop:80 | 0 | 0 | 11 | 0 | 13.69 min |
| 301 | Processed | retail | too_many_errors:5, user_stop:155 | 0 | 1 | 132 | 0 | 20.70 min |
| 301 | Processed | telecom | max_steps:6, too_many_errors:5, user_stop:149 | 0 | 0 | 127 | 0 | 25.19 min |

## Conclusion

Processed SFT minus raw-all SFT on the two-seed mean is `+2.62pp/-1.50pp/+5.00pp` for pass@1/pass@4(any)/pass^4, with action/DB accuracy deltas of `+0.28pp/-1.91pp`.

The [quality audit](QUALITY_AUDIT.md) adds data-level structural counts and
100,000-draw task-paired intervals. Overall pass@1 is
`+2.62pp [-2.12,+7.38]` and pass^4 is `+5.00pp [-1.50,+11.00]`; Telecom
pass@1 is `+9.69pp [+2.81,+17.19]`.
