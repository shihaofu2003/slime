# official-native expanded SFT checkpoint evaluation

## Selection

Selected checkpoint: `iter_0003795`. Policy: seed300 pass@1, then pass^4, then earlier checkpoint.

## Seed 300 overall

| Checkpoint | Updates | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `iter_0000399` | 400 | 34.25% | 62.00% | 14.00% | 66.18% | 33.06% | 0.66 h |
| `iter_0000799` | 800 | 28.50% | 57.00% | 7.00% | 70.92% | 36.44% | 0.52 h |
| `iter_0001199` | 1200 | 48.00% | 78.00% | 15.00% | 73.63% | 36.76% | 0.45 h |
| `iter_0001599` | 1600 | 45.25% | 71.00% | 21.00% | 67.82% | 39.33% | 0.44 h |
| `iter_0001897` | 1898 | 48.00% | 78.00% | 18.00% | 72.28% | 40.72% | 0.41 h |
| `iter_0001999` | 2000 | 47.00% | 74.00% | 18.00% | 69.27% | 45.56% | 0.32 h |
| `iter_0002399` | 2400 | 51.50% | 84.00% | 19.00% | 74.39% | 44.27% | 0.41 h |
| `iter_0002799` | 2800 | 47.50% | 78.00% | 20.00% | 72.00% | 42.20% | 0.47 h |
| `iter_0003199` | 3200 | 53.75% | 79.00% | 27.00% | 72.52% | 43.93% | 0.38 h |
| `iter_0003599` | 3600 | 54.50% | 84.00% | 21.00% | 75.49% | 44.27% | 0.38 h |
| `iter_0003795` | 3796 | 56.25% | 81.00% | 30.00% | 74.27% | 43.70% | 0.36 h |

## Seed 300 domains

| Checkpoint | Domain | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---|---|---:|---:|---:|---:|---:|
| `iter_0000399` | airline | 40.00% | 60.00% | 30.00% | 44.74% | 40.00% |
| `iter_0000399` | retail | 38.75% | 70.00% | 12.50% | 76.45% | 40.51% |
| `iter_0000399` | telecom | 26.88% | 55.00% | 7.50% | 60.94% | 18.85% |
| `iter_0000799` | airline | 20.00% | 45.00% | 5.00% | 42.53% | 31.37% |
| `iter_0000799` | retail | 35.00% | 65.00% | 7.50% | 81.66% | 44.44% |
| `iter_0000799` | telecom | 26.25% | 55.00% | 7.50% | 56.93% | 25.71% |
| `iter_0001199` | airline | 37.50% | 60.00% | 15.00% | 50.68% | 38.46% |
| `iter_0001199` | retail | 52.50% | 80.00% | 25.00% | 85.03% | 55.06% |
| `iter_0001199` | telecom | 48.75% | 85.00% | 5.00% | 68.43% | 16.99% |
| `iter_0001599` | airline | 51.25% | 65.00% | 40.00% | 46.67% | 51.90% |
| `iter_0001599` | retail | 47.50% | 75.00% | 20.00% | 80.87% | 50.94% |
| `iter_0001599` | telecom | 40.00% | 70.00% | 12.50% | 59.82% | 20.53% |
| `iter_0001897` | airline | 40.00% | 60.00% | 20.00% | 57.46% | 40.00% |
| `iter_0001897` | retail | 57.50% | 87.50% | 25.00% | 85.73% | 60.90% |
| `iter_0001897` | telecom | 42.50% | 77.50% | 10.00% | 61.62% | 20.39% |
| `iter_0001999` | airline | 45.00% | 60.00% | 20.00% | 52.55% | 50.00% |
| `iter_0001999` | retail | 56.88% | 87.50% | 27.50% | 86.96% | 61.74% |
| `iter_0001999` | telecom | 38.12% | 67.50% | 7.50% | 47.52% | 24.22% |
| `iter_0002399` | airline | 45.00% | 70.00% | 25.00% | 57.89% | 45.00% |
| `iter_0002399` | retail | 60.62% | 87.50% | 30.00% | 87.01% | 63.69% |
| `iter_0002399` | telecom | 45.62% | 87.50% | 5.00% | 65.27% | 23.13% |
| `iter_0002799` | airline | 41.25% | 65.00% | 25.00% | 52.70% | 43.59% |
| `iter_0002799` | retail | 54.37% | 82.50% | 22.50% | 83.88% | 59.74% |
| `iter_0002799` | telecom | 43.75% | 80.00% | 15.00% | 64.63% | 22.14% |
| `iter_0003199` | airline | 53.75% | 65.00% | 45.00% | 54.26% | 54.43% |
| `iter_0003199` | retail | 57.50% | 82.50% | 27.50% | 86.99% | 62.99% |
| `iter_0003199` | telecom | 50.00% | 82.50% | 17.50% | 61.90% | 19.48% |
| `iter_0003599` | airline | 48.75% | 75.00% | 25.00% | 58.56% | 50.00% |
| `iter_0003599` | retail | 60.62% | 90.00% | 22.50% | 89.70% | 65.81% |
| `iter_0003599` | telecom | 51.25% | 82.50% | 17.50% | 64.54% | 19.21% |
| `iter_0003795` | airline | 51.25% | 65.00% | 35.00% | 51.75% | 51.25% |
| `iter_0003795` | retail | 57.50% | 82.50% | 27.50% | 88.01% | 62.34% |
| `iter_0003795` | telecom | 57.50% | 87.50% | 30.00% | 66.72% | 21.29% |

## Behavior diagnostics

| Checkpoint | Terminations | nonexistent | invalid args | tool errors | parser retries |
|---|---|---:|---:|---:|---:|
| `iter_0000399` | max_steps:34, too_many_errors:6, user_stop:360 | 0 | 5 | 276 | 0 |
| `iter_0000799` | max_steps:78, too_many_errors:75, user_stop:247 | 0 | 50 | 896 | 0 |
| `iter_0001199` | max_steps:8, too_many_errors:3, user_stop:389 | 0 | 8 | 245 | 0 |
| `iter_0001599` | max_steps:8, too_many_errors:3, user_stop:389 | 0 | 0 | 166 | 0 |
| `iter_0001897` | max_steps:6, too_many_errors:6, user_stop:388 | 0 | 25 | 300 | 0 |
| `iter_0001999` | max_steps:16, too_many_errors:35, user_stop:349 | 0 | 29 | 537 | 0 |
| `iter_0002399` | max_steps:8, too_many_errors:8, user_stop:384 | 0 | 10 | 252 | 0 |
| `iter_0002799` | max_steps:18, too_many_errors:10, user_stop:372 | 0 | 1 | 247 | 0 |
| `iter_0003199` | max_steps:4, too_many_errors:9, user_stop:387 | 0 | 2 | 282 | 0 |
| `iter_0003599` | max_steps:8, too_many_errors:8, user_stop:384 | 0 | 2 | 285 | 0 |
| `iter_0003795` | max_steps:3, too_many_errors:8, user_stop:389 | 0 | 6 | 294 | 0 |

## Seed 301 confirmation

| Checkpoint | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---|---:|---:|---:|---:|---:|
| `iter_0003795` | 55.25% | 81.00% | 26.00% | 75.25% | 44.53% |

## Raw Instruct reference

Job `11878`: pass@1 24.75%, pass@4(any) 46.00%, pass^4 8.00%. Job 11878 predates official-native llm_agent evaluation and is retained only as a historical raw-Instruct reference.
