# Turn-filtered SFT data ablation

## Selection

Selected checkpoint: `iter_0003599`. Policy: seed300 pass@1, then pass^4, then earlier checkpoint.

## Seed 300 checkpoint curve

| Checkpoint | Updates | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---|---:|---:|---:|---:|---:|---:|
| `iter_0000399` | 400 | 26.00% | 43.00% | 9.00% | 57.29% | 33.02% |
| `iter_0000799` | 800 | 36.75% | 63.00% | 16.00% | 68.14% | 36.74% |
| `iter_0001199` | 1200 | 42.50% | 71.00% | 21.00% | 71.88% | 41.11% |
| `iter_0001599` | 1600 | 39.50% | 62.00% | 17.00% | 70.51% | 39.22% |
| `iter_0001855` | 1856 | 45.25% | 77.00% | 17.00% | 72.15% | 40.36% |
| `iter_0001999` | 2000 | 45.75% | 74.00% | 21.00% | 71.65% | 37.20% |
| `iter_0002399` | 2400 | 45.25% | 74.00% | 18.00% | 72.98% | 38.50% |
| `iter_0002799` | 2800 | 46.75% | 73.00% | 17.00% | 72.61% | 45.00% |
| `iter_0003199` | 3200 | 45.50% | 74.00% | 22.00% | 70.16% | 43.32% |
| `iter_0003599` | 3600 | 51.75% | 77.00% | 23.00% | 72.95% | 40.31% |
| `iter_0003711` | 3712 | 51.75% | 80.00% | 23.00% | 71.35% | 45.43% |

## Equal-update comparison

`iter_0003599` versus `iter_0003599`.

| Seed | Agent | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | Turn-filtered | 51.75% | 77.00% | 23.00% | 72.95% | 40.31% |
| 300 | Processed | 54.50% | 84.00% | 21.00% | 75.49% | 44.27% |
| 301 | Turn-filtered | 52.50% | 83.00% | 20.00% | 73.04% | 44.30% |
| 301 | Processed | 57.00% | 84.00% | 29.00% | 73.77% | 42.97% |
| 300/301 | Turn-filtered mean | 52.12% | 80.00% | 21.50% | 73.00% | 42.31% |
| 300/301 | Processed mean | 55.75% | 84.00% | 25.00% | 74.63% | 43.62% |

Turn-filtered minus Processed:

| Seed | Scope | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | overall | -2.75pp | -7.00pp | +2.00pp | -2.54pp | -3.96pp |
| 300 | airline | -6.25pp | -10.00pp | -5.00pp | -13.31pp | -5.13pp |
| 300 | retail | -6.25pp | -17.50pp | +7.50pp | -2.86pp | -6.87pp |
| 300 | telecom | +2.50pp | +5.00pp | +0.00pp | +3.06pp | +0.40pp |
| 301 | overall | -4.50pp | -1.00pp | -9.00pp | -0.73pp | +1.33pp |
| 301 | airline | -3.75pp | -15.00pp | -10.00pp | -0.21pp | -3.25pp |
| 301 | retail | -1.87pp | +7.50pp | -12.50pp | -1.55pp | -1.98pp |
| 301 | telecom | -7.50pp | -2.50pp | -5.00pp | +0.80pp | +6.12pp |
| mean | overall | -3.63pp | -4.00pp | -3.50pp | -1.63pp | -1.31pp |
| mean | airline | -5.00pp | -12.50pp | -7.50pp | -6.76pp | -4.19pp |
| mean | retail | -4.06pp | -5.00pp | -2.50pp | -2.21pp | -4.42pp |
| mean | telecom | -2.50pp | +1.25pp | -2.50pp | +1.93pp | +3.26pp |

## Equal-epoch final comparison

`iter_0003711` versus `iter_0003795`.

| Seed | Agent | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | Turn-filtered | 51.75% | 80.00% | 23.00% | 71.35% | 45.43% |
| 300 | Processed | 56.25% | 81.00% | 30.00% | 74.27% | 43.70% |
| 301 | Turn-filtered | 46.00% | 76.00% | 20.00% | 70.81% | 39.52% |
| 301 | Processed | 55.25% | 81.00% | 26.00% | 75.25% | 44.53% |
| 300/301 | Turn-filtered mean | 48.88% | 78.00% | 21.50% | 71.08% | 42.48% |
| 300/301 | Processed mean | 55.75% | 81.00% | 28.00% | 74.76% | 44.12% |

Turn-filtered minus Processed:

| Seed | Scope | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | overall | -4.50pp | -1.00pp | -7.00pp | -2.92pp | +1.73pp |
| 300 | airline | +2.50pp | +15.00pp | -5.00pp | -4.82pp | +2.50pp |
| 300 | retail | +1.88pp | +2.50pp | +2.50pp | -1.21pp | -0.65pp |
| 300 | telecom | -14.37pp | -12.50pp | -17.50pp | -4.49pp | +2.87pp |
| 301 | overall | -9.25pp | -5.00pp | -6.00pp | -4.45pp | -5.01pp |
| 301 | airline | -2.50pp | -5.00pp | -10.00pp | -0.23pp | -0.68pp |
| 301 | retail | -9.38pp | -2.50pp | -12.50pp | -0.86pp | -9.20pp |
| 301 | telecom | -12.50pp | -7.50pp | +2.50pp | -10.09pp | -2.54pp |
| mean | overall | -6.87pp | -3.00pp | -6.50pp | -3.68pp | -1.64pp |
| mean | airline | +0.00pp | +5.00pp | -7.50pp | -2.53pp | +0.91pp |
| mean | retail | -3.75pp | +0.00pp | -5.00pp | -1.04pp | -4.93pp |
| mean | telecom | -13.44pp | -10.00pp | -7.50pp | -7.29pp | +0.16pp |

## Selected checkpoint comparison

`iter_0003599` versus `iter_0003795`.

| Seed | Agent | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | Turn-filtered | 51.75% | 77.00% | 23.00% | 72.95% | 40.31% |
| 300 | Processed | 56.25% | 81.00% | 30.00% | 74.27% | 43.70% |
| 301 | Turn-filtered | 52.50% | 83.00% | 20.00% | 73.04% | 44.30% |
| 301 | Processed | 55.25% | 81.00% | 26.00% | 75.25% | 44.53% |
| 300/301 | Turn-filtered mean | 52.12% | 80.00% | 21.50% | 73.00% | 42.31% |
| 300/301 | Processed mean | 55.75% | 81.00% | 28.00% | 74.76% | 44.12% |

Turn-filtered minus Processed:

| Seed | Scope | pass@1 | pass@4(any) | pass^4 | action acc. | DB acc. |
|---:|---|---:|---:|---:|---:|---:|
| 300 | overall | -4.50pp | -4.00pp | -7.00pp | -1.31pp | -3.39pp |
| 300 | airline | -8.75pp | +0.00pp | -15.00pp | -6.51pp | -6.38pp |
| 300 | retail | -3.12pp | -10.00pp | +2.50pp | -1.17pp | -3.40pp |
| 300 | telecom | -3.75pp | +0.00pp | -12.50pp | +0.88pp | -1.68pp |
| 301 | overall | -2.75pp | +2.00pp | -6.00pp | -2.21pp | -0.23pp |
| 301 | airline | -3.75pp | -5.00pp | -10.00pp | +1.75pp | -2.66pp |
| 301 | retail | -0.62pp | +10.00pp | -15.00pp | -1.56pp | -0.88pp |
| 301 | telecom | -4.37pp | -2.50pp | +5.00pp | -4.75pp | +1.34pp |
| mean | overall | -3.63pp | -1.00pp | -6.50pp | -1.76pp | -1.81pp |
| mean | airline | -6.25pp | -2.50pp | -12.50pp | -2.38pp | -4.52pp |
| mean | retail | -1.88pp | +0.00pp | -6.25pp | -1.36pp | -2.14pp |
| mean | telecom | -4.06pp | -1.25pp | -3.75pp | -1.94pp | -0.17pp |
