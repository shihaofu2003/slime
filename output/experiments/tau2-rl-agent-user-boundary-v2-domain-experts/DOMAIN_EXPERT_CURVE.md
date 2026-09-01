# Domain expert curve

Seed 300; target-domain official test tasks × 4 trials; temperature 0.6; max_steps 200.

## Target-domain checkpoints

| Domain | Model | pass@1 | pass@4(any) | pass^4 | Health |
|---|---|---:|---:|---:|---|
| airline | long100 iter99 | 35.00% | 65.00% | 15.00% | pass |
| airline | mixed iter109 | 36.25% | 55.00% | 20.00% | pass |
| airline | expert iter109 | 31.25% | 60.00% | 10.00% | pass |
| airline | mixed iter119 | 35.00% | 45.00% | 20.00% | pass |
| airline | expert iter119 | 32.50% | 50.00% | 5.00% | pass |
| airline | mixed iter129 | 40.00% | 65.00% | 15.00% | pass |
| airline | expert iter129 | 37.50% | 55.00% | 25.00% | pass |
| retail | long100 iter99 | 30.00% | 47.50% | 12.50% | pass |
| retail | mixed iter109 | 31.87% | 52.50% | 15.00% | pass |
| retail | expert iter109 | 30.00% | 45.00% | 17.50% | pass |
| retail | mixed iter119 | 29.38% | 47.50% | 15.00% | pass |
| retail | expert iter119 | 31.87% | 47.50% | 17.50% | pass |
| retail | mixed iter129 | 30.63% | 47.50% | 20.00% | pass |
| retail | expert iter129 | 30.63% | 50.00% | 15.00% | pass |
| telecom | long100 iter99 | 21.25% | 52.50% | 2.50% | pass |
| telecom | mixed iter109 | 20.62% | 55.00% | 2.50% | pass |
| telecom | expert iter109 | 25.62% | 55.00% | 5.00% | pass |
| telecom | mixed iter119 | 25.62% | 57.50% | 5.00% | pass |
| telecom | expert iter119 | 27.50% | 67.50% | 2.50% | pass |
| telecom | mixed iter129 | 28.12% | 65.00% | 2.50% | pass |
| telecom | expert iter129 | 28.12% | 67.50% | 0.00% | pass |

## Selected checkpoints

| Domain | Iteration | Rule winner |
|---|---:|---|
| airline | 129 | pass@1/pass@4(any)/pass^4/earlier |
| retail | 119 | pass@1/pass@4(any)/pass^4/earlier |
| telecom | 129 | pass@1/pass@4(any)/pass^4/earlier |

## Paired intervals

| Comparison | Metric | Delta | 95% CI |
|---|---|---:|---:|
| airline_expert109-mixed109 | pass@1 / pass^1 | -5.00 pp | [-12.50, +1.25] pp |
| airline_expert109-mixed109 | pass@4(any) | +5.00 pp | [+0.00, +15.00] pp |
| airline_expert109-mixed109 | pass^4 | -10.00 pp | [-25.00, +0.00] pp |
| airline_expert119-mixed119 | pass@1 / pass^1 | -2.50 pp | [-8.75, +3.75] pp |
| airline_expert119-mixed119 | pass@4(any) | +5.00 pp | [-10.00, +20.00] pp |
| airline_expert119-mixed119 | pass^4 | -15.00 pp | [-30.00, +0.00] pp |
| airline_expert129-mixed129 | pass@1 / pass^1 | -2.50 pp | [-12.50, +7.50] pp |
| airline_expert129-mixed129 | pass@4(any) | -10.00 pp | [-30.00, +10.00] pp |
| airline_expert129-mixed129 | pass^4 | +10.00 pp | [-10.00, +30.00] pp |
| airline_expert109-mixed129_equal_domain_samples | pass@1 / pass^1 | -8.75 pp | [-18.75, +0.00] pp |
| airline_expert109-mixed129_equal_domain_samples | pass@4(any) | -5.00 pp | [-15.00, +0.00] pp |
| airline_expert109-mixed129_equal_domain_samples | pass^4 | -5.00 pp | [-20.00, +10.00] pp |
| retail_expert109-mixed109 | pass@1 / pass^1 | -1.88 pp | [-9.38, +4.38] pp |
| retail_expert109-mixed109 | pass@4(any) | -7.50 pp | [-22.50, +7.50] pp |
| retail_expert109-mixed109 | pass^4 | +2.50 pp | [+0.00, +7.50] pp |
| retail_expert119-mixed119 | pass@1 / pass^1 | +2.50 pp | [-3.12, +8.12] pp |
| retail_expert119-mixed119 | pass@4(any) | +0.00 pp | [-15.00, +15.00] pp |
| retail_expert119-mixed119 | pass^4 | +2.50 pp | [-5.00, +10.00] pp |
| retail_expert129-mixed129 | pass@1 / pass^1 | +0.00 pp | [-6.88, +6.88] pp |
| retail_expert129-mixed129 | pass@4(any) | +2.50 pp | [-12.50, +17.50] pp |
| retail_expert129-mixed129 | pass^4 | -5.00 pp | [-17.50, +7.50] pp |
| retail_expert109-mixed129_equal_domain_samples | pass@1 / pass^1 | -0.62 pp | [-6.25, +5.00] pp |
| retail_expert109-mixed129_equal_domain_samples | pass@4(any) | -2.50 pp | [-15.00, +10.00] pp |
| retail_expert109-mixed129_equal_domain_samples | pass^4 | -2.50 pp | [-12.50, +7.50] pp |
| telecom_expert109-mixed109 | pass@1 / pass^1 | +5.00 pp | [-3.12, +13.75] pp |
| telecom_expert109-mixed109 | pass@4(any) | +0.00 pp | [-15.00, +15.00] pp |
| telecom_expert109-mixed109 | pass^4 | +2.50 pp | [-5.00, +10.00] pp |
| telecom_expert119-mixed119 | pass@1 / pass^1 | +1.88 pp | [-8.12, +11.88] pp |
| telecom_expert119-mixed119 | pass@4(any) | +10.00 pp | [-7.50, +27.50] pp |
| telecom_expert119-mixed119 | pass^4 | -2.50 pp | [-12.50, +5.00] pp |
| telecom_expert129-mixed129 | pass@1 / pass^1 | +0.00 pp | [-8.12, +8.12] pp |
| telecom_expert129-mixed129 | pass@4(any) | +2.50 pp | [-12.50, +17.50] pp |
| telecom_expert129-mixed129 | pass^4 | -2.50 pp | [-7.50, +0.00] pp |
| telecom_expert109-mixed129_equal_domain_samples | pass@1 / pass^1 | -2.50 pp | [-11.25, +6.25] pp |
| telecom_expert109-mixed129_equal_domain_samples | pass@4(any) | -10.00 pp | [-25.00, +5.00] pp |
| telecom_expert109-mixed129_equal_domain_samples | pass^4 | +2.50 pp | [-5.00, +10.00] pp |
