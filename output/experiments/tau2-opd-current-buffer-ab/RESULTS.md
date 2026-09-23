# OPD buffer A/B official results

Purpose: compare total buffer32 (A) with unbounded prefetch (B), both LR2e-6/60 updates.

All five models: native three-domain test, seeds300/301, four trials, identical Agent/User and concurrency settings.
B resumed from iter49 after a compute-cluster crash. Pending trajectories were regenerated at the restored policy; its final ten updates are not an uninterrupted unbounded-prefetch control.
Cells are pass@1 / pass@4(any) / pass^4 (%), averaged over both seeds.

| Model | Airline | Retail | Telecom |
|---|---|---|---|
| sft | 44.38 / 62.50 / 32.50 | 50.94 / 78.75 / 22.50 | 42.81 / 80.00 / 8.75 |
| airline | 45.62 / 65.00 / 25.00 | 57.81 / 90.00 / 31.25 | 44.69 / 76.25 / 13.75 |
| retail | 38.75 / 57.50 / 25.00 | 55.31 / 81.25 / 32.50 | 48.44 / 77.50 / 17.50 |
| A | 46.25 / 65.00 / 30.00 | 55.94 / 81.25 / 33.75 | 45.31 / 77.50 / 11.25 |
| B | 45.62 / 67.50 / 22.50 | 57.81 / 81.25 / 28.75 | 48.12 / 82.50 / 13.75 |

## Paired task uncertainty

20,000 task-cluster bootstrap draws; seeds/trials stay within their task. These intervals do not cover training-seed uncertainty.

| Comparison/domain | Δ pass@1 pp | 95% CI pp | Δ pass@4(any) pp | Δ pass^4 pp |
|---|---:|---|---:|---:|
| A-B/airline | +0.62 | [-6.25, +8.12] | -2.50 | +7.50 |
| A-B/retail | -1.88 | [-7.81, +3.75] | +0.00 | +5.00 |
| A-B/telecom | -2.81 | [-9.69, +4.38] | -5.00 | -2.50 |
| A-sft/airline | +1.88 | [-8.12, +13.12] | +2.50 | -2.50 |
| A-sft/retail | +5.00 | [-0.31, +10.31] | +2.50 | +11.25 |
| A-airline/airline | +0.62 | [-6.88, +8.75] | +0.00 | +5.00 |
| A-retail/retail | +0.62 | [-5.94, +7.19] | +0.00 | +1.25 |
| B-sft/airline | +1.25 | [-7.50, +10.00] | +5.00 | -10.00 |
| B-sft/retail | +6.88 | [+0.94, +13.12] | +2.50 | +6.25 |
| B-airline/airline | +0.00 | [-4.38, +4.38] | +2.50 | -2.50 |
| B-retail/retail | +2.50 | [-3.12, +8.44] | +0.00 | -3.75 |
| airline-sft/airline | +1.25 | [-6.25, +8.75] | +2.50 | -7.50 |
| retail-sft/retail | +4.38 | [-0.94, +10.00] | +2.50 | +10.00 |
| airline-retail/retail | +2.50 | [-3.44, +8.44] | +8.75 | -1.25 |

[Per-seed metrics, action/DB diagnostics and all intervals](comparison.json).
Interpret jointly with the consumed-domain counts and lag in the A/B training logs; no automatic recipe winner is declared.
