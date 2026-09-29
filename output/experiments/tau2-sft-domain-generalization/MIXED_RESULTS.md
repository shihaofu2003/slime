# Mixed SFT results

Job 17782 succeeded. Both seeds contain 197 tasks × 4 trials, no missing/duplicate trials or infrastructure failures. Cells: pass@1 / pass@4(any) / pass^4 (%).

| Domain | Seed300 | Seed301 | Mean |
|---|---:|---:|---:|
| airline | 41.25 / 60.00 / 25.00 | 47.50 / 65.00 / 25.00 | 44.38 / 62.50 / 25.00 |
| retail | 59.38 / 90.00 / 27.50 | 61.88 / 85.00 / 32.50 | 60.62 / 87.50 / 30.00 |
| telecom | 48.75 / 85.00 / 12.50 | 50.62 / 87.50 / 17.50 | 49.69 / 86.25 / 15.00 |
| banking_knowledge | 3.87 / 9.28 / 0.00 | 6.70 / 12.37 / 0.00 | 5.28 / 10.82 / 0.00 |
| overall | 28.05 / 46.19 / 10.66 | 30.96 / 47.72 / 12.69 | 29.51 / 46.95 / 11.68 |
| macro | 38.31 / 61.07 / 16.25 | 41.68 / 62.47 / 18.75 | 39.99 / 61.77 / 17.50 |

Historical raw and Qwen3.5 seed300 comparisons: [reference table](REFERENCE_SEED300.md).
Airline expert has completed both seeds; see the [experiment conclusions](README.md)
and [comparison including raw](AIRLINE_VS_RAW.md). Retail was stopped at user request;
Telecom, Banking and raw seed301 remain outstanding.
