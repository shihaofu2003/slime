# Iter69 vs iter99 and SFT — seed300

Job19245 SUCCEEDED, 400/400 simulations, zero infrastructure errors. Each result uses seed300, 100 test tasks, four trials and official-native evaluation. SFT is the historical evaluation of the actual RL initialization iter3795.

Cells: pass@1 / pass@4(any) / pass^4 (%).

| Domain | SFT (no RL) | Mixed iter69 | Mixed iter99 | Airline iter99 |
|---|---|---|---|---|
| overall | 56.25 / 81.00 / 30.00 | 58.25 / 85.00 / 26.00 | 59.25 / 84.00 / 27.00 | 55.00 / 85.00 / 21.00 |
| airline | 51.25 / 65.00 / 35.00 | 43.75 / 70.00 / 15.00 | 46.25 / 70.00 / 20.00 | 46.25 / 70.00 / 20.00 |
| retail | 57.50 / 82.50 / 27.50 | 61.88 / 87.50 / 35.00 | 67.50 / 87.50 / 45.00 | 59.38 / 87.50 / 30.00 |
| telecom | 57.50 / 87.50 / 30.00 | 61.88 / 90.00 / 22.50 | 57.50 / 87.50 / 12.50 | 55.00 / 90.00 / 12.50 |

Mixed iter69 vs SFT: +2/+4/−4pp. Mixed iter99 vs iter69: +1/−1/+1pp. Later training improves Airline/Retail pass@1 by2.5/5.625pp, while Telecom loses4.375pp; Telecom pass^4 drops10pp. No single-seed significance claim.

Sources:
- [SFT (no RL)](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-sft-official-native-expanded/eval/checkpoint-iter_0003795/seed300_0902_224647_summary.json)
- [Mixed iter69](/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2/output/experiments/tau2-areal-async-rl/eval/async-iter69-unbounded-p20-train200-20260914-mixed-all-domains/seed300_0914_113522_summary.json)
- [Mixed iter99](/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2/output/experiments/tau2-areal-async-rl/eval/async-iter99-unbounded-p20-train200-20260914-mixed/seed300_0914_053108_summary.json)
- [Airline iter99](/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2/output/experiments/tau2-areal-async-rl/eval/async-iter99-airline-unbounded-p20-train200-20260914-airline-all-domains/seed300_0914_053601_summary.json)
