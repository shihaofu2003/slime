# Unbounded pool20 iter99 vs SFT — seed300

Both jobs SUCCEEDED, 400/400 simulations each, zero infrastructure errors. Baseline is the actual SFT initialization iter3795, historical seed300 result. Same official-native protocol, Qwen3.6-27B non-thinking User, Agent temperature0.6/top_p1/max_tokens1200, max_steps200, three test domains × four trials.

Each cell: pass@1 / pass@4(any) / pass^4 (%).

| Domain | SFT | Mixed RL99 | Airline RL99 |
|---|---|---|---|
| overall | 56.25 / 81.00 / 30.00 | 59.25 / 84.00 / 27.00 | 55.00 / 85.00 / 21.00 |
| airline | 51.25 / 65.00 / 35.00 | 46.25 / 70.00 / 20.00 | 46.25 / 70.00 / 20.00 |
| retail | 57.50 / 82.50 / 27.50 | 67.50 / 87.50 / 45.00 | 59.38 / 87.50 / 30.00 |
| telecom | 57.50 / 87.50 / 30.00 | 57.50 / 87.50 / 12.50 | 55.00 / 90.00 / 12.50 |

Mixed RL overall deltas: +3/+3/−3pp. Airline RL: −1.25/+4/−9pp. Mixed RL pass@1 gain comes from Retail (+10pp); both RL models lose5pp Airline pass@1. Both lose Airline/Telecom pass^4 by15/17.5pp. Single seed; no statistical significance claim.

Sources:
- [SFT (no RL)](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-sft-official-native-expanded/eval/checkpoint-iter_0003795/seed300_0902_224647_summary.json)
- [Mixed RL iter99 (19097)](/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2/output/experiments/tau2-areal-async-rl/eval/async-iter99-unbounded-p20-train200-20260914-mixed/seed300_0914_053108_summary.json)
- [Airline RL iter99 (19101)](/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2/output/experiments/tau2-areal-async-rl/eval/async-iter99-airline-unbounded-p20-train200-20260914-airline-all-domains/seed300_0914_053601_summary.json)
