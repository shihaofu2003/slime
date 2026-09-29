# Airline expert comparison including raw

Seed300 only: 197 tasks × 4 trials. Raw A/B are repeats of seed300. Cells are pass@1 / pass@4(any) / pass^4 (%). All use official-native Agent, Qwen3.6-27B non-thinking User, Banking BM25, Agent max_tokens 1200.

| Domain | Raw A | Raw B | Mixed | Airline expert |
|---|---:|---:|---:|---:|
| airline | 32.50 / 50.00 / 20.00 | 37.50 / 55.00 / 20.00 | 41.25 / 60.00 / 25.00 | 36.25 / 50.00 / 25.00 |
| retail | 53.75 / 72.50 / 30.00 | 59.38 / 87.50 / 30.00 | 59.38 / 90.00 / 27.50 | 33.75 / 67.50 / 5.00 |
| telecom | 18.75 / 40.00 / 7.50 | 16.25 / 30.00 / 7.50 | 48.75 / 85.00 / 12.50 | 26.88 / 50.00 / 12.50 |
| banking_knowledge | 3.35 / 5.15 / 1.03 | 3.61 / 6.19 / 1.03 | 3.87 / 9.28 / 0.00 | 1.55 / 3.09 / 1.03 |
| overall | 19.67 / 30.46 / 10.15 | 20.94 / 32.49 / 10.15 | 28.05 / 46.19 / 10.66 | 16.75 / 30.46 / 6.60 |

Raw seed301 is not yet available; this table uses seed300 for all models. Mixed and Airline also have completed seed301 evaluations.

Sources:
- Raw A: [seed300_0902_044845_summary.json](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-eval-official-native-full/eval/four-domain-full-bm25/seed300_0902_044845_summary.json)
- Raw B: [seed300_0902_060330_summary.json](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-eval-official-native-full/eval/four-domain-full-bm25/seed300_0902_060330_summary.json)
- Mixed: [summary.json](/mnt/afs/users/fush/projects/ServiceAgent/slime-sft-domain-generalization/output/experiments/tau2-sft-domain-generalization/20260912/mixed/full/eval/seed300/summary.json)
- Airline expert: [summary.json](/mnt/afs/users/fush/projects/ServiceAgent/slime-sft-domain-generalization/output/experiments/tau2-sft-domain-generalization/20260912/airline/full/eval/seed300/summary.json)
