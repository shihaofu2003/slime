# Seed300 reference comparison

All cells are pass@1 / pass@4(any) / pass^4 percentages. Raw A/B are two repeats of seed300, not two seeds. All runs have 197 tasks, four trials and zero reported infrastructure failures; actual task definitions and trial coverage were checked. Official-native Agent, Qwen3.6-27B non-thinking User, Banking BM25. Qwen3.5 uses max_tokens=8192; raw and mixed use 1200, so this is not an equal generation-budget comparison.

| Domain | Raw A | Raw B | Mixed SFT | Qwen3.5 non-thinking | Qwen3.5 thinking |
|---|---:|---:|---:|---:|---:|
| airline | 32.50 / 50.00 / 20.00 | 37.50 / 55.00 / 20.00 | 41.25 / 60.00 / 25.00 | 80.00 / 90.00 / 65.00 | 82.50 / 90.00 / 75.00 |
| retail | 53.75 / 72.50 / 30.00 | 59.38 / 87.50 / 30.00 | 59.38 / 90.00 / 27.50 | 81.88 / 92.50 / 70.00 | 83.75 / 95.00 / 67.50 |
| telecom | 18.75 / 40.00 / 7.50 | 16.25 / 30.00 / 7.50 | 48.75 / 85.00 / 12.50 | 78.12 / 87.50 / 55.00 | 69.38 / 85.00 / 52.50 |
| banking_knowledge | 3.35 / 5.15 / 1.03 | 3.61 / 6.19 / 1.03 | 3.87 / 9.28 / 0.00 | 3.87 / 11.34 / 0.00 | 4.38 / 9.28 / 1.03 |
| overall | 19.67 / 30.46 / 10.15 | 20.94 / 32.49 / 10.15 | 28.05 / 46.19 / 10.66 | 42.51 / 51.27 / 31.98 | 41.62 / 50.25 / 32.49 |

Sources:
- Raw A: [seed300_0902_044845_summary.json](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-eval-official-native-full/eval/four-domain-full-bm25/seed300_0902_044845_summary.json)
- Raw B: [seed300_0902_060330_summary.json](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-eval-official-native-full/eval/four-domain-full-bm25/seed300_0902_060330_summary.json)
- Mixed SFT: [summary.json](/mnt/afs/users/fush/projects/ServiceAgent/slime-sft-domain-generalization/output/experiments/tau2-sft-domain-generalization/20260912/mixed/full/eval/seed300/summary.json)
- Qwen3.5 non-thinking: [seed300_0902_054605_summary.json](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-eval-qwen35-official-native-four-domain/eval/four-domain-full-bm25-nonthinking/seed300_0902_054605_summary.json)
- Qwen3.5 thinking: [seed300_0902_054146_summary.json](/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-eval-qwen35-official-native-four-domain/eval/four-domain-full-bm25-thinking/seed300_0902_054146_summary.json)
