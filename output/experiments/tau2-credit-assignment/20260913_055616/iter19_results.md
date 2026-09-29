# Iter19 与 Processed SFT 对比

同 seed300、100 test tasks × 4 trials；Qwen3.6 User、official-native、Agent temperature 0.6、max_steps 200。

| 范围 | 模型 | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---|---:|---:|---:|---:|---:|
| overall | sft | 56.25% | 81.00% | 30.00% | 74.27% | 43.70% |
| overall | progress | 54.00% | 82.00% | 24.00% | 74.39% | 42.05% |
| airline | sft | 51.25% | 65.00% | 35.00% | 51.75% | 51.25% |
| airline | progress | 42.50% | 65.00% | 25.00% | 52.19% | 42.50% |
| retail | sft | 57.50% | 82.50% | 27.50% | 88.01% | 62.34% |
| retail | progress | 59.38% | 90.00% | 30.00% | 87.52% | 62.42% |
| telecom | sft | 57.50% | 87.50% | 30.00% | 66.72% | 21.29% |
| telecom | progress | 54.37% | 82.50% | 17.50% | 67.18% | 20.92% |

配对任务 bootstrap 100,000 次；整体差值及 95% CI（百分点）：
- pass_at_1: -2.25 [-8.50, +4.00]
- pass_at_4_any: +1.00 [-7.00, +9.00]
- pass_power_4: -6.00 [-16.00, +3.00]

本次为 iter19 中间评测，三项整体 CI 均包含零；不替代 iter39 的预定续训判断。

来源：
- progress: /mnt/afs/users/fush/projects/ServiceAgent/slime-credit-assignment/output/experiments/tau2-credit-assignment/20260913_055616/eval/progress-rtg-v1-iter19/seed300/summary.json
- sft: /mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-sft-official-native-expanded/eval/checkpoint-iter_0003795/seed300_0902_224647_summary.json
