# DB-count mixed smoke + train100

目的：对照时间分组的 mixed-fast-train100，仅替换 turn-level advantage 分组。
实验名：`tau2-db-count-v1 / 20260914_165226-mixed-smoke-train100`。

[启动脚本](run.sh)：1 步 smoke 正常结束后，从相同 Processed SFT 新初始化训练 100 步。
Smoke 使用 `smoke-checkpoints/`，正式训练使用 `checkpoints/`，不继承 smoke 优化器。
两个阶段均运行 30 项测试及 rollout preflight。失败则脚本停止，不自动进入下一阶段。

- 相同 Processed SFT（20260903、torch-dist iter3795），User Qwen3.6-27B、关闭 thinking、温度 0。
- 混合领域、不固定配额；相同 1,982 条训练任务，每步 5 组 × K=8，40 条接受轨迹。
- LR=2e-6 constant，KL/entropy=0，gamma=0.98，outcome/progress/format 权重均为 1。
- Train seed1234、rollout seed42；Agent 温度1、max_tokens1200、max_steps200、训练上限16384。
- 组键为同一任务内动作前的 Agent/User 总 DB 差异条数；原时间分组代码不修改。
- 正式训练每 10 步保存至 iter99。`smoke/`、`train100/` 保存轨迹和 credit.jsonl。
- 8 GPU / 112 CPU / 1584 GB，普通队列，priority normal。

当前时间分组对照：`tau2-credit-assignment/20260914_085453-mixed-fast-train100`（Job19000）。
本次尚未提交官方评测；训练诊断不作为效果胜出的证据。

Job **19187**（pt-seyqmy77），2026-09-14 16:53 已提交。
[运行日志](../jobs/19187-db-count-mixed-smoke-train100-0914-165312975/run_0_20260914_165312975.log)。

## 奖励曲线

[PNG](plots/reward_curve.png) · [SVG](plots/reward_curve.svg) · [CSV](plots/reward_curve.csv)。
只统计正式训练已完成更新的 `rollout/raw_reward`，排除 smoke；满 10 步后显示 trailing 10-step mean。
2026-09-14 17:44 左右的快照包含 iter0–1，奖励为 52.5%、62.5%；属于训练批次指标。
后续进度更新时同步刷新曲线：

```bash
python3 output/experiments/tau2-db-count-v1/20260914_165226-mixed-smoke-train100/plot_reward.py
```

## 持续监控

[watch_reward.py](watch_reward.py) 每 60 秒读取日志，仅在正式训练已完成步数或奖励发生变化时刷新 PNG/SVG/CSV。
排除 smoke，满 10 步后显示滑动均值；训练结束后保持监控，手动停止。
后台进程 PID **656377**，启动时间 2026-09-14 18:43 左右；[监控日志](watch_reward.log)。

```bash
python3 -u output/experiments/tau2-db-count-v1/20260914_165226-mixed-smoke-train100/watch_reward.py --interval 60
# 停止当前后台监控，不影响训练：
kill 656377
```

`--once` 单次刷新；前台运行时 Ctrl-C 停止。已验证实际绘图和仅在数据变化时重绘。
