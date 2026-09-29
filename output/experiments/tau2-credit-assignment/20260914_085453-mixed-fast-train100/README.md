# 优化评分后的混合领域 Credit GRPO 100 步

目的：使用已通过等价性和性能验证的状态评分实现，从 Processed SFT 训练 100 步。
实验名：tau2-credit-assignment / 20260914_085453-mixed-fast-train100。

Job 19000，2026-09-14 08:55 提交，普通队列、priority normal，8 GPU / 112 CPU / 1584 GB。
[运行日志](../jobs/19000-credit-fast-mixed-train100-0914-085503741/run_0_20260914_085503741.log)。
提交和 setup 均显式指定当前 worktree 的 PROJECT_ROOT。

- 新建 checkpoint 目录，从 Processed SFT torch-dist iter3795 初始化，新优化器，100 次更新至 iter99。
- 不固定领域配额，读取共享 1,982 条任务；每次更新 5 组 × K=8，共 40 条接受轨迹。
- progress-rtg-v1：gamma=0.98，outcome/progress/format 权重均为 1，format 错误惩罚 −0.1。
- LR=2e-6 constant，KL/entropy=0，training seed1234、rollout seed42；其他配置沿用混合领域 40 步实验。
- Agent 使用 Processed SFT，User 为 Qwen3.6-27B、thinking 关闭、温度 0。
- 训练前运行 progress 测试和 rollout preflight，每 10 步保存；checkpoint 在 checkpoints/，轨迹和诊断在 train100/。

启动器增加 train100 模式，bash 语法检查通过；评分优化的 20 项测试和 rollout preflight
见 [验证结果](../20260914_085000-scoring-speed/RESULTS.md)。iter49 全域评测已提交，见下。

iter49 全域评测：Job **19107**，2026-09-14 13:37 提交，8 GPU / 112 CPU / 1584 GB，普通队列、priority normal；提交时因用户额度满而排队。先转换 iter49 为 HF，再执行 airline/retail/telecom 全部 test 任务 × 4 trials、seed300；沿用 Qwen3.6-27B User（温度 0、关闭 thinking）、Agent 温度 0.6、max_steps=200、max_tokens=1200。
[启动脚本](eval49.sh)，[运行日志](../jobs/19107-credit-fast-mixed-eval49-s300-0914-133714143/run_0_20260914_133714143.log)；结果输出到 `eval/progress-rtg-v1-iter49/seed300/summary.json`。

训练 Job19000 已于 2026-09-14 17:14 成功完成 100 次更新，iter99 保存成功。完整奖励曲线见 [PNG](plots/reward_curve.png) / [CSV](plots/reward_curve.csv)。

iter99 全域评测：Job **19208**，2026-09-14 17:35 提交，8 GPU / 112 CPU / 1584 GB，普通队列、priority normal。先转换 HF，再沿用 iter49 协议执行三域全部 test 任务 × 4 trials，seed300、Qwen3.6-27B User。[启动脚本](eval99.sh)，[运行日志](../jobs/19208-credit-fast-mixed-eval99-s300-0914-173517403/run_0_20260914_173517403.log)；结果输出到 `eval/progress-rtg-v1-iter99/seed300/summary.json`。

iter79 全域评测：Job **19231**，2026-09-14 18:42 提交，8 GPU / 112 CPU / 1584 GB，普通队列、priority normal。先转换 HF，再沿用 iter49/99 协议执行三域全部 test 任务 × 4 trials，seed300、Qwen3.6-27B User。[启动脚本](eval79.sh)，[运行日志](../jobs/19231-credit-fast-mixed-eval79-s300-0914-184212899/run_0_20260914_184212899.log)；结果输出到 `eval/progress-rtg-v1-iter79/seed300/summary.json`。
