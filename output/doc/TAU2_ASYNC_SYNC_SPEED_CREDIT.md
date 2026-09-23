# TAU2 异步 RL 与无优化同步流程：速度、性能和 credit assignment

## 口径和结论

本文把旧的无优化 control 按项目当前口径作为“同步基线”，把采样并发、数据库缓存、任务补充、重试和异步 Trainer/Generator 重叠后的流程作为“异步 RL + 优化”。

主结论是：完整流水线在匹配的 20 updates 上达到 **4.60×** 加速，稳定窗口达到 **4.79×**；训练耗时减少约 78–79%。这是完整系统的工程加速，不是单独异步调度的因果增益。严格只比较同步和异步调度的另一组实验只有 **1.060×** 加速。

历史错误 SFT（会在 Banking 中超过上下文长度限制）的结果不作为当前主对照。

## 速度对比

速度实验固定了 SFT4673、学习率 `2e-6`、global batch 128、K=8、Trainer2+Generator6 和 external User。旧 control 为 `20260919-progress-db-count-v1-lr2e-6-b128-pw1-external-control`，新流程为 `20260919-progress1-b128-sampling-fix-long30-timeout60`。任务顺序和重试策略发生了变化，因此结果应解释为完整流水线对比。

| 窗口 | 无优化同步口径 | 异步 RL + 优化 | 加速比 | 耗时减少 |
|---|---:|---:|---:|---:|
| 匹配的 20 updates | 370.22 min | 80.46 min | **4.60×** | 78.3% |
| 稳定的 17 updates | 311.66 min | 65.09 min | **4.79×** | 79.1% |

按 4.60× 线性估算相同步数的耗时：

| 训练步数 | 无优化同步 | 异步 RL + 优化 |
|---:|---:|---:|
| 20 updates | 6.17 h | 1.34 h |
| 100 updates | 30.85 h | 6.70 h |
| 110 updates | 33.94 h | 7.37 h |
| 140 updates | 43.19 h | 9.38 h |

当前专家训练的 Airline→Retail→Telecom 主路径实际耗时约 14.15 h。按上述比例折算，无优化流程约需 65.1 h，可节省约 50.9 h。该折算是估计值，不是同一组专家任务上的同步实测；Banking 训练同时运行，不在该主路径的关键耗时内。

| 指标 | 无优化基线 | 异步 RL + 优化 |
|---|---:|---:|
| Batch ready 等待 | 868.65 s/update | 142.63 s/update |
| ready 到 Trainer 启动 | 133.95 s | 0.85 s |
| Trainer compute | 81.85 s | 82.57 s |
| Trainer GPU 利用率 | 7.22% | 33.73% |
| Generator GPU 利用率 | 21.54% | 76.22% |

因此速度收益主要来自减少采样、队列、数据库加载、重试和 Trainer 等待；单步模型计算时间基本不变。

Uniform task control 达到过 5.15×（稳定窗口 5.51×），但 Airline 评测为 `42.50/60/30%`，低于选定的 shuffled/retry 配置 `53.75/80/30%`，所以 5.15× 不作为主结果。

## 异步与同步的性能对比

严格匹配的 20-update sync/async 实验从同一 processed SFT 开始，每臂 800 条接受轨迹，并使用两个评测 seed：

| 指标 | 同步 | 异步 | 异步−同步 |
|---|---:|---:|---:|
| pass@1 | 54.00% | 57.88% | **+3.88 pp** |
| pass@4(any) | 79.50% | 84.00% | **+4.50 pp** |
| pass^4 | 24.00% | 29.50% | **+5.50 pp** |
| Action accuracy | 74.72% | 75.62% | +0.90 pp |
| DB accuracy | 44.88% | 47.28% | +2.40 pp |

任务配对 bootstrap 的 95% CI 为 pass@1 `[+0.25,+7.50]` pp、pass@4(any) `[0.00,+9.00]` pp、pass^4 `[-0.50,+11.50]` pp。点估计支持异步更好，但单组训练配对不能排除训练 seed 和任务采样差异。

在当前修正 SFT 上，boundary RL long100 相比 SFT 的三项变化为 `+5.88/+13.00/+2.00 pp`（pass@1/pass@4(any)/pass^4）。这证明当前长训配方有效，但没有同一修正 SFT 的同步 RL 对照，因此不把它当作异步调度的独立因果证据。

## Credit assignment 与 outcome-only GRPO

strict-single-v1 的相同 SFT、User 和评测协议下，`v2-l000` 是 outcome-only GRPO，`v2-l010` 使用 credit assignment：

| 方法 | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---:|---:|---:|---:|---:|
| Outcome-only GRPO | 26.25% | 50.50% | 7.00% | 62.85% | 26.50% |
| Credit assignment | 28.75% | 54.00% | 11.00% | 66.19% | 30.36% |
| Credit−GRPO | **+2.50 pp** | **+3.50 pp** | **+4.00 pp** | **+3.34 pp** | **+3.86 pp** |

Credit assignment 的 paired 95% CI 仍跨 0：pass@1 `[-1.38,+6.38]` pp、pass@4(any) `[-4.00,+11.00]` pp、pass^4 `[-1.00,+9.00]` pp。因此目前应表述为“正向点估计和更好的动作/DB 指标”，而不是统计上已经确定优于 GRPO。预先设定的严格选择规则因 pass@4 与 v1 持平，最终保留了 v1 arm。

## 证据和限制

- [长跑速度报告](../experiments/tau2-airline-db-count-tuning/reports/speed-diagnosis/long30-timeout60/README.md)；[速度统计 JSON](../experiments/tau2-airline-db-count-tuning/reports/speed-diagnosis/long30-timeout60/speed_comparison.json)
- [Uniform speed control 及质量诊断](../experiments/tau2-airline-db-count-tuning/reports/quality-diagnosis/fast-uniform-control20/README.md)
- [严格 sync/async 性能结果](../experiments/tau2-areal-async-rl/ready-train20-final-results.md)；[纯调度效率对比](../experiments/tau2-areal-async-rl/ready-train20-efficiency.md)
- [修正 SFT 和专家 RL 记录](../experiments/tau2-domain-experts-sft4505-b128/README.md)
- [Credit assignment 对比](../experiments/tau2-agent-single-call-credit-core/CREDIT_COMPARISON.md)
