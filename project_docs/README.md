# Service Agent 核心项目导读

按 2026-09-29 定稿简历整理。面向读者的完整说明在 [项目主页](../project_page/index.html)，各项结果和图表在 [数据与实验依据](../project_page/sources.html)。

## 最终模型主线

Raw Qwen3-4B-Instruct-2507 → 四领域 SFT iter4505 → 2026-09-25 状态进度 Mix RL iter139 → 双教师 OPD iter59（60 次更新）。User 为 Qwen3.6-27B，关闭 thinking。OPD 的 Airline 教师为 expert iter29，其余三域教师为固定 Mix139。

最终四领域官方评测：197 题 × 4 trials，seed300、BM25、统一采样参数。pass@1 / pass@4(any) / pass^4 为 **31.98 / 48.22 / 14.21%**；所有 checkpoint 对比以 [最终实验 README](../output/experiments/tau2-opd-mix139-airline-20260929/README.md) 为准。

## 六项工作的阅读范围

| 工作 | 详细记录 | 当前主线应关注的部分 |
|---|---|---|
| 异步评测 | [01 异步评测](01_异步评测.md) | 共享推理、跨轨迹并发、完成域借槽；4.88× 是旧三域协议与不同 GPU 布局的整体测速 |
| SFT 数据治理 | [02 数据治理](02_AReaL三领域SFT数据质量治理.md) | 第 6–8 节的原生格式重建、30,376 targets 与 Raw-all 控制组；额外双审过滤未被采用 |
| Banking 任务 | [03 任务构造](03_Banking任务构造.md) | 独立合成 647 场景，1,651 完整轨迹、5,672 SFT 行；官方派生诊断题不进入训练 |
| 异步 Agentic RL | [04 异步 RL](04_异步Agentic_RL.md) | 持续采样与真实 token、CPU/采样流水线优化；4.60× 对照两臂都已异步 |
| 状态进度信用分配 | [05 Credit assignment](05_Credit_assignment.md) | progress-db-count-v1，iter139 同预算 Vanilla 对照与 Airline 业务边界诊断 |
| 双教师 OPD | [当前最终实验](../output/experiments/tau2-opd-mix139-airline-20260929/README.md) | Mix139 初始化、Airline29 + Mix139 双教师、buffer32、最终 iter59 |

[06 多教师 OPD](06_多教师OPD.md) 主要记录 2026-09-23 的 SFT 初始化、四教师实验。它的 buffer、学习率和梯度检查提供历史依据；Update80、93.4% 保留率不属于当前最终模型。

## 解释时需要区分的量

- 信用分配：到目标 DB 的字段差异数用于奖励；相对初始 DB 改变的记录数用于分桶。二者不是同一个量。
- OPD：总缓冲 32 组涵盖生成中、待训练和训练中；没有固定策略滞后步数上限。
- 数据规模：源行、对话、展开后的监督目标分开计数。当前核心说明使用可逐步复算的 33,531 → 24,816 → 31,679 → 30,376，不把它们混为删除的低质量轨迹总数。
- 成绩：专项三域数据治理、Banking-only SFT、Airline 速度对照与最终四域评测各自独立。所有增量以同协议匹配模型比较。
