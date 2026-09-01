# tau2 SFT 数据质量：确定性审计

实验名：`tau2-sft-quality-audit`。对象是实际训练文件中的
19,318 条 prefix 样本，对应 2,496
个去重 source dialog；source 全量为 33,531 行、
2,499 个 dialog。

状态说明（2026-08-03）：本文是原始 single-call 领域协议下的历史基线，其中
`multi_tool_policy_violation` 只表示与当时 policy markdown 冲突，不能解释为轨迹质量差。
后续协议中立审计和 matched SFT 评测已完成；最终解释见
[multi-tool 对照](../tau2-sft-multitool-quality-audit/COMPARISON_REPORT.md)与
[筛选结论](../tau2-sft-local-first-relaxed/FINAL_RECOMMENDATION.md)。

## 训练保留率

| domain | source rows | trained rows | retention |
|---|---:|---:|---:|
| airline | 12842 | 8783 | 68.39% |
| retail | 11395 | 8595 | 75.43% |
| telecom | 9294 | 1940 | 20.87% |

| domain | training dialogs | full dialog retained | median per-dialog retention | median source targets | median kept targets |
|---|---:|---:|---:|---:|---:|
| airline | 999 | 423 (42.34%) | 80.00% | 13 | 8 |
| retail | 1,000 | 561 (56.10%) | 100.00% | 11 | 8 |
| telecom | 497 | 27 (5.43%) | 13.33% | 18 | 2 |

转换形状与 8192-token 上限共同排除了
14,213 行；实际训练数据的领域比例
因此不同于 source，尤其需要单独检查 telecom 的能力覆盖。

## Source 标签

| correct | reward | dialogs | training rows |
|---:|---:|---:|---:|
| 0 | 0.0 | 328 | 1308 |
| 0 | 1.0 | 117 | 1121 |
| 1 | 0.0 | 5 | 78 |
| 1 | 1.0 | 2046 | 16811 |

正标签 dialog 为 2,046；同时满足正标签且未命中
确定性硬错误的候选为 315。`reward=1`
只说明 source verifier 的最终结果，不能替代 policy、过程和可模仿性判断。实际训练
文件中有 2,507 行来自非正或冲突标签 dialog。

## 确定性问题

| issue | dialogs | rate |
|---|---:|---:|
| `invalid_tool_name` | 4 | 0.16% |
| `multi_tool_policy_violation`（历史 single-policy） | 1883 | 75.44% |

| domain | dialogs | positive label | same-turn multi | rule-clean dialogs | rule-clean rows | full | state-changing |
|---|---:|---:|---:|---:|---:|---:|---:|
| airline | 999 | 875 | 994 (99.50%) | 5 | 31 | 5 | 3 |
| retail | 1000 | 992 | 856 (85.60%) | 144 | 742 | 22 | 43 |
| telecom | 497 | 179 | 33 (6.64%) | 166 | 547 | 12 | 0 |

所有转换后样本都注入了允许 `one or more tool calls` 的输出协议，而当时 tau2 三个领域
policy 要求每轮最多一个调用；这是历史数据协议冲突，不由最终 reward 自动发现。后续
`dependency-safe-multi` profile 已同步修改 system、领域 policy、筛选和 eval prompt，
不再把调用数量本身作为 hard issue。

同轮多调用 dialog 中有 370 个把写操作
提前放进 batch；另有 1,696 个 dialog 对同一工具使用
至少三组参数，27 个含完全重复调用。这些模式会让
模型学到“提高调用宽度/覆盖”，但不保证参数、依赖或最终状态正确。

## Loss 暴露

- 训练 target 行：19,318
- target 本身为同轮多调用：3,014
- prefix 已含同轮多调用的训练行：12,258
- assistant turn 暴露：124,857
- tool call 暴露：87,216
- 同轮多调用 turn 暴露：16,722
- 写调用暴露：8,333

| domain | target rows | multi target | prefix 已含 multi | multi turn exposure |
|---|---:|---:|---:|---:|
| airline | 8,783 | 1,882 (21.43%) | 6,844 (77.92%) | 10,428 |
| retail | 8,595 | 1,096 (12.75%) | 5,113 (59.49%) | 5,968 |
| telecom | 1,940 | 36 (1.86%) | 301 (15.52%) | 326 |

| source label group | target rows | multi target | prefix 已含 multi |
|---|---:|---:|---:|
| `nonpositive_or_conflict` | 2,507 | 339 (13.52%) | 1,180 (47.07%) |
| `positive` | 16,811 | 2,675 (15.91%) | 11,078 (65.90%) |

这里统计的是 SFT loss 实际看到的重复 prefix，而不是去重 dialog 数。后续模型评审
在 dialog 层进行，最终筛选再映射回原始训练行。

## 后续结果

- 协议中立复审确认：成功轨迹 matched 总质量差（multi − single）为
  `+0.005 [-0.034, +0.045]`；telecom 的 consensus-failure 风险差为
  `+0.94pp [-7.95pp, +9.82pp]`，均不支持按调用数量筛选。
- 本地 Qwen3.6-27B 全量复审加 target-prefix salvage 得到 1,863 dialogs / 15,807
  canonical targets，并构建 19,318-row 预算对齐数据；没有新增外部 API 调用。
- raw-init SFT 的 official tau2 pass@1 / pass@4(any) 为 `27.00% / 54.00%`，相对 raw
  分别提升 `+6.50pp [1.00, 12.00]`、`+11.00pp [2.00, 21.00]`；但 pass^4 为
  `4.00%`，相对旧 SFT 下降 `-6.00pp [-12.00, -1.00]`。
- 因此保留 `dependency-safe-multi` 和 local-first broad tier，但当前训练配方不直接升级为
  唯一默认；下一版增加分域 consistency anchor。
