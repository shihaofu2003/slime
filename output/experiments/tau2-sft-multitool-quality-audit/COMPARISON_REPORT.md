# tau2 SFT 单/多工具协议中立对照

实验名：`tau2-sft-multitool-quality-audit`。本报告把 multi 当作行为事实，
不再把调用数量直接等同于 policy 违规或任务失败。

## 数据视图

- 训练可见 dialog：2,496
- full-source multi：2,295/2,496 (91.95%)
- longest-training-prefix multi：1,883/2,496 (75.44%)
- prefix 看似 single、full source 实为 multi：412

## 标签口径

| group | dialogs |
|---|---:|
| `consensus_success` | 2,046 |
| `consensus_failure` | 328 |
| `reward_only` | 117 |
| `correct_only` | 5 |

## Full-source 描述统计

| domain | call mode | dialogs | reward fail | correct fail | consensus fail | consensus success |
|---|---|---:|---:|---:|---:|---:|
| airline | non-multi | 5 | 0.00% | 0.00% | 0.00% | 100.00% |
| airline | multi | 994 | 0.70% | 11.97% | 0.20% | 87.53% |
| retail | non-multi | 24 | 0.00% | 0.00% | 0.00% | 100.00% |
| retail | multi | 976 | 0.82% | 0.82% | 0.82% | 99.18% |
| telecom | non-multi | 172 | 63.37% | 63.37% | 63.37% | 36.63% |
| telecom | multi | 325 | 64.31% | 64.31% | 64.31% | 35.69% |

## Multi minus non-multi failure risk

下表是观察性 risk difference，不是 multi 的因果效果。CI 为未调整正态近似；
single 样本少于 30 的 domain 标为样本量不足；`n>=30` 也不代表 task-level
common support 已成立。

| domain | label | multi n | single n | risk difference (95% CI) | size check |
|---|---|---:|---:|---:|---|
| all | reward | 2,295 | 201 | -44.47% (-51.46%–-37.47%) | n>=30 |
| all | correct | 2,295 | 201 | -39.59% (-46.63%–-32.55%) | n>=30 |
| all | consensus | 2,295 | 201 | -44.69% (-51.68%–-37.69%) | n>=30 |
| airline | reward | 994 | 5 | +0.70% (+0.18%–+1.22%) | insufficient |
| airline | correct | 994 | 5 | +11.97% (+9.95%–+13.99%) | insufficient |
| airline | consensus | 994 | 5 | +0.20% (-0.08%–+0.48%) | insufficient |
| retail | reward | 976 | 24 | +0.82% (+0.25%–+1.39%) | insufficient |
| retail | correct | 976 | 24 | +0.82% (+0.25%–+1.39%) | insufficient |
| retail | consensus | 976 | 24 | +0.82% (+0.25%–+1.39%) | insufficient |
| telecom | reward | 325 | 172 | +0.94% (-7.95%–+9.82%) | n>=30 |
| telecom | correct | 325 | 172 | +0.94% (-7.95%–+9.82%) | n>=30 |
| telecom | consensus | 325 | 172 | +0.94% (-7.95%–+9.82%) | n>=30 |

## 语义复审集合

- 唯一 dialogs：689
- 非一致成功：450
- full-source single 成功 anchors：92
- 去重后的 multi 成功 controls：147

整体比例只用于描述；airline/retail 缺乏 non-multi common support 时不输出
multi 的因果效果。失败归因由独立 schema 输出，unknown 始终保留在分母中。

## 双 judge 成功轨迹质量

这里比较的是已成功轨迹的示范质量；失败倾向使用上面的全量标签风险，不能从
success-conditioned 样本推断。`strict batch-safe` 要求两位 judge 对每个 multi turn 的
安全、参数依据、写入授权和必要性完全一致；任一低置信、未解决升级或 judge
错误均按 indeterminate 保留在分母中，不能成为 keep candidate。

| cohort | dialogs | both reliable | score determinate | both keep | strict batch-safe | keep candidate | verdict disagree | batch disagree | mean quality /4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single success anchors | 92 | 100.00% | 53.26% | 20.65% | 100.00% | 20.65% | 51.09% | 0.00% | 3.816 |
| multi success controls | 147 | 100.00% | 64.63% | 34.01% | 85.71% | 34.01% | 31.97% | 14.29% | 3.869 |

## Matched success differences

每个 single anchor 先对其最多三个全局不重复的 multi controls 求平均，再以
anchor 为 bootstrap 单位。差值为 multi − single。

| domain | metric | matched anchors | support | mean difference | 95% cluster bootstrap CI |
|---|---|---:|---|---:|---:|
| all | `mean_quality` | 34 | adequate | +0.005 | -0.034–+0.045 |
| all | `business_policy_compliance` | 54 | adequate | -0.181 | -0.346–-0.012 |
| all | `task_completion` | 75 | adequate | +0.000 | -0.053–+0.053 |
| all | `tool_choice` | 60 | adequate | +0.064 | -0.043–+0.171 |
| all | `argument_grounding` | 71 | adequate | +0.136 | +0.074–+0.203 |
| all | `dependency_safety` | 52 | adequate | +0.040 | -0.034–+0.112 |
| all | `call_necessity` | 54 | adequate | +0.122 | -0.005–+0.235 |
| all | `authorization_and_confirmation` | 74 | adequate | +0.010 | -0.034–+0.054 |
| all | `recovery` | 74 | adequate | -0.019 | -0.091–+0.053 |
| all | `communication` | 62 | adequate | -0.056 | -0.198–+0.078 |
| airline | `mean_quality` | 3 | insufficient (<5) | -0.037 | -0.167–+0.056 |
| airline | `business_policy_compliance` | 4 | insufficient (<5) | -0.458 | -1.125–+0.083 |
| airline | `task_completion` | 4 | insufficient (<5) | -0.500 | -0.917–-0.167 |
| airline | `tool_choice` | 4 | insufficient (<5) | -0.042 | -0.125–+0.000 |
| airline | `argument_grounding` | 5 | adequate | +0.100 | +0.000–+0.300 |
| airline | `dependency_safety` | 5 | adequate | -0.033 | -0.100–+0.000 |
| airline | `call_necessity` | 5 | adequate | +0.000 | +0.000–+0.000 |
| airline | `authorization_and_confirmation` | 5 | adequate | +0.000 | +0.000–+0.000 |
| airline | `recovery` | 4 | insufficient (<5) | -0.271 | -0.562–+0.000 |
| airline | `communication` | 5 | adequate | -0.100 | -0.600–+0.300 |
| retail | `mean_quality` | 24 | adequate | -0.010 | -0.042–+0.021 |
| retail | `business_policy_compliance` | 24 | adequate | -0.042 | -0.132–+0.052 |
| retail | `task_completion` | 24 | adequate | -0.014 | -0.035–+0.000 |
| retail | `tool_choice` | 24 | adequate | -0.024 | -0.052–+0.000 |
| retail | `argument_grounding` | 24 | adequate | +0.000 | +0.000–+0.000 |
| retail | `dependency_safety` | 24 | adequate | -0.042 | -0.069–-0.014 |
| retail | `call_necessity` | 24 | adequate | +0.000 | +0.000–+0.000 |
| retail | `authorization_and_confirmation` | 24 | adequate | +0.000 | +0.000–+0.000 |
| retail | `recovery` | 24 | adequate | -0.007 | -0.021–+0.000 |
| retail | `communication` | 24 | adequate | -0.069 | -0.299–+0.174 |
| telecom | `mean_quality` | 7 | adequate | +0.075 | -0.083–+0.194 |
| telecom | `business_policy_compliance` | 26 | adequate | -0.266 | -0.580–+0.032 |
| telecom | `task_completion` | 47 | adequate | +0.050 | -0.014–+0.113 |
| telecom | `tool_choice` | 32 | adequate | +0.143 | -0.052–+0.336 |
| telecom | `argument_grounding` | 42 | adequate | +0.218 | +0.123–+0.317 |
| telecom | `dependency_safety` | 23 | adequate | +0.141 | -0.011–+0.283 |
| telecom | `call_necessity` | 25 | adequate | +0.263 | +0.010–+0.483 |
| telecom | `authorization_and_confirmation` | 45 | adequate | +0.017 | -0.056–+0.094 |
| telecom | `recovery` | 46 | adequate | -0.004 | -0.114–+0.111 |
| telecom | `communication` | 33 | adequate | -0.040 | -0.220–+0.136 |

## 复审覆盖与一致性

- full/full：663；window/full：26
- both reliable：687/689
- all-dimension score concordance (每维差值 ≤1)：320/689
- verdict agreement：378/689
- aggregate batch agreement：594/689
- all-multi-turn exact agreement：383/488 multi dialogs
- strict batch-safe：584/689
- judge keep-candidate intersection：114/689

| route | dialogs | both reliable | verdict agree | batch agree | all-multi-turn exact |
|---|---:|---:|---:|---:|---:|
| full/full | 663 | 99.85% | 55.20% | 86.73% | 79.22% |
| window/full | 26 | 96.15% | 46.15% | 73.08% | 65.38% |
