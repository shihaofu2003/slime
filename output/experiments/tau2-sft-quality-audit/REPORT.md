# tau2 SFT 数据质量审计

实验名：`tau2-sft-quality-audit`。审计实际训练文件的
19,318 条样本（2,496 个去重
dialog），并将 dialog 结论映射回训练行。

状态说明（2026-08-03）：本报告记录原始 single-policy 下的历史筛选结果；其中
`multi_tool_policy_violation` 不能继续解释为数据质量失败。协议中立复审、local-first
重训和 held-out eval 已完成，当前标准见
[FINAL_RECOMMENDATION.md](../tau2-sft-local-first-relaxed/FINAL_RECOMMENDATION.md)。

## 结论

- 完成模型盲审：2,496/2,496 dialogs。
- `keep/review/drop`：235 / 78 / 2,183 dialogs。
- Judge 与确定性事实冲突并被保守覆盖：
  114 dialogs；这些冲突不会自动进入 `keep`。
- 严格筛选后训练行：998/19,318
  (5.17%)。
- `keep` 要求 source `correct=reward=1`、无确定性硬错误、judge verdict 为 `keep`，
  outcome 为成功或仅因 target-tool 边界而不确定，置信度至少 0.75、总分至少
  3.25/4，且任务完成、policy、工具、参数、流程五项均至少 3/4。

## 分领域

| domain | keep dialogs | keep rows | review dialogs | review rows | drop dialogs | drop rows |
|---|---:|---:|---:|---:|---:|---:|
| airline | 5 | 31 | 0 | 0 | 994 | 8752 |
| retail | 132 | 666 | 12 | 76 | 856 | 7853 |
| telecom | 98 | 301 | 66 | 224 | 333 | 1415 |

## 主要筛除原因

| reason | dialogs |
|---|---:|
| `multi_tool_policy_violation`（历史 single-policy） | 1883 |
| `source_correct_not_one` | 445 |
| `source_reward_not_one` | 333 |
| `judge_review_threshold` | 78 |
| `unresolved_judge_escalation` | 8 |
| `invalid_tool_name` | 4 |
| `low_judge_confidence` | 3 |
| `judge:false_success_claim` | 2 |
| `judge_drop` | 2 |
| `judge:policy_violation` | 2 |
| `judge:failed_recovery` | 2 |
| `judge:incomplete_task` | 2 |

## 模型维度诊断

| criterion | all mean | all score < 3 | positive + rule-clean mean | clean score < 3 |
|---|---:|---:|---:|---:|
| `task_completion` | 3.02 | 35.15% | 3.85 | 3.17% |
| `policy_compliance` | 1.29 | 74.45% | 3.98 | 0.32% |
| `tool_choice` | 3.58 | 9.03% | 3.98 | 0.32% |
| `argument_grounding` | 3.80 | 1.37% | 3.98 | 0.63% |
| `workflow_and_dependencies` | 3.02 | 31.68% | 3.95 | 1.59% |
| `authorization_and_confirmation` | 3.80 | 5.88% | 3.98 | 0.95% |
| `recovery` | 3.50 | 20.27% | 3.95 | 1.27% |
| `communication_and_closeout` | 3.53 | 11.45% | 3.93 | 2.22% |

`positive + rule-clean` 是 source 正标签且未命中确定性硬错误的候选；该列用于判断
排除协议/标签污染后，是否仍存在工具选择、参数、流程或沟通质量问题。

| cohort | n | task | policy | tool | args | workflow | recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive + rule-clean | 315 | 3.85 | 3.98 | 3.98 | 3.98 | 3.95 | 3.95 |
| positive + 完整保留 | 988 | 3.72 | 0.62 | 3.90 | 3.99 | 3.27 | 3.94 |
| positive + 截断视图 | 1,049 | 2.29 | 1.30 | 3.25 | 3.63 | 2.65 | 3.08 |
| airline positive + 完整保留 | 409 | 3.55 | 0.53 | 3.78 | 3.97 | 3.05 | 3.88 |
| airline positive + 截断视图 | 460 | 1.78 | 0.29 | 2.95 | 3.39 | 2.10 | 2.65 |
| positive multi（batch 无写） | 1,378 | 2.65 | 0.47 | 3.42 | 3.75 | 2.68 | 3.32 |
| positive multi（batch 含写） | 344 | 3.51 | 0.18 | 3.79 | 3.87 | 3.11 | 3.78 |
| 非正/冲突 source 标签 | 444 | 3.17 | 2.79 | 3.64 | 3.78 | 3.36 | 3.55 |
| airline | 988 | 2.53 | 0.38 | 3.29 | 3.62 | 2.49 | 3.16 |
| retail | 997 | 3.16 | 0.97 | 3.69 | 3.89 | 3.14 | 3.65 |
| telecom | 496 | 3.70 | 3.75 | 3.92 | 3.96 | 3.85 | 3.90 |

同轮 multi 两行只包含 source 正标签；`batch 无写/含写` 由确定性工具集合划分。
由于 multi 在 tau2 中必然违反协议，判断它是否还教坏“如何调用”时应重点比较
tool、args 与 workflow，而不能只看 policy 分数。`n` 是确定性裁决后仍有可用
selected review 的数量。

“完整保留/截断视图”用于分离轨迹本身的语义质量与 max-length/转换造成的缺尾；
即使 prompt 要求不惩罚正常 prefix 边界，两组分差仍应作为 judge 敏感性和训练覆盖
偏差共同解释。

| source label group | reviewed | judge keep / review / drop | outcome success / partial / failure / uncertain |
|---|---:|---:|---:|
| `positive` | 2,037 | 302 / 382 / 1,353 | 836 / 279 / 23 / 899 |
| `nonpositive_or_conflict` | 444 | 277 / 32 / 135 | 20 / 18 / 6 / 400 |

Source 标签是完整 dialog 的结果，而 judge 看的是实际保留下来的最长训练 prefix；
二者不一致不能直接叫“错标”，但能识别成功 dialog 中不值得模仿的步骤，以及失败
dialog 中可能仍可回收的早期步骤。

## Judge 用量

最终选用 provider：`fallback:gemini-2.5-flash` 681, `local:Qwen3.6-27B` 1,800, `none` 15；触发升级
783 条，未解决 114
条，调用错误记录 0 条。本地/API 配对
783 条，verdict 一致
61.43%，
outcome 一致
47.00%。

| dimension | fallback - local mean | paired mean absolute difference |
|---|---:|---:|
| `task_completion` | 0.40 | 0.76 |
| `policy_compliance` | 0.65 | 0.93 |
| `tool_choice` | 0.38 | 0.49 |
| `argument_grounding` | 0.17 | 0.23 |
| `workflow_and_dependencies` | 0.71 | 0.88 |
| `authorization_and_confirmation` | 0.10 | 0.22 |
| `recovery` | 0.48 | 0.67 |
| `communication_and_closeout` | 0.27 | 0.40 |

配对集合由本地低置信、不确定或规则冲突触发，存在选择偏差；差值用于暴露 judge
敏感性，不能解释成 provider 全局优劣。

| provider | calls | prompt tokens | completion tokens | total tokens |
|---|---:|---:|---:|---:|
| `fallback:gemini-2.5-flash` | 1133 | 11004563 | 889425 | 11893988 |
| `local:Qwen3.6-27B` | 2554 | 23918738 | 2160751 | 26079489 |

## 产物

- `selection_manifest.jsonl/csv`：每个 dialog 的可审计决定。
- `filtered_sft_keep.jsonl`：只含 `keep` dialog 的原训练行，未改写对话。
- `SUCCESS_CASES.md`：高质量成功轨迹的模型归因与证据轮次。
- `features.jsonl` / `judge_reviews.jsonl`：规则事实与模型原始结构化判断。
- `judge_adjudication_overrides.jsonl`：模型判断与机器可判事实冲突的保守裁决。

## 限制与下游验证

SFT source 不含可重放的 task DB 与完整 evaluator criteria，因此模型只能结合 policy、
工具 schema、对话和 observation 判断“可模仿质量”；source reward 仍作为独立证据，
不能证明每一步或最终 DB 都正确。筛选结果应通过同配置 SFT 重训和 held-out official
eval 做最终验证。`filtered_sft_keep.jsonl` 为了可审计仍保留原 system prompt，
包括允许 `one or more tool calls` 的冲突文本；它是选择产物，不是可直接训练的
数据。

后续已将 system、领域 policy、筛选和 eval prompt 统一为 `dependency-safe-multi`，并从
raw 模型完成 19,318-row matched-budget SFT。新 SFT 的 pass@1 / pass@4(any) 显著优于
raw 与旧 SFT，但 pass^4 相对旧 SFT 显著下降 6pp；因此本报告的 998-row strict 输出仅作为
历史高精度 seed，不作为最终训练集。完整结果见
[EVAL_COMPARISON.md](../tau2-sft-local-first-relaxed/EVAL_COMPARISON.md)。
