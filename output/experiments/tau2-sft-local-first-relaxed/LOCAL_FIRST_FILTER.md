# Local-first relaxed SFT filter

筛选版本：`local-first-target-prefix-v1`。全量语义复审只使用本地 Qwen3.6-27B；已有
Luna 结果覆盖 239 个成功 dialog，并作为保守否决证据复用，未发起新 API 请求。

## Yield

- 原训练文件：19,318 targets；consensus-success：16,811。
- 最终 canonical：15,807 targets / 1,863 dialogs。
- 领域 target：{'airline': 7277, 'retail': 8243, 'telecom': 287}。
- 多工具 target：2,554，分领域 {'airline': 1488, 'retail': 1057, 'telecom': 9}。
- 训练预算文件：19,318 rows，和旧 SFT 的 row/update budget 对齐。

## Rule

只自动保留 consensus-success。可靠 judge 的 keep/review 均可候选，但 drop、核心维度
低于 3、支持维度低于 2、目标依赖前缀中的机械 hard issue，或任一
multi turn 未通过 dependency/grounding/authorization/necessity strict-safe gate 时拒绝。
已有 Luna 覆盖的样本必须同时通过相同门槛。所有历史 assistant turn 的
`step_loss_mask=0`，只监督获准的最后目标；训练 system 已同步为 dependency-safe multi。

## Artifacts

- `sft_selected_target_only.jsonl`：无重复 canonical targets。
- `sft_train_budget_matched_19318.jsonl`：确定性重复到旧训练的 19,318-row budget。
- `filter_decisions.jsonl`：覆盖原文件每个 target 的无文本 provenance 决策。
- `filter_summary.json`：完整计数、reason 分布和 SHA256。
