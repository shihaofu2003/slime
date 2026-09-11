# dependency-safe multi 筛选草案

实验名：`tau2-sft-multitool-quality-audit`。本文件是复审标准，不是训练 JSONL。

## 候选协议

An assistant turn may contain multiple tool calls only when every call is independently justified by information available before the turn, no call requires another call's result, and every state-changing call is explicitly authorized and order-independent. Otherwise, make one tool call and wait for its result. If you make tool calls, do not respond to the user in the same turn.

## 决策规则

- `keep-candidate`：标签为 `consensus_success`，双 judge 对非调用数量质量结论一致，
  且 `included_targets[].conversation_turn_index` 对应的每个 multi target 在两份
  `multi_turn_assessments` 中逐字段一致并为三类 `safe_independent_*`；参数必须 grounded，
  写入必须已授权且 order-independent，调用必须必要。不得有无效工具/schema、同批
  observation 依赖、未确认写入或无依据重复/fanout。
- `review`：标签冲突、judge 分歧、`batch_safety=uncertain`、长上下文窗口与完整审查
  不一致，或最终 evaluator/DB 证据不可见。
- `drop-candidate`：存在确定性工具错误或双 judge 一致确认的关键流程/参数/依赖错误。
  multi 的存在本身不是 drop reason。

## 后续训练约束

本轮不直接写训练数据。标准确认后从 raw source 重新生成 prompt 一致的数据，使用
`step_loss_mask` 只监督获准 target turn；single-only 与 dependency-safe-multi 对照必须
匹配训练 token、optimizer update、seed 和 official eval 配置。
