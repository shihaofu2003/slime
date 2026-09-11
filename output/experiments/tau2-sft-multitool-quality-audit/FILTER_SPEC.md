# dependency-safe multi 筛选规范

实验名：`tau2-sft-multitool-quality-audit`。状态：audit-v1 定稿，downstream matched SFT
已验收。本文件定义候选数据标准，不是训练 JSONL；`FILTER_SPEC_DRAFT.md` 保留为 judge
前的预审草案。

## 证据结论

- 不再把“同一 assistant turn 有多个 tool call”本身作为违规或 drop reason。成功轨迹的
  matched 总体质量差（multi − single）为 `+0.005`，95% cluster bootstrap CI
  `[-0.034, +0.045]`，没有整体优劣证据；multi 的 argument grounding 更高
  `+0.136 [+0.074, +0.203]`，但 business-policy compliance 更低
  `-0.181 [-0.346, -0.012]`，说明调用数量不能替代语义审查。
- 不能回答“multi 总体更容易失败”。唯一有两类暴露且样本数足够的 telecom 中，
  consensus-failure 为 multi `64.31%`、non-multi `63.37%`，差 `+0.94pp`
  （95% CI `[-7.95pp, +9.82pp]`）；task ID 无交集，仍不是因果估计。airline/retail
  分别只有 5/24 条 non-multi，缺乏 common support。跨 domain 的表观差异禁止解释。
- 450 条非一致成功轨迹的首要原因是 `wrong_workflow`（289/450）；在 341 条
  multi-exposed 失败中，Luna 仅把 66/341（19.35%）归为 multi
  `caused/contributed`，244/341（71.55%）归为 `irrelevant`。决定性错误位于 multi
  turn 为 111/341，位于 single-call turn 为 101/341；“轨迹含 multi”“错误发生在
  multi turn”“batching 导致错误”是三个不同变量。
- 以全部 450 条归因池为固定分母，决定性错误位于 multi turn 为 111/450（24.67%），
  位于 single-call turn 为 200/450（44.44%），non-tool 为 52/450，environment/user
  为 7/450，unknown 为 80/450。这是失败构成，不是两种行为各自的失败概率。
- 失败归因是经 evidence gate 验证的 `gpt-5.6-luna` 主统计，Qwen3.6-27B 仅作敏感性
  检查；它不是 ground truth，也不进入 success keep 的循环 hard rule。

## 实验协议

An assistant turn may contain multiple tool calls only when every call is independently justified by information available before the turn, no call requires another call's result, and every state-changing call is explicitly authorized and order-independent. Otherwise, make one tool call and wait for its result. If you make tool calls, do not respond to the user in the same turn.

这允许只读或写入的 dependency-safe batching，不允许 observation-dependent chain、
未确认写入、顺序相关写入、无依据 fanout/retry，或 tool call 与用户回复同 turn 混合。
领域 policy 中的 “at most one tool call per turn” 只应先在该实验 profile 中替换为上文；
system、领域 policy、数据筛选和 eval prompt 必须使用同一协议。

## 决策单元与标签

- 行为判断使用 full-source dialog；训练暴露另存 longest retained raw prefix；最终监督
  单元是带 source-row、turn index、fingerprint 和 conversion hash 的 converted target。
  三个视图不得互换。
- `consensus_success`（reward 与 correct 均成功）才可自动进入正向 SFT keep；
  `consensus_failure`、`reward_only`、`correct_only` 均先留在固定分母和 review 队列，
  不能因某一个 evaluator 标签成功而自动保留。
- dialog-level 质量通过后，只监督获准 target。若要从失败或部分错误 dialog 中挽救
  早期 turn，必须人工/第三裁判确认该 turn 及其依赖闭包，不能沿用整段 dialog 标签。

## 三层筛选

`keep-candidate` 必须同时满足：

1. 标签为 `consensus_success`，两位 judge 均 reliable 且都给出 quality `keep`；
2. raw/converted non-system messages、system、target fingerprint 与 manifest hash 全部匹配；
3. 不含无效 tool/schema、namespace confusion、tool call 与用户回复同 turn 等机械 hard issue；
4. 每个 multi target 的两份逐 turn 结论逐字段一致，且属于
   `safe_independent_read`、`safe_independent_write` 或 `safe_independent_mixed`；
5. 参数全部由该 turn 前证据 grounded；写入已明确授权且 order-independent；每个调用
   都必要；不存在同批 observation dependency、无依据重复或 fanout；
6. 不含低置信度、unknown、未解决 judge 分歧或 window/full 冲突。

`review` 包含 label 冲突、judge verdict/逐 turn 分歧、低置信、隐藏 DB/evaluator 证据、
长上下文 route 不一致，以及尚未抽到语义复审的 target。`review` 不是 drop。

`not-keep-candidate` 用于当前正向 SFT 自动发布：确定性工具/转换错误、双 judge 一致的
关键流程或依赖错误，以及不满足上述任一 hard gate 的记录均不得自动发布。失败轨迹仍可
单独保留给 failure analysis、RFT/RL 或经 turn-level adjudication 后挽救。

## 当前产物验收

- `filter_decisions_draft.jsonl` 覆盖 2,496 dialogs、19,318 converted targets，且只含
  provenance/decision，不含训练 messages；文件 SHA256 为
  `cbc7679a43f4da12a18ab5281d3008914fcea114337350c5f8eabd3249f8f4ff`。
- 689 个复审 dialog 中有 114 个通过纯 judge 质量交集；加上 label、机械和逐 target
  gate 后只有 69 dialogs / 630 targets 是 `keep-candidate`。其中 50 个 full-source
  multi dialog、19 个 non-multi dialog；57/630 个 target 本身是 multi turn。
- 69 个 dialog 的领域构成为 retail 64、airline 5、telecom 0；15,332 个 target 尚未
  语义复审。因此这 630 条是高精度 seed，不是可直接替换原 SFT 数据的完整训练集。
- 质量复审两侧各 689/689、失败归因两侧各 450/450；quality 和 failure 两个 sealed
  calibration gate 均通过。完整统计见 `COMPARISON_REPORT.md`、`FAILURE_ATTRIBUTION.md`
  和 `summary.json`；共同 payload-manifest SHA256 为
  `ed69566b2b53220010f611eb7165e5b90cf04ab2db4d8999aa4e08e39923033a`。

## 训练与评测验收

- local-first 扩展使用本地 Qwen3.6-27B 覆盖全部 2,046 个 consensus-success dialogs，
  通过 target-prefix salvage 得到 1,863 dialogs / 15,807 canonical targets；19,318-row
  budget 与旧训练更新数对齐。既有 Luna 覆盖只作保守否决，未新增外部 API 调用。
- 新 SFT 从 raw Qwen3-4B-Instruct-2507 iteration 0 训练到 iteration 2,413。三模型 eval
  使用相同 `dependency-safe-multi` profile、v1 STOP user、100 tasks × 4 trials、seed 300，
  三者均为 400 simulations、0 infra errors。
- 新 SFT 的 pass@1 / pass@4(any) / pass^4 为 `27.00% / 54.00% / 4.00%`。相对 raw，
  前两项为 `+6.50pp [1.00, 12.00]`、`+11.00pp [2.00, 21.00]`；相对旧 SFT 为
  `+5.75pp [0.50, 11.00]`、`+13.00pp [2.00, 24.00]`。pass^4 相对旧 SFT 为
  `-6.00pp [-12.00, -1.00]`。
- 这验证了 protocol 与 broad-tier 筛选可保留，但没有完成纯 single-only 训练 arm，不能把
  模型差异解释为 batching 的因果效果。当前数据配方也因 pass^4 回退而不升级为唯一默认。
- 下一版在 broad tier 上增加分域双评审 consistency anchor；第二 eval seed 必须复现 breadth
  增益，且 pass^4 相对旧 SFT 的 paired CI 下界不低于预设 `-2pp` margin。完整决策见
  [FINAL_RECOMMENDATION.md](../tau2-sft-local-first-relaxed/FINAL_RECOMMENDATION.md)。
