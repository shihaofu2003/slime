# tau2 SFT multi-tool 审计设计

实验名：`tau2-sft-multitool-quality-audit`。本实验保留现有 single-policy 结果作为
baseline，不编辑 tau2 policy，也不直接生成训练 JSONL。

## 现有筛选的问题

- legacy deterministic gate 把任意 multi turn 直接记为
  `multi_tool_policy_violation`，并在 judge 前进入 drop；因此不能用该结果检验 multi
  是否安全或更好。
- legacy judge system prompt 同样把 multi 定义为违规，`SUCCESS_CASES.md` 又把
  “每轮最多一个 tool call”写成成功共性；三者共享同一前提，不是独立证据。
- source `correct` 与 `reward` 曾合并为单一正/负结论；本实验保留
  consensus-success、consensus-failure、reward-only、correct-only 四组。
- raw full dialog、最长 retained raw prefix 和实际 converted training row 是不同对象。
  本实验分别保存行为事实与 converted row/system hash，并逐行验证转换。

## 待回答的问题

- 全量样本分别估计 reward/correct/consensus failure 在 multi/non-multi 下的观察比例；
  domain 缺少 common support 时不作因果判断。
- 对所有非 consensus-success 轨迹做失败归因，区分“轨迹出现 multi”与“决定性错误
  发生在 multi turn”。
- 对所有 single consensus-success 和领域/任务匹配的 multi controls 做双 judge
  示范质量比较；success-conditioned 结果不用于估计失败概率。

## 候选协议

An assistant turn may contain multiple tool calls only when every call is independently justified by information available before the turn, no call requires another call's result, and every state-changing call is explicitly authorized and order-independent. Otherwise, make one tool call and wait for its result. If you make tool calls, do not respond to the user in the same turn.

## 有效性边界

完整轨迹只交给可容纳其长度的 fallback；本地 judge 对超长轨迹读取显式记录省略项的
structured window。full/full 与 window/full 分开报告。payload、prompt version、模型配置、
输入数据和 converted artifact 均带 hash；陈旧、缺失或同 provider 的复审不能进入汇总。
