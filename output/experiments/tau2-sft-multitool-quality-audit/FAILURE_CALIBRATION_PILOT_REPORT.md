# tau2 SFT failure-attribution calibration pilot

seed `20260802` 的 72-dialog stress set 是 failure-attribution diagnostic pilot，selection
hash 为 `ee00c216069b62c8fc764c9db4627388bfd0e35348564a2bd9a55a5c8c9f4c15`。它不用于
验证根据本次诊断修订的 Luna-primary evidence gate。

## Result

- Qwen3.6-27B 与 `gpt-5.6-luna`：67/72 双方 reliable。
- 强制 exact consensus 时，primary cause 32/72、causal call mode 35/72、multi role
  12/58 resolved；原 gate 正确失败，未启动 full failure review。
- 分歧主要集中在 multi-exposed、airline/reward-discordant 和 window/full；non-multi
  margin 的 cause/mode exact resolution 分别为 11/14、10/14。
- Luna 单方 reliable 69/72；其中 causal mode 与 `decisive_turns` 的实际 turn 类型
  68/69（98.55%）机械一致。Qwen/Luna decisive-turn sets 仅 7/67 完全相同，说明主要
  不稳定项是“必须选同一个唯一主因”，不是 Luna 引用与自身分类不一致。

## Decision

后续把用户指定的 Luna 作为主归因器，Qwen 作为独立 sensitivity/contradiction audit。
gate 校验 Luna reliability、unknown 覆盖和 causal-mode/decisive-turn 机械一致性；不会把
模型归因称为 ground truth。所有 judge disagreement 和 unknown 仍显式报告，且 failure
cause 不进入 keep-candidate 的硬筛选规则。新 gate 只在不重叠 validation set 上验收。
