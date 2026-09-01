# tau2 SFT multi-tool quality validation v1

seed `20260803` 的 72-dialog cohort 与 diagnostic pilot 完全不重叠，selection hash 为
`30b6141492987d40c2cfbfbd8e6d695989bb6a2eebcb23eb4d759d9e5a9e59a4`。该集合冻结为
validation-v1 diagnostic，不用于验证根据其结果修订的非对称 gate。

## Result

- Qwen3.6-27B 与 `gpt-5.6-luna`：70/72 双方 reliable；full/full 66，window/full 6。
- Luna 的 19 个 keep 全部被 Qwen keep 支持；机械 hard both-keep 0/30。
- batch safety 一致 63/72，multi-turn exact 一致 37/48，维度平均绝对差 0.4476，
  median confidence 0.95。
- 唯一失败项为对称 `keep/drop <= 5%`：4/70（5.71%），超过阈值 1 条。

| Qwen \ Luna | keep | review | drop |
|---|---:|---:|---:|
| keep | 19 | 17 | 4 |
| review | 0 | 18 | 14 |
| drop | 0 | 0 | 0 |

## Directionality diagnosis

四个极端冲突全是 `Qwen=keep, Luna=drop`，没有 `Luna=keep, Qwen!=keep`。它们是
telecom `10`、`374` 与 airline `airline_dialog_109`、`airline_dialog_721`；2 条 single、
2 条 multi，争议均为非 batching 的领域流程/信息约束。最终 keep 采用两位 judge 的交集，
所以这四条都被保守否决，不会形成 unsafe keep。v2 gate 只约束可能进入交集的方向：
fallback keep 必须得到 local keep 支持；v2 阈值在 seed `20260804` 的第三个不重叠 cohort
上验证。
