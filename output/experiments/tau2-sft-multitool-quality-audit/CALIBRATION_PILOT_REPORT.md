# tau2 SFT multi-tool quality calibration pilot

该 72-dialog 集合是 r7 的 diagnostic pilot，不用于验证随后依据其诊断结果修订的 gate。
它与 seed `20260803` 生成的正式 validation cohort 必须完全不重叠。pilot selection hash
为 `828942da48b12db21c33075324896eda4a5065e44ba31cfcce95f38f5fffd21a`。

## Result

- Qwen3.6-27B 与 `gpt-5.6-luna` 均为 72/72 reliable；full/full 64 条，window/full 8 条。
- 严格三分类 verdict 一致为 36/72，但不存在 `keep/drop` 极端冲突。
- Luna 的 14 个 keep 全部同时被 Qwen 判为 keep；Qwen 比 Luna 恰好宽松一级 35/72，
  反向只出现 1/72，表明主要差异是系统性的 severity scale，而非安全结论翻转。
- batch safety 一致 61/72，multi-turn exact 一致 38/51，维度平均绝对差 0.4244，
  median confidence 0.95，机械 hard both-keep 0/28。

| Qwen \ Luna | keep | review | drop |
|---|---:|---:|---:|
| keep | 14 | 23 | 0 |
| review | 0 | 22 | 12 |
| drop | 0 | 1 | 0 |

## Interpretation

原 gate 仅因 `verdict_agreement >= 0.80` 失败，其余七项均通过。最终候选本来就取两个
judge 的 keep 交集，因此正式 validation gate 改为检验：Luna keep 是否被本地 judge
支持、是否存在 keep/drop 极端冲突，以及原有 batch/per-turn/reliability/dimension/
mechanical 条件。新 gate 的阈值只在独立、预先封存的 72 条 validation cohort 上验收。
