# tau2 SFT multi-tool judge calibration

这是与两个 frozen diagnostic cohort 都不重叠、由 prepare 前置封存的
72-dialog validation-v2 set。
最终筛选只保留 local 与 fallback 的 keep 交集，因此 verdict gate 检验更严格
fallback keep 是否得到 local 支持；local keep / fallback drop 会被交集安全否决，
只作为方向性诊断报告。三分类 exact agreement 仍报告但不把 judge 的系统性
尺度偏移误当成不安全 keep。
一致性不等于 ground-truth accuracy；gate 失败时不得发布自动 keep 数据。

## Gate

Status：`pass`。

| gate | pass |
|---|---|
| `declared_quality_calibration_size_72` | yes |
| `fallback_keep_coverage_nonzero` | yes |
| `fallback_keep_supported_by_local_at_least_0.90` | yes |
| `batch_agreement_at_least_0.80` | yes |
| `both_reliable_at_least_0.90` | yes |
| `per_turn_agreement_at_least_0.70` | yes |
| `mean_dimension_difference_at_most_0.75` | yes |
| `median_confidence_at_least_0.70` | yes |
| `mechanical_hard_coverage_nonzero` | yes |
| `mechanical_false_keep_at_most_0.05` | yes |

## Metrics

- dialogs：72 (full/full 64; window/full 8)
- verdict agreement：40/72
- fallback keep supported by local：21/21
- extreme keep/drop disagreements：3/72
- local keep / fallback drop (conservative veto)：3/72
- fallback keep / local non-keep：0/21
- batch agreement：68/72
- both reliable：72/72
- all-multi-turn exact agreement：44/48 multi dialogs
- mean absolute dimension difference：0.4243827160493827
- median confidence：0.95
- mechanical hard both-keep：0/33

| route | dialogs | both reliable | verdict agree | batch agree | all-multi-turn exact |
|---|---:|---:|---:|---:|---:|
| full/full | 64 | 100.00% | 56.25% | 95.31% | 92.50% |
| window/full | 8 | 100.00% | 50.00% | 87.50% | 87.50% |
