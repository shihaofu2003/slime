# tau2 SFT failure-attribution calibration

这是与 exact-consensus diagnostic pilot 不重叠的 72-dialog validation set。
用户指定的 gpt-5.6-luna 是主归因器；Qwen 只用于 sensitivity/contradiction
审计。gate 校验 Luna reliability、unknown 覆盖及 causal mode 与 sealed turn-type
证据的一致性，不把模型归因误称为 ground truth。

## Gate

Status：`pass`。

| gate | pass |
|---|---|
| `declared_failure_calibration_size_72` | yes |
| `fallback_reliable_at_least_0.90` | yes |
| `causal_mode_evidence_consistency_at_least_0.95` | yes |
| `fallback_primary_cause_unknown_at_most_0.30` | yes |
| `fallback_causal_mode_unknown_at_most_0.30` | yes |
| `all_n_ge_12_one_dimensional_margins_pass` | yes |

## Overall

- dialogs：72
- Luna reliable：70/72
- Luna causal-mode / decisive-turn consistency：68/70
- Luna primary cause unknown：11/70
- Luna causal mode unknown：11/70
- local/Luna exact agreement is diagnostic only：primary_failure_cause 41/72, causal_call_mode 39/72, multi_causal_role 26/72

## One-dimensional margins

只有 n >= 12 的 margin 进入 reliability/evidence-consistency gate。

| margin | dialogs | Luna reliable | mode evidence-consistent | gate |
|---|---:|---:|---:|---|
| `route=full/full` | 63 | 63 | 61 | pass |
| `route=window/full` | 9 | 7 | 7 | not_gated_n_lt_12 |
| `domain=airline` | 30 | 28 | 27 | pass |
| `domain=telecom` | 42 | 42 | 41 | pass |
| `label_group=consensus_failure` | 42 | 42 | 41 | pass |
| `label_group=reward_only` | 30 | 28 | 27 | pass |
| `multi_exposure=multi_exposed` | 51 | 49 | 47 | pass |
| `multi_exposure=non_multi_exposed` | 21 | 21 | 21 | pass |
