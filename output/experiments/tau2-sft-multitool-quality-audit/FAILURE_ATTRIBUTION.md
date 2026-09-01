# tau2 SFT 失败归因

本报告以 `label_group != consensus_success` 的 full-source dialog 为归因分母；
reward、correct 与 consensus failure 另行报告。用户指定的 gpt-5.6-luna
给出主归因，Qwen 仅用于独立敏感性检查。Luna 低置信度或 causal mode
与 sealed turn type 不一致时计入 unknown，不从分母移除；模型归因不是
ground truth。

## 标签口径与覆盖

- 归因池（任一标签非成功）：450
- reward failure：333
- correct failure：445
- consensus failure：328
- discordant：reward-only 117；correct-only 5
- Luna reliable：434/450；evidence-usable：421/450
- Qwen reliable：432/450；evidence-usable：396/450

## Qwen sensitivity audit

Exact agreement 只作敏感性诊断，不决定 Luna 主归因。

| field | both reliable denominator | exact | exact rate |
|---|---:|---:|---:|
| `primary_failure_cause` | 416 | 265 | 63.70% |
| `causal_call_mode` | 416 | 236 | 56.73% |
| `multi_causal_role` | 310 | 62 | 20.00% |

| route | dialogs | Luna usable | Qwen usable | exact cause | exact mode |
|---|---:|---:|---:|---:|---:|
| full/full | 424 | 397 | 373 | 259 | 226 |
| window/full | 26 | 24 | 23 | 6 | 10 |

## Primary cause by label stratum

| cause | all non-consensus | consensus failure | reward-only | correct-only |
|---|---:|---:|---:|---:|
| `wrong_workflow` | 289 | 274 | 15 | 0 |
| `unknown` | 80 | 15 | 65 | 0 |
| `incomplete_task` | 37 | 22 | 12 | 3 |
| `tool_namespace_or_nonexistent` | 12 | 6 | 6 | 0 |
| `environment_or_user` | 9 | 2 | 5 | 2 |
| `wrong_arguments` | 9 | 1 | 8 | 0 |
| `false_success_claim` | 6 | 5 | 1 | 0 |
| `missing_precondition` | 4 | 0 | 4 | 0 |
| `confirmation_violation` | 3 | 3 | 0 | 0 |
| `unrelated_or_extra_action` | 1 | 0 | 1 | 0 |

## Decisive call mode by trajectory exposure

这里区分“轨迹含 multi”与“决定性错误发生在 multi turn”；两者不能互换。

| exposure | failures | multi-turn cause | single-call cause | non-tool | env/user | unknown |
|---|---:|---:|---:|---:|---:|---:|
| multi_exposed | 341 | 32.55% | 29.62% | 12.61% | 2.05% | 23.17% |
| non_multi_exposed | 109 | 0.00% | 90.83% | 8.26% | 0.00% | 0.92% |

| domain | exposure | failures | top primary cause | top-cause share | multi-turn cause | single-call cause | non-tool | env/user | unknown |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| airline | multi_exposed | 124 | `unknown` | 52.42% | 4.84% | 15.32% | 21.77% | 5.65% | 52.42% |
| airline | non_multi_exposed | 0 | `n/a` | n/a | n/a | n/a | n/a | n/a | n/a |
| retail | multi_exposed | 8 | `wrong_workflow` | 50.00% | 37.50% | 25.00% | 12.50% | 0.00% | 25.00% |
| retail | non_multi_exposed | 0 | `n/a` | n/a | n/a | n/a | n/a | n/a | n/a |
| telecom | multi_exposed | 209 | `wrong_workflow` | 83.73% | 48.80% | 38.28% | 7.18% | 0.00% | 5.74% |
| telecom | non_multi_exposed | 109 | `wrong_workflow` | 87.16% | 0.00% | 90.83% | 8.26% | 0.00% | 0.92% |

## Multi causal role

Full-source multi non-consensus trajectories：341/450。
其中 Luna 归因为 caused/contributed：66/341。
这回答的是失败轨迹内部的归因构成，不等于 multi 相对 single 的失败风险；
失败风险应看确定性、按 domain/label 分层或匹配后的比较。

| role | dialogs | rate over multi-exposed failures |
|---|---:|---:|
| `irrelevant` | 244 | 71.55% |
| `caused` | 57 | 16.72% |
| `uncertain` | 29 | 8.50% |
| `contributed` | 9 | 2.64% |
| `beneficial` | 2 | 0.59% |
