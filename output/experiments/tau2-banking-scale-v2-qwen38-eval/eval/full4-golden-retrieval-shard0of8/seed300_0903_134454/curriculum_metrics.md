# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 173/251
- Accuracy: 68.92%
- Mean reward: 0.6892

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| evidence_given | 87 | 137 | 63.50% |
| single_action | 86 | 114 | 75.44% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 87 | 137 | 63.50% |
| L3 | 86 | 114 | 75.44% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 173 | 251 | 68.92% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| 交易争议、购买保护与补卡 | 11 | 12 | 91.67% |
| 信用卡保留、销户与竞品比较 | 5 | 6 | 83.33% |
| 信用卡推荐奖励 | 2 | 4 | 50.00% |
| 信用卡选择与申请资格 | 5 | 5 | 100.00% |
| 借记卡拒付与 PIN 锁定 | 13 | 23 | 56.52% |
| 储蓄利息核算 | 27 | 43 | 62.79% |
| 卡片遗失或被盗 | 39 | 49 | 79.59% |
| 账户推荐、开户、销户与入金 | 47 | 69 | 68.12% |
| 返现与奖励核算 | 24 | 40 | 60.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 114, "mean": 0.7543859649122807}, "COMMUNICATE": {"checks": 137, "mean": 0.635036496350365}, "DB": {"checks": 12, "mean": 0.9166666666666666}}`
- Atomic checks: `{"ACTION": {"checks": 114, "pass_rate": 0.7543859649122807, "passed": 86}, "COMMUNICATE": {"checks": 137, "pass_rate": 0.635036496350365, "passed": 87}, "DB": {"checks": 251, "pass_rate": 0.9601593625498008, "passed": 241}}`
- Action checks by requestor: `{"assistant": {"checks": 99, "pass_rate": 0.8282828282828283, "passed": 82}, "user": {"checks": 15, "pass_rate": 0.26666666666666666, "passed": 4}}`
- Termination reasons: `{"user_stop": 251}`
- Failed task count: 78
