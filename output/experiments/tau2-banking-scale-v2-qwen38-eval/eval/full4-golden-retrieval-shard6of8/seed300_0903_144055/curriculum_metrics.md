# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 144/220
- Accuracy: 65.45%
- Mean reward: 0.6545

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| evidence_given | 57 | 105 | 54.29% |
| single_action | 87 | 115 | 75.65% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 57 | 105 | 54.29% |
| L3 | 87 | 115 | 75.65% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 144 | 220 | 65.45% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| 交易争议、购买保护与补卡 | 20 | 25 | 80.00% |
| 信用卡保留、销户与竞品比较 | 12 | 16 | 75.00% |
| 信用卡选择与申请资格 | 4 | 5 | 80.00% |
| 借记卡拒付与 PIN 锁定 | 16 | 30 | 53.33% |
| 储蓄利息核算 | 12 | 24 | 50.00% |
| 卡片遗失或被盗 | 22 | 32 | 68.75% |
| 知识缺失与异常转人工 | 3 | 4 | 75.00% |
| 账户推荐、开户、销户与入金 | 35 | 47 | 74.47% |
| 返现与奖励核算 | 20 | 37 | 54.05% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 115, "mean": 0.7565217391304347}, "COMMUNICATE": {"checks": 105, "mean": 0.5428571428571428}, "DB": {"checks": 10, "mean": 0.9}}`
- Atomic checks: `{"ACTION": {"checks": 115, "pass_rate": 0.7565217391304347, "passed": 87}, "COMMUNICATE": {"checks": 105, "pass_rate": 0.5428571428571428, "passed": 57}, "DB": {"checks": 220, "pass_rate": 0.9409090909090909, "passed": 207}}`
- Action checks by requestor: `{"assistant": {"checks": 101, "pass_rate": 0.8217821782178217, "passed": 83}, "user": {"checks": 14, "pass_rate": 0.2857142857142857, "passed": 4}}`
- Termination reasons: `{"user_stop": 220}`
- Failed task count: 76
