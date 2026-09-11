# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 160/244
- Accuracy: 65.57%
- Mean reward: 0.6557

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| evidence_given | 63 | 116 | 54.31% |
| single_action | 97 | 128 | 75.78% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 63 | 116 | 54.31% |
| L3 | 97 | 128 | 75.78% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 160 | 244 | 65.57% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 10 | 21 | 47.62% |
| 交易争议、购买保护与补卡 | 22 | 29 | 75.86% |
| 信用卡保留、销户与竞品比较 | 24 | 34 | 70.59% |
| 信用卡推荐奖励 | 3 | 6 | 50.00% |
| 信用卡选择与申请资格 | 6 | 6 | 100.00% |
| 借记卡交易争议 | 13 | 27 | 48.15% |
| 借记卡拒付与 PIN 锁定 | 27 | 38 | 71.05% |
| 账户推荐、开户、销户与入金 | 31 | 41 | 75.61% |
| 账户推荐奖励 | 9 | 17 | 52.94% |
| 返现与奖励核算 | 15 | 25 | 60.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 128, "mean": 0.7578125}, "COMMUNICATE": {"checks": 116, "mean": 0.5431034482758621}, "DB": {"checks": 11, "mean": 0.9090909090909091}}`
- Atomic checks: `{"ACTION": {"checks": 128, "pass_rate": 0.7578125, "passed": 97}, "COMMUNICATE": {"checks": 116, "pass_rate": 0.5431034482758621, "passed": 63}, "DB": {"checks": 244, "pass_rate": 0.9672131147540983, "passed": 236}}`
- Action checks by requestor: `{"assistant": {"checks": 112, "pass_rate": 0.8125, "passed": 91}, "user": {"checks": 16, "pass_rate": 0.375, "passed": 6}}`
- Termination reasons: `{"user_stop": 244}`
- Failed task count: 84
