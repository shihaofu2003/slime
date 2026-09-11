# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 221/221
- Accuracy: 100.00%
- Mean reward: 1.0000

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| retrieval_only | 221 | 221 | 100.00% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L2 | 221 | 221 | 100.00% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| bm25 | 221 | 221 | 100.00% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 10 | 10 | 100.00% |
| 交易争议、购买保护与补卡 | 17 | 17 | 100.00% |
| 信用卡保留、销户与竞品比较 | 16 | 16 | 100.00% |
| 信用卡推荐奖励 | 5 | 5 | 100.00% |
| 信用卡选择与申请资格 | 9 | 9 | 100.00% |
| 借记卡交易争议 | 12 | 12 | 100.00% |
| 借记卡拒付与 PIN 锁定 | 23 | 23 | 100.00% |
| 储蓄利息核算 | 15 | 15 | 100.00% |
| 卡片遗失或被盗 | 11 | 11 | 100.00% |
| 知识缺失与异常转人工 | 1 | 1 | 100.00% |
| 账户推荐、开户、销户与入金 | 59 | 59 | 100.00% |
| 账户推荐奖励 | 12 | 12 | 100.00% |
| 返现与奖励核算 | 31 | 31 | 100.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 221, "mean": 1.0}, "COMMUNICATE": {"checks": 221, "mean": 1.0}}`
- Atomic checks: `{"ACTION": {"checks": 221, "pass_rate": 1.0, "passed": 221}, "COMMUNICATE": {"checks": 221, "pass_rate": 1.0, "passed": 221}, "DB": {"checks": 221, "pass_rate": 1.0, "passed": 221}}`
- Action checks by requestor: `{"assistant": {"checks": 221, "pass_rate": 1.0, "passed": 221}}`
- Termination reasons: `{"user_stop": 221}`
- Failed task count: 0
