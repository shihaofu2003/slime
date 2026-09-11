# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 255/256
- Accuracy: 99.61%
- Mean reward: 0.9961

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| retrieval_only | 255 | 256 | 99.61% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L2 | 255 | 256 | 99.61% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| bm25 | 255 | 256 | 99.61% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 33 | 33 | 100.00% |
| 交易争议、购买保护与补卡 | 4 | 4 | 100.00% |
| 信用卡保留、销户与竞品比较 | 19 | 19 | 100.00% |
| 信用卡选择与申请资格 | 22 | 22 | 100.00% |
| 信用额度调整 | 7 | 7 | 100.00% |
| 借记卡交易争议 | 14 | 14 | 100.00% |
| 借记卡拒付与 PIN 锁定 | 12 | 12 | 100.00% |
| 储蓄利息核算 | 13 | 13 | 100.00% |
| 卡片遗失或被盗 | 13 | 13 | 100.00% |
| 知识缺失与异常转人工 | 3 | 3 | 100.00% |
| 账户推荐、开户、销户与入金 | 54 | 54 | 100.00% |
| 账户推荐奖励 | 39 | 40 | 97.50% |
| 返现与奖励核算 | 22 | 22 | 100.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 256, "mean": 1.0}, "COMMUNICATE": {"checks": 256, "mean": 0.99609375}}`
- Atomic checks: `{"ACTION": {"checks": 256, "pass_rate": 1.0, "passed": 256}, "COMMUNICATE": {"checks": 256, "pass_rate": 0.99609375, "passed": 255}, "DB": {"checks": 256, "pass_rate": 1.0, "passed": 256}}`
- Action checks by requestor: `{"assistant": {"checks": 256, "pass_rate": 1.0, "passed": 256}}`
- Termination reasons: `{"user_stop": 256}`
- Failed task count: 1
