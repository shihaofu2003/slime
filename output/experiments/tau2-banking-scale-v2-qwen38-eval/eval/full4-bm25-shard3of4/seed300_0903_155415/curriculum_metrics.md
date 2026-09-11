# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 212/212
- Accuracy: 100.00%
- Mean reward: 1.0000

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| retrieval_only | 212 | 212 | 100.00% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L2 | 212 | 212 | 100.00% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| bm25 | 212 | 212 | 100.00% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 13 | 13 | 100.00% |
| 交易争议、购买保护与补卡 | 14 | 14 | 100.00% |
| 信用卡保留、销户与竞品比较 | 19 | 19 | 100.00% |
| 信用卡推荐奖励 | 5 | 5 | 100.00% |
| 信用卡选择与申请资格 | 11 | 11 | 100.00% |
| 借记卡交易争议 | 10 | 10 | 100.00% |
| 借记卡拒付与 PIN 锁定 | 26 | 26 | 100.00% |
| 储蓄利息核算 | 23 | 23 | 100.00% |
| 卡片遗失或被盗 | 11 | 11 | 100.00% |
| 知识缺失与异常转人工 | 1 | 1 | 100.00% |
| 账户推荐、开户、销户与入金 | 52 | 52 | 100.00% |
| 账户推荐奖励 | 12 | 12 | 100.00% |
| 身份验证与资料变更 | 1 | 1 | 100.00% |
| 返现与奖励核算 | 14 | 14 | 100.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 212, "mean": 1.0}, "COMMUNICATE": {"checks": 212, "mean": 1.0}}`
- Atomic checks: `{"ACTION": {"checks": 212, "pass_rate": 1.0, "passed": 212}, "COMMUNICATE": {"checks": 212, "pass_rate": 1.0, "passed": 212}, "DB": {"checks": 212, "pass_rate": 1.0, "passed": 212}}`
- Action checks by requestor: `{"assistant": {"checks": 212, "pass_rate": 1.0, "passed": 212}}`
- Termination reasons: `{"user_stop": 212}`
- Failed task count: 0
