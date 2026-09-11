# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 244/244
- Accuracy: 100.00%
- Mean reward: 1.0000

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| retrieval_only | 244 | 244 | 100.00% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L2 | 244 | 244 | 100.00% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| bm25 | 244 | 244 | 100.00% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 25 | 25 | 100.00% |
| 交易争议、购买保护与补卡 | 8 | 8 | 100.00% |
| 信用卡保留、销户与竞品比较 | 3 | 3 | 100.00% |
| 信用卡推荐奖励 | 5 | 5 | 100.00% |
| 信用卡选择与申请资格 | 10 | 10 | 100.00% |
| 信用额度调整 | 6 | 6 | 100.00% |
| 借记卡交易争议 | 7 | 7 | 100.00% |
| 借记卡拒付与 PIN 锁定 | 10 | 10 | 100.00% |
| 储蓄利息核算 | 36 | 36 | 100.00% |
| 卡片遗失或被盗 | 19 | 19 | 100.00% |
| 知识缺失与异常转人工 | 1 | 1 | 100.00% |
| 账户推荐、开户、销户与入金 | 73 | 73 | 100.00% |
| 账户推荐奖励 | 21 | 21 | 100.00% |
| 身份验证与资料变更 | 1 | 1 | 100.00% |
| 返现与奖励核算 | 19 | 19 | 100.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 244, "mean": 1.0}, "COMMUNICATE": {"checks": 244, "mean": 1.0}}`
- Atomic checks: `{"ACTION": {"checks": 244, "pass_rate": 1.0, "passed": 244}, "COMMUNICATE": {"checks": 244, "pass_rate": 1.0, "passed": 244}, "DB": {"checks": 244, "pass_rate": 1.0, "passed": 244}}`
- Action checks by requestor: `{"assistant": {"checks": 244, "pass_rate": 1.0, "passed": 244}}`
- Termination reasons: `{"user_stop": 244}`
- Failed task count: 0
