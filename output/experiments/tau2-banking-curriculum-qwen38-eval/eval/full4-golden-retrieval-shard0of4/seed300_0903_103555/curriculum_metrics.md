# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 54/95
- Accuracy: 56.84%
- Mean reward: 0.5684

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| decision_only | 22 | 24 | 91.67% |
| evidence_given | 15 | 24 | 62.50% |
| single_action | 14 | 24 | 58.33% |
| two_skill_composition | 3 | 23 | 13.04% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 15 | 24 | 62.50% |
| L3 | 36 | 48 | 75.00% |
| L4 | 3 | 23 | 13.04% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 54 | 95 | 56.84% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 3 | 4 | 75.00% |
| 交易争议、购买保护与补卡 | 6 | 8 | 75.00% |
| 信用卡保留、销户与竞品比较 | 1 | 4 | 25.00% |
| 信用卡推荐奖励 | 6 | 8 | 75.00% |
| 信用卡选择与申请资格 | 5 | 7 | 71.43% |
| 信用额度调整 | 2 | 4 | 50.00% |
| 借记卡交易争议 | 1 | 4 | 25.00% |
| 借记卡拒付与 PIN 锁定 | 1 | 4 | 25.00% |
| 储蓄利息核算 | 4 | 8 | 50.00% |
| 卡片遗失或被盗 | 2 | 4 | 50.00% |
| 知识缺失与异常转人工 | 3 | 4 | 75.00% |
| 账户推荐、开户、销户与入金 | 11 | 20 | 55.00% |
| 账户推荐奖励 | 1 | 4 | 25.00% |
| 身份验证与资料变更 | 3 | 4 | 75.00% |
| 返现与奖励核算 | 5 | 8 | 62.50% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 46, "mean": 0.391304347826087}, "COMMUNICATE": {"checks": 72, "mean": 0.6388888888888888}, "DB": {"checks": 45, "mean": 0.5555555555555556}}`
- Atomic checks: `{"ACTION": {"checks": 68, "pass_rate": 0.4117647058823529, "passed": 28}, "COMMUNICATE": {"checks": 74, "pass_rate": 0.6486486486486487, "passed": 48}, "DB": {"checks": 95, "pass_rate": 0.7789473684210526, "passed": 74}}`
- Action checks by requestor: `{"assistant": {"checks": 53, "pass_rate": 0.4339622641509434, "passed": 23}, "user": {"checks": 15, "pass_rate": 0.3333333333333333, "passed": 5}}`
- Termination reasons: `{"user_stop": 95}`
- Failed task count: 41
