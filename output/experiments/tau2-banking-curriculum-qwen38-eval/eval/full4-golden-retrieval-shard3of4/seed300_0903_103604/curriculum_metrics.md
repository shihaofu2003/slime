# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 51/92
- Accuracy: 55.43%
- Mean reward: 0.5543

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| decision_only | 20 | 23 | 86.96% |
| evidence_given | 14 | 23 | 60.87% |
| single_action | 12 | 23 | 52.17% |
| two_skill_composition | 5 | 23 | 21.74% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 14 | 23 | 60.87% |
| L3 | 32 | 46 | 69.57% |
| L4 | 5 | 23 | 21.74% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 51 | 92 | 55.43% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 1 | 4 | 25.00% |
| 交易争议、购买保护与补卡 | 8 | 12 | 66.67% |
| 信用卡保留、销户与竞品比较 | 5 | 8 | 62.50% |
| 信用卡推荐奖励 | 2 | 4 | 50.00% |
| 信用卡选择与申请资格 | 4 | 8 | 50.00% |
| 借记卡交易争议 | 2 | 4 | 50.00% |
| 借记卡拒付与 PIN 锁定 | 5 | 8 | 62.50% |
| 储蓄利息核算 | 2 | 4 | 50.00% |
| 卡片遗失或被盗 | 0 | 4 | 0.00% |
| 知识缺失与异常转人工 | 3 | 4 | 75.00% |
| 账户推荐、开户、销户与入金 | 10 | 16 | 62.50% |
| 账户推荐奖励 | 2 | 4 | 50.00% |
| 身份验证与资料变更 | 3 | 4 | 75.00% |
| 返现与奖励核算 | 4 | 8 | 50.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 46, "mean": 0.391304347826087}, "COMMUNICATE": {"checks": 69, "mean": 0.6231884057971014}, "DB": {"checks": 40, "mean": 0.525}}`
- Atomic checks: `{"ACTION": {"checks": 67, "pass_rate": 0.417910447761194, "passed": 28}, "COMMUNICATE": {"checks": 73, "pass_rate": 0.6301369863013698, "passed": 46}, "DB": {"checks": 92, "pass_rate": 0.7934782608695652, "passed": 73}}`
- Action checks by requestor: `{"assistant": {"checks": 54, "pass_rate": 0.46296296296296297, "passed": 25}, "user": {"checks": 13, "pass_rate": 0.23076923076923078, "passed": 3}}`
- Termination reasons: `{"user_stop": 92}`
- Failed task count: 41
