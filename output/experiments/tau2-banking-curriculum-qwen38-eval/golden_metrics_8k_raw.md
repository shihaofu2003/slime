# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 203/375
- Accuracy: 54.13%
- Mean reward: 0.5413

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| decision_only | 84 | 94 | 89.36% |
| evidence_given | 58 | 94 | 61.70% |
| single_action | 48 | 94 | 51.06% |
| two_skill_composition | 13 | 93 | 13.98% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 58 | 94 | 61.70% |
| L3 | 132 | 188 | 70.21% |
| L4 | 13 | 93 | 13.98% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 203 | 375 | 54.13% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 8 | 20 | 40.00% |
| 交易争议、购买保护与补卡 | 21 | 36 | 58.33% |
| 信用卡保留、销户与竞品比较 | 15 | 28 | 53.57% |
| 信用卡推荐奖励 | 11 | 16 | 68.75% |
| 信用卡选择与申请资格 | 21 | 35 | 60.00% |
| 信用额度调整 | 4 | 8 | 50.00% |
| 借记卡交易争议 | 7 | 16 | 43.75% |
| 借记卡拒付与 PIN 锁定 | 12 | 24 | 50.00% |
| 储蓄利息核算 | 10 | 20 | 50.00% |
| 卡片遗失或被盗 | 5 | 16 | 31.25% |
| 知识缺失与异常转人工 | 14 | 20 | 70.00% |
| 账户推荐、开户、销户与入金 | 40 | 68 | 58.82% |
| 账户推荐奖励 | 11 | 20 | 55.00% |
| 身份验证与资料变更 | 6 | 8 | 75.00% |
| 返现与奖励核算 | 18 | 40 | 45.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 186, "mean": 0.3548387096774194}, "COMMUNICATE": {"checks": 280, "mean": 0.6535714285714286}, "DB": {"checks": 171, "mean": 0.5380116959064327}}`
- Atomic checks: `{"ACTION": {"checks": 270, "pass_rate": 0.40370370370370373, "passed": 109}, "COMMUNICATE": {"checks": 287, "pass_rate": 0.6585365853658537, "passed": 189}, "DB": {"checks": 374, "pass_rate": 0.7834224598930482, "passed": 293}}`
- Action checks by requestor: `{"assistant": {"checks": 204, "pass_rate": 0.45098039215686275, "passed": 92}, "user": {"checks": 66, "pass_rate": 0.25757575757575757, "passed": 17}}`
- Termination reasons: `{"infrastructure_error": 1, "user_stop": 374}`
- Failed task count: 172
- Framework retry events: 9 across 5 tasks
- Conservative first-attempt exact accuracy: 201/375 (53.60%)
