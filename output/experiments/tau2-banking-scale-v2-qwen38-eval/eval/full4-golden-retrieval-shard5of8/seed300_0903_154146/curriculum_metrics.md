# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 157/224
- Accuracy: 70.09%
- Mean reward: 0.7009

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| evidence_given | 91 | 134 | 67.91% |
| single_action | 66 | 90 | 73.33% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 91 | 134 | 67.91% |
| L3 | 66 | 90 | 73.33% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 157 | 224 | 70.09% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 26 | 27 | 96.30% |
| 信用卡保留、销户与竞品比较 | 16 | 22 | 72.73% |
| 信用卡选择与申请资格 | 15 | 18 | 83.33% |
| 信用额度调整 | 17 | 20 | 85.00% |
| 借记卡交易争议 | 14 | 32 | 43.75% |
| 储蓄利息核算 | 12 | 22 | 54.55% |
| 知识缺失与异常转人工 | 8 | 11 | 72.73% |
| 账户推荐、开户、销户与入金 | 28 | 40 | 70.00% |
| 账户推荐奖励 | 19 | 24 | 79.17% |
| 返现与奖励核算 | 2 | 8 | 25.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 90, "mean": 0.7333333333333333}, "COMMUNICATE": {"checks": 134, "mean": 0.6791044776119403}, "DB": {"checks": 11, "mean": 0.7272727272727273}}`
- Atomic checks: `{"ACTION": {"checks": 90, "pass_rate": 0.7333333333333333, "passed": 66}, "COMMUNICATE": {"checks": 134, "pass_rate": 0.6791044776119403, "passed": 91}, "DB": {"checks": 224, "pass_rate": 0.9419642857142857, "passed": 211}}`
- Action checks by requestor: `{"assistant": {"checks": 76, "pass_rate": 0.7763157894736842, "passed": 59}, "user": {"checks": 14, "pass_rate": 0.5, "passed": 7}}`
- Termination reasons: `{"user_stop": 224}`
- Failed task count: 67
