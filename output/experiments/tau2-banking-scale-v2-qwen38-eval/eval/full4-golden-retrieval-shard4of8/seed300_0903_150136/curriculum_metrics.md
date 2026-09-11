# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 139/197
- Accuracy: 70.56%
- Mean reward: 0.7056

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| evidence_given | 73 | 107 | 68.22% |
| single_action | 66 | 90 | 73.33% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 73 | 107 | 68.22% |
| L3 | 66 | 90 | 73.33% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 139 | 197 | 70.56% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 22 | 28 | 78.57% |
| 交易争议、购买保护与补卡 | 20 | 29 | 68.97% |
| 信用卡推荐奖励 | 3 | 5 | 60.00% |
| 信用卡选择与申请资格 | 7 | 7 | 100.00% |
| 信用额度调整 | 14 | 19 | 73.68% |
| 借记卡交易争议 | 9 | 17 | 52.94% |
| 储蓄利息核算 | 12 | 20 | 60.00% |
| 知识缺失与异常转人工 | 4 | 6 | 66.67% |
| 账户推荐、开户、销户与入金 | 25 | 37 | 67.57% |
| 账户推荐奖励 | 19 | 25 | 76.00% |
| 身份验证与资料变更 | 4 | 4 | 100.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 90, "mean": 0.7333333333333333}, "COMMUNICATE": {"checks": 107, "mean": 0.6822429906542056}, "DB": {"checks": 11, "mean": 0.8181818181818182}}`
- Atomic checks: `{"ACTION": {"checks": 90, "pass_rate": 0.7333333333333333, "passed": 66}, "COMMUNICATE": {"checks": 107, "pass_rate": 0.6822429906542056, "passed": 73}, "DB": {"checks": 197, "pass_rate": 0.934010152284264, "passed": 184}}`
- Action checks by requestor: `{"assistant": {"checks": 80, "pass_rate": 0.7125, "passed": 57}, "user": {"checks": 10, "pass_rate": 0.9, "passed": 9}}`
- Termination reasons: `{"user_stop": 197}`
- Failed task count: 58
