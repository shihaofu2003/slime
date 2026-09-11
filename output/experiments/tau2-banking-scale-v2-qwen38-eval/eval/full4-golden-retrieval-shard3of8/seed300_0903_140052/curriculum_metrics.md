# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 134/198
- Accuracy: 67.68%
- Mean reward: 0.6768

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| evidence_given | 56 | 91 | 61.54% |
| single_action | 78 | 107 | 72.90% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 56 | 91 | 61.54% |
| L3 | 78 | 107 | 72.90% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 134 | 198 | 67.68% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 16 | 26 | 61.54% |
| 交易争议、购买保护与补卡 | 15 | 19 | 78.95% |
| 信用卡保留、销户与竞品比较 | 20 | 27 | 74.07% |
| 信用卡推荐奖励 | 5 | 7 | 71.43% |
| 信用卡选择与申请资格 | 6 | 7 | 85.71% |
| 借记卡交易争议 | 11 | 22 | 50.00% |
| 借记卡拒付与 PIN 锁定 | 20 | 33 | 60.61% |
| 知识缺失与异常转人工 | 3 | 6 | 50.00% |
| 账户推荐、开户、销户与入金 | 27 | 33 | 81.82% |
| 账户推荐奖励 | 10 | 16 | 62.50% |
| 身份验证与资料变更 | 1 | 2 | 50.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 107, "mean": 0.7289719626168224}, "COMMUNICATE": {"checks": 91, "mean": 0.6153846153846154}, "DB": {"checks": 10, "mean": 0.7}}`
- Atomic checks: `{"ACTION": {"checks": 107, "pass_rate": 0.7289719626168224, "passed": 78}, "COMMUNICATE": {"checks": 91, "pass_rate": 0.6153846153846154, "passed": 56}, "DB": {"checks": 198, "pass_rate": 0.9393939393939394, "passed": 186}}`
- Action checks by requestor: `{"assistant": {"checks": 99, "pass_rate": 0.7373737373737373, "passed": 73}, "user": {"checks": 8, "pass_rate": 0.625, "passed": 5}}`
- Termination reasons: `{"user_stop": 198}`
- Failed task count: 64
