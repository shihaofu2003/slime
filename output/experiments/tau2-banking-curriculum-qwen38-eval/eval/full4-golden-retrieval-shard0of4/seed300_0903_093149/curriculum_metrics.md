# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 52/95
- Accuracy: 54.74%
- Mean reward: 0.5474

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| decision_only | 22 | 24 | 91.67% |
| evidence_given | 16 | 24 | 66.67% |
| single_action | 11 | 24 | 45.83% |
| two_skill_composition | 3 | 23 | 13.04% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 16 | 24 | 66.67% |
| L3 | 33 | 48 | 68.75% |
| L4 | 3 | 23 | 13.04% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 52 | 95 | 54.74% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 2 | 4 | 50.00% |
| 交易争议、购买保护与补卡 | 7 | 8 | 87.50% |
| 信用卡保留、销户与竞品比较 | 3 | 4 | 75.00% |
| 信用卡推荐奖励 | 4 | 8 | 50.00% |
| 信用卡选择与申请资格 | 5 | 7 | 71.43% |
| 信用额度调整 | 2 | 4 | 50.00% |
| 借记卡交易争议 | 1 | 4 | 25.00% |
| 借记卡拒付与 PIN 锁定 | 1 | 4 | 25.00% |
| 储蓄利息核算 | 4 | 8 | 50.00% |
| 卡片遗失或被盗 | 2 | 4 | 50.00% |
| 知识缺失与异常转人工 | 2 | 4 | 50.00% |
| 账户推荐、开户、销户与入金 | 10 | 20 | 50.00% |
| 账户推荐奖励 | 1 | 4 | 25.00% |
| 身份验证与资料变更 | 3 | 4 | 75.00% |
| 返现与奖励核算 | 5 | 8 | 62.50% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 46, "mean": 0.3695652173913043}, "COMMUNICATE": {"checks": 72, "mean": 0.6805555555555556}, "DB": {"checks": 45, "mean": 0.5777777777777777}}`
- Atomic checks: `{"ACTION": {"checks": 68, "pass_rate": 0.4264705882352941, "passed": 29}, "COMMUNICATE": {"checks": 74, "pass_rate": 0.6891891891891891, "passed": 51}, "DB": {"checks": 95, "pass_rate": 0.8, "passed": 76}}`
- Action checks by requestor: `{"assistant": {"checks": 53, "pass_rate": 0.49056603773584906, "passed": 26}, "user": {"checks": 15, "pass_rate": 0.2, "passed": 3}}`
- Termination reasons: `{"user_stop": 95}`
- Failed task count: 43
