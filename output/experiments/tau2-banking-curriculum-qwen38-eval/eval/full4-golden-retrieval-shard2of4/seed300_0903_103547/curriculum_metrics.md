# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 48/92
- Accuracy: 52.17%
- Mean reward: 0.5217

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| decision_only | 20 | 23 | 86.96% |
| evidence_given | 14 | 23 | 60.87% |
| single_action | 11 | 23 | 47.83% |
| two_skill_composition | 3 | 23 | 13.04% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 14 | 23 | 60.87% |
| L3 | 31 | 46 | 67.39% |
| L4 | 3 | 23 | 13.04% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 48 | 92 | 52.17% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 1 | 4 | 25.00% |
| 交易争议、购买保护与补卡 | 6 | 12 | 50.00% |
| 信用卡保留、销户与竞品比较 | 5 | 8 | 62.50% |
| 信用卡推荐奖励 | 3 | 4 | 75.00% |
| 信用卡选择与申请资格 | 4 | 8 | 50.00% |
| 借记卡交易争议 | 2 | 4 | 50.00% |
| 借记卡拒付与 PIN 锁定 | 5 | 8 | 62.50% |
| 储蓄利息核算 | 2 | 4 | 50.00% |
| 卡片遗失或被盗 | 1 | 4 | 25.00% |
| 知识缺失与异常转人工 | 2 | 4 | 50.00% |
| 账户推荐、开户、销户与入金 | 8 | 16 | 50.00% |
| 账户推荐奖励 | 3 | 4 | 75.00% |
| 返现与奖励核算 | 6 | 12 | 50.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 46, "mean": 0.34782608695652173}, "COMMUNICATE": {"checks": 68, "mean": 0.6764705882352942}, "DB": {"checks": 42, "mean": 0.5476190476190477}}`
- Atomic checks: `{"ACTION": {"checks": 67, "pass_rate": 0.417910447761194, "passed": 28}, "COMMUNICATE": {"checks": 69, "pass_rate": 0.6811594202898551, "passed": 47}, "DB": {"checks": 92, "pass_rate": 0.7934782608695652, "passed": 73}}`
- Action checks by requestor: `{"assistant": {"checks": 50, "pass_rate": 0.48, "passed": 24}, "user": {"checks": 17, "pass_rate": 0.23529411764705882, "passed": 4}}`
- Termination reasons: `{"user_stop": 92}`
- Failed task count: 44
