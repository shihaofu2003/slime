# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 329/564
- Accuracy: 58.33%
- Mean reward: 0.5833
- Recovered tasks: 16 (`banking_curriculum_task_071_full_composition, banking_scale_task_022_full_composition, banking_scale_task_041_full_composition, banking_scale_task_058_full_composition, banking_scale_task_060_full_composition, banking_scale_task_063_full_composition, banking_scale_task_064_full_composition, banking_scale_task_067_full_composition, banking_scale_task_069_full_composition, banking_scale_task_073_full_composition, banking_scale_task_074_full_composition, banking_scale_task_086_evidence_given, banking_scale_task_090_full_composition, banking_scale_task_091_full_composition, banking_scale_task_095_full_composition, banking_scale_task_096_full_composition`)

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| decision_only | 84 | 94 | 89.36% |
| evidence_given | 58 | 94 | 61.70% |
| full_composition | 32 | 94 | 34.04% |
| retrieval_only | 93 | 94 | 98.94% |
| single_action | 48 | 94 | 51.06% |
| two_skill_composition | 14 | 94 | 14.89% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 58 | 94 | 61.70% |
| L2 | 93 | 94 | 98.94% |
| L3 | 132 | 188 | 70.21% |
| L4 | 19 | 104 | 18.27% |
| L5 | 14 | 31 | 45.16% |
| L6 | 13 | 53 | 24.53% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| bm25 | 126 | 189 | 66.67% |
| golden_retrieval | 203 | 375 | 54.13% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 14 | 30 | 46.67% |
| 交易争议、购买保护与补卡 | 32 | 54 | 59.26% |
| 信用卡保留、销户与竞品比较 | 25 | 42 | 59.52% |
| 信用卡推荐奖励 | 17 | 24 | 70.83% |
| 信用卡选择与申请资格 | 36 | 54 | 66.67% |
| 信用额度调整 | 8 | 12 | 66.67% |
| 借记卡交易争议 | 11 | 24 | 45.83% |
| 借记卡拒付与 PIN 锁定 | 20 | 36 | 55.56% |
| 储蓄利息核算 | 17 | 30 | 56.67% |
| 卡片遗失或被盗 | 9 | 24 | 37.50% |
| 知识缺失与异常转人工 | 24 | 30 | 80.00% |
| 账户推荐、开户、销户与入金 | 59 | 102 | 57.84% |
| 账户推荐奖励 | 18 | 30 | 60.00% |
| 身份验证与资料变更 | 9 | 12 | 75.00% |
| 返现与奖励核算 | 30 | 60 | 50.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 289, "mean": 0.5778546712802768}, "COMMUNICATE": {"checks": 376, "mean": 0.7367021276595744}, "DB": {"checks": 256, "mean": 0.46484375}, "NL_ASSERTION": {"checks": 1, "mean": 0.0}}`
- Atomic checks: `{"ACTION": {"checks": 1232, "pass_rate": 0.6103896103896104, "passed": 752}, "COMMUNICATE": {"checks": 383, "pass_rate": 0.7389033942558747, "passed": 283}, "DB": {"checks": 563, "pass_rate": 0.7531083481349912, "passed": 424}, "NL_ASSERTION": {"checks": 1, "pass_rate": 0.0, "passed": 0}}`
- Action checks by requestor: `{"assistant": {"checks": 1065, "pass_rate": 0.631924882629108, "passed": 673}, "user": {"checks": 167, "pass_rate": 0.47305389221556887, "passed": 79}}`
- Termination reasons: `{"max_steps": 1, "user_stop": 563}`
- Failed task count: 235
- Framework retry events: 116 across 50 tasks
- Conservative first-attempt exact accuracy: 315/564 (55.85%)
