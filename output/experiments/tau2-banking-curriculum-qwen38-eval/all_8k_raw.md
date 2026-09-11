# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 327/564
- Accuracy: 57.98%
- Mean reward: 0.5798

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| decision_only | 84 | 94 | 89.36% |
| evidence_given | 58 | 94 | 61.70% |
| full_composition | 30 | 94 | 31.91% |
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
| L6 | 11 | 53 | 20.75% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| bm25 | 124 | 189 | 65.61% |
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
| 借记卡拒付与 PIN 锁定 | 19 | 36 | 52.78% |
| 储蓄利息核算 | 16 | 30 | 53.33% |
| 卡片遗失或被盗 | 9 | 24 | 37.50% |
| 知识缺失与异常转人工 | 24 | 30 | 80.00% |
| 账户推荐、开户、销户与入金 | 59 | 102 | 57.84% |
| 账户推荐奖励 | 18 | 30 | 60.00% |
| 身份验证与资料变更 | 9 | 12 | 75.00% |
| 返现与奖励核算 | 30 | 60 | 50.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 289, "mean": 0.5778546712802768}, "COMMUNICATE": {"checks": 375, "mean": 0.7386666666666667}, "DB": {"checks": 241, "mean": 0.4854771784232365}, "NL_ASSERTION": {"checks": 1, "mean": 0.0}}`
- Atomic checks: `{"ACTION": {"checks": 1068, "pass_rate": 0.5842696629213483, "passed": 624}, "COMMUNICATE": {"checks": 382, "pass_rate": 0.7408376963350786, "passed": 283}, "DB": {"checks": 547, "pass_rate": 0.7696526508226691, "passed": 421}, "NL_ASSERTION": {"checks": 1, "pass_rate": 0.0, "passed": 0}}`
- Action checks by requestor: `{"assistant": {"checks": 920, "pass_rate": 0.6119565217391304, "passed": 563}, "user": {"checks": 148, "pass_rate": 0.41216216216216217, "passed": 61}}`
- Termination reasons: `{"infrastructure_error": 16, "max_steps": 1, "user_stop": 547}`
- Failed task count: 237
- Framework retry events: 96 across 50 tasks
- Conservative first-attempt exact accuracy: 315/564 (55.85%)
