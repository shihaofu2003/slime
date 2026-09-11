# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 163/240
- Accuracy: 67.92%
- Mean reward: 0.6792

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| evidence_given | 72 | 121 | 59.50% |
| single_action | 91 | 119 | 76.47% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 72 | 121 | 59.50% |
| L3 | 91 | 119 | 76.47% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 163 | 240 | 67.92% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| 交易争议、购买保护与补卡 | 23 | 30 | 76.67% |
| 信用卡保留、销户与竞品比较 | 19 | 26 | 73.08% |
| 信用卡选择与申请资格 | 6 | 7 | 85.71% |
| 借记卡拒付与 PIN 锁定 | 21 | 31 | 67.74% |
| 储蓄利息核算 | 21 | 35 | 60.00% |
| 卡片遗失或被盗 | 24 | 35 | 68.57% |
| 账户推荐、开户、销户与入金 | 36 | 50 | 72.00% |
| 返现与奖励核算 | 13 | 26 | 50.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 119, "mean": 0.7647058823529411}, "COMMUNICATE": {"checks": 121, "mean": 0.5950413223140496}, "DB": {"checks": 10, "mean": 0.8}}`
- Atomic checks: `{"ACTION": {"checks": 119, "pass_rate": 0.7647058823529411, "passed": 91}, "COMMUNICATE": {"checks": 121, "pass_rate": 0.5950413223140496, "passed": 72}, "DB": {"checks": 240, "pass_rate": 0.95, "passed": 228}}`
- Action checks by requestor: `{"assistant": {"checks": 111, "pass_rate": 0.8198198198198198, "passed": 91}, "user": {"checks": 8, "pass_rate": 0.0, "passed": 0}}`
- Termination reasons: `{"user_stop": 240}`
- Failed task count: 77
