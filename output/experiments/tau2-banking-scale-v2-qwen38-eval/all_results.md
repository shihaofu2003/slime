# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 2163/2740
- Accuracy: 78.94%
- Mean reward: 0.7894

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| evidence_given | 578 | 933 | 61.95% |
| retrieval_only | 932 | 933 | 99.89% |
| single_action | 653 | 874 | 74.71% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 578 | 933 | 61.95% |
| L2 | 932 | 933 | 99.89% |
| L3 | 653 | 874 | 74.71% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| bm25 | 932 | 933 | 99.89% |
| golden_retrieval | 1231 | 1807 | 68.12% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 167 | 201 | 83.08% |
| 交易争议、购买保护与补卡 | 162 | 200 | 81.00% |
| 信用卡保留、销户与竞品比较 | 176 | 215 | 81.86% |
| 信用卡推荐奖励 | 28 | 37 | 75.68% |
| 信用卡选择与申请资格 | 108 | 114 | 94.74% |
| 信用额度调整 | 44 | 52 | 84.62% |
| 借记卡交易争议 | 90 | 141 | 63.83% |
| 借记卡拒付与 PIN 锁定 | 184 | 254 | 72.44% |
| 储蓄利息核算 | 171 | 231 | 74.03% |
| 卡片遗失或被盗 | 171 | 216 | 79.17% |
| 知识缺失与异常转人工 | 25 | 35 | 71.43% |
| 账户推荐、开户、销户与入金 | 499 | 595 | 83.87% |
| 账户推荐奖励 | 157 | 189 | 83.07% |
| 身份验证与资料变更 | 7 | 8 | 87.50% |
| 返现与奖励核算 | 174 | 252 | 69.05% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 1807, "mean": 0.8776978417266187}, "COMMUNICATE": {"checks": 1866, "mean": 0.8092175777063236}, "DB": {"checks": 86, "mean": 0.8372093023255814}}`
- Atomic checks: `{"ACTION": {"checks": 1807, "pass_rate": 0.8776978417266187, "passed": 1586}, "COMMUNICATE": {"checks": 1866, "pass_rate": 0.8092175777063236, "passed": 1510}, "DB": {"checks": 2740, "pass_rate": 0.9664233576642336, "passed": 2648}}`
- Action checks by requestor: `{"assistant": {"checks": 1705, "pass_rate": 0.9049853372434018, "passed": 1543}, "user": {"checks": 102, "pass_rate": 0.4215686274509804, "passed": 43}}`
- Termination reasons: `{"user_stop": 2740}`
- Failed task count: 577
