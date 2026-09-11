# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 161/233
- Accuracy: 69.10%
- Mean reward: 0.6910

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| evidence_given | 79 | 122 | 64.75% |
| single_action | 82 | 111 | 73.87% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 79 | 122 | 64.75% |
| L3 | 82 | 111 | 73.87% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 161 | 233 | 69.10% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| ATM 费用与旅行支票账户 | 12 | 18 | 66.67% |
| 交易争议、购买保护与补卡 | 8 | 13 | 61.54% |
| 信用卡保留、销户与竞品比较 | 23 | 27 | 85.19% |
| 信用卡选择与申请资格 | 7 | 7 | 100.00% |
| 借记卡拒付与 PIN 锁定 | 16 | 28 | 57.14% |
| 卡片遗失或被盗 | 32 | 46 | 69.57% |
| 知识缺失与异常转人工 | 1 | 2 | 50.00% |
| 账户推荐、开户、销户与入金 | 32 | 40 | 80.00% |
| 账户推荐奖励 | 16 | 22 | 72.73% |
| 返现与奖励核算 | 14 | 30 | 46.67% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 111, "mean": 0.7387387387387387}, "COMMUNICATE": {"checks": 122, "mean": 0.6475409836065574}, "DB": {"checks": 11, "mean": 0.9090909090909091}}`
- Atomic checks: `{"ACTION": {"checks": 111, "pass_rate": 0.7387387387387387, "passed": 82}, "COMMUNICATE": {"checks": 122, "pass_rate": 0.6475409836065574, "passed": 79}, "DB": {"checks": 233, "pass_rate": 0.9527896995708155, "passed": 222}}`
- Action checks by requestor: `{"assistant": {"checks": 94, "pass_rate": 0.7872340425531915, "passed": 74}, "user": {"checks": 17, "pass_rate": 0.47058823529411764, "passed": 8}}`
- Termination reasons: `{"user_stop": 233}`
- Failed task count: 72
