# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 0/1
- Accuracy: 0.00%
- Mean reward: 0.0000

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| full_composition | 0 | 1 | 0.00% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L6 | 0 | 1 | 0.00% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| bm25 | 0 | 1 | 0.00% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| 账户推荐、开户、销户与入金 | 0 | 1 | 0.00% |

## Diagnostics

- Reward components: `{"DB": {"checks": 1, "mean": 0.0}}`
- Atomic checks: `{"ACTION": {"checks": 6, "pass_rate": 0.8333333333333334, "passed": 5}, "DB": {"checks": 1, "pass_rate": 0.0, "passed": 0}}`
- Action checks by requestor: `{"assistant": {"checks": 6, "pass_rate": 0.8333333333333334, "passed": 5}}`
- Termination reasons: `{"user_stop": 1}`
- Failed task count: 1
