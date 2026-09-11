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
| 交易争议、购买保护与补卡 | 0 | 1 | 0.00% |

## Diagnostics

- Reward components: `{"DB": {"checks": 1, "mean": 0.0}}`
- Atomic checks: `{"ACTION": {"checks": 25, "pass_rate": 0.56, "passed": 14}, "DB": {"checks": 1, "pass_rate": 0.0, "passed": 0}}`
- Action checks by requestor: `{"assistant": {"checks": 21, "pass_rate": 0.47619047619047616, "passed": 10}, "user": {"checks": 4, "pass_rate": 1.0, "passed": 4}}`
- Termination reasons: `{"user_stop": 1}`
- Failed task count: 1
