# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 3/3
- Accuracy: 100.00%
- Mean reward: 1.0000

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| decision_only | 1 | 1 | 100.00% |
| evidence_given | 1 | 1 | 100.00% |
| single_action | 1 | 1 | 100.00% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 1 | 1 | 100.00% |
| L3 | 2 | 2 | 100.00% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 3 | 3 | 100.00% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| 信用卡选择与申请资格 | 3 | 3 | 100.00% |

## Diagnostics

- Reward components: `{"COMMUNICATE": {"checks": 3, "mean": 1.0}, "DB": {"checks": 1, "mean": 1.0}}`
- Atomic checks: `{"ACTION": {"checks": 1, "pass_rate": 1.0, "passed": 1}, "COMMUNICATE": {"checks": 4, "pass_rate": 1.0, "passed": 4}, "DB": {"checks": 3, "pass_rate": 1.0, "passed": 3}}`
- Termination reasons: `{"user_stop": 3}`
- Failed task count: 0
