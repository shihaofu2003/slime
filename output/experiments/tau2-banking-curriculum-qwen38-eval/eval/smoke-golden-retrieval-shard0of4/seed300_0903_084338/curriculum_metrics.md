# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 6/7
- Accuracy: 85.71%
- Mean reward: 0.8571

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| decision_only | 2 | 2 | 100.00% |
| evidence_given | 2 | 2 | 100.00% |
| single_action | 2 | 2 | 100.00% |
| two_skill_composition | 0 | 1 | 0.00% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L1 | 2 | 2 | 100.00% |
| L3 | 4 | 4 | 100.00% |
| L4 | 0 | 1 | 0.00% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| golden_retrieval | 6 | 7 | 85.71% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| 信用卡选择与申请资格 | 3 | 3 | 100.00% |
| 身份验证与资料变更 | 3 | 4 | 75.00% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 2, "mean": 0.5}, "COMMUNICATE": {"checks": 6, "mean": 0.8333333333333334}, "DB": {"checks": 3, "mean": 0.6666666666666666}}`
- Atomic checks: `{"ACTION": {"checks": 4, "pass_rate": 0.5, "passed": 2}, "COMMUNICATE": {"checks": 7, "pass_rate": 0.8571428571428571, "passed": 6}, "DB": {"checks": 7, "pass_rate": 0.8571428571428571, "passed": 6}}`
- Termination reasons: `{"user_stop": 7}`
- Failed task count: 1
