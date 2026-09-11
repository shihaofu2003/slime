# Banking Curriculum Evaluation

- Model: `openai/Qwen3.8-27B-banking-agent-xhigh`
- Correct: 2/7
- Accuracy: 28.57%
- Mean reward: 0.2857

## By variant

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| full_composition | 2 | 3 | 66.67% |
| retrieval_only | 0 | 3 | 0.00% |
| two_skill_composition | 0 | 1 | 0.00% |

## By level

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| L2 | 0 | 3 | 0.00% |
| L4 | 1 | 2 | 50.00% |
| L5 | 1 | 2 | 50.00% |

## By retrieval mode

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| bm25 | 2 | 7 | 28.57% |

## By business category

| Group | Correct | Tasks | Accuracy |
|---|---:|---:|---:|
| 信用卡选择与申请资格 | 2 | 7 | 28.57% |

## Diagnostics

- Reward components: `{"ACTION": {"checks": 4, "mean": 1.0}, "COMMUNICATE": {"checks": 4, "mean": 0.0}, "DB": {"checks": 3, "mean": 0.6666666666666666}}`
- Termination reasons: `{"user_stop": 7}`
- Failed task count: 5
