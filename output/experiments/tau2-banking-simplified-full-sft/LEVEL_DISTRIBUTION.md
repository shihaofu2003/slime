# Banking SFT level distribution

Counts from exported training JSONL metadata.task_id; level/form mapping follows
examples/tau2-bench/analysis/banking_synthetic/constants.py FORM_LEVEL.
Rows are target-turn training examples, not complete trajectories or supervised-token weights.

| Form | Full rows | Row share | Full trajectories | Trajectory share | Unique tasks | Partial rows | Partial share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_evidence | 269 | 4.74% | 215 | 13.02% | 54 | 5 | 0.28% |
| l2_retrieval | 245 | 4.32% | 107 | 6.48% | 27 | 31 | 1.74% |
| l3_decision | 322 | 5.68% | 303 | 18.35% | 81 | 3 | 0.17% |
| l3_single_action | 1270 | 22.39% | 354 | 21.44% | 107 | 58 | 3.25% |
| l4_two_skill | 1815 | 32.00% | 439 | 26.59% | 161 | 877 | 49.21% |
| l5_medium_workflow | 1165 | 20.54% | 168 | 10.18% | 74 | 444 | 24.92% |
| l6_full_workflow | 586 | 10.33% | 65 | 3.94% | 26 | 364 | 20.43% |

Full: 5,672 rows, 1,651 trajectories, 530 tasks. Partial: 1,782 rows,
329 trajectories, 207 tasks. L5–L6 row share drops from 45.34% to 30.87%;
L6 tasks rise only from 21 to 26. Of 3,890 additional rows, 2,009 (51.65%)
are L1–L3 and 222 (5.71%) are L6. BM25 rows change from 1,716/1,782
(96.30%) to 3,811/5,672 (67.19%); remaining rows use golden retrieval.

Training shuffles all levels together; the run does not implement an
L1→L6 staged curriculum or per-level quota. The partial snapshot was
length-ordered processing output, not a random subsample. These are changes
in composition, not a controlled sample-size scaling experiment.

Official test successes change from 16/388 to 17/388; tasks with any success
remain 9/97. Distribution shifts and limited additional full-workflow task
coverage are plausible explanations, not a proven causal attribution.
Per-level held-out tests and matched-mixture size comparisons would be needed
to separate component-skill learning, transfer, and data-size effects.
