# Raw-all versus processed SFT quality audit

## Data-level audit

Rates are descriptive. Raw-all uses one source answer per row; processed may expand one source answer into multiple atomic targets.

| Metric | Raw-all | Processed |
|---|---:|---:|
| Double-success provenance | 24,120/32,548 (74.11%) | 30,376/30,376 (100.00%) |
| Failure-labeled rows | 8,428/32,548 (25.89%) | 0/30,376 (0.00%) |
| Multi-call targets / tool-call rows | 4,058/12,211 (33.23%) | 0/16,397 (0.00%) |
| Mixed text + call targets / tool-call rows | 1,898/12,211 (15.54%) | 0/16,397 (0.00%) |
| Rows with a multi-call prefix | 20,247/32,548 (62.21%) | 0/30,376 (0.00%) |
| Calls from failure-labeled rows | 3,846/20,489 (18.77%) | 0/16,397 (0.00%) |
| Observed User-only target calls | 16/20,489 (0.08%) | 0/16,397 (0.00%) |
| Current Agent namespace calls | 20,473/20,489 (99.92%) | 16,397/16,397 (100.00%) |
| Current JSON-schema-valid calls | 20,473/20,489 (99.92%) | 16,397/16,397 (100.00%) |
| Current tool-schema rows | n/a | 30,376/30,376 (100.00%) |
| Target-only loss-mask rows | n/a | 30,376/30,376 (100.00%) |
| Complete prefix call/result pairing | n/a | 30,376/30,376 (100.00%) |

## Paired official evaluation

The interval resamples 100 matched tasks. The two seeds are averaged within each task before 100,000 bootstrap draws.

| Scope | Metric | Processed - raw-all | 95% paired CI | Tasks better / tied / worse | Excludes 0 |
|---|---|---:|---:|---:|---:|
| overall | pass@1 | +2.62pp | [-2.12, +7.38]pp | 39 / 28 / 33 | no |
| overall | pass@4(any) | -1.50pp | [-8.00, +5.00]pp | 13 / 72 / 15 | no |
| overall | pass^4 | +5.00pp | [-1.50, +11.00]pp | 20 / 70 / 10 | no |
| airline | pass@1 | +1.88pp | [-9.38, +13.75]pp | 5 / 7 / 8 | no |
| airline | pass@4(any) | +5.00pp | [-15.00, +25.00]pp | 5 / 11 / 4 | no |
| airline | pass^4 | +0.00pp | [-10.00, +10.00]pp | 2 / 16 / 2 | no |
| retail | pass@1 | -4.06pp | [-11.25, +2.81]pp | 14 / 9 / 17 | no |
| retail | pass@4(any) | -10.00pp | [-20.00, -1.25]pp | 2 / 30 / 8 | yes |
| retail | pass^4 | +6.25pp | [-5.00, +16.25]pp | 9 / 27 / 4 | no |
| telecom | pass@1 | +9.69pp | [+2.81, +17.19]pp | 20 / 12 / 8 | yes |
| telecom | pass@4(any) | +3.75pp | [-3.75, +11.25]pp | 6 / 31 / 3 | no |
| telecom | pass^4 | +6.25pp | [-3.75, +16.25]pp | 9 / 27 / 4 | no |

## Interpretation

The direct data-level evidence supports cleaner outcome provenance, atomic targets, and current-runtime compatibility. The paired intervals are conditional on the two observed generation seeds and do not isolate which processing operation caused them, because filtering, target expansion, prompt/schema replacement, and the domain mix changed together.

Raw-all target calls were already 99.92% valid against the current Agent namespace and JSON schemas. Target-call schema validity alone is therefore not the main differentiator; outcome filtering and atomic context are the larger measured changes.

A double-success source label is episode-level evidence; it does not prove that every retained intermediate action is optimal. Action and DB accuracy therefore remain necessary counter-metrics.
