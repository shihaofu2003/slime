# strict-single credit assignment comparison

## Decision

Selected arm: `v1-matched`; reason: `mixed_or_no_strict_primary_winner_retain_v1`. The decision uses only the two-seed mean pass@1 and pass@4(any); pass^4 and bootstrap intervals are report-only.

## Overall

| Model | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---:|---:|---:|---:|---:|
| sft | 24.38% | 47.00% | 9.00% | 64.23% | 28.81% |
| v1-matched | 27.00% | 54.00% | 8.00% | 64.11% | 27.44% |
| v2-l000 | 26.25% | 50.50% | 7.00% | 62.85% | 26.50% |
| v2-l010 | 28.75% | 54.00% | 11.00% | 66.19% | 30.36% |

## Domains

| Model | Domain | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---|---:|---:|---:|---:|---:|
| sft | airline | 31.87% | 42.50% | 22.50% | 42.24% | 32.28% |
| sft | retail | 27.50% | 50.00% | 10.00% | 75.30% | 31.84% |
| sft | telecom | 17.50% | 46.25% | 1.25% | 52.22% | 18.24% |
| v1-matched | airline | 30.63% | 45.00% | 7.50% | 49.13% | 31.01% |
| v1-matched | retail | 25.94% | 51.25% | 10.00% | 75.90% | 30.60% |
| v1-matched | telecom | 26.25% | 61.25% | 6.25% | 52.39% | 20.87% |
| v2-l000 | airline | 29.38% | 45.00% | 10.00% | 44.28% | 29.79% |
| v2-l000 | retail | 26.88% | 46.25% | 8.75% | 75.42% | 31.33% |
| v2-l000 | telecom | 24.06% | 57.50% | 3.75% | 50.04% | 16.02% |
| v2-l010 | airline | 32.50% | 45.00% | 25.00% | 37.83% | 33.33% |
| v2-l010 | retail | 28.12% | 51.25% | 13.75% | 76.09% | 32.70% |
| v2-l010 | telecom | 27.50% | 61.25% | 1.25% | 64.43% | 24.61% |

## Termination and rule diagnostics

Counts pool both evaluation seeds. Empty diagnostics are shown as `none`.

| Model | Scope | Terminations | Rule diagnostics | Protocol diagnostics |
|---|---|---|---|---|
| sft | overall | termination_error=109, termination_max_steps=80, termination_user_stop=611 | none | namespace.affected_trajectory_count=142, namespace.executed_call_count=1076, namespace.namespace_attributed_termination_count=94, namespace.parsed_call_count=1076, namespace.raw_attempt_count=1114, single_call.attempted_calls_in_multi_call_outputs=7, single_call.multi_call_output_turns=3, single_call.single_call_protocol_error_turns=2, single_call.trajectories_with_multi_call_output=2 |
| sft | airline | termination_max_steps=2, termination_user_stop=158 | none | none |
| sft | retail | termination_error=9, termination_user_stop=311 | none | none |
| sft | telecom | termination_error=100, termination_max_steps=78, termination_user_stop=142 | none | namespace.affected_trajectory_count=142, namespace.executed_call_count=1076, namespace.namespace_attributed_termination_count=94, namespace.parsed_call_count=1076, namespace.raw_attempt_count=1114, single_call.attempted_calls_in_multi_call_outputs=7, single_call.multi_call_output_turns=3, single_call.single_call_protocol_error_turns=2, single_call.trajectories_with_multi_call_output=2 |
| v1-matched | overall | termination_error=4, termination_max_steps=82, termination_user_stop=714 | none | namespace.affected_trajectory_count=25, namespace.executed_call_count=34, namespace.parsed_call_count=34, namespace.raw_attempt_count=45, single_call.attempted_calls_in_multi_call_outputs=2, single_call.multi_call_output_turns=1, single_call.trajectories_with_multi_call_output=1 |
| v1-matched | airline | termination_error=1, termination_max_steps=1, termination_user_stop=158 | none | none |
| v1-matched | retail | termination_error=3, termination_user_stop=317 | none | none |
| v1-matched | telecom | termination_max_steps=81, termination_user_stop=239 | none | namespace.affected_trajectory_count=25, namespace.executed_call_count=34, namespace.parsed_call_count=34, namespace.raw_attempt_count=45, single_call.attempted_calls_in_multi_call_outputs=2, single_call.multi_call_output_turns=1, single_call.trajectories_with_multi_call_output=1 |
| v2-l000 | overall | termination_error=84, termination_max_steps=48, termination_user_stop=668 | none | namespace.affected_trajectory_count=136, namespace.executed_call_count=925, namespace.namespace_attributed_termination_count=77, namespace.parsed_call_count=925, namespace.raw_attempt_count=1053, single_call.attempted_calls_in_multi_call_outputs=309, single_call.multi_call_output_turns=12, single_call.single_call_protocol_error_turns=2, single_call.trajectories_with_multi_call_output=6 |
| v2-l000 | airline | termination_max_steps=2, termination_user_stop=158 | none | none |
| v2-l000 | retail | termination_error=4, termination_user_stop=316 | none | none |
| v2-l000 | telecom | termination_error=80, termination_max_steps=46, termination_user_stop=194 | none | namespace.affected_trajectory_count=136, namespace.executed_call_count=925, namespace.namespace_attributed_termination_count=77, namespace.parsed_call_count=925, namespace.raw_attempt_count=1053, single_call.attempted_calls_in_multi_call_outputs=309, single_call.multi_call_output_turns=12, single_call.single_call_protocol_error_turns=2, single_call.trajectories_with_multi_call_output=6 |
| v2-l010 | overall | termination_max_steps=115, termination_user_stop=685 | none | namespace.affected_trajectory_count=2, namespace.executed_call_count=2, namespace.parsed_call_count=2, namespace.raw_attempt_count=2 |
| v2-l010 | airline | termination_max_steps=4, termination_user_stop=156 | none | none |
| v2-l010 | retail | termination_max_steps=2, termination_user_stop=318 | none | none |
| v2-l010 | telecom | termination_max_steps=109, termination_user_stop=211 | none | namespace.affected_trajectory_count=2, namespace.executed_call_count=2, namespace.parsed_call_count=2, namespace.raw_attempt_count=2 |

## Paired task bootstrap

Descriptive only; 100,000 resamples over 100 matched tasks. Each task delta is averaged across seeds 300 and 301 before resampling.

| Comparison | Metric | Delta | 95% paired bootstrap CI |
|---|---|---:|---:|
| v1-matched-sft | pass_at_1 | +2.62 pp | [-1.00, +6.25] pp |
| v1-matched-sft | pass_at_4_any | +7.00 pp | [+1.00, +13.00] pp |
| v1-matched-sft | pass_power_4 | -1.00 pp | [-6.00, +4.00] pp |
| v2-l000-sft | pass_at_1 | +1.88 pp | [-1.75, +5.50] pp |
| v2-l000-sft | pass_at_4_any | +3.50 pp | [-4.00, +10.50] pp |
| v2-l000-sft | pass_power_4 | -2.00 pp | [-6.50, +2.50] pp |
| v2-l010-sft | pass_at_1 | +4.38 pp | [+0.75, +8.12] pp |
| v2-l010-sft | pass_at_4_any | +7.00 pp | [-1.00, +15.00] pp |
| v2-l010-sft | pass_power_4 | +2.00 pp | [-1.00, +5.50] pp |
| v2-l000-v1-matched | pass_at_1 | -0.75 pp | [-4.50, +3.12] pp |
| v2-l000-v1-matched | pass_at_4_any | -3.50 pp | [-11.00, +4.00] pp |
| v2-l000-v1-matched | pass_power_4 | -1.00 pp | [-5.00, +3.00] pp |
| v2-l010-v1-matched | pass_at_1 | +1.75 pp | [-1.88, +5.38] pp |
| v2-l010-v1-matched | pass_at_4_any | +0.00 pp | [-7.50, +7.50] pp |
| v2-l010-v1-matched | pass_power_4 | +3.00 pp | [-1.00, +7.50] pp |
| v2-l010-v2-l000 | pass_at_1 | +2.50 pp | [-1.38, +6.38] pp |
| v2-l010-v2-l000 | pass_at_4_any | +3.50 pp | [-4.00, +11.00] pp |
| v2-l010-v2-l000 | pass_power_4 | +4.00 pp | [-1.00, +9.00] pp |
