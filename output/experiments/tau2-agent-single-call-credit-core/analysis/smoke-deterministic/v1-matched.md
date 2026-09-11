# Training diagnostics: v1-matched

## Dumped trajectories

| Metric | Value |
|---|---:|
| Trajectories | 48 |
| Groups | 6 |
| Removed | 0 |
| Official reward | 0.0625 |
| Official reward, not removed | 0.0625 |
| Binary zero-variance groups | 5 (0.833333) |

The trajectory view includes candidates later replaced by the dynamic filter.

## Reward by domain

| Domain | Trajectories | Successes | Reward mean | Not-removed mean |
|---|---:|---:|---:|---:|
| airline | 16 | 3 | 0.1875 | 0.1875 |
| retail | 8 | 0 | 0 | 0 |
| telecom | 24 | 0 | 0 | 0 |

## Turn credit

| Metric | Value |
|---|---:|
| Versions | {"turn-credit-v1": 48} |
| Reallocation weight | n/a |
| V2 reconstruction available | 0 |
| Raw directional activation | 0 (n/a) |
| Modifier allocation activation | 0 (n/a) |
| Weighted local-channel activation | n/a (n/a) |
| Weighted local L1 budget per modifier-active trajectory | n/a |
| Weighted local L1 budget across dumped modifier-active trajectories | n/a |
| Complementary-neutral turns | 0 |
| Stored/reconstructed mismatches | 0 / 0 |

## Reference actions

| Tier | Count | Rate |
|---|---:|---:|
| exact | 43 | 0.206731 |
| partial_arguments | 35 | 0.168269 |
| tool_name_only | 17 | 0.0817308 |
| unmatched | 113 | 0.543269 |

## Rule errors

| Error | Events | Affected trajectories | Rate |
|---|---:|---:|---:|
| wrong_namespace_tool | 79 | 12 | 0.25 |
| nonexistent_tool | 5 | 2 | 0.0416667 |
| malformed_json | 10 | 5 | 0.104167 |
| wrong_argument_fields | 65 | 23 | 0.479167 |
| tool_execution_error | 4 | 3 | 0.0625 |
| repetition | 27 | 12 | 0.25 |
| max_steps | 13 | 13 | 0.270833 |

## Partial components

| Component | Available trajectories | Mean |
|---|---:|---:|
| tool_name | 48 | 0.30754 |
| argument | 40 | 0.269196 |
| action | 0 | n/a |
| db | 31 | 0.0967742 |
| env_assertion | 7 | 0.190476 |
| communicate | 0 | n/a |

## Termination and sampling

| Metric | Value |
|---|---:|
| Termination reasons | {"max_steps": 13, "too_many_errors": 4, "user_stop": 31} |
| max_steps | 13 (0.270833) |
| Truncated status | 13 (0.270833) |
| replacement_draw_trajectories | 0 |
| replacement_draw_groups | 0 |
| removed_trajectories | 0 |
| dropped_too_long | 0 |
| permanently_too_long | 0 |
| retried_trajectories | 0 |
| extra_rollout_attempts | 0 |

## Accepted-update log metrics

Parsed 1 update steps from 1 log files.
Dynamic-filter drops recorded: 0.

| Metric | Updates | Min | Mean | Final | Final step |
|---|---:|---:|---:|---:|---:|
| rollout/kl | 1 | 0 | 0 | 0 | 0 |
| rollout/raw_reward | 1 | 0.0625 | 0.0625 | 0.0625 | 0 |
| rollout/truncated | 1 | 0.270833 | 0.270833 | 0.270833 | 0 |
| rollout/truncated_ratio | 1 | 0.270833 | 0.270833 | 0.270833 | 0 |
| rollout/zero_std/count_0.0 | 1 | 5 | 5 | 5 | 0 |
| train/grad_norm | 1 | 0.47014 | 0.47014 | 0.47014 | 0 |
| train/kl_loss | 1 | 0 | 0 | 0 | 0 |
| train/loss | 1 | 0.0293425 | 0.0293425 | 0.0293425 | 0 |
| train/pg_loss | 1 | 0.0293425 | 0.0293425 | 0.0293425 | 0 |
| train/ppo_kl | 1 | 0 | 0 | 0 | 0 |
