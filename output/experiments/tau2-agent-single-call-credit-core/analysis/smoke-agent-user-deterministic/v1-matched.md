# Training diagnostics: v1-matched

## Dumped trajectories

| Metric | Value |
|---|---:|
| Trajectories | 48 |
| Groups | 6 |
| Removed | 0 |
| Official reward | 0.0416667 |
| Official reward, not removed | 0.0416667 |
| Binary zero-variance groups | 5 (0.833333) |

The trajectory view includes candidates later replaced by the dynamic filter.

## Reward by domain

| Domain | Trajectories | Successes | Reward mean | Not-removed mean |
|---|---:|---:|---:|---:|
| airline | 16 | 2 | 0.125 | 0.125 |
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
| exact | 59 | 0.283654 |
| partial_arguments | 31 | 0.149038 |
| tool_name_only | 16 | 0.0769231 |
| unmatched | 102 | 0.490385 |

## Rule errors

| Error | Events | Affected trajectories | Rate |
|---|---:|---:|---:|
| wrong_namespace_tool | 82 | 11 | 0.229167 |
| nonexistent_tool | 6 | 2 | 0.0416667 |
| malformed_json | 25 | 6 | 0.125 |
| wrong_argument_fields | 57 | 24 | 0.5 |
| tool_execution_error | 24 | 10 | 0.208333 |
| repetition | 52 | 20 | 0.416667 |
| max_steps | 19 | 19 | 0.395833 |

## Partial components

| Component | Available trajectories | Mean |
|---|---:|---:|
| tool_name | 48 | 0.371693 |
| argument | 40 | 0.356591 |
| action | 0 | n/a |
| db | 26 | 0.115385 |
| env_assertion | 3 | 0.5 |
| communicate | 0 | n/a |

## Termination and sampling

| Metric | Value |
|---|---:|
| Termination reasons | {"max_steps": 19, "too_many_errors": 3, "user_stop": 26} |
| max_steps | 19 (0.395833) |
| Truncated status | 19 (0.395833) |
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
| rollout/raw_reward | 1 | 0.0416667 | 0.0416667 | 0.0416667 | 0 |
| rollout/truncated | 1 | 0.395833 | 0.395833 | 0.395833 | 0 |
| rollout/truncated_ratio | 1 | 0.395833 | 0.395833 | 0.395833 | 0 |
| rollout/zero_std/count_0.0 | 1 | 5 | 5 | 5 | 0 |
| train/grad_norm | 1 | 0.449766 | 0.449766 | 0.449766 | 0 |
| train/kl_loss | 1 | 0 | 0 | 0 | 0 |
| train/loss | 1 | 0.0340686 | 0.0340686 | 0.0340686 | 0 |
| train/pg_loss | 1 | 0.0340686 | 0.0340686 | 0.0340686 | 0 |
| train/ppo_kl | 1 | 0 | 0 | 0 | 0 |
