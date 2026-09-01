# Training diagnostics: v1-matched

## Dumped trajectories

| Metric | Value |
|---|---:|
| Trajectories | 48 |
| Groups | 6 |
| Removed | 0 |
| Official reward | 0.0208333 |
| Official reward, not removed | 0.0208333 |
| Binary zero-variance groups | 5 (0.833333) |

The trajectory view includes candidates later replaced by the dynamic filter.

## Reward by domain

| Domain | Trajectories | Successes | Reward mean | Not-removed mean |
|---|---:|---:|---:|---:|
| airline | 16 | 1 | 0.0625 | 0.0625 |
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
| exact | 54 | 0.259615 |
| partial_arguments | 36 | 0.173077 |
| tool_name_only | 21 | 0.100962 |
| unmatched | 97 | 0.466346 |

## Rule errors

| Error | Events | Affected trajectories | Rate |
|---|---:|---:|---:|
| wrong_namespace_tool | 65 | 11 | 0.229167 |
| nonexistent_tool | 4 | 3 | 0.0625 |
| malformed_json | 4 | 3 | 0.0625 |
| wrong_argument_fields | 65 | 24 | 0.5 |
| tool_execution_error | 7 | 7 | 0.145833 |
| repetition | 51 | 17 | 0.354167 |
| max_steps | 18 | 18 | 0.375 |

## Partial components

| Component | Available trajectories | Mean |
|---|---:|---:|
| tool_name | 48 | 0.37996 |
| argument | 40 | 0.338654 |
| action | 0 | n/a |
| db | 25 | 0.04 |
| env_assertion | 2 | 0.666667 |
| communicate | 0 | n/a |

## Termination and sampling

| Metric | Value |
|---|---:|
| Termination reasons | {"max_steps": 18, "too_many_errors": 5, "user_stop": 25} |
| max_steps | 18 (0.375) |
| Truncated status | 18 (0.375) |
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
| rollout/raw_reward | 1 | 0.0208333 | 0.0208333 | 0.0208333 | 0 |
| rollout/truncated | 1 | 0.375 | 0.375 | 0.375 | 0 |
| rollout/truncated_ratio | 1 | 0.375 | 0.375 | 0.375 | 0 |
| rollout/zero_std/count_0.0 | 1 | 5 | 5 | 5 | 0 |
| train/grad_norm | 1 | 0.44309 | 0.44309 | 0.44309 | 0 |
| train/kl_loss | 1 | 0 | 0 | 0 | 0 |
| train/loss | 1 | 0.027721 | 0.027721 | 0.027721 | 0 |
| train/pg_loss | 1 | 0.027721 | 0.027721 | 0.027721 | 0 |
| train/ppo_kl | 1 | 0 | 0 | 0 | 0 |
