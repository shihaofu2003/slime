# Training diagnostics: v1-matched

## Dumped trajectories

| Metric | Value |
|---|---:|
| Trajectories | 976 |
| Groups | 122 |
| Removed | 4 |
| Official reward | 0.217213 |
| Official reward, not removed | 0.218107 |
| Binary zero-variance groups | 68 (0.557377) |

The trajectory view includes candidates later replaced by the dynamic filter.

## Reward by domain

| Domain | Trajectories | Successes | Reward mean | Not-removed mean |
|---|---:|---:|---:|---:|
| airline | 256 | 73 | 0.285156 | 0.289683 |
| retail | 240 | 74 | 0.308333 | 0.308333 |
| telecom | 480 | 65 | 0.135417 | 0.135417 |

## Turn credit

| Metric | Value |
|---|---:|
| Versions | {"turn-credit-v1": 976} |
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
| exact | 2492 | 0.629293 |
| partial_arguments | 353 | 0.0891414 |
| tool_name_only | 112 | 0.0282828 |
| unmatched | 1003 | 0.253283 |

## Rule errors

| Error | Events | Affected trajectories | Rate |
|---|---:|---:|---:|
| wrong_namespace_tool | 805 | 173 | 0.177254 |
| nonexistent_tool | 46 | 29 | 0.0297131 |
| malformed_json | 258 | 103 | 0.105533 |
| wrong_argument_fields | 636 | 314 | 0.321721 |
| tool_execution_error | 319 | 151 | 0.154713 |
| repetition | 456 | 176 | 0.180328 |
| max_steps | 302 | 302 | 0.309426 |

## Partial components

| Component | Available trajectories | Mean |
|---|---:|---:|
| tool_name | 784 | 0.581924 |
| argument | 744 | 0.533508 |
| action | 0 | n/a |
| db | 636 | 0.333333 |
| env_assertion | 152 | 0.495614 |
| communicate | 0 | n/a |

## Termination and sampling

| Metric | Value |
|---|---:|
| Termination reasons | {"max_steps": 302, "too_many_errors": 38, "user_stop": 636} |
| max_steps | 302 (0.309426) |
| Truncated status | 306 (0.313525) |
| replacement_draw_trajectories | 16 |
| replacement_draw_groups | 2 |
| removed_trajectories | 4 |
| dropped_too_long | 4 |
| permanently_too_long | 4 |
| retried_trajectories | 14 |
| extra_rollout_attempts | 21 |

## Accepted-update log metrics

Parsed 20 update steps from 2 log files.
Dynamic-filter drops recorded: 2.

| Metric | Updates | Min | Mean | Final | Final step |
|---|---:|---:|---:|---:|---:|
| rollout/dynamic_filter/drop_permanently_too_long | 20 | 0 | 0.1 | 0 | 19 |
| rollout/kl | 20 | 0 | 0 | 0 | 19 |
| rollout/raw_reward | 20 | 0.0416667 | 0.220833 | 0.395833 | 19 |
| rollout/truncated | 20 | 0.208333 | 0.314583 | 0.395833 | 19 |
| rollout/truncated_ratio | 20 | 0.208333 | 0.314583 | 0.395833 | 19 |
| rollout/zero_std/count_0.0 | 20 | 1 | 3.05 | 3 | 19 |
| rollout/zero_std/count_1.0 | 20 | 0 | 0.25 | 2 | 19 |
| train/grad_norm | 20 | 0.341885 | 0.438445 | 0.341885 | 19 |
| train/kl_loss | 20 | 0 | 0.00112894 | 0.00170191 | 19 |
| train/loss | 20 | -0.0126081 | 0.0143735 | -0.0126081 | 19 |
| train/pg_loss | 20 | -0.0126251 | 0.0143622 | -0.0126251 | 19 |
| train/ppo_kl | 20 | 0 | 0 | 0 | 19 |
