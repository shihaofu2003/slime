# Training diagnostics: v1-matched

## Dumped trajectories

| Metric | Value |
|---|---:|
| Trajectories | 496 |
| Groups | 62 |
| Removed | 4 |
| Official reward | 0.195565 |
| Official reward, not removed | 0.197154 |
| Binary zero-variance groups | 38 (0.612903) |

The trajectory view includes candidates later replaced by the dynamic filter.

## Reward by domain

| Domain | Trajectories | Successes | Reward mean | Not-removed mean |
|---|---:|---:|---:|---:|
| airline | 176 | 41 | 0.232955 | 0.238372 |
| retail | 80 | 29 | 0.3625 | 0.3625 |
| telecom | 240 | 27 | 0.1125 | 0.1125 |

## Turn credit

| Metric | Value |
|---|---:|
| Versions | {"turn-credit-v1": 496} |
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
| exact | 1138 | 0.580612 |
| partial_arguments | 209 | 0.106633 |
| tool_name_only | 53 | 0.0270408 |
| unmatched | 560 | 0.285714 |

## Rule errors

| Error | Events | Affected trajectories | Rate |
|---|---:|---:|---:|
| wrong_namespace_tool | 567 | 99 | 0.199597 |
| nonexistent_tool | 31 | 17 | 0.0342742 |
| malformed_json | 179 | 71 | 0.143145 |
| wrong_argument_fields | 370 | 166 | 0.334677 |
| tool_execution_error | 176 | 84 | 0.169355 |
| repetition | 327 | 116 | 0.233871 |
| max_steps | 147 | 147 | 0.296371 |

## Partial components

| Component | Available trajectories | Mean |
|---|---:|---:|
| tool_name | 408 | 0.580527 |
| argument | 392 | 0.518676 |
| action | 0 | n/a |
| db | 317 | 0.305994 |
| env_assertion | 68 | 0.470588 |
| communicate | 0 | n/a |

## Termination and sampling

| Metric | Value |
|---|---:|
| Termination reasons | {"max_steps": 147, "too_many_errors": 32, "user_stop": 317} |
| max_steps | 147 (0.296371) |
| Truncated status | 151 (0.304435) |
| replacement_draw_trajectories | 16 |
| replacement_draw_groups | 2 |
| removed_trajectories | 4 |
| dropped_too_long | 4 |
| permanently_too_long | 4 |
| retried_trajectories | 14 |
| extra_rollout_attempts | 21 |

## Accepted-update log metrics

Parsed 10 update steps from 1 log files.
Dynamic-filter drops recorded: 2.

| Metric | Updates | Min | Mean | Final | Final step |
|---|---:|---:|---:|---:|---:|
| rollout/dynamic_filter/drop_permanently_too_long | 10 | 0 | 0.2 | 0 | 9 |
| rollout/kl | 10 | 0 | 0 | 0 | 9 |
| rollout/raw_reward | 10 | 0.0416667 | 0.202083 | 0.208333 | 9 |
| rollout/truncated | 10 | 0.208333 | 0.30625 | 0.3125 | 9 |
| rollout/truncated_ratio | 10 | 0.208333 | 0.30625 | 0.3125 | 9 |
| rollout/zero_std/count_0.0 | 10 | 1 | 3.3 | 2 | 9 |
| rollout/zero_std/count_1.0 | 10 | 0 | 0.3 | 0 | 9 |
| train/grad_norm | 10 | 0.366078 | 0.446932 | 0.432246 | 9 |
| train/kl_loss | 10 | 0 | 0.00042807 | 0.000875551 | 9 |
| train/loss | 10 | -0.00514595 | 0.0172783 | 0.0231442 | 9 |
| train/pg_loss | 10 | -0.00515054 | 0.017274 | 0.0231355 | 9 |
| train/ppo_kl | 10 | 0 | 0 | 0 | 9 |
