# Training diagnostics: v1-matched

## Dumped trajectories

| Metric | Value |
|---|---:|
| Trajectories | 4846 |
| Groups | 606 |
| Removed | 10 |
| Official reward | 0.260421 |
| Official reward, not removed | 0.260959 |
| Binary zero-variance groups | 306 (0.50495) |

The trajectory view includes candidates later replaced by the dynamic filter.

## Reward by domain

| Domain | Trajectories | Successes | Reward mean | Not-removed mean |
|---|---:|---:|---:|---:|
| airline | 1552 | 519 | 0.334407 | 0.336576 |
| retail | 1520 | 520 | 0.342105 | 0.342105 |
| telecom | 1774 | 223 | 0.125705 | 0.125705 |

## Turn credit

| Metric | Value |
|---|---:|
| Versions | {"turn-credit-v1": 4846} |
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
| exact | 15224 | 0.637788 |
| partial_arguments | 2412 | 0.101047 |
| tool_name_only | 370 | 0.0155006 |
| unmatched | 5864 | 0.245664 |

## Rule errors

| Error | Events | Affected trajectories | Rate |
|---|---:|---:|---:|
| wrong_namespace_tool | 947 | 271 | 0.0559224 |
| nonexistent_tool | 81 | 60 | 0.0123813 |
| malformed_json | 520 | 290 | 0.0598432 |
| wrong_argument_fields | 3910 | 1959 | 0.404251 |
| tool_execution_error | 1497 | 782 | 0.16137 |
| repetition | 857 | 438 | 0.0903838 |
| max_steps | 1133 | 1133 | 0.233801 |

## Partial components

| Component | Available trajectories | Mean |
|---|---:|---:|
| tool_name | 4223 | 0.655353 |
| argument | 4063 | 0.599911 |
| action | 0 | n/a |
| db | 3674 | 0.340773 |
| env_assertion | 640 | 0.501302 |
| communicate | 0 | n/a |

## Termination and sampling

| Metric | Value |
|---|---:|
| Termination reasons | {"max_steps": 1133, "too_many_errors": 39, "user_stop": 3674} |
| max_steps | 1133 (0.233801) |
| Truncated status | 1141 (0.235452) |
| replacement_draw_trajectories | 48 |
| replacement_draw_groups | 6 |
| removed_trajectories | 10 |
| dropped_too_long | 10 |
| permanently_too_long | 10 |
| retried_trajectories | 48 |
| extra_rollout_attempts | 65 |

## Accepted-update log metrics

Parsed 100 update steps from 3 log files.
Dynamic-filter drops recorded: 6.

| Metric | Updates | Min | Mean | Final | Final step |
|---|---:|---:|---:|---:|---:|
| rollout/dynamic_filter/drop_permanently_too_long | 100 | 0 | 0.04 | 0 | 99 |
| rollout/dynamic_filter/drop_rollout_error | 100 | 0 | 0.02 | 0 | 99 |
| rollout/kl | 100 | 0 | 0 | 0 | 99 |
| rollout/raw_reward | 100 | 0 | 0.262917 | 0.3125 | 99 |
| rollout/truncated | 100 | 0.0416667 | 0.233958 | 0.208333 | 99 |
| rollout/truncated_ratio | 100 | 0.0416667 | 0.233958 | 0.208333 | 99 |
| rollout/zero_std/count_0.0 | 100 | 0 | 2.73 | 1 | 99 |
| rollout/zero_std/count_1.0 | 100 | 0 | 0.27 | 0 | 99 |
| train/grad_norm | 100 | 0.294368 | 0.437941 | 0.43712 | 99 |
| train/kl_loss | 100 | 0 | 0.00897976 | 0.0279207 | 99 |
| train/loss | 100 | -0.0126081 | 0.0106989 | 0.0121912 | 99 |
| train/pg_loss | 100 | -0.0126251 | 0.0106091 | 0.011912 | 99 |
| train/ppo_kl | 100 | 0 | 0 | 0 | 99 |
