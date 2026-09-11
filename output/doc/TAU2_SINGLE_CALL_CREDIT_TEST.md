# strict-single-v1 credit assignment test

Date: 2026-08-14. Experiment:
[`tau2-agent-single-call-credit-core`](../experiments/tau2-agent-single-call-credit-core/README.md).

## Question and controls

The test asks whether fixed-budget turn-local credit improves held-out official
tau2 success over both the current v1 credit recipe and outcome-only GRPO. All
three arms start from the same strict-single-v1 SFT and hold the User model,
prompt quotas, rollout budget, seeds, LR, KL, and zero-signal handling fixed.
Strict-single-v1 means at most one native tool call in each Assistant message;
a task can still contain multiple Assistant turns.

## Arms

| Arm | Trajectory reward and token advantage |
|---|---|
| `v1-matched` | Current binary + field trajectory score, GRPO normalization, then v1 raw turn penalty. Fresh training control, not raw GRPO. |
| `v2-l000` | Official binary outcome GRPO with turn-local weight `0`; outcome-only control on the v2 code path. |
| `v2-l010` | Official binary outcome GRPO plus zero-sum, fixed-L1 turn-local weight `0.1`. |

Every arm sets `TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0`; invalid and permanently
over-cap groups are still replaced. Training uses seed `1234`, rollout seed
`42`, deterministic sglang sampling, K=8, six groups per update, LR `2e-6`,
K2 KL `0.01`, and 100 updates.

## Execution

Three 8-GPU smoke jobs were submitted in parallel: `v1-matched` job `1474`,
`v2-l000` job `1472`, and `v2-l010` job `1473`. Formal stage-a, stage-b, and
stage-final jobs start from separate fresh roots only after all smoke runtime
checks pass. These jobs succeeded, but showed that `rollout_seed` did not enter
sampling without sglang deterministic inference. Jobs `1483`/`1484`/`1485`
were stopped after finding that a deterministic over-cap retry reused its
original seed. The retry now advances by K=8 seeds per attempt. Isolated jobs
`1487`/`1488`/`1489` passed runtime checks, but showed that the unseeded User
server caused the first behavioral divergence in 37-39 of 48 trajectories.
Jobs `1490`/`1491`/`1492` therefore enable deterministic inference for both
Agent and User servers. They passed: all 48 behavior trajectories matched
exactly across arms, every loss/gradient was finite, and each arm saved iter0.
Formal stage-a jobs are `1498`/`1499`/`1500`. Smoke artifacts are not resumed
by formal training. All three Stage A jobs completed iter9; Stage B jobs are
`1515`/`1516`/`1517` and completed iter19. Stage Final jobs are
`1543`/`1542`/`1544` for v1/lambda0/lambda0.1 respectively and resume those
separate roots; all completed iter99.

## Training results

In the final smoke, all arms had official reward `0.0417`, truncation `0.3958`,
and five binary zero-variance groups. `v2-l000` had zero actual weighted local
activation. `v2-l010` activated the local channel on 28/48 trajectories and
used exactly `0.1` L1 budget per active trajectory. Stage A completed without
OOM, non-finite training, or checkpoint failure:

| Arm | Mean raw reward | First/last five | Final | Mean grad norm | Mean K2 loss |
|---|---:|---:|---:|---:|---:|
| `v1-matched` | 20.21% | 20.83% / 19.58% | 20.83% | 0.447 | 0.000428 |
| `v2-l000` | 21.25% | 20.00% / 22.50% | 39.58% | 0.303 | 0.000385 |
| `v2-l010` | 20.21% | 20.42% / 20.00% | 22.92% | 0.314 | 0.000475 |

Stage B also completed without a health failure:

| Arm | Stage B raw reward | First/last five | Cumulative 20-update reward | Final grad norm | Final K2 loss |
|---|---:|---:|---:|---:|---:|
| `v1-matched` | 23.96% | 25.83% / 22.08% | 22.08% | 0.342 | 0.00170 |
| `v2-l000` | 23.13% | 26.67% / 19.58% | 22.19% | 0.286 | 0.00164 |
| `v2-l010` | 20.42% | 22.92% / 17.92% | 20.31% | 0.236 | 0.00415 |

`rollout/raw_reward` is the accepted trajectories' mean official binary
`task_reward`; lambda only changes token advantages and is intentionally not
added to this metric. The equal early v2 aggregate rewards therefore did not
mean equal policies or an inactive lambda: after the shared initial rollout,
only `1/48`, `0/48`, and `2/48` behaviors matched in updates 1-3. For
`v2-l010`, the pre-filter dump reconstructs local per-token
absolute advantage p99 `1.005` and maximum `19.27`. The fixed reducer-level
L1 budget is therefore active but can concentrate on short turns. The maximum
is a nine-token malformed-JSON turn in an accepted Telecom trajectory; after
the sample reducer it still contributes only `0.05` negative L1 budget. In the
reconstructed accepted view, `|local advantage| > 2.5` covers `0.298%` of
tokens and `> 10` covers `0.0086%`. This is monitored as a scale diagnostic
rather than a continuation gate. Across accepted Stage A+B, the local/global
total L1 is `34.7/348.31` (`10.0%`); local credit gives 53/66 official
zero-variance groups a usable signal, while only `0.235%` of all accepted
tokens exceed `2.5`. Thus lambda `0.1` is active and auxiliary at the reducer
level.

All Stage Final jobs completed 100 finite updates. Every arm trained on exactly
600 accepted K8 groups (4,800 trajectories); no incomplete group entered an
update. Final accepted diagnostics are:

| Arm | Raw / last-20 | Exact action | DB | Zero-var | Truncation | Mean / max K2 |
|---|---:|---:|---:|---:|---:|---:|
| `v1-matched` | 26.29% / 25.10% | 63.86% | 34.41% | 50.00% | 23.40% | .00898 / .05099 |
| `v2-l000` | 26.77% / 26.88% | 64.14% | 36.50% | 49.50% | 21.44% | .02558 / .19109 |
| `v2-l010` | 26.00% / 28.13% | 62.14% | 34.72% | 51.67% | 24.88% | .01053 / .03750 |

`v2-l010` activates local credit on 1,071/4,800 accepted trajectories and
rescues 179/310 binary-zero-variance groups, leaving 131/600 truly signal-free
groups. All active trajectories satisfy zero-sum allocation and local L1
exactly `0.1`; accepted local/global L1 is `107.1/1806.95` (`5.93%`). Its
per-token absolute local advantage p99/max is `.6644/19.27`; only `.115%` of
tokens exceed `2.5`, and the maximum is the same nine-token malformed turn.
It strongly reduces rule violations relative to `v2-l000`, but full-run raw,
action, and DB do not improve over the controls. This establishes that
lambda `0.1` is active, not that it is optimal; held-out evaluation remains the
selection evidence.

## Official evaluation

Pending. The fixed protocol is all 100 test tasks, four trials, seeds 300/301,
temperature `0.6`, `max_steps=200`, strict-single-v1, and the v1 User SFT with
the Qwen parser enabled. All four checkpoints use candidate mode with the
historical random-sampling semantics: deterministic inference is disabled and
the tau2 trial seed is not forwarded as the Agent's SGLang `sampling_seed`.
The SFT is rerun rather than imported from a historical summary. Report pass@1,
pass@4(any), pass^4, action accuracy, DB accuracy, termination, rule errors,
KL, truncation, and zero-variance rates.

## Decision rule

An arm wins only if its two-seed mean pass@1 and pass@4(any) are both strictly
higher than the other two arms. Pass^4 and paired bootstrap intervals are
reported but are not gates. Exact primary ties use DB accuracy and then action
accuracy. Mixed or unresolved results retain `v1-matched`; no OPD job starts.
