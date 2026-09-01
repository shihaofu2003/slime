# tau2-rl-agent-user-boundary-v2-long200

Purpose: continue the selected long100 iter99 state through updates 100-199 while
holding the boundary-v2 protocol and optimization recipe fixed, then compare
iter199 against selected SFT and iter99 in a two-seed held-out evaluation.

## Fixed continuation

- Source root: `Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804`
  at exact iter99, verified by the completed long100 checkpoint and training
  record.
- Destination root:
  `Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805`.
- Updates 100-199 use LR `2e-6`, K2 KL `0.01`, K=8, quota `2/2/2`, the existing
  reward/penalties, `max_steps=60`, v1 User, signed Agent-owned Contract, and
  turn-credit-v1. The scheduler horizon is overridden while optimizer/RNG and
  domain-sampling state resume from iter99.
- Checkpoints are saved at iter109/119/.../199. A retry resumes only from a
  complete destination checkpoint; before the first save it restarts from the
  unchanged source iter99. Every attempt writes a distinct trajectory artifact.

## Run-control policy

Partial health checks process logs and trajectory attempts in order and let the
latest complete copy of a repeated update win. The original run-control policy
stopped on OOM, non-finite metrics, KL thresholds, quota drift, corrupt resume,
or Contract/turn-credit misalignment; behavior errors and `max_steps` remained
diagnostics. For retrospective interpretation, KL thresholds indicate training
drift and trigger review, but do not by themselves invalidate a finite,
complete checkpoint or its held-out evaluation.

## Jobs

- `pt-7xxg28c1`, `fsh-boundary-v2-long200-final-0805-125417`: updates 100-199,
  8-GPU normal priority; deliberately stopped after the predeclared late-KL
  condition was confirmed at update175 (update176 completed while deletion
  propagated) —
  [run log](jobs/fsh-boundary-v2-long200-final-0805-125417/run_20260805_125417.log).
- `pt-76a8tys1`, `fsh-boundary-v2-long200-kl-waiver-resume-0805-164301`:
  user-requested 8-GPU normal-priority diagnostic continuation from the complete
  iter169 checkpoint; redoes updates 170-176 and continues through 199 while
  treating only the two KL checks as diagnostics —
  [run log](jobs/fsh-boundary-v2-long200-kl-waiver-resume-0805-164301/run_20260805_164302.log).
- `pt-750lmyik`, `fsh-boundary-v2-long200-kl-waiver-convert-0805-181416`:
  1-GPU conversion of the completed iter199 diagnostic checkpoint
  to the isolated KL-waiver HF root —
  [run log](jobs/fsh-boundary-v2-long200-kl-waiver-convert-0805-181416/run_20260805_181416.log).
- `pt-ravwja7o`, `fsh-boundary-v2-long200-kl-waiver-eval-seed300-0805-182006`:
  succeeded 2-GPU full-test evaluation, seed300, 400/400 simulations, with
  namespace probes —
  [run log](jobs/fsh-boundary-v2-long200-kl-waiver-eval-seed300-0805-182006/run_20260805_182006.log).
- `pt-xjhzn0v1`, `fsh-boundary-v2-long200-kl-waiver-eval-seed301-0805-182006`:
  succeeded 2-GPU full-test evaluation, seed301, 400/400 simulations —
  [run log](jobs/fsh-boundary-v2-long200-kl-waiver-eval-seed301-0805-182006/run_20260805_182006.log).

## Evaluation and decision

The predeclared strict path stopped after update176 and recorded `reject` in
[`ITER199_LONG200_DECISION.json`](ITER199_LONG200_DECISION.json); this is the
historical run-control outcome, not a finding that an iter199 checkpoint was
invalid. The later diagnostic continuation resumed the exact iter169 state,
reached iter199, converted it to the isolated
`Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_kl_waiver_hf_20260805`
root, and evaluated seeds 300/301 on 100 tasks × 4 trials. Seed300 also completed
the 60 deterministic plus 240 temperature-0.6 namespace probes.

[`ITER199_KL_WAIVER_RESULT.json`](ITER199_KL_WAIVER_RESULT.json) preserves the
original metadata while recording a complete eval and namespace probe. Its
`formal_decision_unchanged=reject` field refers to the original stopping rule.
The capability result below is interpreted directly rather than discarded by
that label.

## Result

The exact cross-root recovery worked: optimizer/RNG and rollout sampler state
loaded from long100 iter99, and complete destination checkpoints were saved at
iter109/119/129/139/149/159/169 with 12 nonempty distcp shards plus optimizer,
metadata, and rollout state. Updates 100–176 had complete finite metrics, exact
`2/2/2` accepted domain quota, aligned Contract/turn-credit, and no OOM, NaN, or
Inf. Raw binary task success averaged `28.81%` over those 77 training batches;
the last 10 averaged `31.67%`, which is diagnostic rather than held-out eval.

KL rose late in training. Updates 167–176 were
`0.070/0.041/0.060/0.096/0.151/0.134/0.175/0.152/0.289/0.260`: the final
10-step mean was `0.14287` (required `<0.10`) and updates 175–176 were both
`>=0.20`. The historical
[`EARLY_STOP_HEALTH_GATE.json`](EARLY_STOP_HEALTH_GATE.json) therefore fails
exactly `last_10_k2_kl` and `no_consecutive_high_kl`; behavior and `max_steps`
remain diagnostics only.

Historical run-control outcome: **`reject`** by `early_health_stop`. At that
point, latest was the complete iter169 checkpoint and iter199 did not yet exist.
This explains why the original decision file contains no iter199 capability
evidence; it should not be read as a retrospective judgment on the later
checkpoint. Long100 iter99 remained the selected default.

After that stopping decision, the diagnostic continuation loaded the complete
iter169 model, optimizer, RNG, and domain-sampler state. It redid only updates
170–176, which had no saved checkpoint, then generated new updates 177–199;
updates 0–169 were not sampled or trained again. The ordered checker lets the
retry own duplicate updates 170–176 and found all 100 updates complete, exact
quota `2/2/2`, no OOM/NaN/Inf, aligned Contract/turn-credit, and a complete
iter199 checkpoint with 12 nonempty distributed shards plus optimizer,
metadata, and rollout state.

| Updates | Raw binary success | Truncated | K2 KL | Grad norm |
|---|---:|---:|---:|---:|
| 100–169 | 28.39% | 19.29% | 0.0331 | 0.4970 |
| 170–179 | 35.63% | 20.42% | 0.0868 | 0.5761 |
| 180–189 | 33.54% | 15.21% | 0.0967 | 0.5179 |
| 190–199 | 31.87% | 19.17% | 0.3237 | 0.8086 |
| **100–199** | **29.98%** | **18.98%** | **0.0739** | **0.5381** |

The last-10 KL mean was `0.32374`, and updates 193–197 contained four adjacent
high-KL pairs. Those are recorded as the two flagged diagnostics with
`enforced=false` in
[`ITER199_KL_WAIVER_HEALTH_GATE.json`](ITER199_KL_WAIVER_HEALTH_GATE.json); all
non-KL health checks passed.

Held-out results (`pass@1 / pass@4(any) / pass^4`):

| Model | Seed300 | Seed301 | Two-seed mean |
|---|---:|---:|---:|
| Selected boundary SFT | 23.75 / 42.00 / 8.00% | 23.50 / 45.00 / 9.00% | 23.63 / 43.50 / 8.50% |
| Long100 iter99 | 27.50 / 53.00 / 9.00% | 31.50 / 60.00 / 12.00% | 29.50 / 56.50 / 10.50% |
| **Long200 iter199, diagnostic continuation** | **31.00 / 55.00 / 11.00%** | **33.75 / 61.00 / 12.00%** | **32.38 / 58.00 / 11.50%** |
| iter199 − SFT | +7.25 / +13.00 / +3.00pp | +10.25 / +16.00 / +3.00pp | **+8.75 / +14.50 / +3.00pp** |
| iter199 − iter99 | +3.50 / +2.00 / +2.00pp | +2.25 / +1.00 / 0.00pp | **+2.88 / +1.50 / +1.00pp** |

Two-seed mean pass@1 by domain shows where the gain came from:

| Model | Airline | Retail | Telecom |
|---|---:|---:|---:|
| Selected boundary SFT | 31.25% | 25.63% | 17.81% |
| Long100 iter99 | 38.13% | 31.25% | 23.44% |
| **Long200 iter199, diagnostic continuation** | **33.13%** | **32.19%** | **32.19%** |
| iter199 − iter99 | **−5.00pp** | **+0.94pp** | **+8.75pp** |

Paired task-bootstrap intervals for the overall deltas are:

| Seed | Reference | Δ pass@1 (95% CI) | Δ pass@4(any) (95% CI) | Δ pass^4 (95% CI) |
|---|---|---:|---:|---:|
| 300 | SFT | +7.25 `[+2.25,+12.25]`pp | +13.00 `[+3.00,+23.00]`pp | +3.00 `[−3.00,+9.00]`pp |
| 301 | SFT | +10.25 `[+4.50,+16.25]`pp | +16.00 `[+5.00,+27.00]`pp | +3.00 `[−3.00,+10.00]`pp |
| 300 | iter99 | +3.50 `[−1.25,+8.50]`pp | +2.00 `[−8.00,+12.00]`pp | +2.00 `[−3.00,+8.00]`pp |
| 301 | iter99 | +2.25 `[−2.50,+7.00]`pp | +1.00 `[−7.00,+9.00]`pp | 0.00 `[−6.00,+6.00]`pp |

Termination diagnostics for iter199 versus iter99 were `max_steps / too_many_errors`
of `39/2` versus `35/2` at seed300 and `38/1` versus `33/4` at seed301. Telecom
namespace `affected trajectories / raw attempts / attributed terminations` were
`8/12/0` versus `8/8/0` at seed300 and `10/19/0` versus `10/13/0` at seed301;
the separate 300-case probe had zero namespace attempts at both temperatures.

Capability point estimates improved overall on both seeds, with most of the
gain concentrated in telecom. Seed301 airline pass@1 fell `−7.50pp` versus
iter99, and the overall iter199 − iter99 intervals include zero. The evidence
therefore does not support an unconditional replacement of the better-balanced
long100 model, but iter199 remains a valid high-KL checkpoint and a strong
Telecom-oriented teacher candidate. **Long100 iter99 remains the default model;
this experiment schedules no 200–299 continuation.**
