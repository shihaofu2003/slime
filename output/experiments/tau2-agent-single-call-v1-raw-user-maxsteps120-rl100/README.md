# tau2-agent-single-call-v1-raw-user-maxsteps120-rl100

Purpose: measure whether increasing the rollout limit from 60 to 120 steps
reduces artificial truncation and improves strict-single Agent RL when using
the raw `Qwen3-4B-Instruct-2507` User.

Name: tau2-agent-single-call-v1-raw-user-maxsteps120-rl100.

## Design

This run starts from the same single-call SFT and repeats the completed
raw-User RL100 recipe. It preserves LR `2e-6`, K=8, K2 KL `0.01`, domain
quotas, reward shaping, penalties, token limits, retry policy, and all other
training settings. The only change is `TAU2_MAX_STEPS=120` instead of `60`.

Checkpoints are isolated under
`Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812`.
The run covers updates 0-99 through stage-a, stage-b, and stage-final.

## Tasks

- `pt-e4a6ryoo` / RL updates 0-99 —
  [run log](jobs/fsh-tau2-single-v1-raw-user-maxsteps120-rl100-0812-164141/run_20260812_164141.log)
  (completed step 99 and saved the complete `iter_0000099` checkpoint at
  2026-08-12 22:34 CST; `TAU2_MAX_STEPS=120`).
- job-manager `993` / iter99 HF conversion —
  [run log](jobs/993-tau2-single-v1-raw-user-maxsteps120-rl99-convert-0813-001618744/run_20260813_001618744.log)
  (failed before conversion because the updated submit helper's default image
  lacks `sglang`; no output directory was created).
- job-manager `995` / iter99 HF conversion retry —
  [run log](jobs/995-tau2-single-v1-raw-user-maxsteps120-rl99-convert-r2-0813-001857383/run_20260813_001857383.log)
  (succeeded; two safetensors shards).
- job-manager `1000` / raw-User formal evaluation, seed 300 —
  [run log](jobs/1000-tau2-single-v1-raw-user-maxsteps120-rl99-eval-s300-0813-002527085/run_20260813_002527085.log);
  [summary](eval/raw-user/seed300_summary.json) (succeeded; 400/400 simulations).
- job-manager `999` / raw-User formal evaluation, seed 301 —
  [run log](jobs/999-tau2-single-v1-raw-user-maxsteps120-rl99-eval-s301-0813-002526976/run_20260813_002526976.log);
  [summary](eval/raw-user/seed301_summary.json) (succeeded; 400/400 simulations).

## Status

Training completed on 2026-08-12; the final step-99 checkpoint was written
successfully. Both formal evaluation jobs completed on 2026-08-13.

The same checkpoint root was continued for another 100 updates with
`TAU2_MAX_STEPS=120`. The resulting iter199 checkpoint and formal raw-User
evaluation score `30.13/58.50/7.00%`; see the separate
[maxsteps120-rl200 experiment](../tau2-agent-single-call-v1-raw-user-maxsteps120-rl200/README.md).

## Training results

All 100 updates completed with finite metrics and no OOM. The comparison holds
the raw User and training recipe fixed; only rollout `max_steps` changes.
Reward and truncation entries are percentages.

| Rollout max_steps | Raw reward all / final 10 | Truncation all / final 10 | K2 all / final 10 / max | Grad norm mean / max |
|---:|---:|---:|---:|---:|
| 60 | 25.79 / **33.13** | 18.38 / 17.92 | **0.00587** / **0.01333** / 0.02021 | **0.47225** / 0.74230 |
| **120** | **26.63** / 32.92 | **1.19 / 1.04** | 0.00607 / 0.01343 / **0.01574** | 0.48510 / **0.69122** |
| 120 − 60 | +0.83 / −0.21pp | **−17.19 / −16.88pp** | +0.00020 / +0.00010 / −0.00447 | +0.01285 / −0.05108 |

Doubling the rollout step budget almost eliminates training truncation, but it
does not materially raise final-10 training reward.

## Formal evaluation

Both seeds use the same strict-single-v1 Agent protocol, the raw
`Qwen3-4B-Instruct-2507` User, the official `test` split, all 100 tasks, four
trials, temperature `0.6`, and evaluation `max_steps=200`. The two jobs cover
800 simulations in total: zero infrastructure errors, zero strict-single
protocol errors, and no multi-call Agent outputs.

Each cell is pass@1 / pass@4(any) / pass^4. The comparison keeps evaluation
settings fixed; `max_steps=120` refers only to the training rollout budget.

| Agent checkpoint (training User; evaluation User) | Overall result |
|---|---:|
| Single-call SFT (raw; raw) | 28.00 / 51.00 / 9.00% |
| Raw-User RL99 max60 (raw; raw) | 29.25 / 54.00 / 6.50% |
| **Raw-User RL99 max120 (raw; raw)** | **28.38 / 56.50 / 9.50%** |
| Raw-User RL199 max60 (raw; raw) | 31.50 / 58.00 / 8.00% |

| Seed | pass@1 | pass@4(any) | pass^4 |
|---:|---:|---:|---:|
| 300 | 26.50% | 55.00% | 7.00% |
| 301 | 30.25% | 58.00% | 12.00% |
| **Mean** | **28.38%** | **56.50%** | **9.50%** |

### Domain results

| Agent | Airline | Retail | Telecom |
|---|---:|---:|---:|
| SFT (raw User) | 42.50 / 57.50 / 30.00% | 35.62 / 61.25 / 7.50% | 13.12 / 37.50 / 0.00% |
| Raw RL99 max60 | 41.25 / 62.50 / 20.00% | 33.12 / 57.50 / 6.25% | 19.38 / 46.25 / 0.00% |
| **Raw RL99 max120** | **38.12 / 60.00 / 17.50%** | **33.75 / 63.75 / 11.25%** | **18.13 / 47.50 / 3.75%** |
| Raw RL199 max60 | 42.50 / 70.00 / 12.50% | 36.56 / 62.50 / 10.00% | 20.94 / 47.50 / 3.75% |

The overall max120 deltas (task-level, seed-averaged paired bootstrap; 100,000
resamples, master seed 20260813; 95% percentile intervals) are:

| Comparison | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| Max120 RL99 − raw RL99 max60 | −0.88 pp [-5.00, +3.25] | +2.50 pp [-5.50, +10.50] | +3.00 pp [-2.00, +8.00] |
| Max120 RL99 − SFT | +0.38 pp [-3.75, +4.50] | +5.50 pp [0.00, +11.50] | +0.50 pp [-4.00, +5.00] |

The longer rollout budget reduces training truncation by `17.19pp` overall
(`18.38%` to `1.19%`), but the held-out pass@1 point estimate is `0.88pp`
below the max60 control and every overall pass@1 interval includes zero.
Pass@4(any) and pass^4 improve pointwise, with uncertainty still spanning
zero. Thus `max_steps=120` fixes a training-termination diagnostic but does not
yet establish a task-success improvement. Mean evaluation diagnostics are
action accuracy `66.00%`, DB accuracy `30.38%`, and 41 evaluation max-step
terminations per seed (82/800 total); evaluation itself used `max_steps=200`.
