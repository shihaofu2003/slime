# tau2-agent-single-call-v1-raw-user-rl200

Purpose: continue the raw-User strict-single-v1 GRPO lineage from iter99 to iter199 while preserving the 16,384-token per-trajectory training cap and the established training recipe.

Name: `tau2-agent-single-call-v1-raw-user-rl200`.

## Configuration

- Agent checkpoint root: `Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_20260812`; resume from the complete iter99 distributed checkpoint with optimizer, RNG, and rollout sampler state.
- User: unmodified `Qwen3-4B-Instruct-2507`.
- Updates: 100–199; LR `2e-6`, K2 KL `0.01`, K=8, three-domain quota `2:2:2`, rollout `max_steps=60`, and training cap `16,384` tokens.

## Tasks

- `pt-x6p7gg51` / `fsh-tau2-single-v1-raw-user-rl-long200-0812-184511` — [run log](jobs/fsh-tau2-single-v1-raw-user-rl-long200-0812-184511/run_20260812_184511.log) (completed updates 100–199; step 199 and a complete `iter_0000199` distributed checkpoint were saved at 2026-08-12 23:43 CST; final logged loss and gradient norm were finite).
- job-manager `994` / iter199 HF conversion — [run log](jobs/994-tau2-single-v1-raw-user-rl199-convert-0813-001618854/run_20260813_001618854.log) (failed before conversion because the updated submit helper's default image lacks `sglang`; no output directory was created).
- job-manager `996` / iter199 HF conversion retry — [run log](jobs/996-tau2-single-v1-raw-user-rl199-convert-r2-0813-001857493/run_20260813_001857493.log) (succeeded; two safetensors shards).
- job-manager `997` / raw-User formal evaluation, seed 300 — [run log](jobs/997-tau2-single-v1-raw-user-trained-rl199-eval-s300-0813-002526730/run_20260813_002526730.log); [summary](eval/raw-user/seed300_summary.json) (succeeded; 400/400 simulations).
- job-manager `998` / raw-User formal evaluation, seed 301 — [run log](jobs/998-tau2-single-v1-raw-user-trained-rl199-eval-s301-0813-002526873/run_20260813_002526873.log); [summary](eval/raw-user/seed301_summary.json) (succeeded; 400/400 simulations).

## Training results

All 100 continuation updates completed with finite metrics and no OOM. The
matched control below is the original single-call iter100–199 continuation;
only the rollout User differs. Reward and truncation entries are percentages.

| Training User, updates 100–199 | Raw reward all / final 10 | Truncation all / final 10 | K2 all / final 10 / max | Grad norm mean / max |
|---|---:|---:|---:|---:|
| v1 User SFT | 29.81 / 32.50 | 20.90 / 22.29 | 0.02643 / 0.04252 / 0.07400 | 0.46001 / 0.68881 |
| **Raw User** | **32.67 / 33.75** | **18.17 / 18.54** | **0.02472 / 0.03579 / 0.04094** | **0.44400 / 0.64181** |
| Raw − v1 | +2.85 / +1.25pp | −2.73 / −3.75pp | −0.00171 / −0.00674 / −0.03306 | −0.01601 / −0.04700 |

Raw-User continuation is healthier on these training diagnostics, especially
late K2 drift. The matched raw-User held-out evaluation completed successfully.

## Formal evaluation

Both seeds use the same strict-single-v1 Agent protocol, the raw
`Qwen3-4B-Instruct-2507` User, the official `test` split, all 100 tasks, four
trials, temperature `0.6`, and evaluation `max_steps=200`. The two jobs cover
800 simulations in total: zero infrastructure errors, zero strict-single
protocol errors, and no multi-call Agent outputs.

Each cell is pass@1 / pass@4(any) / pass^4. The historical rows use the same
raw-User evaluation protocol; `v1 User RL199` was trained with the v1 User and
is included as a descriptive cross-User reference.

| Agent checkpoint (training User; evaluation User) | Overall result |
|---|---:|
| Single-call SFT (raw; raw) | 28.00 / 51.00 / 9.00% |
| Raw-User RL99 iter99 (raw; raw) | 29.25 / 54.00 / 6.50% |
| v1-User RL199 iter199 (v1; raw) | 26.88 / 52.50 / 8.00% |
| **Raw-User RL199 iter199 (raw; raw)** | **31.50 / 58.00 / 8.00%** |

| Seed | pass@1 | pass@4(any) | pass^4 |
|---:|---:|---:|---:|
| 300 | 31.75% | 58.00% | 12.00% |
| 301 | 31.25% | 58.00% | 4.00% |
| **Mean** | **31.50%** | **58.00%** | **8.00%** |

### Domain results

| Agent | Airline | Retail | Telecom |
|---|---:|---:|---:|
| SFT (raw User) | 42.50 / 57.50 / 30.00% | 35.62 / 61.25 / 7.50% | 13.12 / 37.50 / 0.00% |
| Raw RL99 max60 | 41.25 / 62.50 / 20.00% | 33.12 / 57.50 / 6.25% | 19.38 / 46.25 / 0.00% |
| v1 RL199 (raw eval) | 41.87 / 62.50 / 25.00% | 29.69 / 58.75 / 7.50% | 16.56 / 41.25 / 0.00% |
| **Raw RL199 max60** | **42.50 / 70.00 / 12.50%** | **36.56 / 62.50 / 10.00%** | **20.94 / 47.50 / 3.75%** |

The overall raw-RL199 deltas (task-level, seed-averaged paired bootstrap;
100,000 resamples, master seed 20260813; 95% percentile intervals) are:

| Comparison | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| Raw RL199 − SFT | +3.50 pp [-1.00, +8.12] | +7.00 pp [0.00, +14.50] | −1.00 pp [-6.00, +4.00] |
| Raw RL199 − raw RL99 | +2.25 pp [-2.12, +6.75] | +4.00 pp [-3.50, +11.50] | +1.50 pp [-2.50, +5.50] |
| Raw RL199 − v1 RL199 (raw eval) | +4.62 pp [+0.75, +8.62] | +5.50 pp [-2.00, +13.00] | 0.00 pp [-5.00, +5.00] |

Raw-User RL199 is the best raw-User row on pass@1 and pass@4(any), improving
the raw-User RL99 point estimates by `+2.25/+4.00pp`. The intervals for the
controlled raw-User SFT and RL99 comparisons include zero, so this is a
promising point estimate rather than a statistically established overall win.
Pass^4 remains below SFT (`8.00%` vs `9.00%`), indicating no consistency gain.
The strongest pass@1 movement is Telecom (`+7.81pp` versus SFT); Airline
pass^4 falls while pass@4(any) rises, so the improvement is not uniform across
success criteria.

Against the same-iteration v1-User-trained RL199 under the same raw-User
evaluation, raw-User training improves pass@1 by `+4.62pp`, with interval
`[+0.75,+8.62]pp`. This supports a training-User effect within the raw-User
evaluation distribution. The crossed V1-User evaluation is now complete:
raw-User training changes pass@1/pass@4(any)/pass^4 by
`-6.13/-8.50/-4.50pp` there. The direction therefore reverses with the
evaluation User, supporting User-distribution specialization rather than a
gain that generalizes across both User models.

Mean diagnostic values over the two seeds are action accuracy `68.14%`, DB
accuracy `31.53%`, and 36.5 evaluation max-step terminations per 400-trial
seed (73/800 total). These are evaluation diagnostics with `max_steps=200`,
not the training rollout truncation metric.
