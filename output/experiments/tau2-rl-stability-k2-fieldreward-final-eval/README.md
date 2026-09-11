# tau2-rl-stability-k2-fieldreward-final-eval

Purpose: convert the final `iteration 99` checkpoints from the
`2e-6/3e-6/5e-6` stability sweep to Hugging Face format, then compare them with
tau2 official held-out evaluation using the same v1 STOP-trained User as RL
training.

## Configuration

- Agent checkpoints: final `iter_0000099` from all three LR arms.
- User: `Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf`.
- Eval: official tau2 runner, airline/retail/telecom test splits, all tasks,
  four trials, seed 300, Agent temperature 0.6/top-p 1.0/max tokens 8192.
- Resources: conversion uses one GPU per job; evaluation uses two GPUs per job,
  with the Agent on GPU 0 and the v1 User simulator on GPU 1 (TP=1 each).

## Jobs

- 2026-07-29 22:59 CST: initial conversion submissions for all three LR arms
  were rejected before job creation with `MEMBER_QUOTA_EXCEEDED`:
  [2e-6 submit log](jobs/fsh-convert-rl-stability-lr2e6-iter99-0729-225956/submit_20260729_225956.log),
  [3e-6 submit log](jobs/fsh-convert-rl-stability-lr3e6-iter99-0729-225956/submit_20260729_225956.log),
  [5e-6 submit log](jobs/fsh-convert-rl-stability-lr5e6-iter99-0729-225956/submit_20260729_225956.log).
- A 23:28 reserved retry was also rejected; spot quota accepted the subsequent
  conversion jobs.
- `pt-kypbn3ab` / `2e-6` final-checkpoint HF conversion (spot):
  [run log](jobs/fsh-convert-rl-stability-lr2e6-iter99-spot-0729-232852/run_20260729_232852.log).
  `SUCCEEDED`; output `iter_0000099_hf`.
- `pt-lowyera9` / `3e-6` final-checkpoint HF conversion (spot):
  [run log](jobs/fsh-convert-rl-stability-lr3e6-iter99-spot-0729-232900/run_20260729_232900.log).
  `SUCCEEDED`; output `iter_0000099_hf`.
- `pt-bp8pyydd` / `5e-6` final-checkpoint HF conversion (spot):
  [run log](jobs/fsh-convert-rl-stability-lr5e6-iter99-spot-0729-232900/run_20260729_232900.log).
  `SUCCEEDED`; output `iter_0000099_hf`.
- `pt-jzgqt8mt` / `2e-6` official Pass@4 evaluation (2 GPU, spot):
  [submit log](jobs/fsh-eval-rl-stability-lr2e6-iter99-0729-235135/submit_20260729_235135.log).
  Deleted before scheduling after reserved quota became available.
- `pt-q5hspj5u` / `3e-6` official Pass@4 evaluation (2 GPU, spot):
  [submit log](jobs/fsh-eval-rl-stability-lr3e6-iter99-0729-235135/submit_20260729_235135.log).
  Deleted before scheduling after reserved quota became available.
- `pt-a1oj86gj` / `5e-6` official Pass@4 evaluation (2 GPU, spot):
  [submit log](jobs/fsh-eval-rl-stability-lr5e6-iter99-0729-235135/submit_20260729_235135.log).
  Deleted before scheduling after reserved quota became available.
- `pt-l1kxga4w` / `2e-6` official Pass@4 evaluation (2 GPU, reserved):
  [run log](jobs/fsh-eval-rl-stability-lr2e6-iter99-reserved-0729-235441/run_20260729_235441.log).
  `SUCCEEDED`; 400 simulations and zero infrastructure errors;
  [summary](../../../examples/tau2-bench/eval/official/outputs/Qwen3-4B-tau2-agent-rl-stability-lr2e6-iter0000099/user_stop_parser_pass4_summary.json).
- `pt-5n0a6qhc` / `3e-6` official Pass@4 evaluation (2 GPU, reserved):
  [run log](jobs/fsh-eval-rl-stability-lr3e6-iter99-reserved-0729-235458/run_20260729_235458.log).
  Evaluation completed with 400 simulations and zero infrastructure errors;
  deleted after result persistence because Agent sglang cleanup hung;
  [summary](../../../examples/tau2-bench/eval/official/outputs/Qwen3-4B-tau2-agent-rl-stability-lr3e6-iter0000099/user_stop_parser_pass4_summary.json).
- `pt-h905ln01` / `5e-6` official Pass@4 evaluation (2 GPU, reserved):
  [run log](jobs/fsh-eval-rl-stability-lr5e6-iter99-reserved-0729-235458/run_20260729_235458.log).
  Evaluation completed with 400 simulations and zero infrastructure errors;
  deleted after result persistence because Agent sglang cleanup hung;
  [summary](../../../examples/tau2-bench/eval/official/outputs/Qwen3-4B-tau2-agent-rl-stability-lr5e6-iter0000099/user_stop_parser_pass4_summary.json).

## Conversion verification

All three HF directories contain `config.json`, tokenizer files,
`model.safetensors.index.json`, and two complete weight shards of
5,324,169,368 and 2,720,812,592 bytes.

## Results

Task-level `pass@k(any)` uses `1-C(4-c,k)/C(4,k)` for a task with `c`
successes in four trials:

| Agent | pass@1 | pass@2 | pass@3 | pass@4(any) |
|---|---:|---:|---:|---:|
| multitool SFT | 23.25% | 33.33% | 40.25% | 46.00% |
| RL `2e-6` iter 99 | 25.50% | 37.50% | 44.75% | 49.00% |
| `2e-6` − SFT | +2.25 pp | +4.17 pp | +4.50 pp | +3.00 pp |
| RL `3e-6` iter 99 | 23.50% | 35.00% | 42.00% | 47.00% |
| `3e-6` − SFT | +0.25 pp | +1.67 pp | +1.75 pp | +1.00 pp |
| RL `5e-6` iter 99 | 18.50% | 28.33% | 34.50% | 39.00% |
| `5e-6` − SFT | -4.75 pp | -5.00 pp | -5.75 pp | -7.00 pp |

Official tau2 `pass^k` estimates all `k` attempts succeeding:

| Agent | pass^1 | pass^2 | pass^3 | pass^4 |
|---|---:|---:|---:|---:|
| multitool SFT | 23.25% | 13.17% | 10.00% | 8.00% |
| RL `2e-6` iter 99 | 25.50% | 13.50% | 8.75% | 7.00% |
| RL `3e-6` iter 99 | 23.50% | 12.00% | 7.50% | 5.00% |
| RL `5e-6` iter 99 | 18.50% | 8.67% | 5.00% | 3.00% |

Domain `pass@1`:

| Agent | airline | retail | telecom |
|---|---:|---:|---:|
| multitool SFT | 28.75% | 23.75% | 20.00% |
| RL `2e-6` iter 99 | 31.25% | 21.25% | 26.88% |
| RL `3e-6` iter 99 | 27.50% | 26.88% | 18.12% |
| RL `5e-6` iter 99 | 23.75% | 24.38% | 10.00% |

Termination totals over 400 simulations:

| Agent | user stop | max steps | error | infrastructure error |
|---|---:|---:|---:|---:|
| multitool SFT | 362 | 14 | 24 | 0 |
| RL `2e-6` iter 99 | 355 | 21 | 24 | 0 |
| RL `3e-6` iter 99 | 340 | 24 | 36 | 0 |
| RL `5e-6` iter 99 | 258 | 6 | 136 | 0 |

All four evaluations have the same 100 tasks, four trials, seed 300,
`max_steps=200`, v1 User (`iter_0006899`, temperature 0), and Agent decoding
(temperature 0.6, top-p 1.0, max tokens 8192). Each has 400 simulations and
zero infrastructure errors.

`2e-6` is the best RL arm and shows weak positive task coverage, but does not
establish a real capability gain: versus SFT its paired task-level
`pass@1` difference is `+2.25 pp` (normal 95% CI `[-2.84, +7.34]`) and
`pass@4(any)` difference is `+3.00 pp` (`[-8.09, +14.09]`), while official
`pass^4` decreases by 1 pp. `3e-6` is effectively flat on `pass@1` and worse
on consistency. `5e-6` is degraded across aggregate metrics; telecom has 111
`too_many_errors` terminations out of 160. It emits 2,377 tool-error messages
across 303 simulations, versus 611 across 199 simulations for SFT, dominated
by nonexistent tool names. The LR ordering and identical User control point
to RL optimization instability/high-LR policy degeneration, not the User SFT
model, as the primary cause of the poor sweep outcome.

## Monitoring

- 2026-07-30 00:25 CST: all three reserved eval jobs are `RUNNING`. In every
  job, the Agent server is ready on GPU 0 and the training-time v1 User server
  is ready on GPU 1; the official runner confirms all test tasks, four trials.
- `2e-6` and `3e-6` are progressing through retail trial 1. `5e-6` reached
  airline task 44 but had spent about 15 minutes in that single simulation,
  consistent with the high-LR repetition/long-trajectory risk seen during
  training. The job remains live and is retained for the controlled
  comparison.
- Repeated litellm “model isn't mapped” lines are cost-accounting warnings for
  the local User model, not inference failures.
- 2026-07-30 00:56 CST: all jobs remain healthy. `2e-6` has completed airline
  and retail and is at telecom `22/160`; `3e-6` is at telecom `8/160`.
  `5e-6` eventually exited the long airline task 44 but is only at airline
  trial 2 `28/80`, confirming substantially lower evaluation throughput from
  long trajectories. Live average reward is domain-local and is not used as
  the final cross-model metric.
- 2026-07-30 01:27 CST: `2e-6` reached telecom `87/160`. `3e-6` remained in
  one telecom simulation for about 33 minutes, while `5e-6` remained in
  airline task 24 trial 2 for about 32 minutes. Both slow jobs continue to
  issue local User requests, so this is Agent looping/long generation rather
  than a dead sglang service. The official `max_steps=200` is retained for
  comparability with the existing SFT official baseline.
- 2026-07-30 02:17 CST: `2e-6` completed successfully: 400 simulations,
  zero infrastructure errors, and airline/retail/telecom `pass^1` of
  `31.25/21.25/26.88%`. Its task-weighted overall `pass^1/pass^4` is
  `25.50/7.00%`, versus `23.25/8.00%` for the matching multitool SFT
  baseline. This mixed `+2.25/-1.00 pp` result is not evidence of a clear
  capability gain. `3e-6` is at telecom `82/160`; `5e-6` is at airline trial
  2 `37/80`, again slowed by a single trajectory lasting over 35 minutes.
- 2026-07-30 02:56 CST: `3e-6` reached telecom `156/160` and is close to
  completion. `5e-6` is only at airline `40/80`; another single simulation
  has run for about 28 minutes. Both jobs remain `RUNNING`, with no
  infrastructure failure observed.
- 2026-07-30 03:02 CST: the complete `3e-6` summary appeared with 400
  simulations and zero infrastructure errors; the platform job was still
  performing final cleanup. `5e-6` remains incomplete.
- 2026-07-30 03:33 CST: `5e-6` completed airline and reached retail `23/160`,
  with normal short-episode throughput at this point. The `3e-6` platform job
  remains in cleanup after its summary was safely written; `run_eval.sh`
  emitted both cleanup messages and is waiting for the Agent sglang process
  to exit. This does not affect the completed metrics.
- 2026-07-30 04:03 CST: `5e-6` reached retail `39/160` and entered another
  long simulation (about 28 minutes so far). The completed `3e-6` job had
  remained stuck in Agent sglang cleanup for over one hour, so job
  `pt-5n0a6qhc` was deleted to release its two GPUs after verifying that its
  summary, all 400 simulations, and logs were persisted.
- 2026-07-30 04:34 CST: the deleted `3e-6` job disappeared from the active
  job list, confirming resource release. `5e-6` reached retail `107/160`,
  completing about 68 simulations in this interval without a new long stall
  or infrastructure error.
- 2026-07-30 05:04 CST: `5e-6` completed retail and reached telecom `79/160`.
  The in-progress telecom average reward is about `0.11`; this domain-local
  live value is not used as a final aggregate. The job remains healthy with
  no infrastructure error.
- 2026-07-30 05:35 CST: `5e-6` completed all 400 simulations and wrote its
  final summary with zero infrastructure errors. All three LR evaluations are
  complete.
- 2026-07-30 05:43 CST: `5e-6` reproduced the post-evaluation Agent sglang
  cleanup hang. After verifying its summary, all simulations, and logs, job
  `pt-h905ln01` was deleted to release its two GPUs.
