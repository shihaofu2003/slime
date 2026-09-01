# tau2-eval-qwen36-user-async-timed

Purpose: validate the official tau2 asynchronous evaluation, measure elastic
domain scheduling, and compare cache-aware with round-robin User routing.

## Tasks

- Round-robin 2-Agent/3-User full evaluation `11878`: all 100 test tasks, four
  trials, seed 300, eight GPUs —
  [run log](jobs/11878-qwen36-round-robin-2a3u-full-seed300-0901-182544913/run_*_20260901_182544913.log).
  Submitted with Agent routing `cache_aware`, User routing `round_robin`,
  initial domain concurrency `2:2:5`, and elastic slot borrowing under a global
  trajectory concurrency of 9.
- Round-robin 2-Agent/3-User smoke `11851`: 15 tasks per domain, four trials,
  seed 300, eight GPUs —
  [run log](jobs/11851-qwen36-round-robin-2a3u-smoke-15each-0901-170707804/run_*_20260901_170707804.log),
  [summary](eval/round-robin-smoke/seed300_0901_091241_summary.json).
  Passed: 45 tasks / 180 simulations, zero infrastructure errors, evaluation
  wall time 13.81 minutes, and effective trajectory parallelism 6.01x. User
  workers each served 784 successful requests including one readiness probe.
- Elastic 2-Agent/3-User smoke `11819`: 15 tasks per domain, four trials,
  seed 300, eight GPUs —
  [run log](jobs/11819-qwen36-elastic-2a3u-smoke-15each-0901-154850600/run_*_20260901_154850600.log).
  Passed: 45 tasks / 180 simulations, zero infrastructure errors, evaluation
  wall time 15.51 minutes, and effective trajectory parallelism 7.05x. Slots
  moved `2:2:5 -> 0:3:6 -> 0:0:9`. Under `cache_aware`, the three User workers
  served `1033/1470/0` evaluation requests; the next wrapper therefore uses
  `round_robin` for User routing.
- Smoke `11752`: two tasks per domain, two trials, seed 300, eight GPUs —
  [run log](jobs/11752-qwen3-4b-qwen36-async-smoke-0901-122626274/run_*_20260901_122626274.log).
  Invalid: all 12 simulations ended with infrastructure errors because the
  Agent `/generate` request used unsupported `seed` instead of
  `sampling_seed`. Job-manager reported success because tau2 returned results
  and the wrapper exited 0; the evaluator now fails the process when any
  infrastructure error is present.
- Corrected smoke `11753`: two tasks per domain, two trials, seed 300, eight
  GPUs — [run log](jobs/11753-qwen3-4b-qwen36-async-smoke-v2-0901-124257188/run_*_20260901_124257188.log).
  Passed: 6 tasks / 12 simulations, zero infrastructure errors, no missing or
  duplicate trials, and all trajectories recorded Agent/User timing.
- Full evaluation `11756`: all 100 test tasks, four trials, seed 300, eight
  GPUs — [run log](jobs/11756-qwen3-4b-qwen36-async-full-seed300-0901-125330452/run_*_20260901_125330452.log),
  [summary](eval/full/seed300_0901_045448_summary.json).
  Passed: 100 tasks / 400 simulations, all task/trial pairs present exactly
  once, zero infrastructure errors, and all Agent/User timings positive.

## Protocol

- Current topology: two TP1 Qwen3-4B-Instruct-2507 Agent replicas on GPUs 0–1
  with `cache_aware`, and three TP2 Qwen3.6-27B User replicas on GPUs 2–7 with
  `round_robin`.
- Domains run in independent processes with four trials per task. Smoke runs
  use 15 tasks per domain; full runs use the official airline/retail/telecom
  quotas of `20/40/40`.
- Initial concurrency is `2:2:5`; completed-domain slots are lent to unfinished
  domains under a global limit of 9.
- Agent temperature is 0.6, User temperature is 0, and SGLang deterministic
  inference is disabled.

## Status

Corrected smoke `11753` passed. Service initialization took 323 seconds and
the three-domain evaluation took 48.38 seconds. Across 255.97 trajectory
seconds, Agent requests used 75.50 seconds (29.50%), User requests used 179.64
seconds (70.18%), and other work used 0.84 seconds (0.33%); effective trajectory
parallelism was 5.29×. The smoke pass@1 result was 41.67% on this six-task
sample and is not a model-quality estimate.

Full evaluation `11756` succeeded. Job wall time was 52.84 minutes and the
wrapper measured 52.75 minutes: 5.38 minutes to make all services ready, 47.30
minutes for evaluation and summary generation, and about 0.07 minutes for
cleanup. Domain wall times were 10.43 minutes for airline, 19.18 minutes for
retail, and 47.07 minutes for telecom, so telecom determined the tail.

| Scope | pass@1 | pass@4(any) | pass^4 | Wall time |
|---|---:|---:|---:|---:|
| Airline | 31.25% | 65.00% | 15.00% | 10.43 min |
| Retail | 46.88% | 80.00% | 20.00% | 19.18 min |
| Telecom | 3.75% | 7.50% | 2.50% | 47.07 min |
| Overall | 26.50% | 48.00% | 12.00% | 47.29 min |

Across 12,841.90 cumulative trajectory seconds, Agent inference used 5,649.69
seconds (43.99%), User inference used 7,150.56 seconds (55.68%), and other work
used 41.65 seconds (0.32%). Agent p50/p95 latency was 0.63/2.09 seconds; User
p50/p95 was 1.17/2.69 seconds. Effective trajectory parallelism was 4.53×.

The previous four-GPU sequential job `11728` took 174.26 minutes end to end;
`11756` is 3.30× faster. Evaluation wall time fell from about 168 minutes to
47.29 minutes (3.55×), while GPU-hours fell from 11.62 to 7.05 (39.4%). The
two runs are separate stochastic samples, so their score differences are not
used as a model-quality comparison.

Cache-aware routing did not balance all replicas: Agent workers handled
approximately `1/3359/1/3703` requests including warmups, while User workers
handled `2049/3420`. The result is valid, but two Agent replicas were nearly
idle. Any routing-policy or GPU-allocation change needs a matched timing smoke;
the measured production budget for this configuration is about one hour per
100-task, four-trial evaluation.

Smoke `11851` validates round-robin User routing: all three User workers served
exactly 784 successful requests including readiness probes, versus
`1033/1470/0` under cache-aware routing in `11819`. Evaluation wall time fell
from 15.51 to 13.81 minutes (11.0%), while the matched stochastic sample changed
and therefore is not a model-quality comparison. Full evaluation `11878` uses
the validated topology and routing policy.

## Correctness comparison

The serial TP1, serial TP2, and asynchronous runs used the same Agent/User
model families, test split, 100-task quota, four trials, seed 300, and
`max_steps=200`. Their official summaries are
[11723](../tau2-eval-qwen36-user-timing/eval/seed300_summary.json),
[11728](../tau2-eval-qwen36-user-timing-tp2/eval/seed300_summary.json), and
[11756](eval/full/seed300_0901_045448_summary.json).

| Run | User topology | Scope | Tasks | Trajectories | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---|---|---:|---:|---:|---:|---:|
| 11723 | TP1 serial | Airline | 20 | 80 | 38.75% (31/80) | 65.00% (13/20) | 15.00% (3/20) |
| 11723 | TP1 serial | Retail | 40 | 160 | 48.75% (78/160) | 80.00% (32/40) | 25.00% (10/40) |
| 11723 | TP1 serial | Telecom | 40 | 160 | 1.875% (3/160) | 5.00% (2/40) | 0.00% (0/40) |
| 11723 | TP1 serial | Overall | 100 | 400 | 28.00% (112/400) | 47.00% (47/100) | 13.00% (13/100) |
| 11728 | TP2 serial | Airline | 20 | 80 | 32.50% (26/80) | 60.00% (12/20) | 0.00% (0/20) |
| 11728 | TP2 serial | Retail | 40 | 160 | 46.25% (74/160) | 85.00% (34/40) | 15.00% (6/40) |
| 11728 | TP2 serial | Telecom | 40 | 160 | 5.00% (8/160) | 10.00% (4/40) | 0.00% (0/40) |
| 11728 | TP2 serial | Overall | 100 | 400 | 27.00% (108/400) | 50.00% (50/100) | 6.00% (6/100) |
| 11756 | 2 × TP2 asynchronous | Airline | 20 | 80 | 31.25% (25/80) | 65.00% (13/20) | 15.00% (3/20) |
| 11756 | 2 × TP2 asynchronous | Retail | 40 | 160 | 46.88% (75/160) | 80.00% (32/40) | 20.00% (8/40) |
| 11756 | 2 × TP2 asynchronous | Telecom | 40 | 160 | 3.75% (6/160) | 7.50% (3/40) | 2.50% (1/40) |
| 11756 | 2 × TP2 asynchronous | Overall | 100 | 400 | 26.50% (106/400) | 48.00% (48/100) | 12.00% (12/100) |

All three runs contain 100 tasks and 400 trajectories, with four trials per
task, zero infrastructure errors, and zero multi-call output violations. The
asynchronous overall pass@1 is 0.50 percentage points below the closer serial
result; pass@4(any) and pass^4 lie between the two serial results. Exact score
equality is not expected because
SGLang deterministic inference was disabled and concurrent request ordering
changes the sampled trajectories.

| Run | Action accuracy | DB accuracy | `max_steps` terminations | Job wall time | GPU-hours |
|---|---:|---:|---:|---:|---:|
| 11723 | 43.97% | 36.46% | 27 | 229.94 min | 7.66 |
| 11728 | 43.34% | 37.60% | 25 | 174.26 min | 11.62 |
| 11756 | 42.48% | 35.64% | 22 | 52.84 min | 7.05 |

The asynchronous action and DB accuracies are within 1.49 and 1.96 percentage
points of both serial references, and its termination distribution is also
close. Together with complete trial coverage and zero infrastructure errors,
this supports using the asynchronous path as the faster replacement for the
serial evaluation.
