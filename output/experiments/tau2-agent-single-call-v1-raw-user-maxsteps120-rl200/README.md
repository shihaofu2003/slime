# tau2-agent-single-call-v1-raw-user-maxsteps120-rl200

Purpose: continue the raw-User strict-single-v1 GRPO run from iter99 to
iter199 while keeping the training rollout limit at `max_steps=120`.

Name: `tau2-agent-single-call-v1-raw-user-maxsteps120-rl200`.

## Design

The continuation resumes the complete distributed checkpoint and optimizer/RNG
state at
`Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812/iter_0000099`.
It keeps the unmodified `Qwen3-4B-Instruct-2507` User, 8-GPU layout, LR
`2e-6`, K2 KL `0.01`, K=8, reward shaping, penalties, 16,384-token cap, domain
quotas, and all other settings from the max120 RL100 run. Updates 100–199 were
completed with the constant-LR horizon extended to 200.

## Tasks

- `1109` / first max120 raw-User continuation submission — [run log](jobs/1109-tau2-single-v1-raw-user-maxsteps120-rl-long200-0813-0813-092159889/run_20260813_092159889.log) (failed in preflight because the default image lacked `sglang_router`; no update ran).
- `1110` / max120 raw-User continuation retry, updates 100–199 — [run log](jobs/1110-tau2-single-v1-raw-user-maxsteps120-rl-long200-r2-0813-0813-092713318/run_20260813_092713318.log) (succeeded at step 199; complete distributed checkpoint saved).
- `1232` / iter199 distributed checkpoint to HF — [run log](jobs/1232-tau2-single-v1-raw-user-maxsteps120-rl199-convert-0813-164727830/run_20260813_164727830.log) (succeeded; `iter_0000199_hf` is available).
- `1234` / iter199 raw-User formal evaluation, seed 300 — [run log](jobs/1234-tau2-single-v1-raw-user-maxsteps120-rl199-eval-s300-0813-165506397/run_20260813_165506397.log); [summary](eval/raw-user/seed300_summary.json) (succeeded; 400/400 simulations).
- `1235` / iter199 raw-User formal evaluation, seed 301 — [run log](jobs/1235-tau2-single-v1-raw-user-maxsteps120-rl199-eval-s301-0813-165506690/run_20260813_165506690.log); [summary](eval/raw-user/seed301_summary.json) (succeeded; 400/400 simulations).

## Status

Training, HF conversion, and both formal evaluations are complete. The two
evaluation jobs cover 800 simulations with zero infrastructure errors, zero
strict-single protocol errors, and zero multi-call Agent outputs.

## Training results

The 100-update continuation remained finite and had no OOM. Values below are
means over updates 100–199; percentages refer to rollout batches.

| Rollout max_steps | Raw reward all / final 10 | Truncation all / final 10 | K2 all / final 10 / max | Grad norm mean / max |
|---:|---:|---:|---:|---:|
| 60 (matched raw-User continuation) | 32.67 / 33.75% | 18.17 / 18.54% | 0.02472 / 0.03579 / 0.04094 | 0.44400 / 0.64181 |
| **120 (this run)** | **34.10 / 31.25%** | **2.56 / 2.50%** | 0.02620 / 0.03324 / 0.04525 | 0.45947 / 0.62735 |

At step 199, raw reward was `47.92%`, train loss `0.00714`, K2 loss
`0.04525`, and gradient norm `0.48228`. Increasing the rollout budget still
substantially reduces training truncation, but this diagnostic does not by
itself imply a held-out capability gain.

## Formal evaluation

Both seeds use the strict-single-v1 Agent protocol, raw
`Qwen3-4B-Instruct-2507` evaluation User, the official `test` split, 100 tasks,
four trials, Agent temperature `0.6`, and evaluation `max_steps=200`.

| Seed | pass@1 | pass@4(any) | pass^4 | max-step terminations |
|---:|---:|---:|---:|---:|
| 300 | 29.50% | 61.00% | 7.00% | 75 / 400 |
| 301 | 30.75% | 56.00% | 7.00% | 65 / 400 |
| **Mean** | **30.13%** | **58.50%** | **7.00%** | **140 / 800** |

Mean action accuracy is `69.40%` and DB accuracy is `39.39%`; the evaluation
max-step count is the only notable termination diagnostic (`140/800`).

Each result cell below is pass@1 / pass@4(any) / pass^4. Every evaluation uses
`max_steps=200` and the Qwen User tool-call parser. Both User columns in a row
use the same seed coverage; `★` marks the cell that is higher on all three
metrics. Rows follow raw Qwen3, single-call SFT, iter99, iter199, and external
Qwen3.5.

### User ablation matrix

| Agent checkpoint | Training User | Training max_steps | Comparison seeds | Evaluation User = V1, parser ON | Evaluation User = RAW, parser ON |
|---|---|---:|---:|---:|---:|
| [Raw Qwen3-4B-Instruct-2507](../tau2-raw-agent-raw-user-parser-on/README.md) | N/A (raw Agent) | N/A | 300/301 mean | 23.88 / 45.50 / 6.50% ★ | 17.38 / 35.50 / 2.50% |
| Single-call SFT final | N/A (offline SFT) | N/A | 300/301 mean | 25.88 / 49.50 / 8.50% | 28.00 / 51.00 / 9.00% ★ |
| V1-RL iter99 | V1 | 60 | 300/301 mean | 27.50 / 53.50 / 10.00% ★ | 25.25 / 50.50 / 5.50% |
| RAW-RL iter99 | RAW | 60 | 300/301 mean | 24.25 / 49.50 / 6.00% | 29.25 / 54.00 / 6.50% ★ |
| RAW-RL iter99 max120 | RAW | 120 | 300/301 mean | 25.25 / 51.00 / 6.00% | 28.38 / 56.50 / 9.50% ★ |
| V1-RL iter199 | V1 | 60 | 300/301 mean | 31.13 / 61.50 / 10.50% ★ | 26.88 / 52.50 / 8.00% |
| RAW-RL iter199 | RAW | 60 | 300/301 mean | 25.00 / 53.00 / 6.00% | 31.50 / 58.00 / 8.00% ★ |
| RAW-RL iter199 max120 (this run) | RAW | 120 | 300/301 mean | 25.25 / 49.00 / 5.50% | 30.13 / 58.50 / 7.00% ★ |
| [Raw Qwen3.5-4B thinking-on](../tau2-raw-agent-raw-user-parser-on/README.md) | N/A (raw Agent) | N/A | 300/301 mean | 48.50 / 76.00 / 22.00% | 53.25 / 78.00 / 25.00% ★ |
| [Raw Qwen3.5-4B non-thinking](../tau2-raw-agent-raw-user-parser-on/README.md) | N/A (raw Agent) | N/A | 300/301 mean | 32.75 / 61.00 / 9.00% ★ | 25.63 / 53.50 / 7.00% |

This matrix separates the User used during RL rollout from the User used only
for evaluation. Jobs `1284`–`1288` and `1302`–`1306` filled all five formerly
missing RL cells at both seeds, and every RL Agent did better on all three
metrics with the evaluation User matching its rollout User. The raw-Agent rows
are external protocol references; both User columns use parser ON and are
complete at both seeds.

### Domain results

| Agent | Airline | Retail | Telecom |
|---|---:|---:|---:|
| Raw-User SFT | 42.50 / 57.50 / 30.00% | 35.62 / 61.25 / 7.50% | 13.12 / 37.50 / 0.00% |
| Raw-User RL99 max120 | 38.12 / 60.00 / 17.50% | 33.75 / 63.75 / 11.25% | 18.13 / 47.50 / 3.75% |
| Raw-User RL199 max60 | 42.50 / 70.00 / 12.50% | 36.56 / 62.50 / 10.00% | 20.94 / 47.50 / 3.75% |
| V1-User RL199, raw evaluation | 41.88 / 62.50 / 25.00% | 29.69 / 58.75 / 7.50% | 16.56 / 41.25 / 0.00% |
| **Raw-User RL199 max120 (this run)** | **40.00 / 67.50 / 12.50%** | **38.75 / 72.50 / 10.00%** | **16.56 / 40.00 / 1.25%** |

### Comparison

| Comparison | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| This run − Raw-User RL99 max120 | **+1.75pp** | **+2.00pp** | **−2.50pp** |
| This run − Raw-User RL99 max60 | +0.88pp | +4.50pp | +0.50pp |
| This run − Raw-User SFT | +2.13pp | +7.50pp | −2.00pp |
| This run − Raw-User RL199 max60 | −1.38pp | +0.50pp | −1.00pp |
| This run − V1-User RL199 (raw evaluation) | +3.25pp | +6.00pp | −1.00pp |

The direct max120 continuation improves pass@1 and pass@4(any) over its iter99
checkpoint, but pass^4 decreases. Against the matched max60 iter199 run, it has
slightly higher pass@4(any) and lower pass@1/pass^4; therefore the max60 iter199
checkpoint remains the stronger default single-call RL model, while this run is
the useful max120 truncation/length diagnostic. Evaluation max-step terminations
are 140/800 here versus 73/800 for raw-User RL199 max60, so the longer training
rollout budget did not remove the held-out long-interaction failure mode.
