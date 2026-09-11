# tau2-agent-single-call-v1

Purpose: retrain the selected tau2 Agent SFT and its GRPO continuation through iter199
with exactly one native Agent tool call per Assistant message, while preserving
the source conversations, real tool results, task data, User model, reward, and
evaluation protocol.

## Data

`contract_boundary_19318.jsonl` has 19,318 rows and expands to 25,761
target-only single-call candidates. A multi-call target is serialized in source
order; each later target sees the earlier calls and their observed tool results.
Prefix batches use the same call/result serialization. Only converted targets
above 16,384 rendered tokens are removed, without removing their siblings.

Outputs:

- `data/single_call_v1_25708.jsonl`: 25,708 training rows after 53 over-cap
  airline targets are removed; retained rows are airline 13,298, retail 10,460,
  and telecom 1,950, with 14,588 tool-call and 11,120 text targets.
- `data/single_call_v1_longest32_smoke.jsonl`: the 32 longest retained rows.
  Their rendered lengths span 14,567 to 16,382 tokens.

The new rows use native Agent-only tool schemas, target-only `qwen3_full` loss,
the Agent/User ownership boundary, and minimal source metadata. They do not
contain protocol signatures or content hashes.

## Configuration

- SFT: raw `Qwen3-4B-Instruct-2507`, two epochs, batch 16, LR `1e-5`,
  16,384-token cap, independent checkpoint root ending in
  `tau2_agent_sft_single_call_v1_20260809`.
- RL: new SFT init, v1 User SFT as the rollout User, LR `2e-6`, K=8, K2 KL
  `0.01`, turn-credit-v1 and the existing field reward/penalties; quotas
  `3/2/1` for updates 0-9, `3/1/2` for 10-19, and `2/2/2` for 20-99. Stages
  resume the latest checkpoint in the same new root and use no authorization
  gate.
- Continuation: updates 100-199 resume iter99 in the same root with optimizer
  and RNG state, keep the `2/2/2` quota and all other RL settings unchanged,
  and extend the constant-LR scheduler horizon to 200.
- Eval: new SFT, iter99 RL, and iter199 RL only, seeds 300/301, all 100 three-domain test
  tasks, four trials, temperature `0.6`, `max_steps=200`, and the v1 User.

## Jobs

- SFT longest-32 smoke: job `pt-x810r135`,
  [`fsh-tau2-single-v1-sft-smoke-0809-151042`](jobs/fsh-tau2-single-v1-sft-smoke-0809-151042/run_20260809_151042.log)
  (Ray training succeeded through iter3; SCO reported failed only after the Ray
  job exited because the live outer shell file was edited during teardown;
  checkpoint artifacts removed after validation on 2026-08-10).
- Full two-epoch SFT: job `pt-9l22mtj6`,
  [`fsh-tau2-single-v1-sft-full-0809-151743`](jobs/fsh-tau2-single-v1-sft-full-0809-151743/run_20260809_151743.log)
  (succeeded; retained epoch boundary iter1605 and final iter3211).
- SFT iter3211 to `final_hf`: job `pt-no68os6t`,
  [`fsh-tau2-single-v1-sft-convert-0809-173628`](jobs/fsh-tau2-single-v1-sft-convert-0809-173628/run_20260809_173628.log)
  (succeeded).
- RL one-update smoke: job `pt-zir66gbb`,
  [`fsh-tau2-single-v1-rl-smoke-0809-175354`](jobs/fsh-tau2-single-v1-rl-smoke-0809-175354/run_20260809_175354.log)
  (succeeded; iter0).
- RL stage-a, updates 0-9: job `pt-1so44amm`,
  [`fsh-tau2-single-v1-rl-stage-a-0809-181504`](jobs/fsh-tau2-single-v1-rl-stage-a-0809-181504/run_20260809_181504.log)
  (succeeded; iter9).
- RL stage-b, updates 10-19: job `pt-8ar76pjb`,
  [`fsh-tau2-single-v1-rl-stage-b-0809-191553`](jobs/fsh-tau2-single-v1-rl-stage-b-0809-191553/run_20260809_191553.log)
  (succeeded; iter19).
- RL stage-final, updates 20-99: job `pt-1mgxh2za`,
  [`fsh-tau2-single-v1-rl-stage-final-0809-201231`](jobs/fsh-tau2-single-v1-rl-stage-final-0809-201231/run_20260809_201231.log)
  (succeeded; iter99).
- RL continuation, updates 100-199: job `pt-zyt04a72`,
  [`fsh-tau2-single-v1-rl-long200-0810-110007`](jobs/fsh-tau2-single-v1-rl-long200-0810-110007/run_20260810_110007.log)
  (training completed and iter199 saved; manually deleted during the stalled
  W&B log-upload cleanup on 2026-08-10).
- RL iter199 to HF: job `pt-nwckhnsl`,
  [`fsh-tau2-single-v1-rl199-convert-0810-153458`](jobs/fsh-tau2-single-v1-rl199-convert-0810-153458/run_20260810_153458.log)
  (succeeded).
- RL iter199 formal eval, seed300: job `pt-q2fto57u`,
  [`fsh-tau2-single-v1-eval-rl199-300-0810-154200`](jobs/fsh-tau2-single-v1-eval-rl199-300-0810-154200/run_20260810_154200.log)
  (succeeded in 2h42m; [summary](eval/rl199/seed300_summary.json)).
- RL iter199 formal eval, seed301: job `pt-bu0ahbqr`,
  [`fsh-tau2-single-v1-eval-rl199-301-0810-154200`](jobs/fsh-tau2-single-v1-eval-rl199-301-0810-154200/run_20260810_154200.log)
  (succeeded in 2h39m; [summary](eval/rl199/seed301_summary.json)).
- SFT formal eval, seed300: job `pt-mg2h9tfy`,
  [`fsh-tau2-single-v1-eval-sft-300-0809-192140`](jobs/fsh-tau2-single-v1-eval-sft-300-0809-192140/run_20260809_192140.log)
  (succeeded).
- SFT formal eval, seed301: job `pt-i4k53fvy`,
  [`fsh-tau2-single-v1-eval-sft-301-0809-192140`](jobs/fsh-tau2-single-v1-eval-sft-301-0809-192140/run_20260809_192140.log)
  (succeeded).
- RL iter99 to HF: job `pt-62z7f509`,
  [`fsh-tau2-single-v1-rl-convert-0810-002240`](jobs/fsh-tau2-single-v1-rl-convert-0810-002240/run_20260810_002240.log)
  (succeeded).
- RL iter99 formal eval, seed300: job `pt-2yoee8o4`,
  [`fsh-tau2-single-v1-eval-rl-300-0810-010601`](jobs/fsh-tau2-single-v1-eval-rl-300-0810-010601/run_20260810_010601.log)
  (succeeded).
- RL iter99 formal eval, seed301: job `pt-khjd6xio`,
  [`fsh-tau2-single-v1-eval-rl-301-0810-010601`](jobs/fsh-tau2-single-v1-eval-rl-301-0810-010601/run_20260810_010601.log)
  (succeeded).

## Results

The longest-sample SFT smoke completed three expected training updates and
saved iter3. Maximum mean batch length was 15,520.5 tokens; losses were
`0.49468`, `0.07543`, and `0.0000617`, with finite gradient norms and no OOM.
The outer SCO process failed after Ray had reported success because its live
shell file was edited during teardown; the corrected file passes `bash -n`.

Full SFT job `pt-9l22mtj6` completed 3,211 updates in 2h14m. Loss was `0.9094`
at step1 and `0.1711` at step3211; the final LR was `1.0000e-6`. All recorded
losses and gradient norms were finite, and iter3211 was saved successfully.

SFT iter3211 was converted to
`Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809/final_hf`
by job `pt-no68os6t`; the output contains two safetensors shards plus the base
tokenizer and model configuration. On 2026-08-10, 64 interval checkpoints and
the completed smoke checkpoint root were removed, reclaiming about 3.43 TiB;
`iter_0001605`, `iter_0003211`, and `final_hf` remain.

RL smoke job `pt-zir66gbb` passed the rollout preflight and completed update 0
in 11m43s. Its raw reward was `0.0625`, train loss `0.03387`, gradient norm
`0.45769`, and rollout truncation `0.25`; the checkpoint was saved under the
independent smoke root. Training remained finite and had no OOM.

RL stage-a job `pt-1so44amm` completed updates 0-9 in 35m58s and saved iter9.
Raw task reward ranged from `0.0417` to `0.3958`; train loss ranged from
`0.01544` to `0.03256`, gradient norm from `0.40547` to `0.50754`, and K2 loss
at iter9 was `0.000750`. All updates were finite and had no OOM.

RL stage-b job `pt-8ar76pjb` resumed iter9, completed updates 10-19 in 35m31s,
and saved iter19. Raw task reward ranged from `0.0625` to `0.3750`; train loss
ranged from `0.00618` to `0.02067`, gradient norm from `0.38158` to `0.54246`,
and K2 loss at iter19 was `0.003024`. All updates were finite and had no OOM.

The two SFT evaluations completed in 1h56m and 2h02m. Each cell below is
pass@1 / pass@4(any) / pass^4 in percent.

| Scope | Seed 300 | Seed 301 | Two-seed mean |
|---|---:|---:|---:|
| Overall | 27.25 / 50.00 / 10.00 | 24.50 / 49.00 / 7.00 | 25.88 / 49.50 / 8.50 |
| Airline | 36.25 / 55.00 / 25.00 | 31.25 / 45.00 / 20.00 | 33.75 / 50.00 / 22.50 |
| Retail | 28.75 / 55.00 / 7.50 | 26.25 / 52.50 / 7.50 | 27.50 / 53.75 / 7.50 |
| Telecom | 21.25 / 42.50 / 5.00 | 19.38 / 47.50 / 0.00 | 20.31 / 45.00 / 2.50 |

The primary comparable baseline is the selected Contract + boundary SFT,
because it also has seed300/301 results under a strict protocol. Historical
raw and multitool results are seed300 parser-on references, so they are not a
same-protocol causal comparison.

| Model | Evaluation scope | pass@1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| Raw Qwen3-4B-Instruct-2507 | Historical parser-on, seed300 | 23.50% | 45.00% | 6.00% |
| Old multitool SFT | Historical parser-on, seed300 | 23.25% | 46.00% | 8.00% |
| Raw Qwen3.5-4B non-thinking | Final external current-single, seed300/301 mean | 32.75% | 61.00% | 9.00% |
| Selected Contract + boundary SFT | Strict, seed300/301 mean | 23.63% | 43.50% | 8.50% |
| New single-call SFT | Strict-single-v1, seed300/301 mean | **25.88%** | **49.50%** | **8.50%** |
| New - selected boundary SFT | Same-seed point-estimate delta | **+2.25pp** | **+6.00pp** | **0.00pp** |

New single-call SFT pass@1 improved over the selected boundary SFT in every
domain: airline `+2.50pp`, retail `+1.88pp`, and telecom `+2.50pp`. Both seeds
improved pass@1 and pass@4(any); pass^4 changed `+2pp` at seed300 and `-2pp` at
seed301, leaving the mean unchanged. On the historical seed300 references, the
new seed300 result is `+3.75/+5.00/+4.00pp` over raw Instruct and
`+4.00/+4.00/+2.00pp` over the old multitool SFT, but these deltas remain
cross-protocol context rather than an isolated training-data effect.

The raw Qwen3.5 non-thinking row is the final two-seed external baseline from
`tau2-qwen35-nonthinking-eval`. It uses a different base model and the
`current-single` profile rather than the Agent-only `strict-single-v1` view, so
it is not a training control. Relative to that orientation point, the new SFT
is `-6.87/-11.50/-0.50pp`; RL iter99 is `-5.25/-7.50/+1.00pp`.

Overall action/DB accuracy was `66.90%/32.68%` for seed300 and
`62.93%/27.60%` for seed301; the two-seed aggregate was `64.91%/30.14%`.
There were 65 `max_steps` terminations among 800 simulations. Two trajectories
attempted multi-call output: three turns and six calls total; the strict parser
parsed and executed none of those batched calls.

RL stage-final job `pt-1mgxh2za` resumed iter19 and completed updates 20-99 in
3h27m, saving iter99. Across the stage, raw task reward ranged from `0.0000` to
`0.5208`, K2 loss from `0.00307` to `0.02671`, and gradient norm from `0.27721`
to `0.63842`. At iter99, train loss was `0.00709`, K2 loss `0.01782`, gradient
norm `0.43140`, raw reward `0.1458`, and truncation `0.1875`. All 80 updates
were finite and had no OOM.

RL iter99 was converted to HF by job `pt-62z7f509`. Its two formal evaluations
completed in 2h28m and 2h37m. Each cell below is pass@1 / pass@4(any) / pass^4
in percent.

| Scope | SFT mean | RL seed300 | RL seed301 | RL mean | RL - SFT mean |
|---|---:|---:|---:|---:|---:|
| Overall | 25.88 / 49.50 / 8.50 | 28.25 / 50.00 / 11.00 | 26.75 / 57.00 / 9.00 | **27.50 / 53.50 / 10.00** | **+1.63 / +4.00 / +1.50pp** |
| Airline | 33.75 / 50.00 / 22.50 | 30.00 / 40.00 / 20.00 | 32.50 / 50.00 / 25.00 | 31.25 / 45.00 / 22.50 | -2.50 / -5.00 / 0.00pp |
| Retail | 27.50 / 53.75 / 7.50 | 30.00 / 52.50 / 12.50 | 25.00 / 55.00 / 7.50 | 27.50 / 53.75 / 10.00 | 0.00 / 0.00 / +2.50pp |
| Telecom | 20.31 / 45.00 / 2.50 | 25.63 / 52.50 / 5.00 | 25.63 / 62.50 / 2.50 | **25.63 / 57.50 / 3.75** | **+5.31 / +12.50 / +1.25pp** |

The overall RL-minus-SFT paired task deltas were `+1.00/0.00/+1.00pp` for
seed300, with 95% intervals `[-3.50,+5.75]`, `[-9.00,+9.00]`, and
`[-4.00,+6.00]pp`; seed301 was `+2.25/+8.00/+2.00pp`, with intervals
`[-2.50,+7.00]`, `[-3.00,+19.00]`, and `[0.00,+5.00]pp`. Thus the overall
point estimates improve, but none of the per-seed intervals excludes zero.
Seed300 airline pass@1 regressed `-6.25pp`, interval `[-12.50,-1.25]pp`;
the two-seed airline mean regression was `-2.50pp`.

RL action/DB accuracy aggregated over the two seeds was `60.29%/26.59%`, down
`4.62/3.55pp` from SFT. `max_steps` was essentially flat (`64/800` RL versus
`65/800` SFT). RL produced one multi-call trajectory, one affected turn, and
two attempted calls across 800 simulations, versus `2/3/6` for SFT; the strict
parser executed none in either model. The overall gain is therefore modest and
concentrated in telecom, not a uniform action/DB improvement. Historical
boundary long100 RL scored `29.50/56.50/10.50%`, exceeding this RL by
`2.00/3.00/0.50pp`, but that remains a cross-profile reference.

Historical boundary iter199 scored `32.38/58.00/11.50%`, exceeding the
single-call iter99 by `4.88/4.50/1.50pp`. By domain, single-call iter99 pass@1
versus historical boundary iter99 was `-6.88/-3.75/+2.19pp` for
airline/retail/telecom; versus boundary iter199 it was
`-1.88/-4.69/-6.56pp`. The initial 0-99 plan is complete; updates 100-199 and
iter199, conversion, and both formal evaluations are complete.

The continuation preflight passed, loaded the distributed model, optimizer,
RNG, and sampler checkpoint at iter99, and began at update100. All 100 updates
were finite with LR `2e-6`, no OOM, and a complete iter199 checkpoint. Mean raw
reward was `29.81%` (`32.50%` in the final 10); mean truncation was `20.90%`
(`22.29%` final 10). K2 loss averaged `0.02643`, ranged from `0.01349` to
`0.07400`, and averaged `0.04252` in the final 10. Gradient norm averaged
`0.46001`, with a maximum of `0.68881`.

Historical boundary long200 over the same updates had raw reward
`29.98%` (`31.88%` final 10), truncation `18.98%` (`19.17%` final 10), K2
`0.07389` (`0.32374` final 10, maximum `0.64395`), and gradient norm `0.53812`
(`0.80856` final 10). The new single-call continuation therefore retains
similar training reward with modestly higher truncation but avoids the old
late-stage KL and gradient drift.

### Final iter199 evaluation

Both formal jobs succeeded with 400/400 simulations and zero infrastructure
errors. Each cell below is pass@1 / pass@4(any) / pass^4 in percent.

| Scope | Seed 300 | Seed 301 | Two-seed mean | iter199 - iter99 mean |
|---|---:|---:|---:|---:|
| Overall | 30.75 / 58.00 / 11.00 | 31.50 / 65.00 / 10.00 | **31.13 / 61.50 / 10.50** | **+3.63 / +8.00 / +0.50pp** |
| Airline | 33.75 / 60.00 / 15.00 | 26.25 / 40.00 / 10.00 | 30.00 / 50.00 / 12.50 | -1.25 / +5.00 / -10.00pp |
| Retail | 29.38 / 52.50 / 15.00 | 33.13 / 65.00 / 15.00 | 31.25 / 58.75 / 15.00 | +3.75 / +5.00 / +5.00pp |
| Telecom | 30.63 / 62.50 / 5.00 | 32.50 / 77.50 / 5.00 | **31.56 / 70.00 / 5.00** | **+5.94 / +12.50 / +1.25pp** |

Both seeds improved overall pass@1 and pass@4(any) over iter99. Relative to
the single-call SFT, the iter199 mean delta is
`+5.25/+12.00/+2.00pp`; relative to iter99 it is
`+3.63/+8.00/+0.50pp`. The two-seed paired task-bootstrap 95% intervals are:

| Comparator | pass@1 delta | pass@4(any) delta | pass^4 delta |
|---|---:|---:|---:|
| Single-call SFT | +5.25 `[+1.25,+9.25]`pp | +12.00 `[+4.50,+19.50]`pp | +2.00 `[-2.50,+6.50]`pp |
| Single-call RL iter99 | +3.63 `[+0.13,+7.25]`pp | +8.00 `[+0.50,+15.50]`pp | +0.50 `[-3.50,+4.50]`pp |

The pass@1 and pass@4(any) intervals exclude zero; pass^4 remains uncertain.
The gain is concentrated in retail and telecom, while airline pass^4 falls
10pp from iter99. Aggregate action/DB accuracy is `65.46/27.39%`, versus
`60.29/26.60%` at iter99. Terminations are `user_stop=723`,
`too_many_errors=2`, `max_steps=75`, and infrastructure error `0` over 800
simulations. There are zero raw or parsed multi-call turns and zero strict
single-call protocol errors.

For model-lineage orientation:

| Reference | pass@1 / pass@4(any) / pass^4 | iter199 - reference |
|---|---:|---:|
| Single-call SFT | 25.88 / 49.50 / 8.50% | +5.25 / +12.00 / +2.00pp |
| Single-call RL iter99 | 27.50 / 53.50 / 10.00% | +3.63 / +8.00 / +0.50pp |
| Historical boundary RL iter99 | 29.50 / 56.50 / 10.50% | +1.63 / +5.00 / 0.00pp |
| Historical boundary RL iter199 | 32.38 / 58.00 / 11.50% | -1.25 / +3.50 / -1.00pp |
| Raw Qwen3.5 non-thinking baseline | 32.75 / 61.00 / 9.00% | -1.63 / +0.50 / +1.50pp |

The historical boundary and Qwen3.5 rows use different protocol profiles, so
their deltas are orientation rather than causal comparisons. Within the
strict-single-v1 lineage, iter199 is the selected final RL checkpoint: it
improves both seeds and has a positive paired two-seed result for pass@1 and
pass@4(any). It does not uniformly dominate historical boundary iter199 or the
external Qwen3.5 baseline.

### User-model ablation

The table below consolidates every completed strict-single-v1 formal
evaluation in this lineage and appends the raw-Agent references. Every cell is
the seed300/301 mean over the same 100 test tasks and four trials per seed, with
Agent temperature `0.6` and evaluation `max_steps=200`.

- `V1` is
  `Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf`, the User
  simulator trained from the historical multi-tool trajectory data.
- `RAW` is the unmodified `models/Qwen3-4B-Instruct-2507`.
- Training User means the on-policy rollout User used by RL. Agent SFT is
  offline supervised training and therefore has no rollout User.

The cells below are pass@1 / pass@4(any) / pass^4 and every local User server
uses the Qwen tool-call parser. Each row uses identical seed coverage in both
User columns; `★` marks the better User cell when all three displayed metrics
are higher. Rows follow raw Qwen3, single-call SFT, iter99 variants, iter199
variants, then external Qwen3.5 baselines. Raw-Agent rows remain external
references because their Agent protocol is not `strict-single-v1`.

| Agent checkpoint | Training rollout User | Training max_steps | Comparison seeds | Evaluation User = V1, parser ON | Evaluation User = RAW, parser ON |
|---|---|---:|---:|---:|---:|
| [Raw Qwen3-4B-Instruct-2507](../tau2-raw-agent-raw-user-parser-on/README.md) | N/A (raw Agent) | N/A | 300/301 mean | 23.88 / 45.50 / 6.50% ★ | 17.38 / 35.50 / 2.50% |
| Single-call SFT final | N/A (offline SFT) | N/A | 300/301 mean | 25.88 / 49.50 / 8.50% | 28.00 / 51.00 / 9.00% ★ |
| V1-RL iter99 | V1 | 60 | 300/301 mean | 27.50 / 53.50 / 10.00% ★ | 25.25 / 50.50 / 5.50% |
| RAW-RL iter99 | RAW | 60 | 300/301 mean | 24.25 / 49.50 / 6.00% | 29.25 / 54.00 / 6.50% ★ |
| RAW-RL iter99 max120 | RAW | 120 | 300/301 mean | 25.25 / 51.00 / 6.00% | 28.38 / 56.50 / 9.50% ★ |
| V1-RL iter199 | V1 | 60 | 300/301 mean | 31.13 / 61.50 / 10.50% ★ | 26.88 / 52.50 / 8.00% |
| RAW-RL iter199 | RAW | 60 | 300/301 mean | 25.00 / 53.00 / 6.00% | 31.50 / 58.00 / 8.00% ★ |
| RAW-RL iter199 max120 | RAW | 120 | 300/301 mean | 25.25 / 49.00 / 5.50% | 30.13 / 58.50 / 7.00% ★ |
| [Raw Qwen3.5-4B thinking-on](../tau2-raw-agent-raw-user-parser-on/README.md) | N/A (raw Agent) | N/A | 300/301 mean | 48.50 / 76.00 / 22.00% | 53.25 / 78.00 / 25.00% ★ |
| [Raw Qwen3.5-4B non-thinking](../tau2-raw-agent-raw-user-parser-on/README.md) | N/A (raw Agent) | N/A | 300/301 mean | 32.75 / 61.00 / 9.00% ★ | 25.63 / 53.50 / 7.00% |

All RL rows start from the same single-call SFT. Jobs `1284`–`1288` and
`1302`–`1306` completed the five formerly missing crossed cells at both seeds;
every seed summary has 400/400 simulations and zero infrastructure errors. In
all five two-seed RL comparisons, the evaluation User matching the rollout
User is higher on all three metrics. V1-RL99 loses `2.25/3.00/4.50pp` with RAW
instead of V1. With V1 instead of RAW, the RAW-RL max60/max120 Agents lose
`5.00/4.50/0.50pp` and `3.13/5.50/3.50pp` at iter99, then
`6.50/5.00/2.00pp` and `4.88/9.50/1.50pp` at iter199. This is consistent with
rollout/evaluation User-distribution specialization, not a User-independent
ranking of the Agents.

The raw-Agent rows are parser-ON on both sides. Raw Qwen3 scores
`23.88/45.50/6.50%` with V1 and `17.38/35.50/2.50%` with RAW. Qwen3.5
thinking-on instead favors RAW (`53.25/78.00/25.00%`) over V1
(`48.50/76.00/22.00%`), while non-thinking favors V1
(`32.75/61.00/9.00%`) over RAW (`25.63/53.50/7.00%`). The separately retained
seed300 parser-OFF RAW results are Qwen3 `18.00/35.00/4.00%` and Qwen3.5
thinking-on `38.25/54.00/19.00%`; they are diagnostic references, not cells in
the parser-ON matrix. Historical `pass@1` is normalized to the all-trial
success fraction (official `pass^1`).

Each delta below is pass@1 / pass@4(any) / pass^4. Only the named factor is
changed within each row.

| Fixed evaluation User | Comparison | Changed factor | Delta |
|---|---|---|---:|
| V1 | V1-RL iter199 - V1-RL iter99 | 100 additional RL updates | +3.63 / +8.00 / +0.50pp |
| RAW | RAW-RL iter199 - RAW-RL iter99 | 100 additional RL updates | +2.25 / +4.00 / +1.50pp |
| RAW | RAW-RL iter199 max120 - RAW-RL iter99 max120 | 100 additional RL updates | +1.75 / +2.00 / -2.50pp |
| RAW | RAW-RL iter199 max120 - RAW-RL iter199 max60 | Training rollout max_steps | -1.38 / +0.50 / -1.00pp |
| RAW | RAW-RL iter199 - V1-RL iter199 | Training rollout User | +4.62 / +5.50 / 0.00pp |
| V1 | RAW-RL iter199 - V1-RL iter199 | Training rollout User | -6.13 / -8.50 / -4.50pp |
| RAW | RAW-RL iter99 max120 - max60 | Training rollout max_steps | -0.88 / +2.50 / +3.00pp |
| Same SFT Agent | RAW evaluation - V1 evaluation | Evaluation User | +2.13 / +1.50 / +0.50pp |
| Same V1-RL iter199 Agent | RAW evaluation - V1 evaluation | Evaluation User | -4.25 / -9.00 / -2.50pp |

Changing the evaluation User can reverse the apparent RL gain, so V1-evaluated
and RAW-evaluated rows are not direct capability comparisons. Under the same
RAW evaluation, RAW-RL iter199 exceeds V1-RL iter199 by `+4.62pp` pass@1; its
paired 95% interval is `[+0.75,+8.62]pp`, supporting a training-User effect in
the RAW-User distribution. The reverse two-seed comparison is also complete:
under V1 evaluation, RAW-RL iter199 trails V1-RL iter199 by
`6.13/8.50/4.50pp`. Together, the two directions support distribution matching
more strongly than a User-independent superiority claim. At iter99, max120 did
not improve pass@1 over max60; after the
additional 100 updates, max120 reaches `30.13/58.50/7.00%`, still below the
max60 iter199 result on pass@1 and pass^4. Its training truncation is much
lower, but its evaluation max-step terminations are higher (`140/800` versus
`73/800`), so the longer rollout budget is a diagnostic rather than a reason to
replace the max60 iter199 default.

Evidence is in the
[User ablation README](../tau2-agent-single-call-v1-user-ablation/README.md),
[raw-User RL100 README](../tau2-agent-single-call-v1-raw-user-rl100/README.md),
[raw-User RL200 README](../tau2-agent-single-call-v1-raw-user-rl200/README.md),
and [max_steps120 README](../tau2-agent-single-call-v1-raw-user-maxsteps120-rl100/README.md),
[max_steps120 continuation README](../tau2-agent-single-call-v1-raw-user-maxsteps120-rl200/README.md).

## Validation

Targeted CPU tests cover source-order expansion, prefix serialization,
per-target length filtering, one-call Assistant turns, strict parser rejection,
unsigned metadata, and preservation of the older profiles. The rollout
preflight, Python/shell syntax checks, whitespace checks, and submission
dry-runs were run before the first GPU job. Final validation rechecked all
25,708 rows, the completed evaluation summaries, zero infrastructure errors,
zero parsed batched calls, Python/shell syntax, and documentation whitespace.
