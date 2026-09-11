# tau2-agent-single-call-v1-raw-user-rl100

Purpose: test whether rollout User SFT affects strict-single Agent RL by
repeating the selected 100-update single-call GRPO recipe with the unmodified
`Qwen3-4B-Instruct-2507` as the rollout User.

Name: tau2-agent-single-call-v1-raw-user-rl100.

## Design

The run starts from the same
`Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809/final_hf`
Agent SFT and preserves the existing stage quotas, LR `2e-6`, K=8, K2 KL
`0.01`, turn-credit-v1, reward shaping, penalties, strict-single-v1 protocol,
optimizer settings, and seeds. The only experimental change is the rollout
User:

- control: `Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf`
- experiment: unmodified `models/Qwen3-4B-Instruct-2507`

The control single-call RL iter99/iter199 lineage was trained with the v1 User
SFT model above. This run changes the training-time rollout User, unlike the
separate raw-User evaluations that only change the evaluation User.

Updates 0-9, 10-19, and 20-99 retain domain quotas `3/2/1`, `3/1/2`, and
`2/2/2` for Telecom/Airline/Retail ordering used by the original runner.
Checkpoints are isolated under
`Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_20260812`.

## Tasks

- `pt-dusx8lrm` / Raw-User RL updates 0-99 —
  [run log](jobs/fsh-tau2-single-v1-raw-user-rl100-0812-111305/run_20260812_111305.log)
  (succeeded; iter99).
- `pt-eh105h5v` / Iter99 to Hugging Face conversion —
  [run log](jobs/fsh-tau2-single-v1-raw-user-rl99-convert-0812-164141/run_20260812_164141.log)
  (succeeded; two complete safetensors shards in `iter_0000099_hf`).
- `pt-0jtzpvlu` / Raw-User formal evaluation, seed 300 —
  [run log](jobs/fsh-tau2-single-v1-raw-user-trained-rl99-eval-s300-0812-164521/run_20260812_164521.log)
  (succeeded; 400/400 simulations; [summary](eval/raw-user/seed300_summary.json)).
- `pt-pauw0icq` / Raw-User formal evaluation, seed 301 —
  [run log](jobs/fsh-tau2-single-v1-raw-user-trained-rl99-eval-s301-0812-164521/run_20260812_164521.log)
  (succeeded; 400/400 simulations; [summary](eval/raw-user/seed301_summary.json)).

## Results

Job `pt-dusx8lrm` succeeded in 4h42m37s. Its rollout and train records cover
every update 0-99 exactly once, with no gaps or duplicates. Stage-a, stage-b,
and stage-final loaded their expected resume points and saved iter9, iter19,
and iter99. All 100 perf records met the configured per-stage domain quotas;
16 filtered groups were replaced one-for-one.

| Window | Updates | Raw reward mean/range | Truncation mean | K2 mean/max | Grad norm mean/max |
|---|---:|---:|---:|---:|---:|
| Stage-a | 0-9 | 14.79% / 2.08-39.58% | 27.08% | 0.00037 / 0.00058 | 0.50982 / 0.61193 |
| Stage-b | 10-19 | 25.42% / 6.25-35.42% | 23.33% | 0.00103 / 0.00139 | 0.46673 / 0.52344 |
| Stage-final | 20-99 | 27.21% / 2.08-60.42% | 16.67% | 0.00717 / 0.02021 | 0.46825 / 0.74230 |
| Final 10 | 90-99 | 33.13% / 18.75-60.42% | 17.92% | 0.01333 / 0.02021 | 0.46325 / 0.53476 |
| All | 0-99 | **25.79%** / 2.08-60.42% | **18.38%** | 0.00587 / 0.02021 | 0.47225 / 0.74230 |

At step99, raw reward was `45.83%`, truncation `18.75%`, train loss
`0.01073`, K2 loss `0.01619`, and gradient norm `0.53476`. All 100 updates
were finite with no OOM. The six tracebacks in the combined log are the same
non-fatal W&B atexit `ConnectionResetError` pair after each successful stage;
W&B synced, Ray succeeded, and SCO reported `SUCCEEDED`.

`iter_0000099` contains all 15 expected distributed-checkpoint files totaling
56,325,827,598 bytes (52.46 GiB). The latest-iteration marker is `99`, and the
matching rollout sampler state is present.

Compared with the original same-recipe single-call RL100 trained with v1 User
SFT, the training diagnostics are:

| Training User | Raw reward all/final10 | Truncation all/final10 | K2 all/final10/max | Grad norm mean/max |
|---|---:|---:|---:|---:|
| v1 User SFT | 24.67% / 31.88% | 22.92% / 18.96% | 0.00871 / 0.01645 / 0.02671 | 0.44993 / 0.63842 |
| Raw User | **25.79% / 33.13%** | **18.38% / 17.92%** | **0.00587 / 0.01333 / 0.02021** | 0.47225 / 0.74230 |
| Raw - v1 | +1.13 / +1.25pp | -4.54 / -1.04pp | -0.00284 / -0.00312 / -0.00650 | +0.02233 / +0.10388 |

Raw-User training is therefore numerically healthy, with slightly higher
training reward, lower truncation, and lower K2 than the v1-User control. The
higher peak gradient norm remained finite.

Both held-out evaluations then succeeded under the raw User, with 800/800
simulations, zero infrastructure errors, and zero strict-single Agent protocol
errors. Each cell below is pass@1 / pass@4(any) / pass^4 in percent.

| Scope | Raw-User SFT | Raw-User-trained RL iter99 | RL - SFT |
|---|---:|---:|---:|
| Overall | 28.00 / 51.00 / 9.00 | **29.25 / 54.00 / 6.50** | **+1.25 / +3.00 / -2.50pp** |
| Airline | 42.50 / 57.50 / 30.00 | 41.25 / 62.50 / 20.00 | -1.25 / +5.00 / -10.00pp |
| Retail | 35.62 / 61.25 / 7.50 | 33.13 / 57.50 / 6.25 | -2.50 / -3.75 / -1.25pp |
| Telecom | 13.12 / 37.50 / 0.00 | **19.38 / 46.25 / 0.00** | **+6.25 / +8.75 / 0.00pp** |

| Seed | pass@1 | pass@4(any) | pass^4 |
|---:|---:|---:|---:|
| 300 | 29.25% | 53.00% | 6.00% |
| 301 | 29.25% | 55.00% | 7.00% |
| Mean | **29.25%** | **54.00%** | **6.50%** |

The paired two-seed task-bootstrap 95% intervals for overall RL-minus-SFT are
`[-2.88,+5.25]pp`, `[-4.00,+10.00]pp`, and `[-6.00,+0.50]pp`; all include
zero. Telecom pass@1 is the only positive domain result whose interval excludes
zero: `+6.25pp [+0.63,+12.50]`. The overall point gain is therefore not an
established improvement, and four-trial consistency regresses.

Action accuracy rises from `63.54%` to `66.57%`, but DB accuracy is unchanged
at `30.69%` versus `30.62%`. `too_many_errors` falls from 20 to 11, while
`max_steps` rises from 73 to 77 over 800 simulations. Raw-User training mainly
improves Telecom action behavior; it does not yet convert that gain into a
uniform final-state or consistency improvement.

For orientation, the original v1-User-trained/evaluated iter99 scored
`27.50/53.50/10.00%` versus its SFT control at `25.88/49.50/8.50%`, a
`+1.63/+4.00/+1.50pp` delta. The raw-User pair above gives
`+1.25/+3.00/-2.50pp`, so replacing the User has not shown a stronger RL effect.
The final crossed evaluations now isolate the training-time User at iter99.
Under RAW evaluation, RAW-RL exceeds V1-RL by `+4.00/+3.50/+1.00pp`; under V1
evaluation, it trails by `3.25/4.00/4.00pp`. The later iter199 comparison shows
the same reversal: RAW-RL exceeds V1-RL by `+4.62/+5.50/0.00pp` under RAW and
trails by `6.13/8.50/4.50pp` under V1. This supports rollout/evaluation
User-distribution matching rather than a User-independent Agent ranking. The
complete matrix is in the
[single-call main README](../tau2-agent-single-call-v1/README.md#user-model-ablation).
