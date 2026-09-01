# tau2-bench RL

GRPO training for the tau2-bench agent SFT checkpoint. This path uses slime's
default sglang rollout loop with a tau2-specific custom generate function.

## Data

Prepare the AReaL RL task file into slime's metadata format:

```bash
PYTHONPATH=/mnt/afs/users/fush/projects/ServiceAgent/tau2-bench/src:$PYTHONPATH \
python3 examples/tau2-bench/rl/prepare_rl_data.py
```

The prepared rows use `prompt=id` and store the full tau2 `Task`, `domain`, and
absolute `db_path` in `metadata`.

## Commands

Four-card smoke (short trajectories; fast pipeline check):

```bash
bash scripts/submit.sh --experiment tau2-rl --gpus 4 \
  examples/tau2-bench/rl/smoke_qwen3_4b_instruct_2507_tau2_rl.sh
```

Eight-card verification run (full data, few rollouts; reproduces the long-tail
train step that used to OOM):

```bash
bash scripts/submit.sh --experiment tau2-rl --gpus 8 \
  examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_verify_8g.sh
```

Eight-card full training run:

```bash
bash scripts/submit.sh --experiment tau2-rl --gpus 8 \
  examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh
```

Gated local-first dependency-safe run (100 updates first; 200 is allowed only
after `ITER99_PROMOTION.json` passes):

```bash
bash scripts/submit.sh \
  --experiment tau2-rl-local-first-dependency-safe-k2-fieldreward \
  --gpus 8 --priority normal \
  examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_local_first_dependency_safe_k2_fieldreward.sh
```

Independent strict-single-v1 stages (no promotion gate; each full stage resumes
the latest checkpoint in its own root):

```bash
bash examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_single_call_v1.sh smoke
bash examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_single_call_v1.sh stage-a
bash examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_single_call_v1.sh stage-b
bash examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_single_call_v1.sh stage-final
```

The scripts intentionally reject non-4-card and non-8-card jobs. By default the
local trained user simulator runs on the last visible GPU. Set `USER_SGLANG=0`
and provide `TAU2_USER_API_BASE` / `TAU2_USER_API_KEY` to use an external user
simulator and let Ray use all cards.

## Defaults

- Generic/legacy wrapper Agent init:
  `checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714`
  with tokenizer/HF path `.../iter_0002413_hf`. Boundary-v2 launchers override
  this with the selected signed Contract + boundary SFT.
- User simulator: `checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf`.
- Reward: tau2 official binary task reward plus field-level shaping for tool
  name, each compared argument, final DB assertion, env assertions, and
  communication checks. Wrong compared arguments, failed Agent-tool
  executions, malformed JSON, nonexistent tools, repeated calls, and
  `max_steps` receive explicit penalties.
- Objective: trajectory-level GRPO; no TIS.
- On-policy training context: the FULL rolled-out conversation is tokenized
  verbatim (no leading-message truncation). Over-long trajectories are
  re-sampled per sample (the rollout retries up to `TAU2_RL_MAX_ROLLOUT_RETRIES`
  times); only if a trajectory still exceeds `TAU2_RL_MAX_TRAIN_TOKENS`
  (default 16384) after retries is its group dropped by
  `filters.drop_zero_std_or_unsampleable` (which also drops zero-shaped-reward-
  std groups). Dynamic sampling repeatedly draws replacements until the batch
  contains the requested number of informative groups; rejected groups never
  enter training.
- Preflight: `test_rollout_logic.py` runs in the container before any GPU work
  and verifies on-policy tokenization, the loss mask, the over-long flag/neutral
  path, the filter, and the active Agent protocol prompt/metadata. Bypass with
  `SKIP_PREFLIGHT=1`.
- Agent protocol: `TAU2_AGENT_PROTOCOL_PROFILE=current-single` by default for
  compatibility. The local-first experiment fixes it to
  `dependency-safe-multi`; the shared profile reaches Ray workers and Agent
  `llm_args`, amends domain policies, and is recorded in trajectory metadata.
- Boundary-v2 fixes the profile to `agent-owned-dependency-safe-multi`, uses
  native Agent-only schemas with `qwen3_full`, and classifies User-owned calls
  separately from nonexistent tools. Any namespace attempt receives the hard
  penalty recorded in `tau2_field_reward_signals`.
- `strict-single-v1` uses the same native Agent-only schemas and ownership
  boundary, permits another call immediately after a tool result, and treats
  multiple tool-call blocks in one model output as a protocol error. Its
  trajectories omit protocol signatures and contract hashes.
- `filters.DomainQuotaDataSource` enforces an exact six-prompt domain quota per
  update. Rejected zero-std or over-cap groups can only draw same-domain
  replacements; domain cursors are checkpointed for deterministic resume.

## Memory and length handling

The previous runs OOM'd in the log-prob forward. Root cause: slime's bin packer
caps each microbatch at `max_tokens_per_gpu`, but a *single* sample longer than
that cap is placed alone in its own oversized microbatch
(`slime/utils/seqlen_balancing.py`); the log-prob forward then materialises a
full `[T, V]` logits tensor for it. The tau2 long tail (multi-turn tool results)
reaches tens of thousands of tokens per trajectory, so those singleton bins OOM.

Two things do **not** fix this: lowering `max_tokens_per_gpu` (a single over-cap
sample still escapes it) and `--log-probs-chunk-size` (it only chunks the
softmax reduction *after* the model already produced the full logits).

The fix keeps every kept trajectory fully on-policy and untruncated, and wastes
as little data as possible:

- `max_tokens_per_gpu == TAU2_RL_MAX_TRAIN_TOKENS`, so no kept sample can form an
  oversized bin.
- Over-long trajectories are **re-sampled per sample** inside the rollout
  (`_run_tau2_rollout_sync` re-runs the orchestrator under temperature sampling,
  up to `TAU2_RL_MAX_ROLLOUT_RETRIES` extra attempts). The other trajectories in
  the GRPO group are untouched — only the one bad sample is redrawn. This
  preserves on-policy data; group-drop would discard 7 good trajectories.
- Only if a trajectory is **still** over the cap after all retries
  (`permanently_too_long`) does the group-level filter
  (`filters.drop_zero_std_or_unsampleable`) drop the whole group and draw a
  different prompt. As a memory-safe backstop, an unsampleable sample is also
  neutralized (1-token placeholder, zero loss mask, `remove_sample`), so even
  with the filter off the `[T, V]` logits cannot OOM and it contributes no
  gradient (slime's masked-whitening ignores zero-mask tokens). Group-level drop
  is required there because slime's GRPO metric path assumes a fixed
  `n_samples_per_prompt`.

At TP=2, a 16384-token microbatch costs ~10 GiB of fp32 logits (+ conversion
transient) plus ~32 GiB optimizer/weights — comfortably inside 80 GB. The cap
keeps ~99% of tau2 trajectories; the ~1% tail is re-sampled, and only prompts
that repeatedly fail are dropped.

## Real 8g Hyperparameters

The full run uses `run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh`.

- Data: full AReaL RL split, 1,982 tasks: airline 1,148, retail 563, telecom 271.
- Steps: `NUM_ROLLOUT=330`, approximately one pass of 1,982 prompt groups at
  `ROLLOUT_BATCH_SIZE=6`.
- Batch/grouping: `ROLLOUT_BATCH_SIZE=6`, `N_SAMPLES_PER_PROMPT=8`,
  `GLOBAL_BATCH_SIZE=48`; with 6 actor GPUs and TP=2, DP=3 requires the global
  batch to be divisible by 3.
- Sampling: agent `ROLLOUT_TEMPERATURE=1.0`, `top_p=1.0`,
  `AGENT_MAX_TOKENS=1024`; user simulator `temperature=0.0`,
  `max_tokens=512`. `TAU2_MAX_STEPS=60`.
- Memory: `TAU2_RL_MAX_TRAIN_TOKENS=16384` (also `max_tokens_per_gpu`),
  `TAU2_RL_MAX_ROLLOUT_RETRIES=2`, `LOG_PROBS_CHUNK_SIZE=1024`,
  `SGLANG_SERVER_CONCURRENCY=8`.
- Reward: **field-level shaping on**
  (`reward_postprocess.tau2_reward_post_process`):
  `shaped = task_reward + 0.25 * (partial_score - penalty)`. Default credit
  weights are tool name `0.25`, compared argument fields `0.35`, final DB
  assertion `0.25`, env assertions `0.10`, and communication `0.05`. Default
  per-event penalties are wrong compared argument field `0.10`, failed
  Agent-tool execution `0.25`, malformed JSON `0.15`, nonexistent tool `0.15`,
  repeated identical call `0.05`, and `max_steps` `0.20` (with caps for repeated
  events). Disable with `USE_REWARD_SHAPING=0`.
- Algorithm: GRPO, `kl_coef=0`, `kl_loss_coef=0.01`, `kl_loss_type=k2`,
  `entropy_coef=0`; dynamic filtering drops permanently-too-long and
  shaped-zero-std groups.
- Optimizer stability scan: Adam at `2e-6`, `3e-6`, and `5e-6`, constant
  schedule, weight decay `0.1`.
- Checkpoints: save every 33 rollout/update steps.
- W&B: enabled by default in the real wrapper; key is supplied from the job
  environment and is not passed on the command line.

Rationale: the AReaL tau2 paper uses SFT user simulators, trajectory-level GRPO,
large effective batch sizes, dynamic filtering, no LR decay, max context 32k,
and temperature 1.0 for both agent and user. On one 8-card node we cannot match
their 256-512 effective batch (64-80 H200); we keep trajectory GRPO + dynamic
filtering + no LR decay + KL=0 (the algorithm), use the verified local user
simulator, and match our eval temperature (`0.65`). The only deviations from
paper-scale are batch size (48 vs 256-512) and a 16384 context cap (vs 32k) that
drops rather than truncates — normal single-node RL compromises, not a length-
truncation hack.

## Boundary-v2 long100 outcome

The current selected lineage is the three-domain turn-aware boundary-v2 path,
not the generic legacy init above. Its first Pilot A reached a healthy iter9 and
improved namespace/capability, but the historical safety gate stopped on
malformed, execution-error, framework-truncation, and `max_steps` rates. Those
diagnostics did not indicate OOM, NaN/Inf, bad resume state, quota drift, or
Contract/turn-credit misalignment, so they could not answer whether the stable
`2e-6` recipe was simply under-trained.

The follow-up restarted from
`Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_boundary_20260804/final_hf`
with a fresh optimizer/root and held LR `2e-6`, K2 KL `0.01`, K=8,
turn-credit-v1, `max_steps=60`, and the v1 User fixed. Health-only gates at
iter9/19 audited complete finite metrics, KL, exact domain quota/replacement,
checkpoint resume, signed Contract, response spans, loss masks, and penalty
vectors. All gates passed and training completed at iter99.

| Same signed protocol, two-seed mean | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| selected Contract + boundary SFT | 23.63% | 43.50% | 8.50% |
| historical boundary RL iter9 | 25.25% | 50.00% | 9.00% |
| **boundary RL long100 iter99** | **29.50%** | **56.50%** | **10.50%** |

The final classification is **`win`**: both seeds strictly improve pass@1 and
pass@4(any) over both controls, every domain's mean pass@1 improves, and all
health/protocol/namespace/non-inferiority checks pass. Paired uncertainty is
wider on seed 300 and all overall pass^4 CIs cross zero, so not every delta is
independently significant. The selected HF checkpoint is
`Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804/iter_0000099_hf`.
See the [experiment README](../../../output/experiments/tau2-rl-agent-user-boundary-v2-long100/README.md)
and [cross-experiment QA](../../../output/doc/EXP_QA.md).

## Boundary-v2 curve and long200 continuation

The follow-up leaves every long100 checkpoint and gate immutable. Four 2-GPU
jobs independently convert and evaluate the same-lineage iter9/39/69/89
checkpoints at seed 300; the conversions live under a separate curve HF root.
The final curve combines those points with the existing iter99 result and
reports paired task-bootstrap intervals versus selected SFT and each adjacent
same-lineage checkpoint. Historical iter9 and raw Instruct are background
controls, and the single-seed curve never selects a model automatically.

All four curve jobs completed. Same-lineage seed-300 pass@1 over
iter9/39/69/89/99 is `25.00/26.25/24.00/29.75/27.50%`, while pass@4(any) is
`52/51/47/53/53%`: the trajectory is non-monotonic and iter89 is the diagnostic
single-seed peak. The iter69→89 pass@1 paired CI is `[+0.25,+11.25]pp`; the
iter89→99 overall CI crosses zero, though telecom falls `-6.88pp` with CI
`[-13.12,-0.62]pp`. This does not override the two-seed iter99 selection.

`long200-final` is one 8-GPU segment covering updates 100-199. `LOAD_DIR`
initially points at the exact passing long100 iter99 root while `SAVE_DIR` points
at the new long200 root; after iter109 exists, retries resume only from the most
recent complete long200 ten-step checkpoint. The lineage marker records both
roots and the SHA256 of the immutable source health gate. Scheduler-horizon
override preserves the restored optimizer state; LR `2e-6`, K2 KL `0.01`, K=8,
quota `2/2/2`, reward/penalties, `max_steps=60`, v1 User, Contract, and
turn-credit-v1 stay fixed.

Use `finalize_agent_boundary_v2_rl_long200_health.sh partial ...` roughly hourly.
Only OOM, non-finite/KL, quota, checkpoint/resume, Contract, or turn-credit
failures are hard stops; behavior and `max_steps` are diagnostics. Once strict
iter199 health passes, convert/evaluate seeds 300/301 and run
`finalize_agent_boundary_v2_rl_long200.sh`. Only a `win` decision replaces
iter99; `usable` and `reject` retain iter99, and no 200-299 run is automatic.
See the [curve experiment](../../../output/experiments/tau2-rl-agent-user-boundary-v2-checkpoint-curve/README.md)
and [long200 experiment](../../../output/experiments/tau2-rl-agent-user-boundary-v2-long200/README.md).

The run recovered the exact long100 state and saved complete checkpoints through
iter169, but the predeclared KL stop fired before iter199. Updates 167–176 had K2
KL `0.070/0.041/0.060/0.096/0.151/0.134/0.175/0.152/0.289/0.260`; the last-10
mean was `0.14287` and updates 175–176 were both at least `0.20`. All finite,
quota, checkpoint, Contract, and turn-credit checks otherwise passed. The job
was stopped after update176, so iter199 conversion/evaluation was skipped. The
strict early-health decision is **`reject`** and keeps long100 iter99 selected;
200–299 remains unauthorized. Use
`finalize_agent_boundary_v2_rl_long200.sh early-stop FAILED_PARTIAL_HEALTH_GATE`
to reproduce this terminal decision path.

An explicit user-requested diagnostic exception uses
`long200-final-kl-waiver`. It resumes from the same latest complete checkpoint
and waives only `last_10_k2_kl` and `no_consecutive_high_kl`; every other health
check remains enforced. Use `final-kl-waiver` with the long200 health finalizer,
then the `long200-kl-waiver` conversion/eval variant and
`finalize_agent_boundary_v2_rl_long200_kl_waiver.sh`. These artifacts live under
separate Gate, authorization, HF, and eval paths and never overwrite the strict
`reject` decision or automatically replace iter99.

That diagnostic path completed. It loaded exact iter169 model/optimizer/RNG and
domain-sampler state, redid only unsaved updates 170–176, and reached a complete
iter199. Training raw binary success over updates 100–199 was `29.98%`; no
non-finite, OOM, quota, Contract, or turn-credit check failed, while the waived
last-10 KL mean reached `0.32374`. Full seed300/301 evaluation gave iter199
pass@1/pass@4(any)/pass^4 means of `32.38/58.00/11.50%`, versus iter99
`29.50/56.50/10.50%`. The point gain came from telecom (`+8.75pp` mean pass@1);
airline fell `−5.00pp` on average and `−7.50pp` at seed301, and overall paired
CIs versus iter99 crossed zero. Namespace probes passed with zero attempts, but
official-eval raw namespace attempts increased from `8/13` to `12/19` across
seeds 300/301. The diagnostic result therefore keeps long100 iter99 selected
and does not authorize updates 200–299. See
[`ITER199_KL_WAIVER_RESULT.json`](../../../output/experiments/tau2-rl-agent-user-boundary-v2-long200/ITER199_KL_WAIVER_RESULT.json).

## Boundary-v2 domain experts

`domain-expert <airline|retail|telecom>` forks the immutable selected long100
iter99 state into three isolated roots and trains updates 100-129 with quotas
`6/0/0`, `0/6/0`, or `0/0/6`. It keeps the full three-domain prompt file and
restores optimizer, RNG, dataset fingerprint, epoch, and all sampler cursors;
only quota and the scheduler horizon change. Before iter109 a retry reloads
iter99, while later retries require this expert's complete iter109 or iter119.

Checkpoint-scoped expert and mixed-control Gates authorize iter109/119/129.
They bind the exact roots, source Gate SHA, domain/quota, fingerprint, restored
cursors, successful save event, rollout state, metadata, optimizer, and distcp
shards. KL is diagnostic (`enforced=false`); OOM, non-finite state, quota,
checkpoint/recovery, Contract, and turn-credit checks remain hard failures.

Conversion and evaluation use explicit variants:

```bash
bash scripts/convert_tau2_agent_rl_boundary_v2_iter_to_hf.sh domain-expert airline 109
bash scripts/convert_tau2_agent_rl_boundary_v2_iter_to_hf.sh mixed-control 109
bash examples/tau2-bench/eval/official/models/run_full_tau2_agent_rl_boundary_v2_user_stop_parser.sh domain-expert airline 109 300 probe
```

Seed300 selects one checkpoint per expert by pass@1, pass@4(any), pass^4, then
earliest iteration. Seed301 confirms only selected checkpoints and their
same-iteration mixed controls; selected experts also run seed300 forgetting
diagnostics. `advantage` and `near-parity` are OPD-eligible within the 5pp
per-seed bounds, while `reject` is not. No OPD job starts automatically. See the
[experiment README](../../../output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts/README.md).

The completed run selected Airline iter129, Retail iter119, and Telecom iter129.
All 12 checkpoint Gates, conversions, protocol checks, task/trial coverage, and
namespace checks passed. The final two-seed decisions are Airline `reject`,
Retail `advantage`, and Telecom `advantage`; only Retail and Telecom are
OPD-eligible. Airline breached the 5pp bound at seed300 pass@4(any) (`-10pp`)
and seed301 pass@1 (`-8.75pp`). Retail stayed within the bound with mean pass@1
advantage `+0.94pp`; Telecom stayed within it with mean pass@1/pass@4(any)
advantages `+0.31/+1.25pp`. These are eligibility decisions, not claims that
the paired CIs exclude zero, and no distillation was launched.

One completed forgetting eval exposed an unrelated teardown issue: both result
files and the summary were finalized, but sglang did not exit after TERM. The
official eval runner now waits at most 30 seconds per server before SIGKILL, so
successful evaluations cannot hold GPUs indefinitely during cleanup.

This experiment is frozen at the reported checkpoints; no longer domain-expert
continuation is scheduled. Airline failed the two-seed gate despite a rising
seed300 curve, Retail pass@1 peaked at iter119, and Telecom's higher truncation
and KL coincided with pass^4 falling to zero. Any future longer-training or
domain-heavy replay study must use a new experiment/lineage and cannot revise
these decisions in place.

## Monitoring

Use the SCO job id from submission:

```bash
sco acp jobs describe <jobid> --workspace-name project-one
sco acp jobs stream-logs <jobid> --workspace-name project-one
```

Track run progress in the experiment README and monitor roughly hourly. Useful
log markers are `rollout/raw_reward`, `train/loss`, `train/grad_norm`,
`rollout/dynamic_filter/drop_*`, W&B initialization, and checkpoint saves.
