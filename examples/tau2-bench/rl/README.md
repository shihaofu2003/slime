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

Four-card smoke:

```bash
bash scripts/submit.sh --experiment tau2-rl --gpus 4 \
  examples/tau2-bench/rl/smoke_qwen3_4b_instruct_2507_tau2_rl.sh
```

Eight-card training run:

```bash
bash scripts/submit.sh --experiment tau2-rl --gpus 8 \
  examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh
```

The scripts intentionally reject non-4-card and non-8-card jobs. By default the
local trained user simulator runs on the last visible GPU. Set `USER_SGLANG=0`
and provide `TAU2_USER_API_BASE` / `TAU2_USER_API_KEY` to use an external user
simulator and let Ray use all cards.

## Defaults

- Agent init: `checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714`.
- Agent tokenizer/HF path: `.../iter_0002413_hf`.
- User simulator: `checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf`.
- Reward: tau2 official DB/env assertion/action/communication evaluators with
  each task's `db_path`.
- Objective: trajectory-level GRPO; no TIS, no per-turn reward shaping.

## Real 8g Hyperparameters

The first real run uses `run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh`.

- Data: full AReaL RL split, 1,982 tasks: airline 1,148, retail 563, telecom 271.
- Steps: `NUM_ROLLOUT=330`, approximately one pass of 1,982 prompt groups at
  `ROLLOUT_BATCH_SIZE=6`.
- Batch/grouping: `ROLLOUT_BATCH_SIZE=6`, `N_SAMPLES_PER_PROMPT=8`,
  `GLOBAL_BATCH_SIZE=48`; with 6 actor GPUs and TP=2, DP=3 requires the global
  batch to be divisible by 3.
- Sampling: agent `ROLLOUT_TEMPERATURE=0.65`, `top_p=1.0`,
  `AGENT_MAX_TOKENS=1024`; user simulator `temperature=0.0`,
  `max_tokens=512`. `TAU2_MAX_STEPS=60`.
- Memory: `MAX_TOKENS_PER_GPU=6144`, `LOG_PROBS_CHUNK_SIZE=1024`, and
  `SGLANG_SERVER_CONCURRENCY=8`. Log-prob temperature scaling is performed
  inside each chunk, so a long trajectory does not allocate a second full
  `[tokens, vocab]` tensor.
- Algorithm: GRPO, `kl_coef=0`, `kl_loss_coef=0`, dynamic filtering enabled via
  `check_reward_nonzero_std` to drop all-correct/all-wrong prompt groups.
- Optimizer: Adam, `lr=1e-6`, constant schedule, weight decay `0.1`.
- Checkpoints: save every 33 rollout/update steps.
- W&B: enabled by default in the real wrapper; key is supplied from the job
  environment and is not passed on the command line.

Rationale: the AReaL tau2 paper uses SFT user simulators, trajectory-level GRPO,
large effective batch sizes, dynamic filtering, no LR decay, max context 32k,
and temperature 1.0 for both agent and user. Their batch ablation shows total
batch size matters most and dynamic filtering improves Airline from 65.0 to
70.5 pass^1 at batch 8x64. This local 4B run is constrained to one 8-card node,
uses the verified local user simulator, and matches our evaluation temperature
(`0.65`) rather than the paper's exploratory temperature (`1.0`).

The paper's 256-512 effective batches and 32k-token microbatches were measured
on 64 or 80 H200 GPUs. They are reference targets, not safe single-node
defaults. This run keeps eight trajectories per prompt and dynamic filtering,
but uses a DP-aligned effective batch of 48 and a 6,144-token packing target.
An individual tau2 trajectory can still exceed the packing target; chunked
log-prob calculation is therefore required independently of dynamic batching.

## Monitoring

Use the SCO job id from submission:

```bash
sco acp jobs describe <jobid> --workspace-name project-one
sco acp jobs stream-logs <jobid> --workspace-name project-one
```

Track run progress in the experiment README and monitor roughly hourly. Useful
log markers are `rollout/raw_reward`, `train/loss`, `train/grad_norm`,
`rollout/dynamic_filter/drop_*`, W&B initialization, and checkpoint saves.
