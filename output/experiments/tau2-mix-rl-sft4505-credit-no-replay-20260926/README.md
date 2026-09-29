# tau2-mix-rl-sft4505-credit-no-replay-20260926

Purpose: restart four-domain credit-assignment mix RL from SFT4505, discarding all-failed groups without scheduling those tasks for retry two updates later.

## Configuration

[Launcher](run_mix.sh) copies the submitted credit run 23971 launcher. Fresh initialization: `Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920/iter_0004505_hf`, Megatron reference step 4505, fresh optimizer and output directory.

Use `random_tasks.RandomTaskDataSource`, which draws uniformly with replacement and has no deferred task-retry queue. Keep `TAU2_DROP_UNIFORM_OUTCOME_GROUPS=1`: all-zero and all-one groups are discarded and replaced by new random draws. Do not use `ShuffledTaskDataSource`, whose all-zero queue schedules retries after two updates. A discarded task can still appear through ordinary random sampling. This matches the existing mix-run sampler; no sampler implementation change is needed.

Unchanged: progress-db-count-v1 credit, progress/format weights 1/1, gamma 0.98; 8 GPUs (Trainer 2 + three TP2 Agent replicas); 140 updates; LR 2e-6; batch 128 (16 groups × 8); KL/entropy 0; train/rollout seeds 1234/42; pool 32, pending 64, unlimited policy lag; 4 environment workers; Agent/step concurrency 48/160; max_steps 200; Agent max_tokens 1200; context cap 16384; two over-cap retries; save interval 10; OPD off. Existing startup preflights remain enabled.

Reuse the same 2524-task four-domain pool and persistent Qwen3.6-27B nonthinking User job 23962 via the original experiment's endpoint file. Only experiment/output names and the endpoint-file expression change relative to job 23971.

## Jobs

| Job | GPUs | Purpose | Run log |
|---|---:|---|---|
| 24150 (`pt-im4oekgy`) | 8 | SFT4505 credit-assignment mix RL, no deferred all-zero task retries; submitted 2026-09-26 11:26 CST | [Log](jobs/24150-tau2-mix-rl-sft4505-credit-no-replay-0926-112643992/run_0_20260926_112643992.log) |

Cluster accepted at 11:26:46 CST; status SUBMITTED, normal priority and normal admission. Shell syntax passed, and launcher comparison confirms unchanged training parameters. Outputs: `arms/async/20260926_credit-no-replay-train/`; plots: `plots/20260926_credit-no-replay-train/`.
