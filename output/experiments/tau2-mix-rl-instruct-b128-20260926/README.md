# tau2-mix-rl-instruct-b128-20260926

Purpose: measure four-domain vanilla GRPO from the original Qwen3-4B-Instruct-2507 initialization, changing only the starting model relative to SFT4505 control job 24088 (resumed from 24077).

## Configuration

[Launcher](run_vanilla_grpo.sh) copies the control's submitted launcher. HF initialization: `../models/Qwen3-4B-Instruct-2507`; Megatron initialization: `../models/Qwen3-4B-Instruct-2507_torch_dist/release` (paths relative to the repository). The SFT checkpoint step 4505 is removed so the loader uses this release. A fresh output directory starts training at update 0 with fresh optimizer/RNG.

Unchanged: 8 GPUs (Trainer 2 + three TP2 Agent generators on 6), 140 updates, LR 2e-6, batch 128 (16 groups × 8), train/rollout seeds 1234/42, KL/entropy 0, binary task reward and ordinary GRPO group normalization, TIS, uniform-outcome/unusable-group filtering, pool 32, pending 64, unlimited policy lag, 4 environment workers, Agent/step concurrency 48/160, max_steps 200, Agent max_tokens 1200, training context cap 16384, two over-cap retries, and saving every 10 updates and at completion. Reward shaping, turn credit, and OPD remain disabled.

Reuse the same 2524-task pool (Airline 1148, Retail 563, Telecom 271, Banking 542), uniform random sampling, and Qwen3.6-27B nonthinking User job 23962 through the [control's endpoint file](../tau2-mix-rl-sft4505-b128-20260925/user_endpoint.env). The User service returned HTTP 200 before submission. User temperature/top_p/max_tokens remain 0/1/512. Agent timeouts remain 60s, with Banking at 600s.

Shell syntax passed. Resolved launch settings differ only in model initialization and experiment/output paths; the copied data source matches the control's task file. The underlying training script matches both saved control snapshots. Existing rollout and asynchronous/loss preflights remain enabled at job startup.

## Jobs

| Job | GPUs | Purpose | Run log |
|---|---:|---|---|
| 24144 (`pt-accb91g4`) | 8 | Original Instruct initialization, vanilla GRPO, 140 updates; submitted 2026-09-26 08:05 CST | [Log](jobs/24144-tau2-mix-rl-instruct-vanilla-grpo-b128-0926-080536916/run_0_20260926_080536916.log) |

The cluster accepted job 24144 at 08:05:40 CST, with normal priority and elastic admission. Stopped on user request on 2026-09-26; status: STOPPED. Latest completed checkpoint: `iter_0000059` (60 updates). Outputs: `arms/async/20260926_instruct-vanilla-grpo-train/`; reward plots: `plots/20260926_instruct-vanilla-grpo-train/`.
