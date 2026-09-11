# Experiment: tau-bench

**Name:** tau-bench
**Purpose:** Get tau-bench (tau1, retail) RL training running end-to-end with
slime on Qwen3-4B-Instruct-2507 — a run-through of the `examples/tau-bench`
integration: mock data generation, Instruct checkpoint conversion to torch_dist,
then GRPO training with the custom tau rollout function. This is the simpler
`examples/tau-bench` path (not `tau-bench-example`).

**Model:** Qwen3-4B-Instruct-2507 (rotary base 5e6).
**Algorithm:** GRPO, retail env, tool-calling agent; user simulator via Gemini
API (litellm).

## Tasks

Each task's run log is linked relative to this README.

### 1. Generate tau-bench mock data

Dump retail/airline task-index JSONL (`retail_train_tasks.jsonl`,
`retail_dev_tasks.jsonl`, ...) for training/eval.

- Script: `slime/scripts/run-tau-bench-mock-data.sh`
- Latest run: [jobs/fsh--run-tau-bench-mock-data-0709-210236/run_20260709_210236.log](jobs/fsh--run-tau-bench-mock-data-0709-210236/run_20260709_210236.log)
- (earlier failed attempt — missing litellm: [jobs/fsh--run-tau-bench-mock-data-0709-204948/](jobs/fsh--run-tau-bench-mock-data-0709-204948/))

### 2. Convert Qwen3-4B-Instruct-2507 → torch_dist

HF safetensors → Megatron-LM torch_dist checkpoint for slime to load.

- Script: `slime/scripts/convert_qwen3_4b_instruct_to_torch_dist.sh`
- Latest run: [jobs/fsh--convert_qwen3_4b_instruct_to_torch_dist-0709-210237/run_20260709_210237.log](jobs/fsh--convert_qwen3_4b_instruct_to_torch_dist-0709-210237/run_20260709_210237.log)
- (earlier failed attempt — wrapper path: [jobs/fsh--convert_qwen3_4b_instruct_to_torch_dist-0709-204949/](jobs/fsh--convert_qwen3_4b_instruct_to_torch_dist-0709-204949/))

### 3. tau-bench GRPO training

Run `train.py` with the custom tau rollout, Instruct-2507, retail data. User
simulator is an OpenAI-compatible endpoint (o4-mini via chatanywhere) instead of
Gemini. Run-through config: 8 GPU, batch 8 × n-samples 8 (global 64),
num-rollout 10, sglang concurrency 16.

- Script: `slime/scripts/run-qwen3-4B-serviceagent-tau.sh`
- Latest run: [jobs/fsh--run-qwen3-4B-serviceagent-tau-0709-212154/run_20260709_212154.log](jobs/fsh--run-qwen3-4B-serviceagent-tau-0709-212154/run_20260709_212154.log)

### 4. Real (non-smoke) GRPO run — 3 gradient steps ✅

First multi-step training run: verifies the loop trains across iterations,
not just the smoke's single step. Real-ish config, fresh `CHECKPOINT_ROOT`
(`checkpoints_real`) so it never resumes a stale smoke ckpt. Dynamic filter
OFF (cold model → ~90% group rejection → slow; re-enable once warm). Eval
skipped. 8 GPU, batch 8 × n-samples 4 (global 32), num-rollout 3, kl_coef 0.

- Script: `slime/scripts/run-qwen3-4B-serviceagent-tau-real.sh`
- Latest run (SUCCEEDED): [jobs/fsh--run-qwen3-4B-serviceagent-tau-real-0710-200409/run_20260710_200409.log](jobs/fsh--run-qwen3-4B-serviceagent-tau-real-0710-200409/run_20260710_200409.log)
- (earlier attempts — filter-on stall + Ray GCS transient: see MONITOR.md #12)

## Resolved — the `--kl-coef` workaround is replaced by proper logprobs

The GRPO advantage step used to crash (`loss.py:700-703`, all-None log-probs):
slime's default `can_reuse_log_probs_in_loss` skips the actor log-prob
pre-forward, and the custom tau generate returned no logprobs. The proper fix
(option 1 below) is now implemented: `trainable_agents.asolve` captures sglang's
per-turn `output_token_logprobs` and sets `Sample.rollout_log_probs`
(`_align_turn_log_probs` pads the Qwen template `\n` / env tokens with 0.0;
`use_rollout_logprobs` stays False, so with `kl_coef==0` these are only the
zero-KL shape template — values never enter the on-policy ratio). `--kl-coef` is
back to 0. Verified by smoke `pt-prux2yjk` (see MONITOR.md #11): 8/8 rollouts via
`asolve`, no `loss.py:703` crash, gradient step + iter-0 checkpoint, SUCCEEDED.

Original options that were considered:
1. ✅ **(chosen)** Have `generate_with_tau`/`trainable_agents` return per-token
   logprobs (`rollout_log_probs`) from the sglang `/generate` calls — lets
   `kl_coef` stay 0.
2. Fix slime's `compute_advantages_and_returns` (loss.py:700-703) to handle the
   all-None case — upstream change, not needed now.
3. Keep `kl_coef=0.001` — the prior verified workaround, now superseded (kept as
   a `KL_COEF=0.001` env fallback in the run script).

## Monitoring

See [MONITOR.md](MONITOR.md) for the ~30-min monitoring log.
