# tau2-banking-expert-sft

Purpose: convert the accepted independent synthetic Banking expert
trajectories into native Qwen3 `qwen3_full` SFT rows and measure a
Banking-only supervised fine-tune from Qwen3-4B-Instruct-2507.

Name: `tau2-banking-expert-sft`.

2026-09-10: completed a [trajectory simplification trial](SIMPLIFICATION_20260910.md):
eight reviewed BM25 lookups use 69.4% fewer tokens with unchanged answers and
replayed search evidence; a separate conservative full-data candidate is also available.

## Data protocol

- Source: `../tau2-banking-independent-synthetic-v1/final-minimal/successful_trajectories.jsonl`;
  only train tasks, successful rows, and `training_eligible=true` are used.
- Runtime prompt and schemas are rebuilt from the 480-document allowlist and
  each task's Banking retrieval contract. Benchmark-derived data, dev tasks,
  and challenge tasks are not included.
- The converter hides User-owned calls/results, removes a terminal User
  `###STOP###`, and keeps visible User reports. Every Agent Assistant turn is
  one target row; a multi-call turn is expanded into ordered atomic calls with
  the observed result inserted before the next target.
- Rows use native `messages` + `tools`, target-only `step_loss_mask`, and the
  same Qwen3 full-template/token-cap check as processed AReaL SFT. Rows above
  16,384 tokens are dropped rather than truncated.
- Full conversion result: 1,914 input trajectories, 13,804 candidate targets,
  7,016 retained rows, and 542 unique train tasks. Four malformed/unknown-tool
  trajectories were excluded; 6,788 over-cap targets were excluded.

## Jobs

- `16318` / `pt-z3cx4mma` — initial 32-trajectory formatter and native-token smoke;
  [run log](jobs/16318-banking-expert-build-smoke-0908-222001933/run_0_20260908_222001933.log).
- `16320` / `pt-mwlycoe1` — corrected target-level cap handling on the same
  32-trajectory native-token smoke; output is 71 rows from 79 candidates;
  [run log](jobs/16320-banking-expert-build-smoke-v2-0908-222354538/run_0_20260908_222354538.log).
- `16333` / `pt-llmzjwyh` — post-refactor 32-trajectory smoke for streaming
  output and compact cap-drop statistics (71 rows from 79 candidates; passed);
  [run log](jobs/16333-banking-expert-build-smoke-v3-0908-231232606/run_*.log).
- `16321` / `pt-rjpzkkds` — full 1,914-trajectory formatter and native-token
  validation; output is 7,016 rows from 13,804 candidates;
  [run log](jobs/16321-banking-expert-build-full-0908-222558619/run_0_20260908_222558619.log).
- `16325` / `pt-6rw7qtt0` — one-epoch, 8-GPU SFT smoke on the 71-row
  Banking-only smoke dataset; it completed three updates and saved
  `Qwen3-4B-Instruct-2507_tau2_agent_sft_banking_expert_smoke_20260908`.
  [Run log](jobs/16325-banking-expert-sft-smoke-0908-223225454/run_0_20260908_223225454.log).
- `16331` / `pt-3tk8ce6u` — full Banking-only SFT from the 7,016-row dataset
  (8 GPUs, completed at step 875); [run log](jobs/16331-banking-expert-sft-full-0908-230009483/run_*.log).
- `16348` — converted final `iter_0000875` to
  `Qwen3-4B-Instruct-2507_tau2_agent_sft_banking_expert_full_20260908/final_hf`;
  [run log](jobs/16348-banking-expert-final-hf-0908-234442366/run_*.log).
- `16382` — strict full four-domain official-native run with the Banking SFT
  checkpoint and Qwen3.6 User; stopped after repeated 262,144-token context
  overflows made the unbounded run non-terminating;
  [run log](jobs/16382-banking-sft-full-tau2-qwen36-0909-0909-004230788/run_0_20260909_004230788.log).
- `16387` — startup-only timeout-run attempt; it was stopped by the CLI
  argument wiring error before any simulation; [run log](jobs/16387-banking-sft-full-tau2-qwen36-timeout300-0909-0909-012311411/run_0_20260909_012311411.log).
- `16403` — corrected 300 s timeout diagnostic; stopped after confirming that
  context-window retries made the full run impractical;
  [run log](jobs/16403-banking-sft-full-tau2-qwen36-timeout300-0909-v2-0909-013104940/run_*.log).
- `16415` — bounded full four-domain run with the same model, seed, BM25
  retrieval, 197 tasks, four trials, and `max_steps=200`; each simulation has
  a 180 s timeout and `max_retries=0` so context/runaway failures are recorded
  once by the evaluator; [run log](jobs/16415-banking-sft-full-tau2-qwen36-bounded180-0909-0909-015238411/run_*.log).
- `16420` — replacement bounded full four-domain run with the same model,
  seed, BM25 retrieval, 197 tasks, four trials, and `max_steps=200`; each
  simulation has a 60 s timeout and `max_retries=0` so the complete comparison
  finishes within a practical runtime; job succeeded and wrote
  `eval/banking-sft-full-qwen36-bm25-bounded60/seed300_0908_181520_summary.json`;
  [run log](jobs/16420-banking-sft-full-tau2-qwen36-bounded60-0909-0909-021351414/run_*.log).
- `16453` — fixed-protocol rerun of the full four-domain comparison with the
  Banking SFT checkpoint; it uses the baseline settings and leaves the
  per-simulation timeout unset; [run log](jobs/16453-banking-sft-full-tau2-qwen36-fixed-0909-091433588/run_0_20260909_091433588.log).
- `16566` — submitted raw-model control using the same fixed four-domain,
  Qwen3.6-User, BM25, four-trial protocol as `16453`; Agent checkpoint is
  `/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3-4B-Instruct-2507`,
  with per-simulation timeout unset; [run log](jobs/16566-banking-raw-full-tau2-qwen36-fixed-0909-0909-141935763/run_*.log).

## Full benchmark result

Job `16420` succeeded with the trained checkpoint
`/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_banking_expert_full_20260908/final_hf`.
The official summary is
`eval/banking-sft-full-qwen36-bm25-bounded60/seed300_0908_181520_summary.json`.
It contains 197 tasks and 788 simulations (four trials per task), with zero
infrastructure errors. The User is the same Qwen3.6-27B text-only simulator as
the baseline (`qwen3_coder`, `enable_thinking=false`).

| split | baseline pass@1 / pass@4(any) / pass^4 | Banking-SFT pass@1 / pass@4(any) / pass^4 | delta (pp) |
| --- | --- | --- | --- |
| airline | 37.50% / 55.00% / 20.00% | 10.00% / 25.00% / 0.00% | -27.50 / -30.00 / -20.00 |
| retail | 59.38% / 87.50% / 30.00% | 10.00% / 27.50% / 0.00% | -49.38 / -60.00 / -30.00 |
| telecom | 16.25% / 30.00% / 7.50% | 18.12% / 47.50% / 0.00% | +1.87 / +17.50 / -7.50 |
| banking_knowledge | 3.61% / 6.19% / 1.03% | 0.00% / 0.00% / 0.00% | -3.61 / -6.19 / -1.03 |
| overall | 20.94% / 32.49% / 10.15% | 6.73% / 17.77% / 0.00% | -14.21 / -14.72 / -10.15 |

Overall action/database accuracy is `26.46% / 12.72%` for Banking SFT versus
`28.50% / 22.65%` in the selected raw baseline. The 60 s timeout is an
evaluator-imposed wall-clock cutoff added for this bounded run, not a model
failure label; its 410/788 timeout count must not be reported as a model error
rate. The run also recorded 24 `too_many_errors` and 8 `max_steps`, with zero
infrastructure errors. All 388 Banking simulations had zero reward. The raw
baseline's 52 simulations above 60 s were already zero-reward, so applying the
same cap offline leaves its pass metrics unchanged.
Timeout trajectories often contain repeated calls and very large agent prompt
histories (and the Agent server logs malformed/undefined tool outputs), which
is evidence of model/evaluator interaction but does not establish that every
timeout was caused by the model.
Among the 81 Banking simulations that ended without a timeout, the SFT model
also had 0 successes; this separates the observed zero from a claim that all
timed-out trials would necessarily have failed.
The baseline action/database diagnostics above come from its original
unbounded results; only the binary pass metrics are cap-equivalent without a
second baseline rollout.
The comparison is therefore a bounded-completion comparison; the stopped
unbounded diagnostic is retained in job `16382`.

The reusable Banking evaluator is
`analysis/banking_synthetic/run_qwen3_4b_eval.sh`. It invokes the official
Tau2 runner and keeps the established Qwen3.6 User protocol fixed for
base-vs-SFT comparisons: the Agent is Qwen3-4B, while the text-only User is
local Qwen3.6-27B with `qwen3_coder` and `enable_thinking=false`. Formal runs
use four trials per task; dev is split into 45 BM25 tasks plus 30
golden-retrieval tasks, and challenge contains 30 BM25 tasks. The standard
full comparison covers airline (20), retail (40), telecom (40), and
banking_knowledge (97): 197 tasks and 788 simulations. The timeout-bounded
runs are completion safeguards for this Banking-only SFT experiment. The
selected raw baseline has no successful samples above 60 s; its metrics are
recomputed with the same 60 s cap for comparison with job 16420.
