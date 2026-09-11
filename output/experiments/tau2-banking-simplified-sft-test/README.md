# tau2-banking-simplified-sft-test

Purpose: train the raw Qwen3-4B-Instruct-2507 on the completed408 simplified
Banking snapshot and compare Banking generation length and evaluation time
against the raw model under the same protocol.
Name: `tau2-banking-simplified-sft-test`.

## Training

Input: [completed408 snapshot](../tau2-banking-simplify-qwen38-v1/snapshots/20260911-161128-completed408/README.md),
329 trajectories / 207 train tasks / 1,782 rows; includes pilot, used once.
Native messages/tools, qwen3_full, target-only loss mask, final answers retained.
Raw initialization: `models/Qwen3-4B-Instruct-2507` (existing torch_dist copy).
Same historical Banking SFT recipe: 8 GPUs, 2 epochs, batch 16, LR 1e-5
(cosine to 1e-6), seed 1234, max tokens/GPU 16384.
Checkpoint: `../checkpoints/Qwen3-4B-Instruct-2507_tau2_banking_simplified_completed408_20260911/final_hf`
(relative to workspace root's slime directory).

## Evaluation

Raw and new SFT: official-native Banking test, all 97 tasks x 4 trials,
seed300, BM25, Qwen3.6-27B non-thinking User, Agent temperature 0.6,
1200 tokens/call, max_steps 200, max_errors 10, max_retries 3, no simulation
timeout. Same 8-GPU topology and Banking concurrency 9 in both arms.
Report pass@1/pass@4(any)/pass^4, trajectory duration p50/p95/max,
Agent output tokens and request latency, length finish reasons, repeated calls,
max_steps/context errors. Separate User latency from Agent generation time.
Historical timeout-bounded SFT results are contextual, not a matched control.
This snapshot changes both sample selection and content; this test alone
cannot attribute differences solely to simplification.

## Jobs

- `17637` — 8-GPU SFT, final HF conversion, then new-model Banking evaluation;
  [run log](jobs/17637-banking-simplified-sft-eval-0911-174932927/run_0_20260911_174932927.log).
- `17638` — matched 8-GPU raw-model Banking control;
  [run log](jobs/17638-banking-raw-length-control-0911-174937653/run_0_20260911_174937653.log).

- `17657` — evaluation-only restart using the completed new SFT checkpoint;
  [run log](jobs/17657-banking-simplified-eval-fixed-0911-195346178/run_0_20260911_195346178.log).

2026-09-11 19:54 Asia/Shanghai: training finished all 222 updates (0–221),
final checkpoint saved and HF conversion completed. Last loss 0.2540,
grad_norm 3.2099. Job 17637 was stopped during evaluation server startup
because it inherited the same invalid single-domain slot-borrowing option as
17638. Job 17638 exited during CLI validation before any simulations:
`--borrow-completed-domain-slots requires parallel evaluation of at least two domains`.
The shared wrapper now permits an explicit override; Banking evaluation sets
it to 0. Shell syntax checks passed. Job 17657 only evaluates; no retraining.

Per user instruction, do not resubmit the raw-model control. Reuse
[existing raw results](../tau2-banking-expert-sft/eval/raw-full-qwen36-bm25-fixed/seed300_0910_raw_fixed_summary.json):
97 Banking tasks / 388 simulations, pass@1 2.5773%, pass@4(any) 4.1237%,
pass^4 1.0309%. Task split, trials, seed, max_steps, max_errors, retrieval,
and timeout settings match. Previous run evaluated four domains concurrently
(Banking slots 4→5→6); this run is Banking-only with 9 slots, so total wall
clock and request latency are not a strictly matched speed benchmark.
Use trajectory output lengths and termination diagnostics to assess runaway
generation; interpret timing with this concurrency difference.

## Completed evaluation

Job 17657 succeeded: 97 tasks / 388 simulations, no simulation timeout, zero
infrastructure errors. [Summary](eval/simplified-sft-banking/seed300_0911_121608_summary.json)
and [length/duration comparison](eval/length_comparison.json).

| Banking metric | Previous raw | Simplified SFT |
| --- | ---: | ---: |
| pass@1 | 2.58% (10/388) | 4.12% (16/388) |
| pass@4(any) | 4.12% (4/97) | 9.28% (9/97) |
| pass^4 | 1.03% (1/97) | 0% (0/97) |
| Mean Agent tokens/call | 135.1 | 107.8 |
| Mean Agent tokens/trajectory | 1849.7 | 1924.9 |
| Trajectory seconds mean / p95 / max | 25.4 / 67.9 / 220.2 | 31.6 / 91.0 / 311.2 |
| Trajectories >60 seconds | 25/388 | 47/388 |
| max_steps terminations | 8/388 | 8/388 |
| Agent length finish reasons | 10 | 17 |

Banking evaluation wall time was 27.8 minutes (raw 41.8 minutes); higher
concurrency prevents interpreting this as model acceleration. This run completed
without the old non-terminating behavior, but long-tail trajectories remain.
Single-shot success improved by 1.55pp, coverage by 5.15pp; four-trial
consistency fell by 1.03pp. Absolute success remains low. This single-seed
comparison does not establish statistical significance.
