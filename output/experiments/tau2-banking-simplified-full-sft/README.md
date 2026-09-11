# tau2-banking-simplified-full-sft

Purpose: train raw Qwen3-4B-Instruct-2507 on all exported simplified Banking
SFT data, then evaluate Banking success and long-output behavior.
Name: `tau2-banking-simplified-full-sft`.

## Data and training

[Full JSONL](../tau2-banking-simplify-qwen38-v1/data/full/banking_simplified_sft.jsonl):
1,651 trajectories, 5,672 rows, 530 train tasks. Use this file once; no
concatenation with pilot or completed408. Export's token/mask/EOS checks reused.
Initialize from raw Qwen3-4B-Instruct-2507 via its existing torch_dist copy;
do not resume the completed408 SFT. Native messages/tools and qwen3_full
with target-only loss mask and final answers retained.
Same recipe as completed408: 8 GPUs, 2 epochs, batch 16, LR 1e-5 cosine to
1e-6, seed1234, 16384 tokens/GPU. Fresh checkpoint directory:
`/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_banking_simplified_full_20260911`.
Training script converts the final checkpoint to `final_hf`, then evaluates.

## Evaluation

Same as completed408: official-native Banking test, 97 tasks x 4 trials,
seed300, BM25, Qwen3.6-27B non-thinking User, Agent temperature0.6,
1200 tokens/call, max_steps200, max_errors10, max_retries3, no simulation
timeout; Banking concurrency9, slot borrowing disabled.
Compare pass@1/pass@4(any)/pass^4, output tokens, trajectory latency and
termination diagnostics against [completed408](../tau2-banking-simplified-sft-test/README.md)
and its linked historical raw-model results. Do not resubmit raw evaluation.
Raw ran multiple domains concurrently, so its wall time is not a matched
speed benchmark. Completed408 uses the same Banking-only serving settings.

## Jobs

- `17675` — 8-GPU full SFT → final HF conversion → Banking evaluation;
  [run log](jobs/17675-banking-full-sft-eval-0911-215724968/run_0_20260911_215724968.log).
  Submitted 2026-09-11 21:57 Asia/Shanghai, priority normal; queued.
  Parsed all 5,672 JSONL rows and checked export counts against stats.json.
  Shell syntax and single-domain slot-borrowing configuration checked.
  Completed successfully; results below.

## Results

Job 17675 succeeded, final iter707 converted to final_hf. Banking evaluation
completed all 388 simulations without a simulation timeout or infrastructure
errors. [Summary](eval/simplified-sft-banking/seed300_0911_142847_summary.json),
[length statistics](eval/length_stats.json).

| Metric | Raw | Completed408 SFT | Full SFT |
| --- | ---: | ---: | ---: |
| pass@1 | 2.58% | 4.12% | 4.38% |
| pass@4(any) | 4.12% | 9.28% | 9.28% |
| pass^4 | 1.03% | 0% | 1.03% |
| Successful simulations | 10/388 | 16/388 | 17/388 |
| max_steps terminations | 8 | 8 | 3 |

Full versus partial SFT adds one successful simulation and restores one
all-four-success task; this single seed does not establish a reliable gain.
Evaluation wall time 30.6 minutes, versus 27.8 for partial SFT. Both use
Banking concurrency9; raw has different concurrency, as described above.

Full SFT trajectory duration mean/p95/max: 35.3/107.9/319.4 seconds; 63/388 exceed 60 seconds. Mean Agent output 124.5 tokens/call and 2027.9 tokens/trajectory; 22 length finish reasons.

## Data-loading verification

Actual job arguments use `tau2-banking-simplify-qwen38-v1/data/full/banking_simplified_sft.jsonl`,
raw torch_dist initialization, qwen3_full, batch16, num_epoch2, no prompt-length
filter and no path slice. Current file parses to 5,672 rows, 1,651 source
trajectories and 530 tasks. All rows have exactly one trainable Assistant
message, at the end: 3,296 tool-call targets and 2,376 text targets.

Final existing rollout state `global_dataset_state_dict_707.pt` records
sample_index11312, epoch_id1, sample_offset5640. The sequential shuffled loader
therefore consumed every one of the 5,672 rows once and 5,640 rows a second
time. Actual logged updates are 1–707, not the previously reported 708:
`floor(5672/16)*2=708` is the rollout bound, and loading base iteration0 sets
start_rollout_id1. This accounts for 16 sample presentations; floor rounding
accounts for another 16. Total shortfall from two exact passes is 32/11344
(0.28%), all in second-pass coverage. No first-pass rows are missing.

The actual runtime's printed example has 4,196 tokens and a matching mask,
36 nonzero target tokens, with the final EOS/newline trainable. Local tokenizer
revalidation could not run because the local transformers/tokenizers versions
cannot load this tokenizer; no claim of a new full token-level revalidation is
made. The existing export checks and actual training execution are the evidence.

Conversion input is iter0000707 in the full-data run directory; evaluation
server logs point to that directory's final_hf. No completed408 checkpoint was
used. This confirms full-data consumption, not that the data improves held-out
performance.
