# tau2-sft-domain-generalization

Purpose: compare four-domain mixed SFT with independently initialized domain
experts, including target-domain gains and cross-domain performance.
Name: `tau2-sft-domain-generalization`. Batch: `20260912`.

## Recipe

All five arms initialize from raw Qwen3-4B-Instruct-2507: two epochs, eight GPUs,
batch 16, LR 1e-5 cosine to 1e-6, 10% warmup, CP=1, qwen3_full, 16384 tokens/GPU.
Synchronous train.py, training/data seeds 1234, start rollout ID 0.
Final checkpoints only; no checkpoint selection on evaluation.

| Arm | Rows | Updates |
|---|---:|---:|
| mixed | 36048 | 4506 |
| airline | 13783 | 1722 |
| retail | 13614 | 1700 |
| telecom | 2979 | 372 |
| banking | 5672 | 708 |

Sources are shared `tau2-sft-official-native-expanded/data/agent_official_native_expanded.jsonl`
and `tau2-banking-simplify-qwen38-v1/data/full/banking_simplified_sft.jsonl`.
No pilot append. JSONL rows retain messages, tools, metadata and target masks.
`20260912/data/token_stats.json` records tokenizer totals and continuous-batch
coverage: all first-pass rows, followed by a partial reshuffled second pass.

## Execution

Entrypoint: `examples/tau2-bench/domain_generalization/run.sh` with
`prepare`, `train ARM smoke|full`, `convert ARM smoke|full`, or
`eval ARM smoke|full SEED`. Run only inside project job containers.
Submit with `scripts/submit.sh --project-root "$PWD" --env PROJECT_ROOT="$PWD"`
plus `--experiment tau2-sft-domain-generalization`; pass entrypoint arguments
following `--`. Keep normal priority, check quota before submitting, and use
at most two eight-GPU jobs. Do not stop unrelated jobs.

Order: tokenizer validation; longest eight rows/domain training smoke (two
updates), conversion, 3 tasks/domain × 1 trial eval; mixed full train/convert/
seeds 300 and 301; four experts with identical evaluation; raw baselines.
Artifacts remain in this worktree under the batch/arm/mode/seed directories.

## Evaluation and status

Official-native Agent, Qwen3.6-27B non-thinking User, Banking BM25; quotas
20/40/40/97; seeds 300 and 301, four trials each. Agent temperature 0.6,
top-p 1, max_tokens 1200, max_steps 200, max_errors 10, retries 3, no timeout.
Use a private benchmark data copy and actual result paths. Report pass@1,
pass@4(any), pass^4 per seed and their means, task-weighted overall and domain
macro means, paired task bootstrap differences stratified by domain.
Report action/DB accuracy, truncation, termination and infrastructure errors;
missing diagnostics are N/A. Incomplete runs are not valid comparisons.

CPU source preparation and six new unit tests passed. Container job 17741
passed all 39 official-eval regression tests. Tokenizer validation job 17740
is running; no training or evaluation results are available yet.

The corrected historical raw seed300 summary `seed300_0902_021944_summary.json`
loads `slime_sglang_agent`, not official-native `llm_agent` (verified in actual
Airline results.info.agent_info.implementation). It cannot serve as this
experiment's raw baseline; rerun raw seeds 300 and 301.

Summarize completed full evaluations with
`python3 examples/tau2-bench/domain_generalization/summarize.py output/experiments/tau2-sft-domain-generalization/20260912`.
This writes `RESULTS.md` and `results.json` only after complete task/trial
coverage, binary rewards, matching protocol and no infrastructure failures.

## Limitations

Compute differs across arms; only one training seed. Banking training retrieval
mix differs from BM25 evaluation. Point estimates alone do not establish
improved generalization. No RL, OPD, or extra ablations in this experiment.

## Jobs

- 17739: CPU tokenizer submission rejected before launch: unsupported 8 CPU / 64 GB combination. [Submission log](jobs/17739-sft-domain-tokenize-0912-012958838/submit_20260912_012958838.log). No run log was created.
- 17740: tokenizer validation, 16 CPU / 64 GB, submitted with normal priority. [Run log](jobs/17740-sft-domain-tokenize-0912-013035205/run_0_20260912_013035205.log).
- 17741: CPU regression tests passed in the training container (five new tests at execution time, 39 official-eval tests; the sixth new report test also passed locally). [Run log](jobs/17741-sft-domain-cpu-tests-0912-014200173/run_0_20260912_014200173.log).
