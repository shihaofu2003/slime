# tau2-banking-simplify-qwen38-v1

Purpose: simplify all 1,914 successful Banking expert trajectories using local
Qwen3.8-27B deletion proposals, independent semantic reviews, sequential tool
replay, and complete-trajectory non-thinking SFT export. No source data,
AReaL data, existing mixed data, training runs, or evaluation protocols change.

Name: `tau2-banking-simplify-qwen38-v1`.

Partial SFT test export: [16:11 snapshot of 408 completed decisions](snapshots/20260911-161128-completed408/README.md)
contains 329 exported trajectories / 1,782 SFT rows; one of 330 accepted
trajectories was excluded by the non-thinking export check. Full processing
continues separately.

## Protocol

- Delete only original Agent calls with paired results and whole Agent text
  fields. Preserve User events, final answers, arguments/results, and native
  multi-call batches. Never split a batch into result-conditioned decisions.
- Review all source trajectories, including tool errors and the four absent
  from the previous SFT conversion. Reconstruct runtime from tasks/contracts.
- Qwen3.8-27B BF16, TP2, 262,144 context, one concurrent request per replica;
  thinking/xhigh, temperature 1, top-p .95, top-k 20, 32,768 output tokens.
  Proposal/review seeds 300/301; one revision uses 302/303. No generated
  reasoning, review commentary, or hidden User information enters SFT.
- Independently check retained prefix evidence and dialogue coherence. Replay
  retained reads and writes in order, compare actual results and final DBs,
  and recompute the task's required official reward components.
- Pilot: 16 trajectories from each retrieval-variant/error stratum, seed 300,
  plus the previous eight lookup examples and four longest judge inputs,
  deduplicated. These examples are not supplied as model answers.
- Pilot uses two GPUs. Full uses four TP2 replicas (eight GPUs total), with
  length-balanced shards; completed valid pilot results are reused. TP1 was
  tested first and could not support the longest request with its output budget.
- Keep accepted complete Agent-visible trajectories within 16,384 target-model
  tokens. Validate every target's loss mask/EOS and absence of thinking markup.
  Over-cap and unresolved candidates are reported, not prefix-truncated.

## Entry points and outputs

- CPU preparation: `python3 examples/tau2-bench/sft/qwen_simplify_banking.py --stage prepare`.
- Jobs: `bash scripts/submit.sh --experiment tau2-banking-simplify-qwen38-v1 --gpus 2 --name banking-simplify-pilot examples/tau2-bench/sft/run_qwen38_banking_simplify.sh -- --stage pilot`.
- TP2 full: use `--gpus 8 --env BANKING_SIMPLIFY_GPUS=8 --env BANKING_SIMPLIFY_TP=2`
  and `--name banking-simplify-full`, with script arguments `-- --stage full`.
- `inventory.json`: all source input lengths/classifications and pilot IDs.
- `decisions/*.jsonl`: deletion plans, review/replay results, failures, timings.
- `requests/*.jsonl`: structured model outputs, token usage, request duration.
- `data/{pilot,full}/`: `banking_simplified_sft.jsonl`,
  `simplified_trajectories.jsonl`, `results.jsonl`, and `stats.json`.
- Statistics separate selection from compression by comparing the same accepted
  source IDs, and include the previous Banking SFT's final-target coverage.

## Validation

Extend the existing registered CPU test `tests/test_tau2_banking_expert_sft.py`
(`NUM_GPUS = 0`); run it directly. Actual tokenizer/mask and environment replay
checks run with the experiment dependencies. Compression ratios are diagnostic,
not acceptance thresholds. Data simplification does not establish that a
retrained model's evaluation latency is fixed.

## Jobs

- `17197` / `pt-ja5fnv25` — one-GPU pilot, failed on the first long request
  after loading; no trajectory decision completed (0.0896 GPU-hours):
  [run log](jobs/17197-banking-simplify-pilot-0910-234824684/run_0_20260910_234824684.log).
- `17200` / `pt-19nl5brq` — two-GPU TP2 pilot, stopped after the longest
  trajectory exhausted the 16K generation budget in both attempts:
  [run log](jobs/17200-banking-simplify-pilot-tp2-0910-235518627/run_0_20260910_235518627.log).
- `17203` / `pt-17ciyn26` — one-GPU allocation used for CPU integration checks,
  succeeded (0.0696 GPU-hours). Replays all source trajectories and checks native batch
  tokenization, required-unlock removal, and temporary-fixture SFT export:
  [run log](jobs/17203-banking-simplify-cpu-check-0911-000115358/run_0_20260911_000115358.log).
- `17209` / `pt-e1hqhdk3` — TP2 pilot resumed with 32K output budget and
  retry of incomplete inference, submitted 2026-09-11 00:14 CST; stopped
  after ten decisions to correct whitespace-text deletion validation:
  [run log](jobs/17209-banking-simplify-pilot-32k-0911-001411620/run_0_20260911_001411620.log).
- `17210` / `pt-nu6j7o2o` — TP2 pilot continuation after the whitespace-text
  correction, succeeded (3.3238 hours, 6.6475 GPU-hours). Reused eight accepted
  sources; the 67-source continuation accepted 56 and rejected 11:
  [run log](jobs/17210-banking-simplify-pilot-textfix-0911-010040711/run_0_20260911_010040711.log).
- `17371` / `pt-9ksq34vi` — full simplification, eight GPUs / four TP2 replicas, submitted
  2026-09-11 10:21 CST at normal priority. Reuses all 75 pilot decisions and
  processes the remaining 1,839 sources. Initially queued for automatic resource
  retry: workspace cluster quota had seven GPUs remaining, below the eight
  requested, despite sufficient personal quota:
  [run log](jobs/17371-banking-simplify-full-0911-102121034/run_0_20260911_102121034.log).
  Running since 10:23 CST. At 11:09 CST all four workers were progressing;
  17 additional sources completed (12 accepted, four rejected, one over cap),
  with no processing errors in these decisions. Full export remains pending.

## Preparation results

- 1,914 trajectories / 542 tasks; zero structural preflight errors, including
  trajectories missing from the old SFT conversion. Maximum judge input:
  194,490 tokens before the current 32,768-token generation allowance.
- Pilot contains 75 unique trajectories: 25 clean BM25, 18 BM25 with errors,
  16 clean golden retrieval, and 16 golden retrieval with errors. All seven
  task forms are covered, including medium/full workflows.
- 15 CPU tests passed. The previous eight lookup simplifications passed new
  sequential replay and actual Qwen3 tokenizer/mask checks (16 SFT targets).
  Additional unchanged business-action/error-recovery examples passed replay.
- Removing the required dispute-tool unlock from task `000011_l4_two_skill`
  was correctly rejected by sequential replay. Task `000004_l3_single_action`
  is already unlocked by its initialization actions, so removing its repeated
  unlock correctly leaves execution unchanged; this is not a missing-dependency
  test case. Checks use actual task initialization, not assumed empty state.
- TP1 allocated 203,904 KV tokens, below the longest input plus reserved output
  (194,490 + 16,384). The first long request failed a CUDA/NCCL 512-MiB allocation.
  TP2 has now completed the longest proposal and review; the full job remains
  budgeted at eight GPUs.
- The cluster's Transformers 5.8 defaults chat-template output to a dictionary.
  Explicit `return_dict=False` fixes token counting (the first worker reported
  two dictionary keys as two tokens); the valid host inventory used Transformers
  4.51.1. A regression test covers both dictionary-default length checks.
- TP2 model review controls passed: accept a grounded answer, reject an answer
  after deleting its fee evidence, and reject a retained tool argument after
  deleting the observation that supplied it. Controls are separate synthetic
  test cases, not production trajectory judgments.
- Cluster CPU checks passed: all 1,914 originals passed sequential replay and
  required official reward checks; real multi-call tokenization, missing-unlock
  rejection, and the eight-example export fixture passed. All 20 integration/unit,
  106 SFT-quality, and 73 Banking-synthetic tests passed under Transformers 5.8.1.
- TP2 provides 831,230 KV slots, sufficient for the 262K context. The longest
  first proposal used 14,824 reasoning tokens plus incomplete structured output;
  both 16K attempts truncated. Generation is now capped at 32K. Such incomplete
  outputs are processing errors, not semantic data rejections. Pilot order tests
  the longest input first and then seed-300 shuffled representative cases.
- The 32K pilot's longest source (`ed6dae29-5169-46ad-a3de-c8f8c323d1e4`)
  passed proposal, sequential replay, and independent review in 439 seconds.
  It removes 56 calls and 19 Agent text fields; the complete candidate has
  13,576 target-model tokens. Seven retained calls, including three searches,
  matched their saved results; final DB equality and required DB reward 1 passed.
  This is an individual result, not a completed pilot or full-dataset export.
- At 7/75 completed pilot sources, six were accepted and one was semantically
  rejected, with no unresolved processing errors. Source `6122b4b4-9fcb-43c3-9413-d80ae42eae74`
  corrected invalid deletion references on its one revision and passed at 13,650
  target tokens. Source `655b57b6-9ccf-42ef-a079-94234bd8f0df` passed DB replay
  but Qwen rejected its retained, unsupported user-tool handoff target. This
  illustrates why original task success alone does not establish SFT quality.
- Pilot diagnostics exposed a validator bug: original Assistant content `\n\n`
  was treated as nonexistent text, incorrectly consuming a model revision.
  All five inspected invalid-reference cases had valid call IDs and only this
  whitespace-text issue. The validator now permits deleting actual whitespace
  content; User events/final-answer/missing-ID protections remain unchanged.
  A regression checks partial native-batch deletion with whitespace removal.
  Direct CPU run: 21 tests, 17 passed and four integration checks skipped;
  shell syntax and the 67-source continuation assignment passed. Both rejected
  sources had consumed a revision on this bug, so their judgments are rerun
  rather than treated as final exclusions. Model prompts/settings are unchanged.

## Completed pilot and full continuation

- On 2026-09-11 14:22 CST, stopped full job `17371` at the user's request
  and submitted concurrency-four continuation `17467` (eight GPUs, four TP2
  replicas, 16 independent clients):
  [run log](jobs/17467-banking-simplify-full-c4-0911-142211671/run_0_20260911_142211671.log).
  Preserves 202 completed decisions (166 accepted, 30 rejected, six over cap);
  processes 1,711 unfinished sources plus one inference error. Disjoint resume
  assignment covers all 1,712 pending IDs without repeating completed decisions.
  In-flight incomplete sources restart at their proposal, not at a token offset.
  Both concurrency-four benchmark arms completed 16 sources with no processing
  or export errors, in 24.04/25.17 minutes versus concurrency-two's 32.38 minutes.
  Single-concurrency baseline was still finishing when switching; this supports
  a practical throughput improvement, not proof of globally optimal concurrency.
  Prompts, review rules, seeds, and token caps remain unchanged; benchmark outputs
  are not imported into production. Benchmark job `17408` continues to finish.

- All 75 pilot sources processed: 64 accepted across 58 tasks, 11 rejected,
  no unresolved processing or export errors. Export contains 220 SFT rows from
  all 64 accepted sources; no `<think>` or `</think>` markup was found.
- Same accepted source IDs: mean complete-trajectory length fell from 22,223
  to 6,770 tokens (69.5% reduction); maximum candidate length is 16,046.
  Simplification changed 49 accepted trajectories, recovered 23 over-cap
  trajectories, and recovered 24 final-answer targets missing from old SFT.
- Acceptance by source stratum: clean BM25 22/25, BM25 with errors 12/18,
  clean golden retrieval 15/16, golden retrieval with errors 15/16.
- Qwen review rejected unsupported tool discovery/handoff, unsupported final
  account-state claims, and an omitted user-facing refusal. These are semantic
  rejections after the allowed revision, not inference or export failures.
  Retained recoverable tool errors are not an automatic rejection criterion.
- Full continuation preserves the pilot model, prompts, review/replay checks,
  deletion-only operations, and 16,384-token SFT cap. Pilot decisions, including
  rejections, are reused. Final full-data export remains pending.
- At the full job's 2.70-hour check, 81/1,839 new sources had decisions:
  65 accepted, 11 rejected, four over cap, and one processing error.
  Source `4551354e-1eb6-4c55-8761-928e9f5a27ff` produced invalid deletion
  references on attempt zero; its revision reached 32,768 output tokens
  (`finish_reason=length`, 32,362 reasoning tokens). It remains an inference
  error requiring retry, not a semantic exclusion. Other workers continue;
  do not restart the full job solely for this isolated incomplete output.
- Pilot artifacts: [statistics](data/pilot/stats.json),
  [per-source results](data/pilot/results.jsonl),
  [SFT data](data/pilot/banking_simplified_sft.jsonl).
