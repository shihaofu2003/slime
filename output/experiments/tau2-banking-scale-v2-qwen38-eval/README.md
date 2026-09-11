# Qwen3.8 Banking Scale V2 Evaluation

Experiment name: `tau2-banking-scale-v2-qwen38-eval`. Purpose: generate and
score one Qwen3.8-27B Agent/User trajectory for each of the 2,740 validated
node-level Banking scale-v2 task specifications.

## Protocol

- Agent and User both use `/mnt/afs/models/Qwen3.8-27B`.
- Agent uses xhigh thinking, 16,384 output tokens, 262,144 logical context,
  and 0.95 GPU memory fraction; User uses non-thinking decoding and 1,024
  output tokens.
- One trial per task, exact Tau2 success, real Banking environment.
- Retrieval-specific evaluation keeps BM25 tasks separate from tasks supplied
  with golden documents.

## Partitions

- Golden retrieval: 1,807 tasks, split into eight source-disjoint shards.
- BM25: 933 tasks, split into four source-disjoint shards.
- All variants derived from one source task remain in one shard for a given
  retrieval mode.

## Results

- Full aggregate: `2163/2740` exact successes (`78.94%`), with all 2,740
  trajectories ending in `user_stop` and zero infrastructure errors.
- Evidence-given: `578/933` (`61.95%`); BM25 retrieval-only: `932/933`
  (`99.89%`); single-action: `653/874` (`74.71%`).
- All 2,740 tasks are benchmark-derived diagnostics. The 2,163 successful
  trajectories are retained for analysis but are ineligible for SFT, GRPO, and
  OPD.
- Artifacts: [aggregate metrics](all_results.md),
  [machine-readable metrics](all_results.json), and
  [legacy-named diagnostic manifest](training_pool_manifest.json).

## Jobs

All twelve logical shards are complete. The first BM25 shard-1 job failed
before inference and was replaced by job 13274; two completed result sets had
post-evaluation wrapper failures, but their full trajectories were preserved
and summarized offline.

- **SUCCEEDED** — golden-retrieval shard 0/8, `173/251`, job 13216:
  [run log](jobs/13216-qwen38-bank-scale-v2-golden-s0of8-0903-205631720/run_0_20260903_205631720.log).
- **SUCCEEDED** — golden-retrieval shard 1/8, `161/233`, job 13218:
  [run log](jobs/13218-qwen38-bank-scale-v2-golden-s1of8-0903-205632438/run_0_20260903_205632438.log).
- **SUCCEEDED** — golden-retrieval shard 2/8, `160/244`, job 13217:
  [run log](jobs/13217-qwen38-bank-scale-v2-golden-s2of8-0903-205632085/run_0_20260903_205632085.log).
- **SUCCEEDED** — golden-retrieval shard 3/8, `134/198`, job 13219:
  [run log](jobs/13219-qwen38-bank-scale-v2-golden-s3of8-0903-205632734/run_0_20260903_205632734.log).
- **SUCCEEDED** — golden-retrieval shard 4/8, `139/197`, job 13223:
  [run log](jobs/13223-qwen38-bank-scale-v2-golden-s4of8-0903-205644062/run_0_20260903_205644062.log).
- **EVALUATION COMPLETE; WRAPPER FAILED AFTERWARD** — golden-retrieval shard
  5/8, `157/224`, zero infrastructure errors, job 13222. All trajectories and
  the official summary were saved; the same live launcher edit caused a shell
  parse error only after service cleanup, and per-shard metrics were regenerated
  from the complete results:
  [run log](jobs/13222-qwen38-bank-scale-v2-golden-s5of8-0903-205638047/run_0_20260903_205638047.log),
  [metrics](eval/full4-golden-retrieval-shard5of8/seed300_0903_154146/curriculum_metrics.md).
- **SUCCEEDED** — golden-retrieval shard 6/8, `144/220`, job 13221:
  [run log](jobs/13221-qwen38-bank-scale-v2-golden-s6of8-0903-205637742/run_0_20260903_205637742.log).
- **SUCCEEDED** — golden-retrieval shard 7/8, `163/240`, job 13220:
  [run log](jobs/13220-qwen38-bank-scale-v2-golden-s7of8-0903-205637343/run_0_20260903_205637343.log).
- **SUCCEEDED** — BM25 shard 0/4, `244/244`, zero infrastructure errors, job
  13225:
  [run log](jobs/13225-qwen38-bank-scale-v2-bm25-s0of4-0903-205713710/run_0_20260903_205713710.log).
- **FAILED BEFORE INFERENCE** — BM25 shard 1/4, 256 tasks, job 13224. One
  assigned GPU had only 40.44 GB free and OOMed while loading the model:
  [run log](jobs/13224-qwen38-bank-scale-v2-bm25-s1of4-0903-205652149/run_0_20260903_205652149.log).
- **SUCCEEDED** — BM25 shard 2/4, `221/221`, zero infrastructure errors, job
  13226:
  [run log](jobs/13226-qwen38-bank-scale-v2-bm25-s2of4-0903-205720526/run_0_20260903_205720526.log).
- **SUCCEEDED** — BM25 shard 3/4, `212/212`, zero infrastructure errors, job
  13227:
  [run log](jobs/13227-qwen38-bank-scale-v2-bm25-s3of4-0903-205727443/run_0_20260903_205727443.log).
- **EVALUATION COMPLETE; WRAPPER FAILED AFTERWARD** — BM25 shard 1/4, job
  13274, explicitly requested 4 GPUs, 56 CPUs, and 792 GB memory. All 256
  trajectories and the official summary were saved (`255/256`, zero
  infrastructure errors); a live edit to the shared launcher caused a shell
  parse error only after service cleanup. The per-shard curriculum metrics
  were regenerated from the complete `results.json`, so no model rerun is
  required:
  [run log](jobs/13274-qwen38-bank-scale-v2-bm25-s1of4-retry-v2-0904-000059878/run_0_20260904_000059878.log),
  [metrics](eval/full4-bm25-shard1of4/seed300_0903_160228/curriculum_metrics.md).
