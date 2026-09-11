# Qwen3.6 Banking Curriculum Evaluation

Experiment name: `tau2-banking-curriculum-qwen36-eval`. Purpose: measure the
single-trial accuracy of local Qwen3.6-27B on all 564 validated Banking
curriculum task specifications.

## Protocol

- Agent: local `Qwen3.6-27B`, official-native Tau2 Agent, thinking enabled,
  temperature 0, top-p 1, and 4,096 output tokens per turn.
- User: local `Qwen3.6-27B`, thinking disabled, temperature 0, top-p 1, and
  512 output tokens per turn.
- Banking environment: real Tau2 environment, one trial per task, seed 300,
  maximum 200 steps.
- Metric: exact Tau2 task success. Results are also grouped by curriculum
  variant, level, retrieval mode, and business category.

## Evaluation partitions

- `golden_retrieval` shard 0: 187 tasks from 47 source tasks.
- `golden_retrieval` shard 1: 188 tasks from 47 source tasks.
- `bm25` shard 0: 189 tasks from 94 source tasks.
- The three partitions are disjoint and cover all 564 task ids. All variants
  derived from one source stay together within each retrieval-mode split.

## Artifacts

Each run directory contains its staged task manifest, official Tau2 summary,
raw `results.json`, and `curriculum_metrics.json` / `.md`. The final aggregate
summary is written after all three full partitions finish.

## Jobs

- **FAILED TO SUBMIT** — unsupported `2 GPU / 32 CPU / 128 GB` resource
  combination; no workload started and no GPU time was consumed:
  [job-manager log](jobs/12816-qwen36-banking-curriculum-bm25-smoke-0903-134326109/jobm.log).
- **FAILED** — seven-task BM25 smoke loaded both model servers successfully,
  then stopped before inference because the isolated data directory omitted
  the Banking retrieval prompt templates:
  [run log](jobs/12817-qwen36-banking-curriculum-bm25-smoke-v2-0903-134341543/run_0_20260903_134341543.log).
- **STOPPED** — seven-task BM25 smoke after adding the prompt templates; stopped
  during its first task when the protocol was changed to require Qwen3.6 for
  both Agent and User:
  [run log](jobs/12820-qwen36-banking-curriculum-bm25-smoke-v3-0903-135211496/run_0_20260903_135211496.log).
- **STOPPED** — seven-task BM25 smoke with Qwen3.6-27B serving both Agent and
  User; stopped when the requested model changed to Qwen3.8-27B:
  [run log](jobs/12823-dual-qwen36-banking-curriculum-bm25-smoke-0903-140434634/run_0_20260903_140434634.log).
