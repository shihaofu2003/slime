# Tau2 Banking Task Curriculum

Experiment name: `tau2-banking-task-curriculum`. Purpose: inventory the 97
runtime Banking tasks, decompose them into capability graphs, and build
executable diagnostic probes. Because all 97 source tasks are reserved for
final benchmark evaluation, every artifact in this experiment is excluded
from SFT, GRPO, and OPD.

## Canonical inputs

- Runtime source: `../tau2-bench/data/tau2/domains/banking_knowledge/tasks/task_*.json`
  (97 tasks).
- Knowledge base: 698 documents.
- Historical evidence: two Qwen3-4B-Instruct-2507 runs and one Qwen3.5-4B
  non-thinking run, four trials per task with BM25 retrieval.
- Runtime task files take precedence over the aggregate `tasks.json`, which
  differs for 13 tasks.

## Artifacts

- `annotations.jsonl` and `catalog.jsonl`: six-part annotations, dependency
  graphs, source drift, and historical evidence for all 97 source tasks.
- `pilot_tasks.json` and `pilot_contracts.jsonl`: 24 hand-curated tasks from
  four representative sources.
- `expanded_tasks.json` and `expanded_contracts.jsonl`: 564 executable task
  specifications and their validation contracts.
- `validation_report.json` and `expanded_validation_report.json`: static and
  real-environment validation results.

## Expanded curriculum

- 94 replayable source tasks: 4 hand-curated and 90 systematically projected.
- Six variants per source: `evidence_given`, `retrieval_only`, `decision_only`,
  `single_action`, `two_skill_composition`, and `full_composition`; 94 tasks per
  variant and 564 total.
- All 15 business categories are represented.
- Level counts: L1 94, L2 94, L3 188, L4 104, L5 31, and L6 53.
- Each of the 470 decomposed variants starts with an explicit user request to
  the Agent; response-format placeholders remain visible while gold values are
  absent. Their User simulator stops after the first substantive answer and
  exposes only User tools evaluated by that bounded task. The 94
  full-composition tasks retain their source conversations and tools.
- All 564 specifications are benchmark-derived diagnostics, not training data.
- Systematic Evidence tasks identify the target by section plus line ordinal or
  by a table row's first-cell key, so exact-string scoring has one objective
  source line without exposing its value.
- `task_051`, `task_077`, and `task_083` remain in the 97-task inventory but
  are excluded from expansion because their current reference chains do not
  replay cleanly in the current environment.

## Validation

- 17 zero-GPU unit tests cover inventory rules, graph validity, variant
  boundaries, source identity, fact selection, query construction, and
  dialogue-entry gold leakage, including bounded User-tool exposure.
- Static validation accepts 564 unique task/contract pairs with no missing
  documents, source mappings, variants, or reward bases.
- All 94 retrieval-only reference queries return the target document in the
  environment's BM25 top 10. An exact local ranking reproduction has worst
  target rank 3.
- The real Tau2 Banking environment validates 564 schemas and completes 564
  expanded reference replays with zero expansion errors. It also preserves the
  seven known source-reference diagnostics across the three excluded tasks.

Rebuild and validate with:

```bash
python3 examples/tau2-bench/analysis/banking_task_curriculum.py build
python3 examples/tau2-bench/analysis/banking_task_curriculum.py expand
bash scripts/submit.sh --experiment tau2-banking-task-curriculum --gpus 0 \
  --cpus 16 --memory 64 \
  examples/tau2-bench/analysis/run_banking_task_curriculum.sh -- validate-expanded
```

## Model evaluation and scale-out

The dual-Qwen3.8 evaluation now covers all 564 tasks with Qwen3.8-27B serving
both Agent and User. After replacing 16 protocol-failure rows with targeted
valid reruns, exact success is `329/564` (`58.33%`) with no remaining
infrastructure errors. The selected trajectories and corrected diagnostic-use
assignments are in the [evaluation report](../tau2-banking-curriculum-qwen38-eval/all_recovered.md)
and [pool manifest](../tau2-banking-curriculum-qwen38-eval/training_pool_manifest.json).

Node-level expansion adds 2,740 independently validated evidence, retrieval,
and single-action tasks. Their dual-Qwen3.8 score is `2163/2740` (`78.94%`).
Together the two diagnostic sets contain 3,304 task specifications and 2,492
verified successful trajectories (`75.42%`). Source-level splitting is not
sufficient because every source belongs to the final benchmark. None of these
tasks or trajectories may be consumed by SFT, OPD, or GRPO.

## Jobs

- **FAILED** — initial source/pilot validator; known source diagnostics were
  incorrectly promoted to pilot failures:
  [run log](jobs/12682-banking-curriculum-env-validate-0903-094315292/run_0_20260903_094315292.log).
- **SUCCEEDED** — corrected 24-task pilot environment validation:
  [run log](jobs/12683-banking-curriculum-env-validate-v2-0903-094726375/run_0_20260903_094726375.log).
- **SUCCEEDED** — final hand-curated pilot replay after contract corrections:
  [run log](jobs/12685-banking-curriculum-env-validate-final-0903-095626036/run_0_20260903_095626036.log).
- **FAILED** — first 564-task replay; actions passed, but 18 title-only BM25
  queries missed their target documents because titles are not indexed:
  [run log](jobs/12706-banking-curriculum-expanded-env-validate-0903-112418796/run_0_20260903_112418796.log).
- **SUCCEEDED** — final 564-task replay with source-grounded BM25 queries;
  expansion errors 0:
  [run log](jobs/12713-banking-curriculum-expanded-env-validate-v2-0903-114359349/run_0_20260903_114359349.log).
- **SUCCEEDED** — bounded-User v3 validation after narrowing User-tool
  exposure: all 564 regenerated schemas and all 564 reference trajectories
  passed in the real Banking environment, with zero expansion errors:
  [run log](jobs/12990-banking-bounded-user-env-validate-v3-0903-163355008/run_0_20260903_163355008.log).
- **SUCCEEDED** — regression validation for symmetric comma normalization in
  Tau2's communicate evaluator: new regression 1/1, existing Banking tests
  14/14, and pilot reference replays 24/24 with zero validation errors:
  [run log](jobs/13014-banking-communicate-evaluator-fix-validate-0903-172138094/run_0_20260903_172138094.log).
