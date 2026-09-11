# tau2 analysis

## SFT quality audit

`audit_sft_quality.py` audits the exact max-8192 SFT file used by the current agent
checkpoint. It reconstructs one training view per AReaL source dialog, counts the
repeated prefix exposures seen by the SFT loss, applies deterministic policy/schema
checks, samples successful trajectories, and builds blinded model-judge inputs.

```bash
python3 examples/tau2-bench/analysis/audit_sft_quality.py prepare
```

The default open-source judge is the downloaded `models/Qwen3.6-27B`, served locally
with SGLang. Explicit thinking is disabled so the 2,048-token completion budget is used
for the constrained JSON; set `SFT_QUALITY_ENABLE_THINKING=true` only with a larger
`SFT_QUALITY_MAX_TOKENS`. It first reviews a 72-dialog stratified calibration set twice. Calibration
requires at least 99% completion, 85% verdict agreement, mean dimension difference at
most 0.5, at most 5% false keeps on deterministic hard errors, and median confidence at
least 0.75. Low-confidence, unexpectedly uncertain, disagreeing, rule-conflicting, or
failed local reviews use an OpenAI-compatible fallback API when configured. A
high-confidence `uncertain` caused only by the known final-tool-call dataset boundary is
not escalated because another judge cannot observe the missing result either. If the
local calibration fails globally, the full review uses the fallback API.

```bash
bash scripts/submit.sh --experiment tau2-sft-quality-audit --gpus 1 \
  --name tau2-sft-quality-qwen36-27b --with-vitabench \
  examples/tau2-bench/analysis/run_sft_quality_audit.sh
```

API fallback can be configured without changing code:

```bash
SFT_QUALITY_FALLBACK_BASE_URL=https://openrouter.ai/api/v1 \
SFT_QUALITY_FALLBACK_MODEL=google/gemini-2.5-flash \
SFT_QUALITY_FALLBACK_API_KEY_ENV=OPENROUTER_API_KEY \
bash examples/tau2-bench/analysis/run_sft_quality_audit.sh
```

## Quality gates

A dialog is `drop` if source `correct` or `reward` is not positive, or if deterministic
checks find mixed text/tool output, same-turn multiple calls, nonexistent/wrong-namespace
tools, argument-schema errors, or inconsistent prefixes. A dialog is `keep` only when it
passes those checks and the selected judge says `keep` with outcome `success`, or with
`uncertain` only when the visible training prefix ends on its target tool call. It also
requires confidence at least 0.75, mean quality at least 3.25/4, and at least 3/4 on
completion, policy, tool choice, argument grounding, and workflow. All remaining dialogs
are `review`; unresolved judge escalation can never silently become `keep`.

The strict filter does not rewrite multi-call trajectories into counterfactual sequential
ones. Such repair would expose later calls to earlier observations that were unavailable
when the source response was generated; those trajectories stay auditable in the
manifest and are excluded from `filtered_sft_keep.jsonl`.

## Dependency-safe multitool audit

`run_multitool_quality_audit.sh` is a separate, protocol-neutral follow-up. It reads the
same raw and converted SFT data but defaults to
`output/experiments/tau2-sft-multitool-quality-audit/` and the
`dependency-safe-multi` profile. Unlike the legacy audit, it judges complete source
dialogs, does not treat call count alone as a hard error, compares successful single and
multi-call trajectories, and attributes failed trajectories to a primary cause. The
existing `run_sft_quality_audit.sh` and its `current-single` outputs remain unchanged.

The default `all` stage prepares two independent, sealed 72-dialog calibrations. The
quality gate checks reliability, verdict/batch agreement, per-multi-turn agreement,
dimension distance, and mechanical false keeps. The failure-attribution gate is declared
before reviews, stratified by domain, label group, multi exposure, and full/window route,
and checks both overall field resolution and every one-dimensional margin with `n >= 12`.
Full quality and failure review start only after their respective hard gate passes;
`summarize` requires both gates to pass.

Every selected normal-length trajectory is reviewed once by local Qwen3.6-27B and once
by the fallback API. A trajectory that exceeds the local context is never truncated: the
local judge receives a prepared structured window and a separately configured
long-context fallback receives the complete trajectory. Failure fields have four mutually
exclusive resolution states: `agreed_resolved`, `agreed_unknown`, `disagreed`, and
`unreliable`; only the first supports a named attribution, and all four remain in the fixed
denominator. Quality disagreement remains `review`.

After three judge self-repair attempts, only allowlisted citation/assessment structure may
be repaired conservatively. The repaired confidence is capped below the reliability
threshold and the record is marked with an error, so it cannot become a keep or resolved
failure attribution. Unselected hard failures are retained in secret-redacted
`*.failures.jsonl` sidecars.

Quality output includes one assessment for every multi-call turn, so reviewed SFT target
rows can be mapped to dependency, grounding, authorization, and necessity evidence. The
matched comparison covers all 92 single consensus-success anchors but only 147 globally
unique controls from 1,954 multi successes; it estimates matched quality differences, not
multi-success safety prevalence or full-corpus filter yield.

```bash
bash scripts/submit.sh --experiment tau2-sft-multitool-quality-audit --gpus 1 \
  --name tau2-sft-multitool-qwen36-27b --with-vitabench \
  examples/tau2-bench/analysis/run_multitool_quality_audit.sh
```

Run or resume one stage with `SFT_MULTITOOL_STAGE=prepare`, `calibrate`, `quality`,
`failure`, or `summarize`. `calibrate` runs both gates; `quality` and `failure` run only
their corresponding gate before the full partition. `prepare` performs no model serving
or API calls and is the supported prepare-only path:

```bash
SFT_MULTITOOL_STAGE=prepare \
bash examples/tau2-bench/analysis/run_multitool_quality_audit.sh
```

The independent fallback keeps a 2048-token output budget for ordinary payloads and
uses 4096 for full long-context payloads; override the latter with
`SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_MAX_TOKENS` when reproducing a prior run.

For cluster submission, pass the stage as a script argument:

```bash
bash scripts/submit.sh --experiment tau2-sft-multitool-quality-audit --gpus 1 \
  --name tau2-sft-multitool-prepare --with-vitabench \
  examples/tau2-bench/analysis/run_multitool_quality_audit.sh -- --stage prepare
```

Dual review requires an OpenAI-compatible fallback. The runner can discover OpenRouter
or an `OPENAI_API_BASE` from the tau2 `.env`, or it can be configured explicitly. Use a
different long-context endpoint/model when the ordinary fallback cannot accept complete
long dialogs:

```bash
SFT_MULTITOOL_FALLBACK_BASE_URL=https://openrouter.ai/api/v1 \
SFT_MULTITOOL_FALLBACK_MODEL=google/gemini-2.5-flash \
SFT_MULTITOOL_FALLBACK_API_KEY_ENV=OPENROUTER_API_KEY \
SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_BASE_URL=https://openrouter.ai/api/v1 \
SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_MODEL=google/gemini-2.5-pro \
SFT_MULTITOOL_LONG_CONTEXT_FALLBACK_API_KEY_ENV=OPENROUTER_API_KEY \
bash examples/tau2-bench/analysis/run_multitool_quality_audit.sh
```

This phase is an audit, not a policy rewrite or training-data release. It does not edit
the tau2 domain policy and does not emit a final training JSONL.

## Outputs

Results live under `output/experiments/tau2-sft-quality-audit/`:

- `DETERMINISTIC_REPORT.md`, `inventory.json`, and `features.jsonl` contain source-label,
  protocol, schema, and repeated-loss-exposure facts.
- `success_sample.jsonl` contains stratified positive, rule-clean trajectories for the
  initial “why did it succeed?” study.
- `calibration_sample.jsonl`, `calibration_report.json`, and `judge_reviews.jsonl` retain
  the local/API audit trail.
- `selection_manifest.jsonl/csv`, `REPORT.md`, and `SUCCESS_CASES.md` explain every final
  decision; `filtered_sft_keep.jsonl` preserves selected training rows verbatim except
  for an added `metadata.sft_quality_tier` provenance field.

`filtered_sft_keep.jsonl` is an audit/selection artifact, not a training-ready
`tau2_official_single` dataset: preserving rows verbatim also preserves the conflicting
`one or more tool calls` system instruction. Regenerate or replay the selected dialogs
under the official single-call protocol before the retraining ablation.

The multitool experiment instead writes full-source/training-visible feature views,
converted-row checks, `comparison_pairs.jsonl`, sealed `calibration_selection.jsonl` and
`failure_calibration_selection.jsonl`, a hashed `payload_manifest.jsonl`, local/long
partitions, structured windows, and separate local/API review trails. Its reports are
`COMPARISON_REPORT.md`, `FAILURE_ATTRIBUTION.md`, `CALIBRATION_REPORT.md`,
`FAILURE_CALIBRATION_REPORT.md`, and `FILTER_SPEC_DRAFT.md`. After both gates pass,
`filter_decisions_draft.jsonl` records one provenance-only decision per converted target,
including hashes and exclusion reasons but no prompts, conversations, target text, or
training messages. It is not a training JSONL; unknown evidence, unreviewed targets, label
conflicts, and judge disagreement never silently become keep decisions.

## Verification

The CPU regression test is:

```bash
python3 tests/test_tau2_sft_quality.py
```

The filtered dataset is a research arm, not a claimed improvement. Compare it against
the existing unfiltered checkpoint with the same SFT budget and the same held-out
official tau2 evaluation before attributing a model change to data quality.
