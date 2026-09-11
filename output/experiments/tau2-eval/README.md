# tau2-eval

**Purpose:** tau2-bench evaluation (airline, retail, telecom) for a
sglang-served policy. Current eval path uses tau2-bench's official
`HalfDuplexAgent` + `TextRunConfig` + runner APIs. Legacy direct-`AgentGymEnv`
results are kept below for comparison.

Code: [`examples/tau2-bench/eval/official/`](../../../examples/tau2-bench/eval/official/).
Submit via `scripts/submit.sh --experiment tau2-eval`.

## Results

Full test split, 100 tasks (airline 20 / retail 40 / telecom 40), 4 trials per
task. Cells in the first table are `pass@1 / pass@4(any)` in **%**, where
`pass@4(any)` means at least one successful trial for that task.

### Current config (agent temp 0.6, user-sim temp 0.0, prose-as-respond)

| model | overall | airline | retail | telecom |
|---|---|---|---|---|
| **Qwen3.5-4B** (thinking ON) | **68.2 / 91.0** | **78.8 / 90.0** | **45.6 / 82.5** | **85.6 / 100.0** |
| Qwen3-4B-Instruct-2507 (base) | 26.2 / 58.0 | 35.0 / 60.0 | 28.7 / 67.5 | 19.4 / 47.5 |
| Qwen3-4B-Instruct-2507-SFT | 24.5 / 43.0 | 33.8 / 45.0 | 24.4 / 52.5 | 20.0 / 32.5 |
| Qwen3-4B-tau2-grpo-v1 (RL) | 7.5 / 17.0 | 5.0 / 15.0 | 5.6 / 7.5 | 10.6 / 27.5 |

### Official pass^k

Cells are tau2 `pass_hat_ks`: `pass^1 / pass^4` in **%**.

| model | overall | airline | retail | telecom |
|---|---|---|---|---|
| **Qwen3.5-4B** (thinking ON) | **68.2 / 39.0** | **78.8 / 60.0** | **45.6 / 10.0** | **85.6 / 57.5** |
| Qwen3-4B-Instruct-2507 (base) | 26.2 / 6.0 | 35.0 / 10.0 | 28.7 / 7.5 | 19.4 / 2.5 |
| Qwen3-4B-Instruct-2507-SFT | 24.5 / 13.0 | 33.8 / 25.0 | 24.4 / 7.5 | 20.0 / 12.5 |
| Qwen3-4B-tau2-grpo-v1 (RL) | 7.5 / 1.0 | 5.0 / 0.0 | 5.6 / 2.5 | 10.6 / 0.0 |

### Δ vs prior config (temp 0.8/0.7, Qwen3.5 `--disable-thinking`, strict parse)

| model | pass@1 (old → new) | pass@4 (old → new) |
|---|---|---|
| Qwen3.5-4B | 13 → **68.2** (+55.2) | 36 → **91.0** (+55.0) |
| Qwen3-4B-tau2-grpo-v1 | 34 → 7.5 (−26.5) | 62 → 17.0 (−45.0) |
| Qwen3-4B-Instruct-2507 | 5 → **26.2** (+21.2) | 22 → **58.0** (+36.0) |

Takeaways: (1) thinking is the right mode for Qwen3.5 on tau2; (2)
prose-as-respond improved the base model by no longer penalizing plain-prose
replies; (3) this GRPO checkpoint underperforms the base model on this official
runner/config and needs follow-up before treating it as a useful policy.

## Tasks

Per-job run logs are added here on submission (relative links, per the doc
conventions).

- **official runner full Pass@4 — `Qwen3-4B-Instruct-2507-SFT`, test split**
  (`pt-8soksnov`, SUCCEEDED) — 1 GPU, tau2 official runner path, full test
  split, `NUM_TRIALS=4`, domains airline/retail/telecom. This checkpoint is the
  SFT model converted from the Qwen3-4B-Instruct-2507 non-thinking model family:
  `checkpoints/Qwen3-4B-Instruct-2507_tau2_sft_areal_strict_no_thinking_epoch2_20260713/iter_0000881_hf`.
  Completed with overall pass@1=24.5%, pass@4=43.0%, infra_error_count=0.
  Wrapper: `examples/tau2-bench/eval/official/models/run_full_this_sft.sh`.
  Summary target:
  `examples/tau2-bench/eval/official/outputs/Qwen3-4B-Instruct-2507-SFT/pass4_summary.json`.
  Run log: [jobs/fsh--tau2-official-full-Qwen3-4B-Instruct-2507-SFT-0714-000047/run_20260714_000047.log](jobs/fsh--tau2-official-full-Qwen3-4B-Instruct-2507-SFT-0714-000047/run_20260714_000047.log).

- **official runner full Pass@4 — `Qwen3.5-4B`, test split** (`pt-vmxrs8rf`,
  SUCCEEDED) — 1 GPU, tau2 official runner path, full test split,
  `NUM_TRIALS=4`, domains airline/retail/telecom. This is a thinking model; the
  wrapper uses `AGENT_MAX_TOKENS=8192` and no default presence penalty.
  Completed with overall pass@1=68.2%, pass@4=91.0%, infra_error_count=0.
  Wrapper: `examples/tau2-bench/eval/official/models/run_full_qwen3.5-4b.sh`. Summary
  target: `examples/tau2-bench/eval/official/outputs/Qwen3.5-4B/pass4_summary.json`.
  Run log: [jobs/fsh--tau2-official-full-qwen35-4b-0713-190824/run_20260713_190824.log](jobs/fsh--tau2-official-full-qwen35-4b-0713-190824/run_20260713_190824.log).

- **official runner full Pass@4 — `Qwen3-4B-tau2-grpo-v1`, test split**
  (`pt-z5l30f58`, SUCCEEDED) — 1 GPU, tau2 official runner path, full test
  split, `NUM_TRIALS=4`, domains airline/retail/telecom. This checkpoint uses
  the Qwen3-4B-Instruct-2507 non-thinking model family. Completed with overall
  pass@1=7.5%, pass@4=17.0%, infra_error_count=0. Wrapper:
  `examples/tau2-bench/eval/official/models/run_full_qwen3-4b-tau2-grpo-v1.sh`.
  Summary target:
  `examples/tau2-bench/eval/official/outputs/Qwen3-4B-tau2-grpo-v1/pass4_summary.json`.
  Run log: [jobs/fsh--tau2-official-full-qwen3-4b-tau2-grpo-v1-0713-190816/run_20260713_190816.log](jobs/fsh--tau2-official-full-qwen3-4b-tau2-grpo-v1-0713-190816/run_20260713_190816.log).

- **official runner full Pass@4 — `Qwen3-4B-Instruct-2507`, test split**
  (`pt-h4cpb2sv`, SUCCEEDED) — 1 GPU, tau2 official runner path, full test
  split, `NUM_TRIALS=4`, domains airline/retail/telecom. This is the
  non-thinking base model. Completed with overall pass@1=26.2%, pass@4=58.0%,
  infra_error_count=0. Wrapper:
  `examples/tau2-bench/eval/official/models/run_full_qwen3-4b-instruct-2507.sh`.
  Summary target:
  `examples/tau2-bench/eval/official/outputs/Qwen3-4B-Instruct-2507/pass4_summary.json`.
  Run log: [jobs/fsh--tau2-official-full-qwen3-4b-instruct-2507-0713-190808/run_20260713_190808.log](jobs/fsh--tau2-official-full-qwen3-4b-instruct-2507-0713-190808/run_20260713_190808.log).

- **official runner smoke raw-adapter v2 — `Qwen3-4B-Instruct-2507`, test split**
  (`pt-xwffu725`, SUCCEEDED) — 1 GPU, tau2 official runner path,
  `TASK_SPLIT=test`, `NUM_TASKS=1`, `NUM_TRIALS=1`, domains
  airline/retail/telecom. Agent calls sglang raw `/generate` and adapts Qwen
  native tool-call text into tau2 `AssistantMessage`. This retry uses the
  timestamped default `SAVE_PREFIX` to avoid tau2's interactive resume prompt.
  All three domains completed through the official `HalfDuplexAgent` runner
  with `infra_error_count=0`; pass^1 was 0/3 for this base-model smoke.
  Wrapper: `examples/tau2-bench/eval/official/run_eval.sh`. Summary target:
  `examples/tau2-bench/eval/official/outputs/Qwen3-4B-Instruct-2507/summary.json`.
  Run log: [jobs/fsh--tau2-official-test-smoke-raw-adapter-v2-0713-174432/run_20260713_174432.log](jobs/fsh--tau2-official-test-smoke-raw-adapter-v2-0713-174432/run_20260713_174432.log).

- **official runner smoke raw-adapter — `Qwen3-4B-Instruct-2507`, test split**
  (`pt-zo3qo6u5`, FAILED) — 1 GPU, tau2 official runner path,
  `TASK_SPLIT=test`, `NUM_TASKS=1`, `NUM_TRIALS=1`, domains
  airline/retail/telecom. Agent calls sglang raw `/generate` and adapts Qwen
  native `<tool_call>` output into tau2 `AssistantMessage`. Failed before
  running simulations because tau2 found an existing `save_to` results file and
  prompted for interactive resume, causing EOF in the batch job. Fixed by making
  the default `SAVE_PREFIX` timestamped. Wrapper:
  `examples/tau2-bench/eval/official/run_eval.sh`. Summary target:
  `examples/tau2-bench/eval/official/outputs/Qwen3-4B-Instruct-2507/summary.json`.
  Run log: [jobs/fsh--tau2-official-test-smoke-raw-adapter-0713-173444/run_20260713_173444.log](jobs/fsh--tau2-official-test-smoke-raw-adapter-0713-173444/run_20260713_173444.log).

- **official runner smoke retry — `Qwen3-4B-Instruct-2507`, test split**
  (`pt-0kd5nxfa`, stopped) — 1 GPU, tau2 official runner path,
  `TASK_SPLIT=test`, `NUM_TASKS=1`, `NUM_TRIALS=1`, domains
  airline/retail/telecom. Stopped before completion after deciding to replace
  the chat/completions path with a raw `/generate` adapter. Wrapper:
  `examples/tau2-bench/eval/official/run_eval.sh`. Summary target:
  `examples/tau2-bench/eval/official/outputs/Qwen3-4B-Instruct-2507/summary.json`.
  Run log: [jobs/fsh--tau2-official-test-smoke-retry-0713-173056/run_20260713_173056.log](jobs/fsh--tau2-official-test-smoke-retry-0713-173056/run_20260713_173056.log).

- **official runner smoke — `Qwen3-4B-Instruct-2507`, test split** (`pt-f7hli2xf`,
  SUCCEEDED with eval infra errors) — 1 GPU, tau2 official runner path,
  `TASK_SPLIT=test`, `NUM_TASKS=1`, `NUM_TRIALS=1`, domains
  airline/retail/telecom. Job infrastructure and sglang startup succeeded, but
  all three simulations ended as tau2 infrastructure errors because the
  OpenAI-compatible response had neither `content` nor `tool_calls`. Superseded
  by `pt-0kd5nxfa`. Run log:
  [jobs/fsh--tau2-official-test-smoke-0713-172447/run_20260713_172447.log](jobs/fsh--tau2-official-test-smoke-0713-172447/run_20260713_172447.log).

> **Re-run with corrected eval config (most recent; supersedes the entries
> below).** New defaults: agent `temperature=0.6`, user-sim `temperature=0.0`,
> Qwen3.5-4B **thinking ON** (`--max-new-tokens 8192 --presence-penalty 1.5`),
> and a **prose-as-respond** fallback in `actions.py::parse_action` (lifts a
> plain-prose reply — common for thinking/base models — into a `respond` action
> instead of failing the parse). The three runs below are this config. The older
> entries further down used temp 0.8/0.7, `--disable-thinking`, and strict
> parsing; their numbers are not comparable.

- **full Pass@4 re-run — `Qwen3.5-4B` (thinking ON)** (`pt-5jat8jfb`, SUCCEEDED) —
  1 GPU. Thinking by default; `--max-new-tokens 8192 --top-p 0.95
  --presence-penalty 1.5`. **Overall: pass@1=39.0%, pass@4=73.0%.** Per-domain:
  airline 55.0%/75.0%, retail 25.0%/60.0%, telecom 45.0%/85.0%. A large gain over
  the prior non-thinking run (13%/36%) and ahead of the tau-trained GRPO
  checkpoint — thinking is the right mode for Qwen3.5 on tau2.
  Wrapper: `examples/tau2-bench/run_full_qwen3.5-4b.sh`. Report →
  `examples/tau2-bench/outputs/eval/Qwen3.5-4B/pass4.json`.
  Run log: [jobs/fsh--examples-tau2-bench-run_full_qwen3.5-4b-0712-193347/run_20260712_193347.log](jobs/fsh--examples-tau2-bench-run_full_qwen3.5-4b-0712-193347/run_20260712_193347.log).

- **full Pass@4 re-run — `Qwen3-4B-tau2-grpo-v1`** (`pt-pjtcahl7`, SUCCEEDED) —
  1 GPU. Non-thinking (no `<think>` in template). **Overall: pass@1=32.0%,
  pass@4=56.0%.** Per-domain: airline 35.0%/45.0%, retail 30.0%/55.0%, telecom
  32.5%/62.5%. Close to the prior run (34%/62%); the prose-as-respond fix is a
  no-op here (the checkpoint always emits `<tool_call>`).
  Wrapper: `examples/tau2-bench/run_full_qwen3-4b-tau2-grpo-v1.sh`. Report →
  `examples/tau2-bench/outputs/eval/Qwen3-4B-tau2-grpo-v1/pass4.json`.
  Run log: [jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-tau2-grpo-v1-0712-193348/run_20260712_193348.log](jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-tau2-grpo-v1-0712-193348/run_20260712_193348.log).

- **full Pass@4 re-run — `Qwen3-4B-Instruct-2507`** (`pt-c81u025s`, SUCCEEDED) —
  1 GPU. Non-thinking base model. **Overall: pass@1=24.0%, pass@4=48.0%.**
  Per-domain: airline 20.0%/55.0%, retail 30.0%/47.5%, telecom 20.0%/45.0%. A big
  jump from the prior 5%/22%: the prose-as-respond fix stopped penalizing the
  base model's plain-prose replies (it replies in prose, not tool calls).
  Wrapper: `examples/tau2-bench/run_full_qwen3-4b-instruct-2507.sh`. Report →
  `examples/tau2-bench/outputs/eval/Qwen3-4B-Instruct-2507/pass4.json`.
  Run log: [jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-instruct-2507-0712-193347/run_20260712_193347.log](jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-instruct-2507-0712-193347/run_20260712_193347.log).

- **full Pass@4 — `Qwen3.5-4B`** (`pt-jjr38v44`, SUCCEEDED) — 1 GPU. Qwen3.5-4B is a
  **multimodal (vision+text), hybrid-arch, thinking** model
  (`Qwen3_5ForConditionalGeneration`). Evaluated **text-only** (tau2 sends no
  images; sglang serves it with the vision encoder idle — the container sglang
  already supports `qwen3_5`). Sampling matches the Qwen3 runs exactly (temp
  0.8 / top_p 1.0 / top_k 20 / k=4 / max_steps 100 / max_new_tokens 1200) **plus
  `--disable-thinking`** (`enable_thinking=False`): with thinking ON, 40% of
  episodes died in `parse_error` (non-convergent `<think>` rambled about a
  nonexistent `send_message` tool and stopped without a tool call); disabling
  thinking makes the model emit the tool call directly and matches the
  non-thinking Qwen3 models for a fair comparison. With thinking off, only 10/400
  attempts were `parse_error` (2.5%). **Overall: pass@1=13%, pass@4=36%.**
  Per-domain: airline 15%/30%, retail 10%/18%, telecom 15%/57% (pass@1 / pass@4).
  vs base 5%/22% → +8 pp pass@1, +14 pp pass@4; but below the tau-trained GRPO
  checkpoint (34%/62%) — task-specific RL still beats the newer arch alone.
  Wrapper: `examples/tau2-bench/run_full_qwen3.5-4b.sh`.
  Report → `examples/tau2-bench/outputs/eval/Qwen3.5-4B/pass4.json`.
  Run log: [jobs/fsh--examples-tau2-bench-run_full_qwen3.5-4b-0712-112834/run_20260712_112834.log](jobs/fsh--examples-tau2-bench-run_full_qwen3.5-4b-0712-112834/run_20260712_112834.log).

- **smoke** (`pt-3ojqjgxv`, SUCCEEDED) — 1 GPU, base `Qwen3-4B-Instruct-2507`,
  proxy user-sim `gpt-4.1-mini`, `--max-tasks-per-domain 1 --num-samples 1`.
  End-to-end run confirmed: sglang served the policy, all 3 domains evaluated,
  report written to `examples/tau2-bench/outputs/eval/tau2_eval.json`.
  `pass_at_1 = 0/3` (airline 25 steps, retail 10 steps — healthy episodes; telecom
  `parse_error` on turn 1 — base model didn't emit `<tool_call>`, handled cleanly).
  Expected for an untrained base model at k=1; infra is sound.
  Run log: [jobs/fsh--examples-tau2-bench-run_eval-0711-195455/run_20260711_195455.log](jobs/fsh--examples-tau2-bench-run_eval-0711-195455/run_20260711_195455.log).

- **full Pass@4 — `Qwen3-4B-tau2-grpo-v1`** (`pt-j4128zbs`, FAILED) — 1 GPU, the
  GRPO checkpoint (trained from `Qwen3-4B-Instruct-2507`), proxy user-sim
  `gemini-2.5-flash`, all 3 domains × full test split × `--num-samples 4`.
  Reached the end of telecom (~560 episodes) then crashed: `httpx.ReadError` on a
  sglang `/generate` call (`eval.py:182`) — a transient dropped keep-alive
  connection against a healthy sglang server. Because `eval.py` then wrote the
  report only at the very end and had no retry, **all results were lost** (no
  `pass4.json`). Fixed `eval.py` (retry transient transport errors; atomic
  incremental write per task) and resubmitted as `pt-t7u66jtq`.
  Wrapper: `examples/tau2-bench/run_full_qwen3-4b-tau2-grpo-v1.sh`.
  Report → `examples/tau2-bench/outputs/eval/Qwen3-4B-tau2-grpo-v1/pass4.json`.
  Failed run log: [jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-tau2-grpo-v1-0711-210252/run_20260711_210252.log](jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-tau2-grpo-v1-0711-210252/run_20260711_210252.log).
  Retry (`pt-t7u66jtq`, SUCCEEDED, with the eval.py robustness fix) — **Overall:
  pass@1=34%, pass@4=62%.** Per-domain: airline 25%/50%, retail 40%/67.5%,
  telecom 32.5%/62.5% (pass@1 / pass@4). vs the base model's 5%/22% overall —
  GRPO lifted pass@1 by +29 pp and pass@4 by +40 pp (retail +40/+55, telecom
  +27.5/+40). Run log: [jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-tau2-grpo-v1-0711-230717/run_20260711_230717.log](jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-tau2-grpo-v1-0711-230717/run_20260711_230717.log).

- **full Pass@4 — `Qwen3-4B-Instruct-2507`** (`pt-nm2duwk2`, SUCCEEDED) — 1 GPU, base
  model, proxy user-sim `gemini-2.5-flash`, all 3 domains × full test split ×
  `--num-samples 4` (100 tasks). Paired baseline for the GRPO checkpoint.
  **Overall: pass@1=5%, pass@4=22%.** Per-domain: airline 15%/40%, retail
  0%/12.5%, telecom 5%/22.5% (pass@1 / pass@4). Ran the pre-fix `eval.py` but
  reached the final write cleanly (no crash).
  Wrapper: `examples/tau2-bench/run_full_qwen3-4b-instruct-2507.sh`.
  Report → `examples/tau2-bench/outputs/eval/Qwen3-4B-Instruct-2507/pass4.json`.
  Run log: [jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-instruct-2507-0711-210258/run_20260711_210258.log](jobs/fsh--examples-tau2-bench-run_full_qwen3-4b-instruct-2507-0711-210258/run_20260711_210258.log).
