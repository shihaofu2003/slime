# tau2-qwen35

**Purpose:** diagnostic smoke runs that stood up tau2-bench evaluation for
**Qwen3.5-4B** — a multimodal (vision+text), hybrid-arch, *thinking* model
(`Qwen3_5ForConditionalGeneration`) — before launching the full Pass@4 run
(which lives under experiment `tau2-eval`). All runs are 1 GPU, 1 task/domain ×
1 sample, user sim `gemini-2.5-flash`.

Code: [`examples/tau2-bench/`](../../../examples/tau2-bench/) (`smoke_qwen3.5-4b.sh`,
`run_full_qwen3.5-4b.sh`).

## Tasks

- **smoke 1 — default thinking** (`pt-cohmizf6`) — bare run with no parser
  fix. All 3 tasks `parse_error`: "Unexpected text outside `<tool_call>` block"
  — the model emits `<think>`/reasoning prose before the tool call, which the
  old strict parser rejected. Confirmed sglang loads `qwen3_5` text-only (8.6 GB,
  vision encoder idle).
  Run log: [jobs/fsh--examples-tau2-bench-smoke_qwen3.5-4b-0712-103002/run_20260712_103002.log](jobs/fsh--examples-tau2-bench-smoke_qwen3.5-4b-0712-103002/run_20260712_103002.log).

- **smoke 2 — raw-output logging** (`pt-3bqb5u48`) — added a parse-failure dump
  to `eval.py`. Showed the model's output is verbose natural-language reasoning
  ("The user wants to book a flight. According to the policy…") before any
  `<tool_call>`.
  Run log: [jobs/fsh--examples-tau2-bench-smoke_qwen3.5-4b-0712-103553/run_20260712_103553.log](jobs/fsh--examples-tau2-bench-smoke_qwen3.5-4b-0712-103553/run_20260712_103553.log).

- **smoke 3 — tolerant multi-format parser** (`pt-p0obq1d7`, thinking ON,
  `max_new_tokens=8192`) — rewrote `actions.py::parse_action` to ignore
  `<think>`/prose around the block and accept both Qwen3 JSON and Qwen3.5
  qwen3_coder `<function>/<parameter>` formats. Worked: 55 tool calls, all 3
  episodes completed (`pass@1=0/3`). But the subsequent thinking-ON *full* run
  (`pt-k61qe1e2`, killed) hit ~40% `parse_error` — `<think>` non-convergently
  rambled about a nonexistent `send_message` tool and stopped with no tool call.
  Run log: [jobs/fsh--examples-tau2-bench-smoke_qwen3.5-4b-0712-104637/run_20260712_104637.log](jobs/fsh--examples-tau2-bench-smoke_qwen3.5-4b-0712-104637/run_20260712_104637.log).

- **smoke 4 — `--disable-thinking`** (`pt-3avof9n4`) — added
  `eval.py --disable-thinking` (`enable_thinking=False`): model emits the tool
  call directly. Clean: 0 `parse_error`, all episodes completed, `pass@1=1/3`
  (telecom solved). This is the config the full run uses.
  Run log: [jobs/fsh--examples-tau2-bench-smoke_qwen3.5-4b-0712-112252/run_20260712_112252.log](jobs/fsh--examples-tau2-bench-smoke_qwen3.5-4b-0712-112252/run_20260712_112252.log).

## Conclusion

Qwen3.5-4B evaluates **text-only** under the shared engine; the needed changes
are (a) `parse_action` tolerates thinking/prose + both tool-call formats, and
(b) `--disable-thinking` avoids non-convergent-`<think>` parse failures and
keeps the run comparable to the non-thinking Qwen3 models. Both changes are
backward-compatible (Qwen3 JSON still parses first; the flag defaults off). Full
Pass@4 run: `pt-jjr38v44` under experiment `tau2-eval`.
