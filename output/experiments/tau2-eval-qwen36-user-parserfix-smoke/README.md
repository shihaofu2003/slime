# tau2-eval-qwen36-user-parserfix-smoke

Purpose: validate the Qwen3.6-27B User simulator with non-thinking text-only
serving and the Qwen3-Coder tool-call parser before rerunning the full official
evaluation.

## Tasks

- Job `12134`: Airline, Retail, and Telecom; five test tasks per domain, one
  trial, seed 300, eight GPUs —
  [run log](jobs/12134-qwen36-user-parserfix-textonly-3domain-5each-0902-095040349/run_*_20260902_095040349.log),
  [summary](eval/three-domain-smoke-parserfix-textonly-5each/seed300_0902_015208_summary.json).
  Submitted with `enable_thinking=false`, `--language-only`,
  `--reasoning-parser qwen3`, and `--tool-call-parser qwen3_coder`.

## Result

- Job state: `SUCCEEDED`; 15/15 simulations completed with zero infrastructure
  errors and zero `max_steps` terminations.
- All three User workers reported `language_only=True`,
  `reasoning_parser='qwen3'`, and `tool_call_parser='qwen3_coder'`.
- The Telecom trajectories contain 89 native User tool calls and 89 matching
  User tool results. They contain zero literal `<tool_call>` tags, zero
  `<think>` tags, and the User-worker logs contain zero
  `Failed to parse JSON part` messages.
- Smoke pass@1 was 80% Airline, 80% Retail, and 60% Telecom. These five-task
  samples validate execution only and are not a model-quality comparison.
