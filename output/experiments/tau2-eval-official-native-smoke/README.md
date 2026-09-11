# tau2-eval-official-native-smoke

Purpose: validate the Tau2 official native Agent evaluation path with the full
regression preflight and a four-domain, three-task, one-trial GPU smoke.

## Tasks

- Cluster preflight `12239`: official evaluation tests, RL rollout regression,
  router import/help checks, Python compilation, and shell syntax checks —
  [run log](jobs/12239-tau2-official-native-preflight-0902-121825958/run_*_20260902_121825958.log).
- Four-domain GPU smoke `12240`: Airline, Retail, Telecom, and Banking
  Knowledge, three tasks per domain, one trial, seed 300, eight GPUs —
  [run log](jobs/12240-examples-tau2-bench-eval-official-models-run_qwen3_4b_qwen36_user_async_four_domain-0902-122051833/run_*_20260902_122051833.log),
  [summary](../tau2-eval-qwen36-user-four-domain/eval/four-domain-smoke-bm25/seed300_0902_042631_summary.json).

## Acceptance

- Preflight passed 38 official-evaluation tests and the RL rollout regression;
  both jobs exited successfully.
- The smoke recorded `official-native`, `llm_agent`, and no protocol profile.
  Every domain completed 3 tasks / 3 simulations with zero infrastructure
  errors; overall completion was 12 / 12.
- Native Assistant tool calls have a matching Tool response ID in every saved
  trajectory. The Agent router handled evaluation requests only through
  `/v1/chat/completions`; the User remained non-thinking, language-only
  Qwen3.6 with the `qwen3_coder` tool parser.
