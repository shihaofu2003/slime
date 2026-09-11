# vitabench-qwen35-smoke

Purpose: install and validate all VitaBench dependencies in a tau2-compatible
isolated environment, then run a one-task VitaBench smoke with local
Qwen3.5-4B and the existing remote API for user/evaluator calls.

Name: `vitabench-qwen35-smoke`.

## Jobs

- `pt-h1oq0wrh` / dependency and coexistence probe: installed all 29 upstream
  requirements and passed the joint imports, then failed because a global
  `pip check` included known base-image metadata gaps:
  [run log](jobs/fsh-vitabench-deps-probe-0731-230256/run_20260731_230256.log).
- `pt-ph7e56j5` / dependency and coexistence probe v2: checks the Vita overlay
  against the pre-install conflict baseline while retaining all other checks.
  Succeeded: 29/29 requirements, tau2 shared pins, joint imports, task loading,
  and both CLIs passed with no new dependency conflicts:
  [run log](jobs/fsh-vitabench-deps-probe-v2-0731-230944/run_20260731_230944.log).
- `pt-u0dflfuy` / Qwen3.5-4B delivery smoke: local structured tool call,
  existing remote API, and Vita delivery task `10711001`. Succeeded end to end;
  the task reached `max_steps` at 30 with reward 0 after 14 agent tool calls:
  [run log](jobs/fsh-vitabench-qwen35-delivery-0731-231700/run_20260731_231700.log).
