# tau2-bench examples

This directory has been reorganized for tau2-bench evaluation and future
training work in slime. The first working eval implementation is archived under
`eval/legacy/`; new eval runs use the official tau2 runner path under
`eval/official/`.

## Layout

- `eval/official/` - recommended evaluation path. It follows tau2-bench's
  official agent and runner APIs: `HalfDuplexAgent`, `TextRunConfig`, and
  `run_domain`.
- `eval/legacy/` - archived first-pass eval test. It drove `AgentGymEnv`
  directly and kept custom prompt, parser, and metric code. Keep it for result
  comparison only.
- `sft/`, `rft/`, `rl/`, `opd/` - reserved for the next tau2-bench training
  integrations.
- `shared/` - small utilities that are genuinely shared across stages.

## Official eval

The official eval path was rewritten to follow tau2-bench's recommended API:
register a `HalfDuplexAgent`, build `TextRunConfig`, and call tau2's
`run_domain`. It does not step `AgentGymEnv` directly and does not own custom
pass^k aggregation.

Smoke run:

```bash
bash scripts/submit.sh --experiment tau2-eval --gpus 2 \
  examples/tau2-bench/eval/official/run_eval.sh
```

Full Pass@4 run:

```bash
bash scripts/submit.sh --experiment tau2-eval --gpus 2 \
  examples/tau2-bench/eval/official/models/run_full_qwen3-4b-instruct-2507.sh
```

The official eval uses the `test` split unless `TASK_SPLIT` is set.
It launches local sglang servers for the policy model and the user simulator.
The agent calls raw `/generate` and adapts Qwen native tool-call text into tau2
`AssistantMessage` objects. The official tau2 `user_simulator` calls the second
server through LiteLLM at `/v1/chat/completions`. Set `USER_SGLANG=0` to route
the user simulator through `tau2-bench/.env` or another OpenAI-compatible base
URL instead.

Qwen3.5 thinking models get a larger generation budget in their wrappers;
Qwen3 non-thinking models use the standard budget.

The latest smoke on the official path was `pt-xwffu725`:
`Qwen3-4B-Instruct-2507`, `TASK_SPLIT=test`, one task per domain, one trial.
It completed airline/retail/telecom with `infra_error_count=0`; model pass^1
was 0/3.

## Legacy eval

The previous implementation is archived under `eval/legacy/` without deletion.
It is useful for comparing old runs, but new evaluations should use
`eval/official/`.
