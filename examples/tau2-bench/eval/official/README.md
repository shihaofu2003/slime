# Official tau2-bench eval

This eval path follows tau2-bench's recommended interface:

1. Register a `HalfDuplexAgent` factory.
2. Build `TextRunConfig` for each domain.
3. Run tau2's official runner and metrics.

The runner does not step `AgentGymEnv` manually and does not compute pass^k
itself. Results are written by tau2 under `data/simulations/<save_to>/`, and a
small slime summary is written under this directory's `outputs/` by default.

This is the default tau2-bench eval path in slime. The archived first-pass eval
test lives in `../legacy/` for comparison only.

## Commands

Smoke:

```bash
bash scripts/submit.sh --experiment tau2-eval --gpus 2 \
  examples/tau2-bench/eval/official/run_eval.sh
```

Full Pass@4:

```bash
bash scripts/submit.sh --experiment tau2-eval --gpus 2 \
  examples/tau2-bench/eval/official/models/run_full_qwen3-4b-instruct-2507.sh
```

Strict-single-v1 SFT or RL, formal seed 300/301:

```bash
bash examples/tau2-bench/eval/official/models/run_full_tau2_agent_single_call_v1_user_stop_parser.sh sft 300
bash examples/tau2-bench/eval/official/models/run_full_tau2_agent_single_call_v1_user_stop_parser.sh rl 301
```

These wrappers run all 100 test tasks with four trials and record pass@1,
pass@4(any), pass^4, multi-call output attempts, action/DB accuracy, and
`max_steps` diagnostics in the summary.

Eight-GPU asynchronous Agent/User pool:

```bash
bash scripts/submit.sh --experiment tau2-eval-qwen36-user-async-timed --gpus 8 \
  examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh
```

This starts two TP1 Agent workers and three TP2 User workers. The Agent router
uses `cache_aware`; the User router uses `round_robin` so all three expensive
User replicas receive traffic. Airline, retail, and telecom run in independent
processes with initial trajectory concurrency `2:2:5` and a global limit of 9.
When a domain completes, its slots are redistributed to unfinished domains.
The summary includes allocation events, Agent/User/environment time shares,
and request latency percentiles.

Qwen3.5 thinking smoke:

```bash
bash scripts/submit.sh --experiment tau2-eval --gpus 2 \
  examples/tau2-bench/eval/official/models/smoke_qwen3.5-4b.sh
```

## Configuration

Important environment variables:

- `MODEL_PATH` - local policy checkpoint.
- `DOMAINS` - comma-separated domains, default `airline,retail,telecom`.
- `TASK_SPLIT` - task split, default `test`.
- `NUM_TASKS` - tasks per domain. The smoke script defaults to `1`; full
  wrappers unset it.
- `NUM_TRIALS` - trials per task. Full wrappers set this to `4`.
- `USER_SGLANG` - start a local user simulator sglang server, default `1`.
- `USER_MODEL_PATH` - local user simulator checkpoint, default
  `/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3-4B-Instruct-2507`.
- `USER_MODEL` - OpenAI-compatible user simulator model name, default
  `basename(USER_MODEL_PATH)`.
- `AGENT_CUDA_VISIBLE_DEVICES` / `USER_CUDA_VISIBLE_DEVICES` - default `0` and
  `1`.
- `AGENT_REPLICA_CUDA_GROUPS` / `USER_REPLICA_CUDA_GROUPS` - semicolon-separated
  replica GPU groups, for example `0;1;2;3` and `4,5;6,7`. When set, the runner
  starts one SGLang worker per group behind a dedicated router.
- `PARALLEL_DOMAINS` - run selected domains in independent processes, default
  `0`; `MAX_CONCURRENCY` remains the per-domain trajectory concurrency.
- `DOMAIN_CONCURRENCY` - optional comma-separated initial concurrency, for
  example `airline:2,retail:2,telecom:5`. When unset, every selected domain uses
  `MAX_CONCURRENCY`.
- `GLOBAL_CONCURRENCY` - global trajectory limit used by elastic domain
  scheduling; the initial domain allocation must sum to this value.
- `BORROW_COMPLETED_DOMAIN_SLOTS` - when `1`, a successfully completed domain's
  slots are redistributed among unfinished domains; default `0`.
- `ROUTER_POLICY` - compatibility default for both routers, default
  `cache_aware`.
- `AGENT_ROUTER_POLICY` / `USER_ROUTER_POLICY` - per-router overrides. The
  Qwen3.6 asynchronous wrapper uses `cache_aware` / `round_robin`.
- `PORT` / `USER_PORT` - default `30000` and `30001`.
- `USER_TOP_P` / `USER_MAX_TOKENS` - optional user sampling controls;
  `USER_MAX_TOKENS` defaults to `512`.
- `TAU2_USER_API_BASE` - optional external/local user simulator base URL when
  `USER_SGLANG=0`.
- `AGENT_SGLANG_EXTRA_ARGS` / `USER_SGLANG_EXTRA_ARGS` - override sglang
  parser/server flags when needed. `SGLANG_EXTRA_ARGS` remains an agent-side
  compatibility alias.
- `AGENT_EXTRA_BODY_JSON` - optional raw sglang sampling params such as
  `{"presence_penalty": 1.5}`. The default presence penalty is `0.0`.
- `RUN_STAMP` / `SAVE_PREFIX` - output run naming. By default, `SAVE_PREFIX`
  includes a timestamp so repeated smoke submissions do not hit tau2's
  interactive resume prompt.

By default the job starts two local sglang services. The agent calls raw
`/generate` and adapts Qwen native `<tool_call>` text into tau2
`AssistantMessage` objects. A timing-only subclass of the official tau2
`user_simulator` calls the user service through LiteLLM at
`/v1/chat/completions`; its prompts, tools, and generation behavior are
unchanged.
Use `USER_SGLANG=0` to fall back to an external OpenAI-compatible user
simulator configured by `.env` or `TAU2_USER_API_BASE`.

Thinking-capable models such as Qwen3.5 use their native thinking template and
get a larger default generation budget in their wrappers; non-thinking Qwen3
models are unaffected.

Validated smoke: `pt-xwffu725` ran `Qwen3-4B-Instruct-2507` on the `test` split
with one task per domain and one trial. Airline, retail, and telecom all
completed with `infra_error_count=0`.
