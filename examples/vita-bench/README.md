# VitaBench integration

VitaBench runs in `/tmp/serviceagent-vitabench-venv`, layered over the cluster
image with system site-packages. The setup verifies all 29 dependencies declared
by Vita and keeps shared versions compatible with tau2 (`litellm==1.81.11`,
`PyYAML==6.0.3`, `tenacity==9.1.2`).

## Commands

Dependency and coexistence probe:

```bash
bash scripts/submit.sh --experiment vitabench-qwen35-smoke --with-vitabench \
  --name vitabench-deps-probe examples/vita-bench/probe_dependencies.sh
```

Qwen3.5-4B single-task smoke (local SGLang agent, existing remote `gpt-4.1`
credentials for the user simulator and evaluator):

```bash
bash scripts/submit.sh --experiment vitabench-qwen35-smoke --with-vitabench \
  --name vitabench-qwen35-delivery examples/vita-bench/run_qwen3_5_4b_smoke.sh
```

Full Qwen3.5-4B evaluation with the user-selected ChatAnywhere `gpt-4.1-ca`
user simulator and `gemini-2.5-flash` evaluator:

```bash
bash scripts/submit.sh --experiment vitabench-qwen35-full-eval \
  --with-vitabench --gpus 4 --name vitabench-qwen35-full-4gpu-gptca-gemini25 \
  examples/vita-bench/run_qwen3_5_4b_full_eval.sh -- --suite-parallelism 4
```

The full run covers four Chinese 100-task suites with four trials per task. It
uses fixed 10-task shards, resumes valid partial results, preserves corrupt
checkpoints, assigns one suite to each GPU at concurrency 4 (16 trajectories in
aggregate), caps aggregate remote API concurrency at 8, and writes a strictly
validated `summary.json` and `summary.csv`. Every successful local Qwen,
GPT-4.1 user-simulator, and Gemini evaluator API response is atomically
journaled; exact SFT JSONL exports and provenance/checksum indexes are generated
after the gate and after the full run. Only the local Qwen agent has thinking
enabled; `gpt-4.1-ca` is the non-thinking user simulator.
The published Qwen score of 22.0 is only a descriptive reference because its
full protocol is undisclosed and this run uses Gemini rather than the paper's
Claude evaluator.

Fresh local-role runs use separate schema-6 `artifacts-v4*` directories and
eight single-GPU BF16 SGLang instances: two thinking agents, two non-thinking
user simulators, and four non-thinking evaluators. Four workers retain 16
aggregate trajectories; their endpoint routes are `agent 0/1/0/1`,
`user 0/1/0/1`, and `evaluator 0/1/2/3`. Gate suites remain worker-pinned,
while full-evaluation shards are atomically claimed from a shared dynamic queue
so idle workers can take shards from any suite. Historical `artifacts-v2` and
`artifacts-v3*` results used thinking evaluators and remain untouched.

```bash
bash scripts/submit.sh --experiment vitabench-qwen35-local-role-eval \
  --with-vitabench --gpus 8 --name vitabench-qwen35-local-2a2u4e \
  examples/vita-bench/run_qwen3_5_4b_full_eval_local_roles.sh
```

Complete raw Qwen3-4B-Instruct-2507 evaluation with non-thinking Agent, User,
and Evaluator roles:

```bash
bash scripts/submit.sh \
  --experiment vitabench-qwen3-4b-instruct-2507-full-eval \
  --with-vitabench --gpus 8 --priority normal \
  --name vitabench-qwen3-4b-instruct-2507-full \
  examples/vita-bench/run_qwen3_4b_instruct_2507_full_eval_local_roles.sh
```

Run the isolated 16-trajectory gate, then the four-suite shard-0 A/B candidate:

```bash
bash scripts/submit.sh --experiment vitabench-qwen35-local-role-eval \
  --with-vitabench --gpus 8 --name vitabench-qwen35-local-2a2u4e-gate \
  examples/vita-bench/run_qwen3_5_4b_full_eval_local_roles.sh \
  -- --run-mode gate
bash scripts/submit.sh --experiment vitabench-qwen35-local-role-eval \
  --with-vitabench --gpus 8 --name vitabench-qwen35-local-2a2u4e-ab-shard0 \
  examples/vita-bench/run_qwen3_5_4b_full_eval_local_roles.sh \
  -- --run-mode ab --ab-shard-index 0
/tmp/serviceagent-vitabench-venv/bin/python \
  examples/vita-bench/compare_local_topology_ab.py \
  --baseline-artifact-dir output/experiments/vitabench-qwen35-local-role-eval/artifacts-v2 \
  --candidate-artifact-dir output/experiments/vitabench-qwen35-local-role-eval/artifacts-v3-ab-shard0 \
  --output output/experiments/vitabench-qwen35-local-role-eval/ab-shard0.json
```

The smoke reads the existing `tau2-bench/.env` without copying credentials into
the submitted script or model YAML. It checks local structured tool calls, the
remote API through Vita's client, one real delivery trajectory, result
redaction, and tau2/slime/tensorboard imports.

Replay the 6,349 persisted local-evaluator windows without rerunning Agent/User,
first with byte-identical frozen prompts and then with non-thinking rubric state
propagation:

```bash
bash scripts/submit.sh --experiment vitabench-evaluator-thinking-ab \
  --with-vitabench --gpus 8 --priority normal \
  --name vitabench-evaluator-thinking-ab \
  examples/vita-bench/run_evaluator_thinking_ab.sh
```

The runner strictly audits the schema-5 source and response journals, gates on
a two-task-per-suite pilot, checkpoints every response for resume, and emits a
task-cluster-bootstrap score comparison plus latency/token decision gates.
Invalid semantic attempts are atomically journaled; the recovery launcher uses
an immutable-base overlay so supplements cannot overwrite primary results.
The chained result measured Thinking on versus off at 23.0625% versus 21.7500%
Avg@4/pass@1, while evaluator latency improved 4.36× and estimated total
critical path improved 2.62×. This quality tradeoff is accepted: the default
Qwen3.6-27B Evaluator entry in `models_qwen35_full.yaml` now sets
`chat_template_kwargs.enable_thinking=false`. See the
[full A/B record](../../output/experiments/vitabench-evaluator-thinking-ab/README.md).
