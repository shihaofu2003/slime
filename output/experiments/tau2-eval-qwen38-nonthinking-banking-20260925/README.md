# tau2-eval-qwen38-nonthinking-banking-20260925

Purpose: evaluate Qwen3.8-27B as a nonthinking Agent on official Banking Knowledge, retaining the SFT4505 expert evaluation settings and adapting GPU placement for the 27B model.

## Model support

The [official model card](https://huggingface.co/Qwen/Qwen3.8-27B#instruct-or-non-thinking-mode) supports `chat_template_kwargs.enable_thinking=false`. The local `/mnt/afs/models/Qwen3.8-27B/chat_template.jinja` implements this setting by closing the thinking block before generation. Prior cluster smoke 12725 successfully generated text and parsed tool calls with thinking disabled; its [run log](../../../../slime/output/experiments/qwen38-27b-inference-smoke/jobs/12725-qwen38-27b-xhigh-compat-smoke-0903-115952222/run_0_20260903_115952222.log) records both checks.

## Protocol

[Launcher](run_eval.sh) reuses the existing asynchronous official evaluation wrapper, matching the Banking settings of [SFT4505 expert evaluation](../tau2-domain-experts-sft4505-b128/eval_checkpoint.sh). Change Agent weights to `/mnt/afs/models/Qwen3.8-27B`, use its `qwen3_coder` tool parser, and explicitly disable thinking.

- All 97 official Banking Knowledge test tasks; 4 trials, seed 300, 388 simulations.
- Agent temperature 0.6, top_p 1.0, max_tokens 1200; max_steps 200, max_errors 10; BM25 retrieval.
- User remains Qwen3.6-27B nonthinking, temperature 0, max_tokens 512, context 65536.
- Banking concurrency remains 4. The global cap remains 9; only the Banking domain is active. Disable completed-domain slot borrowing, which requires multiple domains.
- Eight GPUs: two TP2 Agent replicas on GPUs 0–3 and two TP2 User replicas on GPUs 4–7. Preserve Agent/User memory fractions 0.85/0.90 and cache-aware/round-robin routing. Normal priority.

## Jobs

| Job | GPUs | Purpose | Run log |
|---|---:|---|---|
| 23973 (`pt-c7xdm54r`) | 8 | Qwen3.8-27B nonthinking Banking evaluation, submitted 2026-09-25 10:47 CST | [Log](jobs/23973-tau2-qwen38-nonthinking-banking-eval-0925-104741048/run_0_20260925_104741048.log) |

Status at 10:55 CST: RUNNING; all four TP2 model replicas and both routers are ready. Official Banking evaluation started at 10:53:58; 12/388 simulations are saved, all terminated with `user_stop`. The saved run settings confirm Agent temperature 0.6, top_p 1.0, max_tokens 1200, `enable_thinking=false`, seed 300, four trials, max_steps 200, max_errors 10, and BM25.

A launch-environment comparison confirmed unchanged sampling, User model, token/step limits, seed, trials, retrieval, memory fractions, and routing. Only Agent identity/format, the requested Banking-only scope, and GPU placement differ. Shell syntax and whitespace checks passed. The official Banking task file contains 97 tasks.

Outputs: `eval/banking-full-bm25-nonthinking/`; summary: `seed300_20260925_qwen38_nonthinking_seed300_summary.json`.
