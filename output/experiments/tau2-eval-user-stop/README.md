# tau2-eval-user-stop

Purpose: evaluate tau2-bench official Pass@4 with the STOP-trained local user
simulator, and verify it ends conversations with `###STOP###` instead of driving
episodes to `max_steps`. This is the follow-up to `tau2-eval-user-sft`, where the
local user model almost never stopped.

Code: [`examples/tau2-bench/eval/official/`](../../../examples/tau2-bench/eval/official/).

## Setup

- User Megatron checkpoint:
  `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899`
- User HF target: `.../iter_0006899_hf`
- User served model name: `Qwen3-4B-tau2-user-sft-stop-iter0006899`
- Agent models: airline/retail/telecom, test split, 4 trials per task (400 sims each).

## Tasks

- `pt-1i9dl3zk` / `fsh-convert-user-sft-stop-iter-0006899-0715-153727` — HF
  conversion for the STOP-trained user model:
  [../tau2-hf-convert/jobs/fsh-convert-user-sft-stop-iter-0006899-0715-153727/run_20260715_153727.log](../tau2-hf-convert/jobs/fsh-convert-user-sft-stop-iter-0006899-0715-153727/run_20260715_153727.log).
- `pt-2wkrem3t` / `fsh-tau2-user-stop-agent-sft-multitool-0715-154112` — Pass@4 for
  `Qwen3-4B-tau2-agent-sft-multitool-iter0002413`:
  [jobs/fsh-tau2-user-stop-agent-sft-multitool-0715-154112/run_20260715_154112.log](jobs/fsh-tau2-user-stop-agent-sft-multitool-0715-154112/run_20260715_154112.log).
- `pt-nmhb49uk` / `fsh-tau2-user-stop-agent-areal-sft-0715-154112` — Pass@4 for
  `Qwen3-4B-tau2-sft-areal-strict-iter0000881`:
  [jobs/fsh-tau2-user-stop-agent-areal-sft-0715-154112/run_20260715_154112.log](jobs/fsh-tau2-user-stop-agent-areal-sft-0715-154112/run_20260715_154112.log).
- `pt-8nqh1j5h` / `fsh-tau2-user-stop-agent-qwen3-4b-instruct-0715-154113` — Pass@4
  for `Qwen3-4B-Instruct-2507`:
  [jobs/fsh-tau2-user-stop-agent-qwen3-4b-instruct-0715-154113/run_20260715_154113.log](jobs/fsh-tau2-user-stop-agent-qwen3-4b-instruct-0715-154113/run_20260715_154113.log).
- `pt-6dy67ya1` / `fsh-tau2-user-stop-agent-qwen3-5-4b-0715-154113` — Pass@4 for
  `Qwen3.5-4B`:
  [jobs/fsh-tau2-user-stop-agent-qwen3-5-4b-0715-154113/run_20260715_154113.log](jobs/fsh-tau2-user-stop-agent-qwen3-5-4b-0715-154113.log).

## Outputs

| agent model | wrapper | summary |
|---|---|---|
| Qwen3-4B-tau2-agent-sft-multitool-iter0002413 | `.../models/run_full_tau2_agent_sft_multitool_user_stop.sh` | `outputs/Qwen3-4B-tau2-agent-sft-multitool-iter0002413/user_stop_pass4_summary.json` |
| Qwen3-4B-tau2-sft-areal-strict-iter0000881 | `.../models/run_full_this_sft_user_stop.sh` | `outputs/Qwen3-4B-tau2-sft-areal-strict-iter0000881/user_stop_pass4_summary.json` |
| Qwen3-4B-Instruct-2507 | `.../models/run_full_qwen3-4b-instruct-2507_user_stop.sh` | `outputs/Qwen3-4B-Instruct-2507/user_stop_pass4_summary.json` |
| Qwen3.5-4B | `.../models/run_full_qwen3.5-4b_user_stop.sh` | `outputs/Qwen3.5-4B/user_stop_pass4_summary.json` |

## Results

All four jobs are DONE. `pass^k` is tau2 official `pass_hat_k` (sim-weighted
overall, %). `user_stop` = episodes ended by `###STOP###`/`###TRANSFER###`/
`###OUT-OF-SCOPE###`; `max_steps` = hit the turn guard; `err` =
`termination_error + termination_infrastructure_error`. Each run is 400 sims
(airline 80 / retail 160 / telecom 160).

### STOP-trained user vs the two baselines

The agents with an immutable gemini-API user baseline (same agent, gemini user).
Gemini rows are recomputed from the original simulation directories referenced
by `tau2-eval/README.md`; mutable `outputs/<agent>/pass4_summary.json` files were
later overwritten by a local base-Qwen user run.

| agent | user model | pass^1 | pass^4 | user_stop | max_steps | err |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | gemini | 26.2 | 6.0 | 395 | 0 | 5 |
| Qwen3-4B-Instruct-2507 | user_sft (old) | 1.8 | 0.0 | 25 | 370 | 5 |
| Qwen3-4B-Instruct-2507 | **user_stop (new)** | **18.0** | **4.0** | **391** | **8** | 1 |
| Qwen3.5-4B | gemini | 68.2 | 39.0 | 399 | 0 | 1 |
| Qwen3.5-4B | user_sft (old) | 0.8 | 0.0 | 15 | 385 | 0 |
| Qwen3.5-4B | **user_stop (new)** | **28.7** | **10.0** | **357** | **43** | 0 |

The two tau-SFT agents (no gemini baseline run; old→new only):

| agent | user model | pass^1 | pass^4 | user_stop | max_steps | err |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B-tau2-agent-sft-multitool | user_sft (old) | 3.2 | 1.0 | 43 | 299 | 58 |
| Qwen3-4B-tau2-agent-sft-multitool | **user_stop (new)** | **17.8** | **6.0** | **337** | **14** | 49 |
| Qwen3-4B-tau2-sft-areal-strict | user_sft (old) | 0.0 | 0.0 | 7 | 339 | 54 |
| Qwen3-4B-tau2-sft-areal-strict | **user_stop (new)** | **11.5** | **3.0** | **320** | **50** | 30 |

Per-domain (pass^1 / user_stop / max_steps), STOP-trained user:

| agent | airline | retail | telecom |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 16.2 / 79 / 1 | 29.4 / 160 / 0 | 7.5 / 152 / 7 |
| Qwen3.5-4B | 46.2 / 80 / 0 | 36.9 / 160 / 0 | 11.9 / 117 / 43 |
| multitool | 32.5 / 80 / 0 | 20.0 / 159 / 0 | 8.1 / 98 / 14 |
| areal-strict | 21.2 / 76 / 2 | 18.1 / 155 / 3 | 0.0 / 89 / 45 |

Headline: `user_stop` rose from ~1–7% of episodes to ~80–98% on airline/retail,
and `max_steps` collapsed from ~85–95% to ~0–3%. This proves the STOP data fix
worked, but agent success still trails the immutable gemini baseline: instruct
18.0 vs 26.2 pass^1, and Qwen3.5 28.7 vs 68.2. The largest residual gap is
telecom, where the local user server still lacks a tool-call parser (see
Analysis).

## Analysis: code problem or data problem?

**Data problem, not code.** The tau2 official eval path is correct, and the
STOP-trained user model confirms it.

**Evidence the eval code is fine.** With a gemini-API user the same code path
(`UserSimulator` → `litellm.completion(tools=…)` → `message.tool_calls`) always
terminated correctly (`user_stop` 361–379 / 400) and telecom user-side tools
executed. So the runner, the half-duplex orchestrator, and the termination logic
are not the cause.

**Root cause: the original user-SFT data had no `###STOP###` targets.** The
`issue-stats` diagnostic (`fsh-tau2-user-sft-issue-stats-0715-015710`) measured
the data the old checkpoint (`iter_0006637`) was trained on:

- `areal_tau2_user_sft_no_thinking.jsonl`: `target_stop = 0`.
- `mixed_user_sft_no_thinking.jsonl`: `target_stop = 63` (all from APIGen; AReaL
  contributed 0), but `finalish_text = 4862` — task-completion turns stored as
  plain text ("Thank you for your help. … Have a good day!") rather than as
  `###STOP###`.

So the model was trained to keep talking past completion; it learned exactly the
loop we saw in `tau2-eval-user-sft` (an airline episode ran 201 messages, reward
0, ending at `max_steps` after the agent had already said "have a wonderful day"
repeatedly). This is a data-conversion defect in the original
`prepare_user_sft_data.py`, which dropped the terminal turn instead of converting
it to a `###STOP###` target.

**The fix was a data fix.** `prepare_user_sft_data.py::convert_areal_terminal_stop_row`
now appends one `###STOP###` target per successful AReaL dialog
(`areal_target_stop = 2097`; data rows 28976 → 31073). The `data-stop` job
(`fsh-tau2-user-sft-data-stop-0715-121924`) regenerated the data; the `stop`
training (`fsh-tau2-user-sft-stop-0715-122152`, `iter_0006899`) used it. No eval
or runner code changed — and termination recovered.

**Residual gaps (also data/serving, not eval-code):**

- `Qwen3.5-4B` still trails the immutable gemini baseline (28.7 vs 68.2 pass^1)
  even though `user_stop`/`max_steps` on airline/retail are now clean. The
  remaining gap is user-model *quality*: a weaker customer (info disclosure,
  scenario adherence) than gemini. Likely contributors, all on the data side —
  (1) scenario distribution shift: trained on the thin `areal_scenario` metadata
  ("reason_for_call / task_id / difficulty") but evaluated on the real, rich
  `task.user_scenario`; (2) system-prompt shift: trained with a custom
  `USER_TOOL_FORMAT` preamble + `simulation_guidelines.md`, evaluated with tau2's
  official `simulation_guidelines[_tools].md` and no preamble; (3) 4B vs gemini
  capability.
- Telecom-specific: user tool calls still leak as literal `<tool_call>` text
  (3279 occurrences in the old telecom run) because the user sglang server is
  launched without `--tool-call-parser`, so `TelecomUserTools` are never invoked
  as structured calls. This shows up as residual telecom `max_steps`/`err`
  (multitool telecom `err=48`; areal-strict telecom `max_steps=45`). Impact on
  pass^1 is large: immutable gemini is 85.6/19.4 on telecom (Qwen3.5/instruct),
  while parser-off local user is 11.9/7.5. This is real and worth fixing for
  telecom reliability and evaluation validity.

**Next steps.** (1) Close the scenario/prompt distribution gap in
`prepare_user_sft_data.py` (use the real `user_scenario`; align guidelines file
per domain; drop the custom preamble). (2) Add `--tool-call-parser qwen25` to the
user sglang server in `run_eval.sh` so telecom user tools execute. (3) Drop
`content_and_tool` mixed targets to match tau2's "message XOR tool call" user
protocol.
