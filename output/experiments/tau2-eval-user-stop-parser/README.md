# tau2-eval-user-stop-parser

**Purpose:** re-run the `tau2-eval-user-stop` evaluation with the user sglang
server's Qwen tool-call parser **enabled**, to verify whether telecom user-side
tools (`TelecomUserTools`) get executed instead of leaking `<tool_call>` text,
and to measure the metric impact. Everything else is identical to
`tau2-eval-user-stop` — same STOP-trained user model
`Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899`, same agents, test
split, 4 trials (400 sims each).

**Change:** `examples/tau2-bench/eval/official/run_eval.sh` now defaults the user
server to `--tool-call-parser qwen`. The agent server is untouched (raw
`/generate`, Python-side parse). Wrappers `run_full_*_user_stop_parser.sh` write
to distinct `user_stop_parser_*` artifacts so the `user_stop` baseline is kept.

## Results — official pass^k (sim-weighted, %)

`pass^k` is tau2 `pass_hat_k`. The gemini baseline is recomputed from the
original immutable simulation directories used by `tau2-eval/README.md`
(`tau2_official_<agent>_{domain}_test_4trials`), not from the mutable
`outputs/<agent>/pass4_summary.json`; that summary was later overwritten by a
local base-Qwen user run.

| agent | gemini | parser OFF | **parser ON** |
|---|---|---|---|
| Qwen3.5-4B — pass^1 / pass^4 | 68.2 / 39.0 | 28.7 / 10.0 | **47.2 / 19.0** |
| Qwen3-4B-Instruct-2507 — pass^1 / pass^4 | 26.2 / 6.0 | 18.0 / 4.0 | **23.5 / 6.0** |
| Qwen3-4B tau2 multitool SFT — pass^1 / pass^4 | — | 17.8 / 6.0 | **23.25 / 8.0** |

Termination (user_stop / max_steps / err), parser ON vs OFF:

| agent | user_stop OFF→ON | max_steps OFF→ON | err OFF→ON |
|---|---|---|---|
| Qwen3.5-4B | 357 → 381 | 43 → 18 | 0 → 1 |
| Qwen3-4B-Instruct-2507 | 391 → 388 | 8 → 10 | 1 → 2 |
| Qwen3-4B tau2 multitool SFT | 337 → 362 | 14 → 14 | 49 → 24 |

## Results — pass@k (any-trial, %)

Here `pass@k(any)` means that at least one of `k` sampled trials succeeds. For
a task with `c` successes among `n=4` observed trials, the estimator is
`1 - C(n-c,k) / C(n,k)`. Therefore `pass@1 = c/n`: it uses all four
exchangeable trials, not the arbitrarily numbered trial 0. `pass@4(any)` is 1
iff at least one of the four trials succeeds. Overall averages these per-task
values over 100 tasks (airline 20 / retail 40 / telecom 40). This table is
separate from tau-bench's official `pass^k`, which estimates the probability
that all `k` sampled trials succeed.

Overall:

| agent | gemini | parser OFF | **parser ON** |
|---|---|---|---|
| Qwen3.5-4B — pass@1 / pass@4(any) | 68.2 / 91.0 | 28.7 / 47.0 | **47.2 / 78.0** |
| Qwen3-4B-Instruct-2507 — pass@1 / pass@4(any) | 26.2 / 58.0 | 18.0 / 35.0 | **23.5 / 45.0** |
| Qwen3-4B tau2 multitool SFT — pass@1 / pass@4(any) | — | 17.8 / 36.0 | **23.25 / 46.0** |

Telecom:

| agent — telecom | gemini | parser OFF | **parser ON** |
|---|---|---|---|
| Qwen3.5-4B — pass@1 / pass@4(any) | 85.6 / 100.0 | 11.9 / 22.5 | **53.8 / 90.0** |
| Qwen3-4B-Instruct-2507 — pass@1 / pass@4(any) | 19.4 / 47.5 | 7.5 / 20.0 | **15.6 / 32.5** |
| Qwen3-4B tau2 multitool SFT — pass@1 / pass@4(any) | — | 8.1 / 22.5 | **20.0 / 47.5** |

## Telecom — metric effect

The parser fix targets telecom (the only domain with user tools). Parser ON
more than doubles telecom for instruct and quadruples it for Qwen3.5:

| agent — telecom | gemini | parser OFF | **parser ON** |
|---|---|---|---|
| Qwen3.5-4B — pass^1 / pass^4 / avg_reward | 85.6 / 57.5 / 0.86 | 11.9 / 0.0 / 0.12 | **53.8 / 15.0 / 0.54** |
| Qwen3-4B-Instruct-2507 — pass^1 / pass^4 / avg_reward | 19.4 / 2.5 / 0.19 | 7.5 / 0.0 / 0.07 | **15.6 / 0.0 / 0.16** |
| Qwen3-4B tau2 multitool SFT — pass^1 / pass^4 / avg_reward | — | 8.1 / 0.0 / 0.08 | **20.0 / 0.0 / 0.20** |

airline/retail are flat within 4-trial noise (they have no user tools, so the
parser does not affect them).

## Telecom — user tool execution verification

Count of user turns whose tool call was **executed as a structured tool call**
vs **leaked as literal `<tool_call>` text** (from each telecom `results.json`,
160 sims):

| agent | variant | executed | leaked |
|---|---|---:|---:|
| Qwen3.5-4B | parser OFF | 0 | 5544 |
| Qwen3.5-4B | parser ON | **1010** | 2423 |
| Qwen3-4B-Instruct-2507 | parser OFF | 0 | 1341 |
| Qwen3-4B-Instruct-2507 | parser ON | **1104** | 867 |
| Qwen3-4B tau2 multitool SFT | parser OFF | 0 | 2119 |
| Qwen3-4B tau2 multitool SFT | parser ON | **1715** | 1291 |

User-side tools went from **0 executed → 1010–1715 executed** across
all three agents.
Leakage dropped but is not zero; residual leaks are turns where the model mixes
prose with the tool call in one turn (the `content_and_tool` pattern), which the
sglang Qwen parser does not always extract — addressed by the v2
training data that drops `content_and_tool` targets.

## Conclusion

The parser fix is a large, targeted improvement over the same trained user with
parser OFF:

- telecom user-side tools now execute (0 → ~1000). Qwen3.5 telecom pass@1 /
  pass@4(any) rose 11.9/22.5 → 53.8/90.0; instruct rose 7.5/20.0 →
  15.6/32.5.
- overall Qwen3.5 rose 28.7/47.0 → 47.2/78.0, instruct rose 18.0/35.0 →
  23.5/45.0, and multitool SFT rose 17.8/36.0 → 23.25/46.0. The multitool
  checkpoint has no matching gemini-user run, so its table baseline is left blank.
- airline/retail are unchanged within four-trial noise; termination remains
  healthy. The v2 training run drops `content_and_tool` targets to address
  the remaining leakage.

**Is the user-model training successful? Functionally, yes; quality-wise, it is
not yet at the gemini baseline.** STOP-training + parser gives a usable simulator
that (1) terminates properly (`user_stop` 381–388 / 400), (2) executes telecom
user tools, and (3) provides substantially stronger and more stable evaluation
than the pre-parser model. However, the correct immutable gemini baseline is
68.2/91.0 overall for Qwen3.5 and 26.2/58.0 for instruct, versus our local
user's 47.2/78.0 and 23.5/45.0. The remaining gap is concentrated in customer
behavior/scenario quality and residual prose+tool mixing, not STOP handling or
basic tool execution.

**Baseline provenance warning.** The mutable files
`outputs/<agent>/pass4_summary.json` were overwritten on 2026-07-14 by a local
base-Qwen user run (e.g. Qwen3.5 38.2/54.0). They are not the gemini baseline.
The gemini numbers above come directly from the original simulation directories
referenced by `tau2-eval/README.md`: `tau2_official_<agent>_{domain}_test_4trials`.

## Tasks

- `pt-ct8nsil2` / `fsh-tau2-userstop-parser-qwen35-0720-133310` — full Pass@4,
  agent `Qwen3.5-4B`, user `iter0006899`, parser ON:
  [jobs/fsh-tau2-userstop-parser-qwen35-0720-133310/run_20260720_133310.log](jobs/fsh-tau2-userstop-parser-qwen35-0720-133310/run_20260720_133310.log).
  Summary: `examples/tau2-bench/eval/official/outputs/Qwen3.5-4B/user_stop_parser_pass4_summary.json`.
- `pt-914ezwml` / `fsh-tau2-userstop-parser-instruct-0720-133310` — full Pass@4,
  agent `Qwen3-4B-Instruct-2507`, user `iter0006899`, parser ON:
  [jobs/fsh-tau2-userstop-parser-instruct-0720-133310/run_20260720_133310.log](jobs/fsh-tau2-userstop-parser-instruct-0720-133310/run_20260720_133310.log).
  Summary: `examples/tau2-bench/eval/official/outputs/Qwen3-4B-Instruct-2507/user_stop_parser_pass4_summary.json`.
- `pt-oxg5i3rc` / `fsh-tau2-userstop-parser-sft-multitool-0720-210257` — full
  Pass@4, agent SFT `Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool` (`iter_0002413`), user `iter0006899`, parser ON:
  [jobs/fsh-tau2-userstop-parser-sft-multitool-0720-210257/run_20260720_210257.log](jobs/fsh-tau2-userstop-parser-sft-multitool-0720-210257/run_20260720_210257.log).
  Summary: `examples/tau2-bench/eval/official/outputs/Qwen3-4B-tau2-agent-sft-multitool-iter0002413/user_stop_parser_pass4_summary.json`.
