# tau2-eval-user-stop-v2-parser

**Purpose:** evaluate the aligned STOP-trained v2 local user simulator
`Qwen3-4B-Instruct-2507_tau2_user_sft_stop_v2/iter_0006311_hf` and compare
Agent results across Gemini, the original local Qwen User, v1 `iter_0006899_hf`,
and v2. All three v2 jobs completed successfully on the official
test split: airline 20, retail 40, telecom 40, four trials per task (400
simulations per agent), with no infrastructure errors.

## Metric definitions

- `pass^k` is tau2's official `pass_hat_k`: estimated probability that all `k`
  sampled trials succeed.
- `pass@k` is the probability that at least one of `k` sampled trials succeeds.
  For `c` successes among the four observed trials, it is estimated as
  `1 - C(4-c,k) / C(4,k)`.
- `pass^1 = pass@1`. Overall is the task-weighted average over 100 tasks. All
  values below are percentages.

## V2 Agent comparison

Official `pass^k`:

| Agent | pass^1 | pass^2 | pass^3 | pass^4 |
|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 17.00 | 9.33 | 6.00 | 4.00 |
| Multitool SFT `iter_0002413` | 19.00 | 11.33 | 9.25 | 8.00 |
| **Qwen3.5-4B** | **30.75** | **21.17** | **17.50** | **16.00** |

At-least-one-success `pass@k`:

| Agent | pass@1 | pass@2 | pass@3 | pass@4 |
|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 17.00 | 24.67 | 29.00 | 32.00 |
| Multitool SFT `iter_0002413` | 19.00 | 26.67 | 32.25 | 37.00 |
| **Qwen3.5-4B** | **30.75** | **40.33** | **46.25** | **50.00** |

V2 domain endpoints:

| Agent | Domain | pass^1 | pass^4 | pass@4 |
|---|---|---:|---:|---:|
| Instruct | airline | 16.25 | 0.00 | 35.00 |
| Instruct | retail | 26.88 | 10.00 | 45.00 |
| Instruct | telecom | 7.50 | 0.00 | 17.50 |
| Multitool SFT | airline | 37.50 | 25.00 | 55.00 |
| Multitool SFT | retail | 24.38 | 7.50 | 47.50 |
| Multitool SFT | telecom | 4.38 | 0.00 | 17.50 |
| Qwen3.5 | airline | 40.00 | 20.00 | 65.00 |
| Qwen3.5 | retail | 38.12 | 22.50 | 62.50 |
| Qwen3.5 | telecom | 18.75 | 7.50 | 30.00 |

## Four User simulator comparison

The four User configurations are:

| User | Model | User tool parser | Result source |
|---|---|---|---|
| Gemini | `gemini-2.5-flash` API | not applicable | immutable official simulation directories |
| Original Qwen | `models/Qwen3-4B-Instruct-2507` | historical run, OFF | `outputs/<agent>/pass4_summary.json` references |
| v1 | `..._tau2_user_sft_stop/iter_0006899_hf` | Qwen parser ON | `user_stop_parser_pass4_summary.json` |
| v2 | `..._tau2_user_sft_stop_v2/iter_0006311_hf` | Qwen parser ON | `user_stop_v2_parser_pass4_summary.json` |

All use the official test tasks and four trials. This is a system-level
comparison rather than a checkpoint-only ablation because Gemini uses the API
User implementation and the original Qwen runs predate the parser-ON change.
There is no matching Gemini run for the multitool SFT `iter_0002413` Agent; its
Gemini cells are left blank instead of substituting the different
`iter_0000881` SFT checkpoint.

Common-Agent macro average uses only Instruct and Qwen3.5, the two Agents that
have results under all four Users:

| User | pass^1 | pass^2 | pass^3 | pass^4 | pass@1 | pass@2 | pass@3 | pass@4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gemini** | **47.25** | **32.92** | **26.38** | **22.50** | **47.25** | **61.58** | **69.38** | **74.50** |
| Original Qwen | 28.12 | 19.33 | 14.88 | 11.50 | 28.12 | 36.92 | 41.25 | 44.50 |
| v1 | 35.38 | 22.00 | 16.12 | 12.50 | 35.38 | 48.75 | 56.25 | 61.50 |
| v2 | 23.88 | 15.25 | 11.75 | 10.00 | 23.88 | 32.50 | 37.62 | 41.00 |

Official `pass^k` by Agent:

| Agent | User | pass^1 | pass^2 | pass^3 | pass^4 |
|---|---|---:|---:|---:|---:|
| Instruct | Gemini | 26.25 | 12.00 | 7.75 | 6.00 |
| Instruct | Original Qwen | 18.00 | 9.67 | 6.25 | 4.00 |
| Instruct | v1 | 23.50 | 13.00 | 8.75 | 6.00 |
| Instruct | v2 | 17.00 | 9.33 | 6.00 | 4.00 |
| Multitool SFT | Gemini | — | — | — | — |
| Multitool SFT | Original Qwen | 18.25 | 10.50 | 6.50 | 4.00 |
| Multitool SFT | v1 | 23.25 | 13.17 | 10.00 | 8.00 |
| Multitool SFT | v2 | 19.00 | 11.33 | 9.25 | 8.00 |
| Qwen3.5 | Gemini | 68.25 | 53.83 | 45.00 | 39.00 |
| Qwen3.5 | Original Qwen | 38.25 | 29.00 | 23.50 | 19.00 |
| Qwen3.5 | v1 | 47.25 | 31.00 | 23.50 | 19.00 |
| Qwen3.5 | v2 | 30.75 | 21.17 | 17.50 | 16.00 |

At-least-one-success `pass@k` by Agent:

| Agent | User | pass@1 | pass@2 | pass@3 | pass@4 |
|---|---|---:|---:|---:|---:|
| Instruct | Gemini | 26.25 | 40.50 | 50.50 | 58.00 |
| Instruct | Original Qwen | 18.00 | 26.33 | 31.25 | 35.00 |
| Instruct | v1 | 23.50 | 34.00 | 40.25 | 45.00 |
| Instruct | v2 | 17.00 | 24.67 | 29.00 | 32.00 |
| Multitool SFT | Gemini | — | — | — | — |
| Multitool SFT | Original Qwen | 18.25 | 26.00 | 29.75 | 32.00 |
| Multitool SFT | v1 | 23.25 | 33.33 | 40.25 | 46.00 |
| Multitool SFT | v2 | 19.00 | 26.67 | 32.25 | 37.00 |
| Qwen3.5 | Gemini | 68.25 | 82.67 | 88.25 | 91.00 |
| Qwen3.5 | Original Qwen | 38.25 | 47.50 | 51.25 | 54.00 |
| Qwen3.5 | v1 | 47.25 | 63.50 | 72.25 | 78.00 |
| Qwen3.5 | v2 | 30.75 | 40.33 | 46.25 | 50.00 |

Domain endpoints are `pass^1 / pass^4 / pass@4`:

| Agent | User | Airline | Retail | Telecom |
|---|---|---|---|---|
| Instruct | Gemini | 35.00 / 10.00 / 60.00 | 28.75 / 7.50 / 67.50 | 19.38 / 2.50 / 47.50 |
| Instruct | Original Qwen | 27.50 / 10.00 / 45.00 | 28.12 / 5.00 / 55.00 | 3.12 / 0.00 / 10.00 |
| Instruct | v1 | 21.25 / 5.00 / 45.00 | 32.50 / 12.50 / 57.50 | 15.62 / 0.00 / 32.50 |
| Instruct | v2 | 16.25 / 0.00 / 35.00 | 26.88 / 10.00 / 45.00 | 7.50 / 0.00 / 17.50 |
| Multitool SFT | Gemini | — | — | — |
| Multitool SFT | Original Qwen | 32.50 / 20.00 / 40.00 | 24.38 / 0.00 / 47.50 | 5.00 / 0.00 / 12.50 |
| Multitool SFT | v1 | 28.75 / 20.00 / 45.00 | 23.75 / 10.00 / 45.00 | 20.00 / 0.00 / 47.50 |
| Multitool SFT | v2 | 37.50 / 25.00 / 55.00 | 24.38 / 7.50 / 47.50 | 4.38 / 0.00 / 17.50 |
| Qwen3.5 | Gemini | 78.75 / 60.00 / 90.00 | 45.62 / 10.00 / 82.50 | 85.62 / 57.50 / 100.00 |
| Qwen3.5 | Original Qwen | 67.50 / 50.00 / 85.00 | 49.38 / 22.50 / 70.00 | 12.50 / 0.00 / 22.50 |
| Qwen3.5 | v1 | 42.50 / 20.00 / 70.00 | 43.12 / 22.50 / 70.00 | 53.75 / 15.00 / 90.00 |
| Qwen3.5 | v2 | 40.00 / 20.00 / 65.00 | 38.12 / 22.50 / 62.50 | 18.75 / 7.50 / 30.00 |

## Conclusions

- On the common-Agent macro average, Gemini yields the highest Agent success:
  `pass^1/pass^4/pass@4 = 47.25/22.50/74.50`. Among local Users, v1 is highest
  at `35.38/12.50/61.50`, followed by Original Qwen
  `28.12/11.50/44.50`, then v2 `23.88/10.00/41.00`.
- For Instruct, the ordering is Gemini > v1 > Original Qwen > v2 on `pass^1`
  and `pass@4`; Gemini and v1 tie on `pass^4` at `6.00`.
- For Qwen3.5, Gemini is strongest by a large margin. v1 is the best local User,
  especially telecom `pass@4=90.00`; Original Qwen is strong in airline/retail
  but fails to exercise telecom effectively; v2 is lower overall.
- For the multitool SFT Agent, v1 has the best overall `pass^1/pass@4`
  (`23.25/46.00`), while v1 and v2 tie on `pass^4=8.00`. V2 is strongest in
  airline but weak in telecom. No matching Gemini result exists.
- These rankings measure Agent task completion under each User system, not
  intrinsic User-model quality. Parser and implementation differences make the
  four-way table unsuitable as a checkpoint-only quality ranking.

## Tasks

All evaluations use two GPUs, one each for the Agent and user sglang servers.

- `pt-za475i5m` / `fsh-userstop-v2-parser-qwen35-0720-234413` — **SUCCEEDED**,
  Agent `Qwen3.5-4B`, user v2 `iter_0006311_hf`:
  [jobs/fsh-userstop-v2-parser-qwen35-0720-234413/run_20260720_234413.log](jobs/fsh-userstop-v2-parser-qwen35-0720-234413/run_20260720_234413.log).
  Summary: `examples/tau2-bench/eval/official/outputs/Qwen3.5-4B/user_stop_v2_parser_pass4_summary.json`.
- `pt-hicd06c9` / `fsh-userstop-v2-parser-instruct-0720-234413` — **SUCCEEDED**,
  Agent `Qwen3-4B-Instruct-2507`, user v2 `iter_0006311_hf`:
  [jobs/fsh-userstop-v2-parser-instruct-0720-234413/run_20260720_234414.log](jobs/fsh-userstop-v2-parser-instruct-0720-234413/run_20260720_234414.log).
  Summary: `examples/tau2-bench/eval/official/outputs/Qwen3-4B-Instruct-2507/user_stop_v2_parser_pass4_summary.json`.
- `pt-7m01soiy` / `fsh-userstop-v2-parser-sft-multitool-0720-234414` —
  **SUCCEEDED**, Agent multitool SFT `iter_0002413_hf`, user v2
  `iter_0006311_hf`:
  [jobs/fsh-userstop-v2-parser-sft-multitool-0720-234414/run_20260720_234414.log](jobs/fsh-userstop-v2-parser-sft-multitool-0720-234414/run_20260720_234414.log).
  Summary: `examples/tau2-bench/eval/official/outputs/Qwen3-4B-tau2-agent-sft-multitool-iter0002413/user_stop_v2_parser_pass4_summary.json`.
