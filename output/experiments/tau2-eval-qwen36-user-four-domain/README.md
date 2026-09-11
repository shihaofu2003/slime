# tau2-eval-qwen36-user-four-domain

Purpose: extend the validated two-Agent/three-User asynchronous evaluation to
Airline, Retail, Telecom, and Banking Knowledge while recording the Banking
retrieval configuration explicitly.

## Tasks

- Corrected four-domain BM25 full evaluation `12165`: all selected test tasks,
  four trials, seed 300, eight GPUs; Qwen3.6 User runs text-only with thinking
  disabled and the `qwen3_coder` tool parser —
  [run log](jobs/12165-qwen36-four-domain-bm25-full-parserfix-textonly-seed300-0902-101805474/run_*_20260902_101805474.log).
- Four-domain BM25 full evaluation `11941`: all 197 selected tasks, four
  trials, seed 300, eight GPUs —
  [run log](jobs/11941-qwen36-four-domain-bm25-full-seed300-0901-203831592/run_*_20260901_203831592.log),
  [summary](eval/four-domain-full-bm25/seed300_0901_124000_summary.json).
  Infrastructure completed with 197 tasks / 788 simulations and zero reported
  infrastructure errors, but the quality result is invalid because the User
  service used the incompatible `qwen` tool-call parser.
- Four-domain BM25 smoke `11940`: three tasks per domain, one trial, seed 300,
  eight GPUs —
  [run log](jobs/11940-qwen36-four-domain-bm25-smoke-0901-202704695/run_*_20260901_202704695.log),
  [summary](eval/four-domain-smoke-bm25/seed300_0901_122847_summary.json).
  It has the same parser incompatibility and is not a valid quality smoke.

## Protocol

- Agent: two TP1 Qwen3-4B-Instruct-2507 replicas with cache-aware routing.
- User: three TP2 Qwen3.6-27B replicas with round-robin routing. Qwen3.6 must
  run with `--language-only`, `enable_thinking=false`, and
  `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`; jobs 11940 and
  11941 omitted text-only mode and incorrectly used `--tool-call-parser qwen`.
- Domains: `airline`, `retail`, `telecom`, and `banking_knowledge`.
- Banking retrieval: `bm25`; smoke uses 3 tasks per domain and 1 trial, while
  full evaluation uses every selected task and 4 trials.

## Invalid full results retained for diagnosis

| Domain | Tasks | Trajectories | pass@1 / pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|---:|
| Airline | 20 | 80 | 32.50% (26/80) | 65.00% (13/20) | 5.00% (1/20) |
| Retail | 40 | 160 | 48.13% (77/160) | 77.50% (31/40) | 22.50% (9/40) |
| Telecom | 40 | 160 | 3.13% (5/160) | 10.00% (4/40) | 0.00% (0/40) |
| Banking Knowledge | 97 | 388 | 0.26% (1/388) | 1.03% (1/97) | 0.00% (0/97) |
| **Overall** | **197** | **788** | **13.83% (109/788)** | **24.87% (49/197)** | **5.08% (10/197)** |

The three User-worker logs contain 3,691 `Failed to parse JSON part` messages.
Qwen3.6 emitted Qwen3-Coder-style `<function=...>` calls, which the generic
`qwen` parser left as assistant text instead of returning structured User tool
calls. The table is retained only to identify the affected artifact; none of
these scores should be used for model comparison. The corrected
[three-domain smoke](../tau2-eval-qwen36-user-parserfix-smoke/README.md) passed;
a corrected full evaluation is still required.

## Timing and allocation

- Model/service startup: 322 seconds; evaluation wall time: 3303.92 seconds;
  wrapper wall time: 3632 seconds (60.53 minutes).
- Overlapping domain wall times: Airline 1422.08 seconds, Retail 1653.67
  seconds, Telecom 2881.64 seconds, and Banking 3287.09 seconds. These values
  overlap because the domains run concurrently and must not be summed.
- Aggregate trajectory time was 25854.38 seconds, equivalent to a 7.83x
  parallelism factor. Agent inference accounted for 47.82%, User inference for
  51.84%, and other work for 0.34%.
- Slots changed from `1:2:2:4` to `0:2:2:5` after Airline, then `0:0:3:6`
  after Retail, and finally `0:0:0:9` after Telecom. All released slots were
  reused by unfinished domains.
