# tau2-raw-agent-raw-user-parser-on

Purpose: replace the historical RAW-User parser-OFF raw-Agent references with
matched parser-ON evaluations and complete both User columns at seeds `300`
and `301`. Each task uses the test split, all three domains, four trials, Agent
temperature `0.6`, evaluation `max_steps=200`, and an explicit
`--tool-call-parser qwen` on the User server.

## Tasks

- Job `1279` / raw Qwen3-4B Agent — [run log](jobs/1279-raw-qwen3-agent-raw-user-parser-on-0813-212242603/run_20260813_212242603.log)
  (succeeded; 400/400; [summary](eval/qwen3/seed300_summary.json)).
- Job `1280` / raw Qwen3.5-4B thinking Agent — [run log](jobs/1280-raw-qwen35-thinking-agent-raw-user-parser-on-0813-212248868/run_20260813_212248868.log)
  (succeeded; 400/400; [summary](eval/qwen35-thinking/seed300_summary.json)).
- Job `1281` / raw Qwen3.5-4B non-thinking Agent — [run log](jobs/1281-raw-qwen35-nonthinking-agent-raw-user-parser-on-0813-212253329/run_20260813_212253329.log)
  (succeeded; 400/400; [summary](eval/qwen35-nonthinking/seed300_summary.json)).
- Job `1307` / raw Qwen3 Agent with RAW User, seed301 — [run log](jobs/1307-raw-qwen3-agent-raw-user-parser-on-s301-0814-011411969/run_20260814_011411969.log)
  (succeeded; 400/400; [summary](eval/qwen3/seed301_summary.json)).
- Job `1308` / raw Qwen3.5 thinking Agent with RAW User, seed301 — [run log](jobs/1308-raw-qwen35-thinking-agent-raw-user-parser-on-s301-0814-011412507/run_20260814_011412507.log)
  (succeeded; 400/400; [summary](eval/qwen35-thinking/seed301_summary.json)).
- Job `1309` / raw Qwen3.5 non-thinking Agent with RAW User, seed301 — [run log](jobs/1309-raw-qwen35-nonthinking-agent-raw-user-parser-on-s301-0814-011413056/run_20260814_011413056.log)
  (succeeded; 400/400; [summary](eval/qwen35-nonthinking/seed301_summary.json)).
- Job `1310` / raw Qwen3 Agent with V1 User, seed301 — [run log](jobs/1310-raw-qwen3-agent-v1-user-parser-on-s301-0814-011413607/run_20260814_011413607.log)
  (succeeded; 400/400; [summary](eval/qwen3-v1-user/seed301_summary.json)).
- Job `1311` / raw Qwen3.5 thinking Agent with V1 User, seed301 — [run log](jobs/1311-raw-qwen35-thinking-agent-v1-user-parser-on-s301-0814-011414778/run_20260814_011414778.log)
  (succeeded; 400/400; [summary](eval/qwen35-thinking-v1-user/seed301_summary.json)).

## Results

Each cell is the seed300/301 mean pass@1 / pass@4(any) / pass^4. Both User
columns use parser ON; `★` marks the User cell that is higher on all three
metrics. Every new summary contains 400/400 simulations and zero infrastructure
errors.

| Raw Agent | Seeds | V1 User, parser ON | RAW User, parser ON |
|---|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 300/301 mean | 23.88 / 45.50 / 6.50% ★ | 17.38 / 35.50 / 2.50% |
| Qwen3.5-4B thinking | 300/301 mean | 48.50 / 76.00 / 22.00% | 53.25 / 78.00 / 25.00% ★ |
| Qwen3.5-4B non-thinking | 300/301 mean | 32.75 / 61.00 / 9.00% ★ | 25.63 / 53.50 / 7.00% |

The historical seed300 parser-OFF RAW references remain Qwen3
`18.00/35.00/4.00%` and Qwen3.5 thinking `38.25/54.00/19.00%`; they are not
used in the matrix. The parser is required for correct User-tool execution,
but it does not guarantee a uniform score increase. Raw Qwen3 and Qwen3.5
non-thinking favor V1 User, while Qwen3.5 thinking favors RAW User. Thus the
evaluation User materially changes the raw-Agent baseline and must always be
reported. Non-thinking uses the current-single Agent protocol; Qwen3 and
thinking retain their raw-Agent serving profiles. The non-thinking V1 seed301
control comes from its original two-seed evaluation.
