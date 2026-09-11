# tau2-sft-full-domain-eval

Purpose: evaluate the later-half checkpoint curve from the complete four-domain
Agent SFT run under one fixed official four-domain protocol.

Name: `tau2-sft-full-domain-eval`.

2026-09-10: jobs `16762` and `17090` (requeued from `16763`, using its log
directory) were stopped at the user's request after estimated remaining
Banking runtime exceeded one hour. Banking progress was 93/388 and 87/388;
the other three domains had completed. `iter_0004673` had not started.
See [Banking data diagnosis](../tau2-banking-expert-sft/DATA_DIAGNOSIS_20260910.md).

## Protocol

- Domains: Airline, Retail, Telecom, and Banking Knowledge.
- Full test split: 197 tasks, 4 trials each, 788 simulations per checkpoint.
- Seed: 300; Banking retrieval: BM25.
- Agent: two TP1 replicas on GPUs 0-1, temperature 0.6.
- User: three TP2 Qwen3.6-27B non-thinking replicas on GPUs 2-7.

## Checkpoints

`iter_0002399`, `iter_0002799`, `iter_0003199`, `iter_0003599`,
`iter_0003999`, `iter_0004399`, and final `iter_0004673`.

## Conversion tasks

- Job `16696`: `iter_0002399`. [Run log](jobs/16696-full-domain-convert-0002399-0909-173750816/run_*.log).
- Job `16697`: `iter_0002799`. [Run log](jobs/16697-full-domain-convert-0002799-0909-173904586/run_*.log).
- Job `16698`: `iter_0003199`. [Run log](jobs/16698-full-domain-convert-0003199-0909-173911083/run_*.log).
- Job `16699`: `iter_0003599`. [Run log](jobs/16699-full-domain-convert-0003599-0909-173917254/run_*.log).
- Job `16702`: `iter_0003999`. [Run log](jobs/16702-full-domain-convert-0003999-0909-173918674/run_*.log).
- Job `16700`: `iter_0004399`. [Run log](jobs/16700-full-domain-convert-0004399-0909-173917732/run_*.log).
- Job `16701`: `iter_0004673`. [Run log](jobs/16701-full-domain-convert-0004673-0909-173918199/run_*.log).

## Evaluation tasks

- Job `16705`: `iter_0002399`. [Run log](jobs/16705-full-domain-eval-0002399-0909-174113367/run_*.log).
- Job `16706`: `iter_0002799`. [Run log](jobs/16706-full-domain-eval-0002799-0909-174310579/run_*.log).
- Job `16707`: `iter_0003199`. [Run log](jobs/16707-full-domain-eval-0003199-0909-174311065/run_*.log).
- Jobs `16709`, `16711`, and `16713` were cancelled before execution and
  superseded by explicit-resource submissions.
- Job `16761`: `iter_0003599`, 8 GPUs, 112 CPUs, 1,584 GB memory.
  [Run log](jobs/16761-full-domain-eval-0003599-r2-0909-184036305/run_*.log).
- Job `16762`: `iter_0004399`, 8 GPUs, 112 CPUs, 1,584 GB memory.
  [Run log](jobs/16762-full-domain-eval-0004399-r2-0909-184039134/run_*.log).
- Job `16763`: `iter_0003999`, then final `iter_0004673`, evaluated sequentially
  with 8 GPUs, 112 CPUs, and 1,584 GB memory.
  [Run log](jobs/16763-full-domain-eval-0003999-0004673-r2-0909-184040172/run_*.log).
