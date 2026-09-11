# audio-01

Purpose: queue-facing alias for a mapped evaluation workload.

Name: `audio-01`. The authoritative real-name mapping is in
[EXPERIMENT_ALIASES.md](../../doc/EXPERIMENT_ALIASES.md). Review decisions and
final data remain under the mapped real experiment directory.

Aggregate GPU demand across all nonterminal jobs is limited to 24 GPUs.

## Jobs

- `15801` / `pt-sko7zptx` — six-H100 resume for mapped shards 0–5,
  preserving all completed decisions from the stopped jobs; succeeded in 3.13
  hours.
  [Run log](jobs/15801-audio-01-a-0907-220324051/run_0_20260907_220324051.log).
- `15802` / `pt-e0f267br` — six-H100 resume for mapped shards 6–11,
  preserving all completed decisions from the stopped jobs; succeeded in 3.17
  hours.
  [Run log](jobs/15802-audio-01-b-0907-220333731/run_0_20260907_220333731.log).
- `15903` / `pt-r29stnjw` — four-H100 resume for mapped shards 12–15,
  submitted when aggregate nonterminal demand fell from 24 to 20 GPUs;
  succeeded in 3.30 hours.
  [Run log](jobs/15903-audio-01-c-0908-004902400/run_0_20260908_004902400.log).

No submission raised aggregate nonterminal demand above 24 GPUs.
