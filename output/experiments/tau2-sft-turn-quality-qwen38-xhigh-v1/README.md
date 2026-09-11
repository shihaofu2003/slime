# tau2-sft-turn-quality-qwen38-xhigh-v1

Purpose: filter the already-selected official-native Agent SFT data at the
individual supervised-turn level with two independent local Qwen3.8-27B
`xhigh` reviews. Remove only unanimous material errors and never reconstruct or
repair a target.

Name: `tau2-sft-turn-quality-qwen38-xhigh-v1`.

## Protocol

- Input: `../tau2-sft-official-native-expanded/data/agent_official_native_expanded.jsonl`.
- One input JSONL row is one decision unit; only its final Assistant message
  with `step_loss_mask=1` is judged against that row's visible prefix, policy,
  schemas, and observations.
- Prior Assistant claims are untrusted context: a target that correctly recovers
  is retained, while a target that endorses or depends on the earlier error is
  judged on that dependency. Explicitly conditional options remain conditional
  when the entity or eligibility facts are unresolved.
- The source `correct` and `reward` labels are hidden from the judge.
- Qwen3.8-27B runs TP1 in BF16 with a 32,768-token context and explicit nested
  `reasoning_effort=xhigh`. Passes A/B use seeds 300/301.
- Within each domain, pending rows are scheduled by Assistant turn ordinal and
  then prompt length. Each GPU serves up to eight independent requests; each
  request still judges exactly one turn.
- Only `drop/drop` is eligible for removal. Every disagreement and explicit
  `review` is retained and listed separately; an established manual `retain`
  calibration also remains retained as review if stochastic full-run judgments
  drift to `drop/drop`.
- The filtered training file is written by copying original input lines in
  their original order. No message, metadata, or tool call is serialized again.

## Artifacts

- `shards/` — prepared 16-way token-balanced inputs and the 128-row pilot.
- `reviews/` — per-pass structured decisions, merged decisions, and retained
  Review entries. Model reasoning text is not stored.
- `data/agent_official_native_expanded_turn_filtered.jsonl` — final verbatim
  input subset.
- `summary.json` — verdict, domain/type, issue, usage, latency, and error counts.

## Jobs

- `15144` — first one-H100 pilot, stopped after both passes incorrectly accepted
  the known Basic Economy cabin-change workaround at input line 4,768. Its
  outputs are retained under `reviews/pilot_v1_failed/`.
  [Run log](jobs/15144-turn-quality-pilot-0907-004155283/run_0_20260907_004155283.log).
- `15191` / `pt-eijcy8j3` — second one-H100 pilot using prompt
  `tau2-turn-quality-v2`, stopped after pass B incorrectly treated cancellation
  as an available alternative for input line 21,429. Pass A dropped that turn,
  and both passes retained its later correction at input line 14,336. Outputs
  are retained under `reviews/pilot_v2_failed/`.
  [Run log](jobs/15191-turn-quality-pilot-v2-0907-011633552/run_0_20260907_011633552.log).
- `15235` — third one-H100 pilot using prompt `tau2-turn-quality-v3`, stopped
  after pass B miscomputed the May 11 to May 15 interval as within 24 hours at
  input line 21,429. Pass A correctly dropped it, and both passes retained the
  later correction. Outputs are retained under `reviews/pilot_v3_failed/`.
  [Run log](jobs/15235-turn-quality-pilot-v3-0907-020818551/run_0_20260907_020818551.log).
- `15284` / `pt-czhiwofp` — fourth one-H100 pilot using prompt
  `tau2-turn-quality-v4`; the automated calibration passed, but manual review
  rejected its unanimous drop of input line 23,976. That turn only describes
  cancellation conditionally while the requested reservation is unresolved.
  Outputs are retained under `reviews/pilot_v4_failed/`.
  [Run log](jobs/15284-turn-quality-pilot-v4-0907-023153195/run_0_20260907_023153195.log).
- `15432` / `pt-yk6kbjy5` — one-H100 pilot using prompt
  `tau2-turn-quality-v5`; completed 128 rows in 1.92 GPU-hours. All nine
  calibration rows matched their expected outcomes.
  [Run log](jobs/15432-turn-quality-pilot-v5-retry2-0907-113822164/run_0_20260907_113822164.log).
- `15433` / `pt-o72y3e0d` — duplicate v5 submission stopped during environment
  setup before any review was written; job 15432 is the retained run.
  [Run log](jobs/15433-turn-quality-pilot-v5-0907-113823807/run_0_20260907_113823807.log).
- `15515` / `pt-fty123yz` — initial full-data review for shards 0–7 on eight
  H100s. It was stopped after 3,436 pass-A and 3,431 pass-B decisions so the
  remaining rows could resume with ordinal grouping and concurrency eight.
  [Run log](jobs/15515-turn-quality-full-00-07-0907-134714022/run_0_20260907_134714022.log).
- `15516` / `pt-uhrfl7g8` — initial full-data review for shards 8–15 on eight
  H100s. It was stopped after 3,567 pass-A and 3,561 pass-B decisions; all
  completed rows were retained for the concurrency-eight resume.
  [Run log](jobs/15516-turn-quality-full-08-15-0907-134720009/run_0_20260907_134720009.log).
- `15737` / `pt-ssfkhag3` — concurrency-eight resume for shards 0–7, reusing
  every valid decision from job 15515 and scheduling only missing decisions.
  It was stopped for queue-facing alias migration after reaching 8,094 pass-A
  and 8,083 pass-B decisions.
  [Run log](jobs/15737-turn-quality-full-00-07-c8-resume-0907-200142906/run_0_20260907_200142906.log).
- `15745` / `pt-n0ssl4bk` — concurrency-eight resume for shards 8–15, reusing
  every valid decision from job 15516 and scheduling only missing decisions.
  It was stopped for queue-facing alias migration after reaching 7,655
  decisions in each pass.
  [Run log](jobs/15745-turn-quality-full-08-15-c8-resume-0907-201209399/run_0_20260907_201209399.log).
- `15801` / `pt-sko7zptx` — alias `audio-01-a`, six-H100 resume for shards 0–5.
  It writes decisions back to this experiment while its job log remains under
  the alias experiment. It succeeded in 3.13 hours.
  [Run log](../audio-01/jobs/15801-audio-01-a-0907-220324051/run_0_20260907_220324051.log).
- `15802` / `pt-e0f267br` — alias `audio-01-b`, six-H100 resume for shards 6–11
  with the same artifact mapping. It succeeded in 3.17 hours.
  [Run log](../audio-01/jobs/15802-audio-01-b-0907-220333731/run_0_20260907_220333731.log).
- `15903` / `pt-r29stnjw` — alias `audio-01-c`, four-H100 resume for shards
  12–15. It was submitted only after aggregate nonterminal demand fell to 20
  GPUs and succeeded in 3.30 hours.
  [Run log](../audio-01/jobs/15903-audio-01-c-0908-004902400/run_0_20260908_004902400.log).

## Result

The implementation passes eleven CPU unit tests. Preparation covered all 30,376
input rows exactly once across 16 shards; v5 per-shard row counts are
1,873–1,917 and estimated token workload differs by less than 0.2%. The largest
v5 judge prompt is 18,550 tokens, so every row fits the 32,768-token service
context with the 8,192-token completion budget and no truncation. The v5 pilot
produced 117 unanimous keeps, 8 unanimous drops, and 3 retained review cases.
Manual review confirmed the only non-calibration unanimous drop: input line
22,052 invents an unsupported website procedure for adding a credit card. The
three disagreements or invalid-judge cases remain retained by construction.
The initial concurrency-eight resume windows sustained 327–332 aggregate
decode tokens/s per GPU versus about 93 tokens/s with concurrency two; both
resume jobs reached batch size eight without OOM or transport failure.

Both full passes cover all 30,376 rows. The final merge contains 29,015
unanimous keeps, 675 drops, and 686 retained review rows, producing a
29,701-row byte-identical source-order subset. There were 676 `drop/drop`
pairs; input line 23,976 is the sole non-removal because its established manual
`retain` calibration takes precedence over the stochastic full-run drift. Its
raw pass verdicts remain recorded as `drop/drop`, while the merged decision is
`review` with basis `manual_calibration_retain`. The 66 pass-A and 64 pass-B
invalid responses are also retained for review.
