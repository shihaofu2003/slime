# Banking SFT test snapshot: completed408

Purpose: fixed partial export for SFT testing, using decisions completed by
2026-09-11 16:11:28 Asia/Shanghai. This is not the completed full dataset.

- 408 completed decisions: 330 accepted, 69 rejected, 9 over cap. The separate
  unresolved inference error is outside this snapshot.
- Final export: **329 trajectories, 207 tasks, 1,782 SFT rows**. Rejected and
  over-cap trajectories are not included in the training file.
- One accepted trajectory failed the non-thinking export check:
  `fffaa486-556b-4efe-862d-e0966d2dd08c`. Source User message 17 contains a literal
  `</think>` and duplicated text. It is excluded, not rewritten; production
  decisions remain unchanged. Finalize exited nonzero to report this exclusion.
- Every exported target passed complete-trajectory length, target-only loss
  mask, EOS, and thinking-markup checks. Maximum row length: 16,368 tokens
  with the Qwen3-4B-Instruct-2507 tokenizer; configured cap: 16,384.
- Export preserves native tool-call batches and the final Agent answer.
  Use the existing native `messages`/`tools`, `qwen3_full` loss-mask path and
  final Assistant `step_loss_mask=1`; do not flatten tool batches or train
  all prefix Assistant turns again. Recheck lengths if using another tokenizer.
- This partial, length-ordered processing snapshot is not a random sample.
  It includes the earlier pilot; do not concatenate that pilot again.

Files: [training JSONL](data/full/banking_simplified_sft.jsonl),
[simplified trajectories](data/full/simplified_trajectories.jsonl),
[all 408 results](data/full/results.jsonl), [statistics](data/full/stats.json).
Request counts in snapshot statistics are zero because model request logs were
not copied; the original proposal/review decisions are in
[snapshot decisions](decisions/snapshot.jsonl).

Export ran on local CPU with the existing `qwen_simplify_banking.py --stage
finalize --scope full --output-dir <this directory>`; no new job was submitted
and running job 17467 was not modified.
