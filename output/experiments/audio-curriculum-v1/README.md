# audio-curriculum-v1

Purpose: queue-facing alias for the independent synthetic curriculum experiment.
The authoritative alias-to-real-name mapping is in
[EXPERIMENT_ALIASES.md](../../doc/EXPERIMENT_ALIASES.md). Generated artifacts
remain under the real experiment directory linked by that mapping.

Before each submission, aggregate GPU demand across all nonterminal jobs is
limited to 24 GPUs. This supersedes the later temporary 88-GPU limit.
The retained data target is 2,000 train, 500 dev, and 500 challenge tasks;
non-selected scale candidates remain diagnostic only.

## Jobs

Future submissions using this alias are recorded here. Jobs through 15754 were
submitted under the original experiment name and remain documented in its
README.

- Pilot BM25 protocol-recovery round 4, 46 exact task IDs (job 15811):
  [run log](jobs/15811-audio-pilot-r18-recovery-r4-0907-222251217/run_0_20260907_222251217.log).
- Scale golden-retrieval protocol-recovery round 1, 20 exact task IDs (job
  15816):
  [run log](jobs/15816-audio-scale-golden-recovery-r1-0907-223209428/run_0_20260907_223209428.log).
- Scale BM25 first trial, shard 19/26 (job 15834):
  [run log](jobs/15834-audio-scale-r18-s300-19of26-0907-224307841/run_0_20260907_224307841.log).
- Scale BM25 first trial, shard 20/26 (job 15843):
  [run log](jobs/15843-audio-scale-r18-s300-20of26-0907-225443940/run_0_20260907_225443940.log).
- Pilot BM25 protocol-recovery round 5, 34 exact task IDs; 18 valid records
  retained (job 15866):
  [run log](jobs/15866-audio-pilot-r18-recovery-r5-0907-233444682/run_0_20260907_233444682.log).
- Pilot BM25 protocol-recovery round 6, 22 exact task IDs; six valid records
  retained (job 15907):
  [run log](jobs/15907-audio-pilot-r18-recovery-r6-0908-011307493/run_0_20260908_011307493.log).
- Scale BM25 first trial, shard 21/26; user-stopped before resubmission (job
  15912):
  [run log](jobs/15912-audio-scale-r18-s300-21of26-0908-013537153/run_0_20260908_013537153.log).
- Pilot BM25 protocol-recovery round 7, 18 exact task IDs; four valid records
  retained (job 15916):
  [run log](jobs/15916-audio-pilot-r18-recovery-r7-0908-015557036/run_0_20260908_015557036.log).
- Scale BM25 first trial, shard 22/26; user-stopped before resubmission (job
  15923):
  [run log](jobs/15923-audio-scale-r18-s300-22of26-0908-021710614/run_0_20260908_021710614.log).
- Pilot BM25 protocol-recovery round 8, 16 exact task IDs; two valid records
  retained (job 15927):
  [run log](jobs/15927-audio-pilot-r18-recovery-r8-0908-025134817/run_0_20260908_025134817.log).
- Pilot BM25 protocol-recovery round 9, 15 exact task IDs; user-stopped before
  resubmission (job 15935):
  [run log](jobs/15935-audio-pilot-r18-recovery-r9-0908-033536515/run_0_20260908_033536515.log).
- Scale BM25 first trial, shard 21/26, full replacement for the preserved but
  excluded partial payload from job 15912, retry 1 (job 15953):
  [run log](jobs/15953-audio-scale-r18-s300-21of26-retry1-0908-041612934/run_0_20260908_041612934.log).
- Scale BM25 first trial, shard 22/26, full replacement for the preserved but
  excluded partial payload from job 15923, retry 1 (job 15954):
  [run log](jobs/15954-audio-scale-r18-s300-22of26-retry1-0908-041624475/run_0_20260908_041624475.log).
- Pilot BM25 protocol-recovery round 9, 15 exact task IDs, full replacement for
  the preserved but excluded partial payload from job 15935; 15/15 records,
  three valid reward-one trajectories and 12 protocol failures (job 15955):
  [run log](jobs/15955-audio-pilot-r18-recovery-r9-retry1-0908-041630860/run_0_20260908_041630860.log).
- Scale BM25 first trial, shard 23/26 (job 15972):
  [run log](jobs/15972-audio-scale-r18-s300-23of26-0908-045503112/run_0_20260908_045503112.log).
- Scale BM25 first trial, shard 24/26; stopped after the final target was
  reduced to 2,000 train, with partial records retained (job 15997):
  [run log](jobs/15997-audio-scale-r18-s300-24of26-0908-072145808/run_0_20260908_072145808.log).
- Scale BM25 first trial, shard 25/26; stopped after the final target was
  reduced to 2,000 train, with partial records retained (job 15998):
  [run log](jobs/15998-audio-scale-r18-s300-25of26-0908-072524708/run_0_20260908_072524708.log).
- Pilot BM25 protocol-recovery round 10, 15 exact task IDs (job 16004):
  [run log](jobs/16004-audio-pilot-r18-recovery-r10-0908-075633619/run_0_20260908_075633619.log).
- Reduced 3,000-task set, BM25 seed-300 recovery round 1, shard 1/2
  (job 16013):
  [run log](jobs/16013-audio-reduced-r18-recovery-r1-1of2-0908-081840286/run_0_20260908_081840286.log).
- Reduced 3,000-task set, BM25 seed-300 recovery round 1, shard 0/2
  (job 16014):
  [run log](jobs/16014-audio-reduced-r18-recovery-r1-0of2-0908-081840722/run_0_20260908_081840722.log).
- Pilot BM25 protocol-recovery round 11, 14 exact task IDs (job 16018):
  [run log](jobs/16018-audio-pilot-r18-recovery-r11-0908-083119880/run_0_20260908_083119880.log).
