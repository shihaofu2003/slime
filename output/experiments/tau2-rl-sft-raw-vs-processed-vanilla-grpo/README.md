# tau2-rl-sft-raw-vs-processed-vanilla-grpo

Purpose: compare raw-all and processed official-native SFT initializations with
one paired 100-update vanilla GRPO run per arm and fixed two-seed official
evaluation.

| Arm | SFT initialization | Formal RL output |
|---|---|---|
| Raw-all | `Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903/final_hf` | `Qwen3-4B-Instruct-2507_tau2_agent_rl_raw_all_vanilla_grpo_qwen36_20260903` |
| Processed | `Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903/final_hf` | `Qwen3-4B-Instruct-2507_tau2_agent_rl_official_native_expanded_vanilla_grpo_qwen36_20260903` |

## Tasks

- Job `12984` was the first raw-all smoke attempt; it stopped in preflight
  before training because the test itself shadowed its `json` import.
  [Run log](jobs/12984-raw-all-smoke-0903-160153240/run_*_20260903_160153240.log).
- Job `12985` was the corresponding processed smoke attempt and stopped at the
  same preflight error before training.
  [Run log](jobs/12985-processed-smoke-0903-160153495/run_*_20260903_160153495.log).
- Job `13041` was the first raw-all retry after fixing preflight; it was stopped
  before start so the resubmission could request CPU and memory explicitly.
  It therefore has no run log.
  [Scheduler log](jobs/13041-raw-all-smoke-0903-182432132/jobm.log).
- Job `13040` was the corresponding processed retry and was stopped before
  start for the same resource-specification change.
  It therefore has no run log.
  [Scheduler log](jobs/13040-processed-smoke-0903-182431838/jobm.log).
- Job `13055` ran the raw-all smoke with 8 GPUs, 112 CPUs, and 1,584 GB;
  succeeded with a finite step-0 update and iter0 checkpoint.
  [Run log](jobs/13055-raw-all-smoke-resourced-0903-183932643/run_*_20260903_183932643.log).
- Job `13054` ran the processed smoke with the same explicit resources;
  succeeded with a finite step-0 update and iter0 checkpoint.
  [Run log](jobs/13054-processed-smoke-resourced-0903-183932380/run_*_20260903_183932380.log).
- Job `13117` completed the raw-all 100-update training with 8 GPUs, 112 CPUs,
  and 1,584 GB. Ray finished successfully and wrote iter99; after conversion
  and both evaluations completed, the outer job was stopped because its User
  server ignored shutdown and left the wrapper blocked in `wait`.
  [Run log](jobs/13117-raw-all-train100-0903-190750299/run_*_20260903_190750299.log).
- Job `13118` completed the paired processed 100-update training with the same
  explicit resources.
  [Run log](jobs/13118-processed-train100-0903-190750596/run_*_20260903_190750596.log).
- Job `13166` converts the complete raw-all iter9 checkpoint to Hugging Face
  format with 1 GPU, 14 CPUs, and 198 GB.
  [Run log](jobs/13166-raw-all-convert-iter9-0903-203812829/run_*_20260903_203812829.log).
- Job `13167` converts the complete processed iter9 checkpoint with the same
  explicit resources.
  [Run log](jobs/13167-processed-convert-iter9-0903-203813159/run_*_20260903_203813159.log).
- Job `13177` evaluates raw-all iter9 at seed 300 with 8 GPUs, 112 CPUs, and
  1,584 GB under the fixed official-native protocol.
  [Run log](jobs/13177-raw-all-eval-iter9-seed300-0903-204133949/run_*_20260903_204133949.log).
- Job `13176` evaluates processed iter9 at seed 300 with the same explicit
  resources and protocol.
  [Run log](jobs/13176-processed-eval-iter9-seed300-0903-204133599/run_*_20260903_204133599.log).
- Job `13248` converts the complete processed iter39 checkpoint with 1 GPU,
  14 CPUs, and 198 GB.
  [Run log](jobs/13248-processed-convert-iter39-0903-232000805/run_*_20260903_232000805.log).
- Job `13250` evaluates processed iter39 at seed 300 with 8 GPUs, 112 CPUs,
  and 1,584 GB under the fixed official-native protocol.
  [Run log](jobs/13250-processed-eval-iter39-seed300-0903-232315483/run_*_20260903_232315483.log).
- Job `13253` converts the complete raw-all iter39 checkpoint with 1 GPU,
  14 CPUs, and 198 GB.
  [Run log](jobs/13253-raw-all-convert-iter39-0903-232703553/run_*_20260903_232703553.log).
- Job `13255` evaluates raw-all iter39 at seed 300 with 8 GPUs, 112 CPUs, and
  1,584 GB under the fixed official-native protocol.
  [Run log](jobs/13255-raw-all-eval-iter39-seed300-0903-233100004/run_*_20260903_233100004.log).
- Job `13296` converts the complete processed iter69 checkpoint with 1 GPU,
  14 CPUs, and 198 GB.
  [Run log](jobs/13296-processed-convert-iter69-0904-010605182/run_*_20260904_010605182.log).
- Job `13298` evaluates processed iter69 at seed 300 with 8 GPUs, 112 CPUs,
  and 1,584 GB under the fixed official-native protocol.
  [Run log](jobs/13298-processed-eval-iter69-seed300-0904-010935984/run_*_20260904_010935984.log).
- Job `13300` converts the complete raw-all iter69 checkpoint with 1 GPU,
  14 CPUs, and 198 GB.
  [Run log](jobs/13300-raw-all-convert-iter69-0904-011905385/run_*_20260904_011905385.log).
- Job `13301` evaluates raw-all iter69 at seed 300 with 8 GPUs, 112 CPUs, and
  1,584 GB under the fixed official-native protocol.
  [Run log](jobs/13301-raw-all-eval-iter69-seed300-0904-012235291/run_*_20260904_012235291.log).
- Job `13326` converts the complete processed iter99 checkpoint with 1 GPU,
  14 CPUs, and 198 GB.
  [Run log](jobs/13326-processed-convert-iter99-0904-035822749/run_*_20260904_035822749.log).
- Job `13328` evaluates processed iter99 at seed 300 with 8 GPUs, 112 CPUs,
  and 1,584 GB under the fixed official-native protocol.
  [Run log](jobs/13328-processed-eval-iter99-seed300-0904-040132367/run_*_20260904_040132367.log).
- Job `13327` evaluates processed iter99 at seed 301 with the same explicit
  resources and protocol.
  [Run log](jobs/13327-processed-eval-iter99-seed301-0904-040132033/run_*_20260904_040132033.log).
- Job `13336` converts the complete raw-all iter99 checkpoint with 1 GPU,
  14 CPUs, and 198 GB.
  [Run log](jobs/13336-raw-all-convert-iter99-0904-074257660/run_*_20260904_074257660.log).
- Job `13338` evaluates raw-all iter99 at seed 301 with 8 GPUs, 112 CPUs, and
  1,584 GB under the fixed official-native protocol.
  [Run log](jobs/13338-raw-all-eval-iter99-seed301-0904-075402984/run_*_20260904_075402984.log).
- Job `13339` evaluates raw-all iter99 at seed 300 with the same explicit
  resources and protocol.
  [Run log](jobs/13339-raw-all-eval-iter99-seed300-0904-075403295/run_*_20260904_075403295.log).

## Fixed protocol

Both arms use training seed 1234, rollout seed 42, K=8, five prompt groups per
update, the Airline/Retail/Telecom `1/2/2` quota, LR `2e-6`, constant Adam,
clip `0.2`, and zero KL/entropy loss coefficients. Training uses only binary
task reward and slime's built-in GRPO group normalization; zero-variance groups
are retained with zero advantage. Rollout uses the unsigned Tau2
official-native Agent protocol and non-thinking Qwen3.6-27B User, with Agent
temperature 1.0, User temperature 0, 1,200/512 token budgets, 200 steps, a
16,384-token training cap, and two over-cap retries.

Each eight-GPU job assigns GPUs 0–3 to the Agent, GPUs 4–5 to the TP2 User,
and leaves GPUs 6–7 unused. The matched SFT controls score as follows under the
same official evaluation protocol:

| SFT control | Two-seed pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| Raw-all | 53.12% | 82.50% | 23.00% |
| Processed | 55.75% | 81.00% | 28.00% |

Formal evaluation covers 100 test tasks and four trials at seed 300 for
iterations 9/39/69/99, plus seed 301 for iteration 99. The existing matched SFT
seed-300/301 results are reused. The final comparison is based on the two-seed
iteration-99 overall pass@1 mean; intermediate checkpoints are diagnostic only.

## Behavior analysis

The causal diagnosis of Raw-all's update-40--69 collapse, its post-collapse
schema-copy/repetition loop, and Processed RL's conservative tool-use pattern
relative to its SFT control is in [Behavior analysis](BEHAVIOR_ANALYSIS.md).

## Status

The initial smoke attempts failed only in the rollout-logic preflight and did
not create checkpoint roots. The `json` shadowing defect was fixed and the
local 14-test suite and syntax checks pass. Unresourced retries `13040`/`13041`
were stopped before start at the user's request. Resourced replacements
`13054`/`13055` both succeeded: each retained its zero-variance groups,
accepted exactly `1/2/2` prompt groups, trained global batch 40 with finite
loss and grad norm, reported zero truncation, and saved iter0. Their raw reward
was `0.475`/`0.300` and grad norm was `0.3244`/`0.3841` for
Processed/Raw-all. Formal jobs `13117`/`13118` started at normal priority from
the untouched SFT roots with the same explicit resources. Both cluster
preflights passed and both loaded their SFT weights at RL iteration 0 with new
optimizer and RNG state. Their first updates completed with finite loss and
grad norm, exact `1/2/2` accepted quotas, and zero truncation. Raw-all step 0
had raw reward `0.325`, three retained zero-variance groups, and grad norm
`0.2971`; Processed step 0 had raw reward `0.425`, two retained zero-variance
groups, one same-domain replacement for an unusable Telecom group, and grad
norm `0.3828`. Both jobs saved complete iter9 checkpoints and continued past
the first formal observation point. Conversion jobs `13166`/`13167` produced
complete iter9 Hugging Face checkpoints. Official seed-300 evaluation jobs
`13177`/`13176` then succeeded with 100 tasks, 400 unique task/trial
simulations, exact `20/40/40` domain coverage, binary rewards, and zero
infrastructure errors. Raw-all iter9 scored `52.75/80.00/20.00%` and Processed
iter9 scored `54.75/81.00/27.00%` in pass@1/pass@4(any)/pass^4. These are
diagnostic checkpoints and do not select the final model. Formal training
continues; Processed has also saved its complete iter19 checkpoint.
Processed has now saved iter39, completed its explicitly resourced conversion,
and started the corresponding seed-300 evaluation. Raw-all completed update 39
with finite metrics and zero truncation, saved and converted iter39, and queued
its identically resourced seed-300 evaluation. Processed iter39 seed-300
evaluation completed with the required 100 tasks, 400 unique task/trial
simulations, `20/40/40` domain coverage, binary rewards, and zero infrastructure
errors. It scored `57.25/81.00/29.00%` in
pass@1/pass@4(any)/pass^4, with action accuracy `74.05%` and DB accuracy
`46.41%`. Raw-all iter39 seed-300 evaluation passed the same coverage and
infrastructure checks and scored `56.25/84.00/28.00%`, with action accuracy
`77.35%` and DB accuracy `48.14%`. Both remain diagnostic checkpoints.
Processed training subsequently completed and saved iter69 with finite metrics;
its explicitly resourced conversion succeeded and the corresponding seed-300
evaluation is queued while both formal jobs continue.
Raw-all also completed and saved iter69 with finite loss and grad norm; its
explicitly resourced conversion succeeded and its seed-300 evaluation is
queued. The first client submission attempt for that evaluation reached no job
ID during a transient manager connection failure; retry `13301` is the only
submitted evaluation job.
Processed iter69 seed-300 evaluation completed with the required 100 tasks,
400 unique task/trial simulations, `20/40/40` domain coverage, binary rewards,
and zero infrastructure errors. It scored `52.25/80.00/24.00%` in
pass@1/pass@4(any)/pass^4, with action accuracy `73.73%` and DB accuracy
`44.27%`; iter69 remains diagnostic only.
Raw-all iter69 seed-300 evaluation passed the same coverage and infrastructure
checks and scored `25.50/56.00/7.00%`, with action accuracy `70.63%` and DB
accuracy `38.00%`. It produced 61 `MAX_STEPS` terminations versus five for
Processed iter69, consistent with the large coefficient-zero K2 drift observed
in Raw-all training; this checkpoint is also diagnostic only.
Processed formal training completed all 100 updates with finite final loss and
grad norm, saved a complete 11-file iter99 checkpoint, and exited successfully.
Its explicitly resourced iter99 conversion job `13326` succeeded and produced
the complete two-shard Hugging Face checkpoint. Seed-300/301 evaluation jobs
`13328`/`13327` both passed the required coverage and infrastructure checks.
Seed 300 scored `51.50/81.00/23.00%` and seed 301 scored
`50.00/79.00/22.00%` in pass@1/pass@4(any)/pass^4, for a two-seed mean of
`50.75/80.00/22.50%`. Raw-all formal training completed all 100 updates with
finite loss and grad norm, exact per-update `1/2/2` quotas, global batch 40,
and a complete 11-file iter99 checkpoint. Conversion job `13336` succeeded and
produced the complete two-shard Hugging Face checkpoint. Seed-301/300 official
evaluation jobs `13338`/`13339` both succeeded with 100 tasks, 400 unique
task/trial simulations, exact `20/40/40` domain coverage, and zero
infrastructure errors. Seed 300 scored `21.00/40.00/11.00%` and seed 301
scored `24.50/45.00/14.00%`, for a two-seed mean of
`22.75/42.50/12.50%` in pass@1/pass@4(any)/pass^4.

The predeclared primary comparison favors Processed RL by `+28.00 pp` in
two-seed iter99 pass@1; its task-paired 95% bootstrap interval is
`[+21.75, +34.38] pp`, and both seeds favor Processed. Relative to their own
SFT controls, Raw-all RL changes pass@1 by `-30.37 pp` and Processed RL by
`-5.00 pp`. Raw-all's coefficient-zero K2 drift reached `204.62`, while
Processed stayed at or below `0.0132`; zero-variance accepted groups were
`63.80%` and `48.40%`, respectively. Full metrics, curves, sampling
diagnostics, and paired intervals are in [Final results](RESULTS.md). The
conclusion applies to this one paired training seed and is not a claim of
training-seed-invariant superiority; no OPD job was launched.
