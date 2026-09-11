# Experiments index

One entry per experiment: name, purpose, link to its README.

- [tau2-banking-simplified-full-sft](../experiments/tau2-banking-simplified-full-sft/README.md) —
  raw-model SFT on all 5,672 simplified Banking rows followed by Banking evaluation.

- [tau2-banking-simplified-sft-test](../experiments/tau2-banking-simplified-sft-test/README.md) —
  raw-model SFT on completed408 simplified Banking data; matched Banking length and latency evaluation.

- [tau2-banking-simplify-concurrency-v1](../experiments/tau2-banking-simplify-concurrency-v1/README.md) —
  matched 8-GPU concurrency 1/2/4 benchmark before changing the live Banking simplification job.

- [tau2-banking-simplify-qwen38-v1](../experiments/tau2-banking-simplify-qwen38-v1/README.md) —
  deletion-only Qwen3.8-27B simplification of all Banking expert trajectories,
  with independent semantic review, sequential replay, and complete non-thinking SFT export.
  Pilot complete (64/75 accepted); full job 17371 stopped for concurrency-four
  continuation 17467, preserving completed production decisions.

- [tau2-sft-official-native-expanded-turn-filtered](../experiments/tau2-sft-official-native-expanded-turn-filtered/README.md) —
  two-epoch SFT and matched checkpoint evaluation on the turn-quality-filtered
  official-native data, using the prior Processed SFT as the data-ablation
  control.

- [tau2-rl-sft-raw-vs-processed-vanilla-grpo](../experiments/tau2-rl-sft-raw-vs-processed-vanilla-grpo/README.md) —
  paired 100-update vanilla GRPO runs from raw-all and processed official-native
  SFT initializations, followed by fixed two-seed official evaluation.

- [qwen38-27b-inference-smoke](../experiments/qwen38-27b-inference-smoke/README.md) —
  one-GPU BF16 SGLang compatibility check for the local Qwen3.8-27B Hugging
  Face checkpoint, covering `xhigh` reasoning, text, parsed tool calls, and
  image input.

- [tau2-sft-turn-quality-qwen38-xhigh-v1](../experiments/tau2-sft-turn-quality-qwen38-xhigh-v1/README.md) —
  conservative turn-only quality filtering of the selected official-native
  Agent SFT rows using two local Qwen3.8-27B `xhigh` reviews, with no target
  reconstruction or repair.

- [audio-01](../experiments/audio-01/README.md) — queue-facing alias for the
  Qwen3.8 turn-quality resume jobs; the real experiment name is recorded in
  [EXPERIMENT_ALIASES.md](EXPERIMENT_ALIASES.md).

- [tau2-eval-qwen36-user-parserfix-smoke](../experiments/tau2-eval-qwen36-user-parserfix-smoke/README.md) —
  three-domain, five-task smoke validating non-thinking text-only Qwen3.6-27B
  User serving with the Qwen3-Coder tool-call parser.

- [tau2-eval-official-native-smoke](../experiments/tau2-eval-official-native-smoke/README.md) —
  cluster preflight and four-domain GPU smoke for the Tau2 official native
  `llm_agent` evaluation path.

- [tau2-eval-official-native-full](../experiments/tau2-eval-official-native-full/README.md) —
  formal four-domain, four-trial evaluation through the Tau2 official native
  `llm_agent` path.

- [tau2-eval-qwen35-official-native-four-domain](../experiments/tau2-eval-qwen35-official-native-four-domain/README.md) —
  matched four-domain, four-trial official-native evaluations of Qwen3.5-4B
  with Agent thinking enabled and disabled.

- [tau2-qwen3-qwen35-atomic-gap-analysis](../experiments/tau2-qwen3-qwen35-atomic-gap-analysis/README.md) —
  deterministic and blinded contrastive analysis of two Qwen3-4B-Instruct-2507
  evaluations against Qwen3.5-4B non-thinking, with targeted prefix replay to
  separate local action choice from upstream evidence and state accumulation.

- [tau2-banking-task-curriculum](../experiments/tau2-banking-task-curriculum/README.md) —
  97-task Banking inventory and dependency graphs, plus 564 benchmark-derived
  diagnostic probes over 94 replayable sources; all are excluded from training.

- [tau2-banking-independent-synthetic-v1](../experiments/tau2-banking-independent-synthetic-v1/README.md) —
  accepted independent 542/75/30 Banking train/dev/challenge set with an empty
  DB, a 480-document isolated runtime, reward-one reference replay, and Qwen3.8
  Agent/User evaluation.

- [audio-curriculum-v1](../experiments/audio-curriculum-v1/README.md) —
  historical queue-facing alias for earlier jobs in the independent synthetic
  curriculum; later jobs use the real experiment name recorded in
  [EXPERIMENT_ALIASES.md](EXPERIMENT_ALIASES.md).

- [tau2-banking-task-curriculum-scale-v2](../experiments/tau2-banking-task-curriculum-scale-v2/README.md) —
  node-level diagnostic scale-out to 2,740 Banking specifications: every annotated
  evidence fact, matching BM25 retrieval, and reference action across the 94
  replayable benchmark tasks; all pass reference replay and none is trainable.

- [tau2-banking-scale-v2-qwen38-eval](../experiments/tau2-banking-scale-v2-qwen38-eval/README.md) —
  one Qwen3.8-27B Agent/User trajectory and exact Tau2 score for each of the
  2,740 benchmark-derived diagnostic tasks: 2,163 exact successes (78.94%);
  trajectories are excluded from SFT, GRPO, and OPD.

- [tau2-banking-curriculum-qwen36-eval](../experiments/tau2-banking-curriculum-qwen36-eval/README.md) —
  single-trial local Qwen3.6-27B evaluation of all 564 benchmark-derived
  Banking diagnostic tasks, split by required retrieval mode.

- [tau2-banking-curriculum-qwen38-eval](../experiments/tau2-banking-curriculum-qwen38-eval/README.md) —
  single-trial Qwen3.8-27B evaluation of all 564 benchmark-derived diagnostic
  tasks, using Qwen3.8-27B for both Agent and User; all 16 initial protocol
  failures are recovered and final exact success is 329/564 (58.33%).

- [tau-bench](../experiments/tau-bench/README.md) — tau-bench (tau1, retail)
  RL run-through for Qwen3-4B-Instruct-2507 with slime: generate mock data,
  convert the Instruct checkpoint to torch_dist, then GRPO training with the
  custom tau rollout. Monitoring log: [experiments/tau-bench/MONITOR.md](../experiments/tau-bench/MONITOR.md).

- [tau2-deps-probe](../experiments/tau2-deps-probe/README.md) — probe job that
  discovers which packages the fresh cluster image needs for tau2 (build:
  hatchling, editables; runtime: toml, deepdiff, litellm; [gym]: gymnasium;
  [knowledge] BM25: rank-bm25) and runs dependency smokes.

- [tau2-eval](../experiments/tau2-eval/README.md) — tau2-bench evaluation
  (airline/retail/telecom) for a sglang-served policy. Current path uses tau2's
  official agent/runner APIs; legacy direct-`AgentGymEnv` runs are archived.

- [tau2-eval-local-user](../experiments/tau2-eval-local-user/README.md) —
  tau2-bench official eval smoke with both agent and user simulator served by
  local sglang.

- [tau2-eval-qwen36-user-async-timed](../experiments/tau2-eval-qwen36-user-async-timed/README.md) —
  eight-GPU official eval with parallel domains, elastic concurrency, and
  Agent/User/environment timing breakdowns. The current routing comparison
  uses two TP1 Qwen3-4B Agent replicas and three TP2 Qwen3.6-27B User replicas;
  the User router changes from cache-aware to round-robin.

- [tau2-eval-qwen36-user-four-domain](../experiments/tau2-eval-qwen36-user-four-domain/README.md) —
  eight-GPU asynchronous evaluation of Airline, Retail, Telecom, and
  Banking Knowledge with explicit BM25 retrieval.

- [tau2-eval-user-sft](../experiments/tau2-eval-user-sft/README.md) —
  tau2-bench official Pass@4 evaluations with the trained local user SFT model
  served by sglang.

- [tau2-hf-convert](../experiments/tau2-hf-convert/README.md) — conversion of
  tau2 user and agent SFT torch_dist checkpoints to Hugging Face directories for
  sglang evaluation.

- [tau2-qwen35](../experiments/tau2-qwen35/README.md) — diagnostic smokes that
  stood up tau2-bench eval for the multimodal/thinking Qwen3.5-4B (text-only
  serving + `--disable-thinking` + multi-format parser) before the full run.

- [tau2-qwen35-nonthinking-eval](../experiments/tau2-qwen35-nonthinking-eval/README.md) —
  final two-seed raw Qwen3.5-4B non-thinking, one-tool-per-turn baseline:
  pass@1/pass@4(any)/pass^4 `32.75/61.00/9.00%`, plus the matched seed-300
  thinking-mode comparison and single-call SFT-data conversion check.

- [tau2-agent-single-call-v1](../experiments/tau2-agent-single-call-v1/README.md) —
  independent strict-single-v1 SFT-to-GRPO retraining from the selected boundary
  data, with source-order call/result serialization and matched two-seed formal
  evaluation of the new SFT and RL checkpoints. SFT, RL iter99, and selected RL
  iter199 score `25.88/49.50/8.50%`, `27.50/53.50/10.00%`, and
  `31.13/61.50/10.50%`. Iter199 improves over iter99 by
  `+3.63/+8.00/+0.50pp`; paired two-seed pass@1 and pass@4(any) intervals are
  positive, with gains concentrated in retail and telecom. Its README also
  contains the unified parser-ON matrix separating RL training User,
  evaluation User, and training/evaluation `max_steps`; every cell now has a
  final seed300/301 mean.

- [tau2-agent-single-call-credit-core](../experiments/tau2-agent-single-call-credit-core/README.md) —
  controlled three-arm strict-single-v1 comparison of fresh turn-credit-v1,
  outcome-only GRPO, and fixed-budget turn-credit-v2 from the same SFT, with
  matched zero-signal handling and two-seed official evaluation.

- [tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl](../experiments/tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl/README.md) —
  four parallel 8-GPU strict-single-v1 runs from the same SFT: one mixed and
  three single-domain turn-credit-v2 lambda-0.1 agents, all trained with the
  raw Qwen3 User and rollout `max_steps=120` before targeted official eval.

- [tau2-rl-boundary-v2-domain-continuations-20260816](../experiments/tau2-rl-boundary-v2-domain-continuations-20260816/README.md) —
  cancelled wrong-lineage submission; boundary-v2/v1-User/max60 jobs were
  stopped before training after the intended single-call/raw-User/max120
  configuration was clarified.

- [tau2-agent-single-call-v1-raw-user-maxsteps120-domain-rl130](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-domain-rl130/README.md) —
  stopped invalid submissions that restarted from the single-call SFT instead
  of the required RAW-User maxsteps120 iter99 checkpoint.

- [tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130/README.md) —
  corrected mixed/Retail/Telecom continuations initialized from RAW-User
  maxsteps120 iter99, running updates 100–129 continuously with turn-credit-v1
  and checkpoint-only saves at iter109/119/129.

- [tau2-agent-single-call-v1-user-ablation](../experiments/tau2-agent-single-call-v1-user-ablation/README.md) —
  matched two-seed official evaluation of the strict-single SFT and selected RL
  iter199 Agents with the unmodified Qwen3-4B-Instruct-2507 User, changing only
  the User checkpoint relative to the v1 User-SFT controls. Raw-User SFT and RL
  score `28.00/51.00/9.00%` and `26.88/52.50/8.00%`; under the same raw User,
  RL-minus-SFT is `-1.13/+1.50/-1.00pp`. Its complete two-axis matrix separates
  RL training User (V1/RAW) from evaluation User (V1/RAW); the five crossed RL
  cells collected later are final seed300/301 means.

- [tau2-agent-single-call-v1-rl-crossed-parser-on](../experiments/tau2-agent-single-call-v1-rl-crossed-parser-on/README.md) —
  completed seed300/301 evaluations for the five previously missing
  strict-single RL crossed cells, all 400/400 per seed with parser ON and zero
  infrastructure errors. Every crossed two-seed mean trails its
  rollout-User-matched control on all three aggregate pass metrics.

- [tau2-raw-agent-raw-user-parser-on](../experiments/tau2-raw-agent-raw-user-parser-on/README.md) —
  completed parser-ON raw-Agent evaluations with V1 and RAW Users at seeds
  300/301. RAW-User means are Qwen3 `17.38/35.50/2.50%`, Qwen3.5 thinking
  `53.25/78.00/25.00%`, and Qwen3.5 non-thinking `25.63/53.50/7.00%`; the
  matched V1-User means remain explicit in the experiment table.

- [tau2-agent-single-call-v1-raw-user-rl100](../experiments/tau2-agent-single-call-v1-raw-user-rl100/README.md) —
  100-update strict-single-v1 GRPO ablation from the selected single-call SFT,
  changing only the rollout User from v1 User SFT to the unmodified
  Qwen3-4B-Instruct-2507 model. Iter99 raw-User evaluation scores
  `29.25/54.00/6.50%`, or `+1.25/+3.00/-2.50pp` over the matched raw-User SFT;
  all overall paired intervals include zero and the gain is Telecom-led.

- [tau2-agent-single-call-v1-raw-user-rl200](../experiments/tau2-agent-single-call-v1-raw-user-rl200/README.md) —
  continuation of the raw-User strict-single-v1 RL lineage from iter99 to
  iter199, preserving the 16,384-token training cap and all other recipe
  settings; 8-GPU job `pt-x6p7gg51` reached step199 and saved the complete
  `iter_0000199` checkpoint; HF conversion and both 400/400 raw-User
  evaluations succeeded. Iter199 scores `31.50/58.00/8.00%`, versus raw-User
  RL99 `29.25/54.00/6.50%` (`+2.25/+4.00/+1.50pp`, overall intervals include
  zero). Against v1-User-trained RL199 under the same raw-User evaluation,
  pass@1 improves by `+4.62pp` with interval `[+0.75,+8.62]pp`.

- [tau2-agent-single-call-v1-raw-user-maxsteps120-rl100](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl100/README.md) —
  controlled repeat of raw-User strict-single RL100 with rollout
  `max_steps=120` instead of 60, testing whether the longer interaction budget
  reduces artificial truncation and improves learned behavior; job `pt-e4a6ryoo`
  completed step99 and saved `iter_0000099`; HF conversion and both 400/400
  raw-User evaluations succeeded. It scores `28.38/56.50/9.50%` versus max60
  `29.25/54.00/6.50%`; the pass@1 delta is `−0.88pp` and its overall interval
  includes zero.

- [tau2-agent-single-call-v1-raw-user-maxsteps120-rl200](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl200/README.md) —
  completed 100-update continuation of the max120 raw-User RL checkpoint from
  iter99 to iter199; retry job `1110`, HF conversion `1232`, and both raw-User
  evaluations (`1234`/`1235`, 400/400 each) succeeded. The iter199 mean is
  `30.13/58.50/7.00%`; relative to max120 iter99 it is
  `+1.75/+2.00/-2.50pp`, while relative to max60 iter199 it is
  `-1.38/+0.50/-1.00pp`.

- [tau2-rl](../experiments/tau2-rl/README.md) — tau2-bench GRPO training for
  the agent SFT checkpoint using the trained local user simulator and per-task
  AReaL RL databases. Monitoring log:
  [experiments/tau2-rl/MONITOR.md](../experiments/tau2-rl/MONITOR.md).

- [tau2-rl-usercmp](../experiments/tau2-rl-usercmp/README.md) — A/B/C
  comparison of three user simulators (v2 SFT, v1 SFT, original instruct) for
  tau2-bench GRPO; identical real-8g runs varying only the user model.

- [tau2-rl-stability-k2-fieldreward-lr-sweep](../experiments/tau2-rl-stability-k2-fieldreward-lr-sweep/README.md) —
  airline GRPO stability scan at `2e-6`, `3e-6`, and `5e-6` with `k2` KL,
  field-level reward credit, explicit failure penalties, and shaped-zero-std
  dynamic replacement.

- [tau2-rl-stability-k2-fieldreward-final-eval](../experiments/tau2-rl-stability-k2-fieldreward-final-eval/README.md) —
  Hugging Face conversion and paired tau2 official held-out evaluation of the
  final `2e-6/3e-6/5e-6` stability-sweep checkpoints using the training-time
  v1 STOP-trained User.

- [tau2-rl-local-first-dependency-safe-k2-fieldreward](../experiments/tau2-rl-local-first-dependency-safe-k2-fieldreward/README.md) —
  gated airline GRPO from the local-first relaxed SFT under the shared
  dependency-safe protocol: evaluate at iter99, resume optimizer/RNG state to
  iter199 only after promotion, then select the final eligible checkpoint.

- [tau2-traj-pattern](../experiments/tau2-traj-pattern/README.md) —
  paired deterministic + Gemini trajectory-pattern analysis across raw/SFT/RL.
  Deterministic results show RL gains only on no-Agent-write workflows and that
  SFT/RL multitool behavior violates the one-tool-per-turn domain policies;
  calibrated semantic review covers all 1,200 trajectories with exact rule facts
  taking precedence over contradictory judge labels.

- [tau2-sft-quality-audit](../experiments/tau2-sft-quality-audit/README.md) —
  source-dialog reconstruction, deterministic quality gates, stratified success
  analysis, and calibrated local-first LLM review of the exact tau2 SFT training
  data, with API escalation and a traceable filtered-data manifest.

- [tau2-sft-multitool-quality-audit](../experiments/tau2-sft-multitool-quality-audit/README.md) —
  protocol-neutral dependency-safe-multi follow-up that separates label failure risk,
  matched successful-trajectory quality, and causal failure attribution, with a final
  turn-level filtering specification and provenance-only decisions.

- [tau2-sft-local-first-relaxed](../experiments/tau2-sft-local-first-relaxed/README.md) —
  local Qwen3.6-27B full-corpus, target-prefix filtering under the calibrated
  dependency-safe-multi standard, followed by raw-init Agent SFT and paired official
  tau2 comparison against the raw and previous SFT checkpoints; pass@1/pass@4(any)
  improve significantly, while the pass^4 regression keeps the recipe in broad-tier
  candidate status pending a consistency-anchor ablation.

- [tau2-sft-agent-user-boundary-v2](../experiments/tau2-sft-agent-user-boundary-v2/README.md) —
  signed Agent-owned tool contract, strict User-event provenance, native tool
  tokenization, and matched-budget contract-only versus boundary-anchor SFT.

- [tau2-sft-official-native-expanded](../experiments/tau2-sft-official-native-expanded/README.md) —
  two-pass direct expansion of successful raw AReaL Airline, Retail, and Telecom
  rows into current Tau2 official-native target-only SFT data, followed by a
  two-epoch raw-Instruct SFT checkpoint curve and matched official-native evals.

- [tau2-sft-full-domain](../experiments/tau2-sft-full-domain/README.md) —
  full-domain Agent SFT on the concatenated official-native AReaL and Banking
  expert datasets, using the prior three-domain SFT recipe.

- [tau2-sft-full-domain-eval](../experiments/tau2-sft-full-domain-eval/README.md) —
  official four-domain evaluation curve for the later full-domain SFT checkpoints.

- [tau2-sft-raw-all-max16384](../experiments/tau2-sft-raw-all-max16384/README.md) —
  raw AReaL Tau2 SFT control retaining successful and failed rows, with only
  over-16,384-token examples removed before final-only two-seed evaluation.

- [tau2-rl-agent-user-boundary-v2](../experiments/tau2-rl-agent-user-boundary-v2/README.md) —
  turn-aware three-domain GRPO from the v3-selected Contract + boundary SFT,
  with absolute Assistant-turn penalties, strict span/token alignment,
  same-domain replacement, and iter9/iter19/iter99 promotion gates; Pilot A
  improved namespace and held-out capability but stopped at iter9 on four
  strict behavior/truncation checks.

- [tau2-rl-agent-user-boundary-v2-long100](../experiments/tau2-rl-agent-user-boundary-v2-long100/README.md) —
  fresh 100-update rerun of the same turn-aware recipe, with health-only
  iter9/iter19 continuation gates; iter99 passed both seed evaluations and the
  final comparison against selected SFT and historical iter9 classified `win`.
  Two-seed mean pass@1/pass@4(any)/pass^4 is `29.50/56.50/10.50%`, improving
  over selected SFT by `+5.88/+13.00/+2.00pp` and historical iter9 by
  `+4.25/+6.50/+1.50pp`. The cross-experiment problem/solution chain is in
  [EXP_QA.md](EXP_QA.md).

- [tau2-rl-agent-user-boundary-v2-domain-experts](../experiments/tau2-rl-agent-user-boundary-v2-domain-experts/README.md) —
  three isolated 30-update single-domain continuations from selected long100
  iter99, compared with mixed iter109/119/129 prefixes as future OPD teacher
  candidates without launching distillation. Retail iter119 and Telecom iter129
  show small replicated advantages and remain teacher candidates. Airline
  iter129 trails mixed RL on the target domain and is not selected, providing
  evidence of useful cross-domain transfer rather than an invalid checkpoint.

- [tau2-rl-agent-user-boundary-v2-checkpoint-curve](../experiments/tau2-rl-agent-user-boundary-v2-checkpoint-curve/README.md) —
  seed-300 same-lineage curve at iter9/39/69/89/99 under the fixed signed
  boundary-v2 protocol, with independent HF conversions and paired task-level
  comparisons. The curve is non-monotonic (`25.00/26.25/24.00/29.75/27.50%`
  pass@1); iter89 is the diagnostic single-seed peak, while historical iter9
  and raw Instruct remain background controls only.

- [tau2-rl-agent-user-boundary-v2-long200](../experiments/tau2-rl-agent-user-boundary-v2-long200/README.md) —
  exact cross-root continuation of the selected long100 iter99 optimizer/RNG and
  per-domain sampling state. It was stopped after update176 when 10-step K2 KL
  reached `0.14287` and updates 175–176 were both `>=0.20`; this produced the
  original early-stop label. A diagnostic continuation reached iter199 and
  completed two 400-simulation evals: mean pass@1/pass@4(any)/pass^4 was
  `32.38/58.00/11.50%`, versus iter99 `29.50/56.50/10.50%`. Telecom improved,
  but seed301 airline fell `7.5pp`, overall deltas versus iter99 were uncertain,
  and namespace raw attempts increased. Iter199 is a valid high-KL checkpoint
  and Telecom-oriented candidate; long100 iter99 remains the better-balanced
  default.

- [tau2-sft](../experiments/tau2-sft/README.md) — tau2-bench SFT data
  conversion and Qwen3-4B-Instruct-2507 supervised fine-tuning scripts.

- [tau2-banking-expert-sft](../experiments/tau2-banking-expert-sft/README.md) —
  Banking synthetic expert-trajectory conversion to native Qwen3 SFT rows and
  the Banking-only supervised fine-tuning experiment.

- [tau2-user-sft](../experiments/tau2-user-sft/README.md) — tau2-bench user
  simulator SFT data conversion and Qwen3-4B-Instruct-2507 user-model
  supervised fine-tuning.

- [tau2-eval-user-stop-parser](../experiments/tau2-eval-user-stop-parser/README.md) —
  tau2-bench official Pass@4 with the STOP-trained user model and the user
  sglang Qwen tool-call parser enabled; telecom user tools now execute (0 →
  ~1000) and telecom metrics rise sharply.

- [tau2-eval-user-stop-v2-parser](../experiments/tau2-eval-user-stop-v2-parser/README.md) —
  completed tau2-bench comparison of Qwen3.5-4B, Qwen3-4B-Instruct-2507,
  and the multitool SFT Agent under Gemini, original Qwen, v1, and v2 User
  simulators, including official `pass^k` and at-least-one-success `pass@k`.

- [tau2-eval-qwen36-user-timing-tp2](../experiments/tau2-eval-qwen36-user-timing-tp2/README.md) —
  four-GPU official tau2 evaluation of raw Qwen3-4B-Instruct-2507 with a
  non-thinking Qwen3.6-27B User on TP2, retaining request and stage timing
  evidence for the final wall-time breakdown.

- [vitabench-qwen35-smoke](../experiments/vitabench-qwen35-smoke/README.md) —
  install and compatibility validation for VitaBench's 29 dependencies, plus a
  one-task Qwen3.5-4B smoke using the existing remote user/evaluator API.

- [vitabench-qwen35-full-eval](../experiments/vitabench-qwen35-full-eval/README.md) —
  full Chinese VitaBench evaluation of Qwen3.5-4B across four 100-task suites
  with four trials per task on four GPUs, strict completeness checks, lossless
  Qwen/GPT/Gemini SFT journals, and comparison to the published four-domain
  average score of 22.0.

- [vitabench-qwen35-local-role-eval](../experiments/vitabench-qwen35-local-role-eval/README.md) —
  cost-controlled Chinese VitaBench evaluation of Qwen3.5-4B with local
  Qwen3.6-27B user-simulator and evaluator replicas on eight 80GB GPUs.

- [vitabench-evaluator-thinking-ab](../experiments/vitabench-evaluator-thinking-ab/README.md) —
  paired frozen-prompt and chained-state replay of the 6,349 persisted
  VitaBench evaluator windows; the measured speed/quality tradeoff establishes
  Qwen3.6-27B non-thinking evaluation as the default for fresh runs.

- [vitabench-qwen3-4b-instruct-2507-full-eval](../experiments/vitabench-qwen3-4b-instruct-2507-full-eval/README.md) —
  complete Chinese VitaBench evaluation of raw Qwen3-4B-Instruct-2507 with
  local non-thinking Qwen3.6-27B User and Evaluator roles on eight 80GB GPUs.
