# tau2-rl-local-first-dependency-safe-k2-fieldreward

Purpose: validate and, only after a predeclared promotion gate, extend airline
GRPO from the local-first relaxed Agent SFT under one shared
`dependency-safe-multi` protocol.

## Configuration

- Init and fixed KL reference:
  `Qwen3-4B-Instruct-2507_tau2_agent_sft_local_first_relaxed_20260803`;
  rollout tokenizer/model path is its `final_hf` directory.
- Airline 1,148-task train split; 8 GPUs (6 Agent actor/rollout, 1 v1 User,
  1 spare); group size 8; rollout/global batch `6/48`.
- Constant LR `2e-6`; `k2` KL loss `0.01`; algorithm KL `0`; entropy `0`;
  clip `0.4/0.4`; field reward, penalties, dynamic replacement, 16,384-token
  cap, and two overlength retries.
- Stage `train100` targets total update 100 and saves iter19/39/59/79/99.
  Stage `resume200` requires a passing `ITER99_PROMOTION.json`, restores
  iter99 optimizer/RNG state in place, and targets total update 200.

## Evaluation and gates

- Official eval uses all 100 test tasks across airline/retail/telecom, four
  trials, seed 300, Agent `0.6/1.0/8192`, v1 User `0/512`, and max
  steps/errors `200/10`.
- Historical raw/old-SFT/new-SFT/old-RL evals are reused. The comparison
  records their unsigned protocol provenance, requires the new RL candidate
  to carry the current signature, rejects mismatched non-protocol settings,
  seed, task set, trial indices/count, or infrastructure-error artifacts, and
  reports 100,000 task-paired bootstrap samples. Cross-protocol deltas are
  explicitly descriptive rather than a controlled single-variable result.
- Raw, old-SFT, and new-SFT use their existing unsigned
  `dependency_safe_multi_user_stop_parser` artifacts; old-RL uses its existing
  implicit-current-single `user_stop_parser` artifact.
- `run_tau2_local_first_rl_gate.sh` enforces the predeclared eval, termination,
  final-10-update, latest-384-trajectory, and checkpoint-latest thresholds.
  Failed iter99 promotion stops before continuation.
- Final selection excludes ineligible iter99/iter199 checkpoints, then ranks by
  pass@1, pass@4(any), and pass^4; an exact tie keeps iter99.

## Iter99 result

| Model | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| raw | 20.50% | 43.00% | 6.00% |
| old SFT | 21.25% | 41.00% | 10.00% |
| new SFT | 27.00% | 54.00% | 4.00% |
| old RL100 | 25.50% | 49.00% | 7.00% |
| new RL100 | 19.00% | 38.00% | 5.00% |

- Against new SFT, new RL100 is `-8.00pp` pass@1 (95% paired CI
  `[-13.25, -2.75]`) and `-16.00pp` pass@4(any) (`[-27.00, -5.00]`).
- New RL100 pass@1 by domain is airline `28.75%`, retail `23.12%`, telecom
  `10.00%`; the corresponding new-SFT values are `30.00%`, `29.38%`, and
  `23.12%`.
- The iter99 promotion gate failed: strict performance improvement, retail and
  telecom non-inferiority, eval `too_many_errors` (`100 > 21`), and latest-384
  repetition (`0.6224 > 0.5`) failed. Training remained finite, with final-10
  mean k2 KL `0.02590` and truncation `0.4167%`.
- Per the predeclared flow, no `resume200` job was submitted.

## Diagnosis

- Raw Instruct already confuses telecom User/device tools with Agent tools in
  72/160 simulations (272 calls). New SFT reaches 94/160 (418 calls) and 18
  `too_many_errors`; new RL100 reaches 141/160, 1,154 calls, and 93 error
  terminations. SFT did not create the defect from zero, and RL amplified it.
- The SFT filter rejects invalid namespaces, and none of the 353 supervised
  telecom targets calls a User/device tool. This was only an offline selection
  rule: the model-facing SFT system prompt does not explicitly prohibit User
  tools, omits the detailed User-device policy and Agent tool schema, and SFT
  tokenization passes `tools=None`.
- The Agent-data converter also discards `tool_calls` attached to User turns;
  148/353 telecom rows contain 745 resulting blank User turns before device
  tool results. Telecom was only 353/19,318 training rows (1.83%), with nine
  canonical multitool targets. These gaps leave weak namespace supervision.
- New and old RL use the same reward entrypoint, dynamic filter, and configured
  field-credit/penalty rules and weights. The old artifact did not record a
  source hash, so byte-identical reward source cannot be established.
- Training is airline-only, while the largest loss is telecom. The field reward
  gives partial credit for tool names/arguments and caps nonexistent/repetition
  penalties; it does not teach the telecom Agent/User tool boundary. Under the
  multi-call protocol this permits more speculative calls without improving DB
  correctness.

## Jobs

- `pt-abpzks27`: new-SFT `train100`, 8 GPUs, normal priority —
  [run log](jobs/fsh-tau2-rl-local-first-depsafe-train100-0803-140328/run_20260803_140328.log).
  Completed 100 updates with iter19/39/59/79/99 checkpoints, finite metrics,
  and signed dependency-safe trajectory metadata.
- `pt-tb1i0zwe`: iter99 Megatron-to-HF conversion, 1 GPU, normal priority —
  [run log](jobs/fsh-tau2-rl-local-first-depsafe-convert-iter99-0803-215120/run_20260803_215121.log).
  Completed; `iter_0000099_hf` contains two safetensors shards and the origin
  tokenizer/config files.
- `pt-bu7xsbf2`: iter99 dependency-safe official eval, 2 GPUs, normal priority —
  [run log](jobs/fsh-tau2-rl-local-first-depsafe-iter99-eval-0803-222520/run_20260803_222520.log).
  Completed all 400 simulations with zero infrastructure errors.
- `pt-1tfoum1u`: superseded old-RL eval; cancelled after detecting that it
  started before protocol content signatures were recorded —
  [run log](jobs/fsh-tau2-old-rl100-depsafe-eval-0803-140328/run_20260803_140328.log).
- `pt-wcnupbqd`: cancelled before evaluation after deciding to reuse the
  existing old-RL result —
  [run log](jobs/fsh-tau2-old-rl100-depsafe-sig-eval-0803-141346/run_20260803_141346.log).
- `pt-yzrdesjn`: cancelled before evaluation after deciding to reuse the
  existing raw-Instruct result —
  [run log](jobs/fsh-tau2-raw-depsafe-sig-eval-0803-141346/run_20260803_141346.log).
- `pt-1v5ovdl2`: cancelled before evaluation after deciding to reuse the
  existing old-SFT result —
  [run log](jobs/fsh-tau2-old-sft-depsafe-sig-eval-0803-141347/run_20260803_141347.log).
- `pt-f5m7ax8h`: cancelled before evaluation after deciding to reuse the
  existing new-SFT result —
  [run log](jobs/fsh-tau2-new-sft-depsafe-sig-eval-0803-141347/run_20260803_141347.log).
- New-RL iter99 conversion and official eval are complete. Promotion failed and
  `resume200` was not submitted.

## Artifacts

- Checkpoint root:
  `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_local_first_dependency_safe_k2_fieldreward_lr2e6_20260803`
- Training trajectories: `trajectories/iter0000-0099.jsonl` and, only after
  promotion, `trajectories/iter0100-0199.jsonl`.
- Comparison/gate outputs: `ITER99_COMPARISON.*`, `ITER99_PROMOTION.json`,
  `ITER199_COMPARISON.*`, `ITER199_ELIGIBILITY.json`, and
  `FINAL_SELECTION.json` as stages complete.
