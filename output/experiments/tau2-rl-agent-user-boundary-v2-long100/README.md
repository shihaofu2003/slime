# tau2-rl-agent-user-boundary-v2-long100

Purpose: retrain the v3-selected Contract + boundary SFT for updates 0-99 to
isolate training length, using a fresh optimizer/checkpoint lineage and health-only
iter9/iter19 continuation gates.

## Problem and approach

This run closes a specific gap left by the earlier experiments rather than
changing the reward recipe again:

| Prior problem and evidence | Diagnosis | Long100 treatment |
|---|---|---|
| The airline stability sweep made `2e-6` numerically stable, but its seed-300 result versus the matching multitool SFT was only `+2.25pp` pass@1 with CI `[-2.84,+7.34]pp`, while pass^4 fell `1pp`. KL stayed below `0.06`. | The recipe no longer collapsed, but the evaluation did not establish RL > SFT and left under-training plausible. | Keep LR `2e-6`, K2 KL `0.01`, K=8, reward/penalties, rollout horizon, and v1 User fixed; test training length only. |
| The first boundary-v2 Pilot A improved namespace and held-out capability at iter9, but `ITER9_SAFETY_GATE.json` stopped promotion on framework truncation `30.63%`, malformed trajectories `18.13%`, execution-error trajectories `15.63%`, and `max_steps` `29.58%`. | Those are important behavior diagnostics, but none demonstrated OOM, NaN/Inf, corrupt checkpoints, quota drift, or Contract/turn-credit misalignment. Framework `truncated` also mixed normal horizon endings with timeout, context-window, and token-cap failures. Stopping at iter9 therefore confounded the training-length test. | Retain the historical artifact and its original label for traceability, without treating it as evidence that iter9 was invalid. Start from SFT with a fresh optimizer/root, and keep behavior and termination metrics for the final evaluation. |
| A segmented run must resume exactly while changing domain quota and extending the scheduler horizon. | Reusing the historical iter9 optimizer would not be a fresh 100-update test; loose root/gate matching could silently resume the wrong lineage. | Use isolated `long100-a/b/final` stages, exact checkpoint-root and predecessor-gate checks, stage-specific quota validation, and optimizer/RNG resume only inside the new lineage. |

The resulting checker verifies finite metrics, correct checkpoint recovery,
domain quotas, and rollout/train alignment across resumed stages. The original
run-control policy also stopped on its KL threshold. In interpreting results,
OOM, non-finite training, corrupt recovery, or protocol drift invalidate a run;
KL, malformed calls, execution errors, and termination rates remain diagnostics
to review with held-out capability.

## Fixed configuration

- Checkpoint root: `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804`.
- SFT source: [turn-aware-rl-v3 Contract + boundary selection](../tau2-sft-agent-user-boundary-v2/SFT_PROMOTION.json).
- LR `2e-6`; K2 KL `0.01`; K=8; turn-credit-v1 reward/penalties;
  `max_steps=60`; v1 STOP-trained User.
- `long100-a`: updates 0-9, quota `3/2/1`; `long100-b`: updates 10-19,
  quota `3/1/2`; `long100-final`: updates 20-99, quota `2/2/2` and
  checkpoints every 10 updates.
- Saved RL checkpoints: iter9, iter19, iter29, iter39, iter49, iter59, iter69,
  iter79, iter89, and iter99; converted model: `iter_0000099_hf`.
- The historical [iter9 early-stop artifact](../tau2-rl-agent-user-boundary-v2/ITER9_SAFETY_GATE.json)
  remains unchanged; iter9 is retained as a valid capability control.

## Gates and evaluation

- `ITER9_HEALTH_GATE.json`, `ITER19_HEALTH_GATE.json`, and
  `ITER99_HEALTH_GATE.json` hard-check checkpoint/resume state, complete finite
  metrics, KL, exact quota replacement, signed Contract, and turn-credit spans.
- Framework truncation is reported as `max_steps`, `timeout`, `context_window`,
  `over_token_cap`, or `other`. Behavior-error rates are diagnostic only.
- Iter99 evaluation uses seeds 300 and 301, the full 100-task test split, four
  trials, temperature `0.6`, and `max_steps=200`; seed 300 also runs namespace
  probes. Each eval job uses two GPUs, one each for Agent and User.
- `ITER99_LONG100_DECISION.json` compares selected SFT, historical RL iter9,
  and long100 iter99 and classifies the result as `win`, `usable`, or `reject`.

## Jobs

- `pt-6ljb73fh`, `fsh-boundary-v2-long100-a-0805-003753`: updates 0-9,
  8-GPU normal priority, quota `3/2/1`, succeeded —
  [run log](jobs/fsh-boundary-v2-long100-a-0805-003753/run_20260805_003753.log),
  [health gate](ITER9_HEALTH_GATE.json) (`pass`).
- `pt-87rkqxvh`, `fsh-boundary-v2-long100-b-0805-011948`: updates 10-19,
  8-GPU normal priority, failed before update 10 because Megatron rejected the
  expanded scheduler horizon (`480` vs `960`) —
  [run log](jobs/fsh-boundary-v2-long100-b-0805-011948/run_20260805_011949.log).
- `pt-6qfykywh`, `fsh-boundary-v2-long100-b-r1-0805-013349`: updates 10-19
  retry with scheduler-horizon override, 8-GPU normal priority, succeeded —
  [run log](jobs/fsh-boundary-v2-long100-b-r1-0805-013349/run_20260805_013349.log),
  [health gate](ITER19_HEALTH_GATE.json) (`pass`).
- `pt-iju9stpv`, `fsh-boundary-v2-long100-final-0805-021636`: updates 20-99,
  8-GPU normal priority, balanced quota `2/2/2`, succeeded —
  [run log](jobs/fsh-boundary-v2-long100-final-0805-021636/run_20260805_021636.log),
  [health gate](ITER99_HEALTH_GATE.json) (`pass`).
- `pt-st2no821`, `fsh-boundary-v2-long100-iter99-convert-0805-055858`:
  iter99 to HF, 1-GPU normal priority, succeeded —
  [run log](jobs/fsh-boundary-v2-long100-iter99-convert-0805-055858/run_20260805_055858.log).
- `pt-ac9om8s6`, `fsh-boundary-v2-long100-iter99-seed300-0805-060935`:
  iter99 official seed-300 evaluation plus namespace probes, 2-GPU normal
  priority, succeeded —
  [run log](jobs/fsh-boundary-v2-long100-iter99-seed300-0805-060935/run_20260805_060935.log),
  [summary](eval/iter99/seed300_summary.json),
  [namespace probes](eval/iter99/namespace_probe.json) (`pass`).
- `pt-87g787k1`, `fsh-boundary-v2-long100-iter99-seed301-0805-060935`:
  iter99 official seed-301 evaluation, 2-GPU normal priority, succeeded —
  [run log](jobs/fsh-boundary-v2-long100-iter99-seed301-0805-060935/run_20260805_060935.log),
  [summary](eval/iter99/seed301_summary.json).

## Result

Final [decision](ITER99_LONG100_DECISION.json): **`win`**. All health,
protocol, namespace, and capability non-inferiority conditions passed; iter99
pass@1 and pass@4(any) strictly exceeded both controls in each seed.

| Seed | Model | pass@1 | pass@4(any) | pass^4 |
|---:|---|---:|---:|---:|
| 300 | selected SFT | 23.75% | 42.00% | 8.00% |
| 300 | historical iter9 | 26.25% | 52.00% | 10.00% |
| 300 | long100 iter99 | 27.50% | 53.00% | 9.00% |
| 301 | selected SFT | 23.50% | 45.00% | 9.00% |
| 301 | historical iter9 | 24.25% | 48.00% | 8.00% |
| 301 | long100 iter99 | 31.50% | 60.00% | 12.00% |

Arithmetic means over the two formal seeds make the training-length effect
clear:

| Model | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| selected SFT | 23.63% | 43.50% | 8.50% |
| historical iter9 | 25.25% | 50.00% | 9.00% |
| **long100 iter99** | **29.50%** | **56.50%** | **10.50%** |
| iter99 - selected SFT | **+5.88pp** | **+13.00pp** | **+2.00pp** |
| iter99 - historical iter9 | **+4.25pp** | **+6.50pp** | **+1.50pp** |

Mean domain pass@1 also improved against both same-protocol controls:

| Model | Airline | Retail | Telecom |
|---|---:|---:|---:|
| selected SFT | 31.25% | 25.63% | 17.81% |
| historical iter9 | 32.50% | 26.56% | 20.31% |
| **long100 iter99** | **38.13%** | **31.25%** | **23.44%** |

For historical orientation, the older parser-on/stability evaluations used the
same 100-task × 4-trial scale, seed 300, v1 User, and Agent temperature `0.6`,
but not the signed boundary profile or selected SFT. They are not causal
controls for long100:

| Historical model/evaluation | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| Raw Qwen3-4B-Instruct-2507 | 23.50% | 45.00% | 6.00% |
| Old multitool SFT | 23.25% | 46.00% | 8.00% |
| Old RL `2e-6` iter99 | 25.50% | 49.00% | 7.00% |
| Old RL `3e-6` iter99 | 23.50% | 47.00% | 5.00% |
| Old RL `5e-6` iter99 | 18.50% | 39.00% | 3.00% |
| Long100 iter99, signed profile seed 300 | 27.50% | 53.00% | 9.00% |
| Raw Qwen3.5-4B, different base | 47.20% | 78.00% | 19.00% |

Sources for the historical rows are the
[parser-on baseline evaluation](../tau2-eval-user-stop-parser/README.md) and
[stability-sweep final evaluation](../tau2-rl-stability-k2-fieldreward-final-eval/README.md).

Per-domain metrics, all behavior diagnostics, and paired 95% bootstrap CIs are
in the seed-300 comparison ([JSON](eval/comparisons/iter99_seed300.json),
[Markdown](eval/comparisons/iter99_seed300.md)) and seed-301 comparison
([JSON](eval/comparisons/iter99_seed301.json),
[Markdown](eval/comparisons/iter99_seed301.md)). Telecom namespace
affected/attempts/attributed terminations fell to `8/8/0` and `10/13/0` for
seeds 300 and 301, respectively. The first `long100-b` launch failed before
training and left iter9 unchanged; its successful retry overrode only the
scheduler horizon while retaining optimizer state.

The paired uncertainty is seed-dependent. Against selected SFT, seed 300 has
pass@1 CI `[-1.00,+8.50]pp` and pass@4(any) CI `[+2.00,+20.00]pp`; seed 301 has
`[+2.50,+13.50]pp` and `[+4.00,+26.00]pp`. Against historical iter9, both
coverage intervals exclude zero on seed 301 but not seed 300. All overall
pass^4 intervals include zero. Thus `win` is the predeclared point-estimate,
health, protocol, namespace, and non-inferiority classification; it is not a
claim that every improvement is independently statistically significant.

Validation: 90 related CPU tests, the in-job rollout preflight, `py_compile`,
`bash -n`, strict JSON parsing, and 8-GPU training/2-GPU evaluation dry-runs
passed. No CI file was added; the historical artifact retains its original
`fail` field, which records the old stopping rule rather than checkpoint
invalidity.
