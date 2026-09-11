# tau2-rl-agent-user-boundary-v2-checkpoint-curve

Purpose: evaluate the long100 lineage at iter9/39/69/89 under one fixed signed
seed-300 protocol, then combine those points with the existing iter99 result to
locate improvement, plateau, or regression without selecting a model from one seed.

## Protocol and authorization

- Source: long100 torch-dist checkpoints; curve HF conversions use the isolated
  `Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_checkpoint_curve_hf_20260805`
  root and never modify the long100 root.
- Each checkpoint requires its passing health gate, an explicit successful-save
  event, and a complete file/shard manifest before conversion or evaluation.
- Each 2-GPU job converts one checkpoint and evaluates seed 300 on the full
  100-task test split with four trials, temperature `0.6`, `max_steps=200`, v1
  User, and `agent-owned-dependency-safe-multi`; Agent and User use one GPU each.

## Jobs

- `pt-pkdz4mom`, `fsh-boundary-v2-curve-iter9-seed300-0805-125418`:
  iter9 conversion + evaluation, 2-GPU normal priority —
  [run log](jobs/fsh-boundary-v2-curve-iter9-seed300-0805-125418/run_20260805_125418.log).
- `pt-p7j3tu2f`, `fsh-boundary-v2-curve-iter39-seed300-0805-125418`:
  iter39 conversion + evaluation, 2-GPU normal priority —
  [run log](jobs/fsh-boundary-v2-curve-iter39-seed300-0805-125418/run_20260805_125418.log).
- `pt-tm6urzn3`, `fsh-boundary-v2-curve-iter69-seed300-0805-125419`:
  iter69 conversion + evaluation, 2-GPU normal priority —
  [run log](jobs/fsh-boundary-v2-curve-iter69-seed300-0805-125419/run_20260805_125419.log).
- `pt-p0x6ozbo`, `fsh-boundary-v2-curve-iter89-seed300-0805-125419`:
  iter89 conversion + evaluation, 2-GPU normal priority —
  [run log](jobs/fsh-boundary-v2-curve-iter89-seed300-0805-125419/run_20260805_125419.log).

## Outputs

[`CHECKPOINT_CURVE.json`](CHECKPOINT_CURVE.json) and
[`CHECKPOINT_CURVE.md`](CHECKPOINT_CURVE.md) report selected SFT and same-lineage
iter9/39/69/89/99, with historical iter9 and raw Instruct marked as background
controls. They include overall/domain metrics, behavior/termination, telecom
namespace, and 100,000-sample paired task-bootstrap intervals versus SFT and
adjacent same-lineage checkpoints.

## Result

All four normal-priority jobs succeeded on 2026-08-05 and each produced the
authorized isolated HF conversion plus all 400 evaluation trajectories.

| Seed-300 model | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| selected SFT | 23.75% | 42.00% | 8.00% |
| same-lineage iter9 | 25.00% | 52.00% | 8.00% |
| same-lineage iter39 | 26.25% | 51.00% | 9.00% |
| same-lineage iter69 | 24.00% | 47.00% | 8.00% |
| **same-lineage iter89** | **29.75%** | **53.00%** | **10.00%** |
| same-lineage iter99 | 27.50% | 53.00% | 9.00% |
| historical iter9 (background) | 26.25% | 52.00% | 10.00% |
| raw Instruct (background) | 22.75% | 49.00% | 5.00% |

The same-lineage direction is non-monotonic: pass@1 changes
`+1.25/-2.25/+5.75/-2.25pp` over 9→39→69→89→99. The iter69→89 pass@1 gain has
paired 95% CI `[+0.25,+11.25]pp`; iter89→99 is not significant overall
(`[-6.75,+2.25]pp`) but its telecom pass@1 delta is `-6.88pp`
(`[-13.12,-0.62]pp`). Iter89 is therefore the single-seed curve peak, not an
automatic selection: this diagnostic neither changes the selected two-seed
iter99 model nor stops the independent long200 continuation.
