# Raw-all versus Processed vanilla GRPO results

## Conclusion

Primary result: `processed rl leads with ci excluding zero`. The decision uses the two-seed iter99 overall pass@1 mean; intermediate checkpoints are diagnostic only.

The trajectory-level explanation of Raw-all's collapse and Processed RL's
domain-dependent changes is in [Behavior analysis](BEHAVIOR_ANALYSIS.md).

## Iter99 two-seed results

| Model | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---:|---:|---:|---:|---:|
| Raw-all SFT | 53.12% | 82.50% | 23.00% | 74.48% | 46.03% |
| Raw-all RL iter99 | 22.75% | 42.50% | 12.50% | 52.24% | 31.45% |
| Processed SFT | 55.75% | 81.00% | 28.00% | 74.76% | 44.11% |
| Processed RL iter99 | 50.75% | 80.00% | 22.50% | 73.73% | 46.21% |

## Final deltas

| Comparison | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---:|---:|---:|---:|---:|
| raw-all-rl-minus-sft | -30.37 pp | -40.00 pp | -10.50 pp | -22.24 pp | -14.58 pp |
| processed-rl-minus-sft | -5.00 pp | -1.00 pp | -5.50 pp | -1.02 pp | +2.10 pp |
| processed-rl-minus-raw-all-rl | +28.00 pp | +37.50 pp | +10.00 pp | +21.50 pp | +14.77 pp |

## Paired task bootstrap

Intervals use 100,000 task-level resamples; seed 300 and 301 deltas are averaged within each task before resampling.

| Comparison | Metric | Delta | 95% CI | Excludes zero |
|---|---|---:|---:|---:|
| raw-all-rl-minus-sft | pass@1 | -30.38 pp | [-36.50 pp, -24.25 pp] | yes |
| raw-all-rl-minus-sft | pass@4(any) | -40.00 pp | [-48.50 pp, -31.50 pp] | yes |
| raw-all-rl-minus-sft | pass^4 | -10.50 pp | [-17.00 pp, -4.50 pp] | yes |
| processed-rl-minus-sft | pass@1 | -5.00 pp | [-9.38 pp, -0.62 pp] | yes |
| processed-rl-minus-sft | pass@4(any) | -1.00 pp | [-7.00 pp, +5.00 pp] | no |
| processed-rl-minus-sft | pass^4 | -5.50 pp | [-11.50 pp, +0.50 pp] | no |
| processed-rl-minus-raw-all-rl | pass@1 | +28.00 pp | [+21.75 pp, +34.38 pp] | yes |
| processed-rl-minus-raw-all-rl | pass@4(any) | +37.50 pp | [+28.00 pp, +47.00 pp] | yes |
| processed-rl-minus-raw-all-rl | pass^4 | +10.00 pp | [+3.50 pp, +17.00 pp] | yes |

## Per-seed direction

| Seed | Processed RL - Raw-all RL pass@1 |
|---:|---:|
| 300 | +30.50 pp |
| 301 | +25.50 pp |

## Seed-300 checkpoint curve

| Arm | Iteration | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---:|---:|---:|---:|---:|---:|
| raw-all | 9 | 52.75% | 80.00% | 20.00% | 74.30% | 45.80% |
| raw-all | 39 | 56.25% | 84.00% | 28.00% | 77.35% | 48.14% |
| raw-all | 69 | 25.50% | 56.00% | 7.00% | 70.63% | 38.00% |
| raw-all | 99 | 21.00% | 40.00% | 11.00% | 49.59% | 29.39% |
| processed | 9 | 54.75% | 81.00% | 27.00% | 76.15% | 43.04% |
| processed | 39 | 57.25% | 81.00% | 29.00% | 74.05% | 46.41% |
| processed | 69 | 52.25% | 80.00% | 24.00% | 73.73% | 44.27% |
| processed | 99 | 51.50% | 81.00% | 23.00% | 72.81% | 46.80% |

## Training diagnostics

| Arm | Groups / trajectories | Raw reward mean | KL mean / max | Accepted binary zero-variance groups | Truncation mean / max | Over-cap retried / rate | Permanently over-cap / rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw-all | 500 / 4000 | 0.2905 | 36.488502 / 204.620630 | 319 (63.80%) | 0.0025 / 0.0500 | 274 / 5.92% | 27 / 0.58% |
| processed | 500 / 4000 | 0.4480 | 0.005460 / 0.013122 | 242 (48.40%) | 0.0000 / 0.0000 | 176 / 4.04% | 13 / 0.30% |

Observation points from the full per-update curves stored in `RESULTS.json`:

| Arm | Step | Raw reward | K2 drift | Grad norm | Zero-variance groups | Truncation |
|---|---:|---:|---:|---:|---:|---:|
| raw-all | 0 | 0.3250 | 0.000000 | 0.2971 | 3 | 0.0000 |
| raw-all | 9 | 0.3250 | 0.000677 | 0.4524 | 1 | 0.0000 |
| raw-all | 39 | 0.7000 | 0.049570 | 0.3886 | 3 | 0.0000 |
| raw-all | 69 | 0.0500 | 72.971790 | 2.0641 | 4 | 0.0250 |
| raw-all | 99 | 0.2000 | 58.641052 | 0.0000 | 5 | 0.0000 |
| processed | 0 | 0.4250 | 0.000000 | 0.3828 | 2 | 0.0000 |
| processed | 9 | 0.2250 | 0.000481 | 0.3544 | 2 | 0.0000 |
| processed | 39 | 0.8500 | 0.003556 | 0.3467 | 3 | 0.0000 |
| processed | 69 | 0.4750 | 0.009360 | 0.3011 | 4 | 0.0000 |
| processed | 99 | 0.6750 | 0.010570 | 0.2843 | 3 | 0.0000 |

Both arms completed 100 finite updates and accepted Airline/Retail/Telecom groups in the exact 100/200/200 totals. KL is a coefficient-zero K2 drift diagnostic and entropy does not enter the loss.

## Scope

One paired training seed; evidence describes vanilla-GRPO trainability and final capability, not training-seed-invariant superiority.
