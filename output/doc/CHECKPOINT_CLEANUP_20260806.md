# Checkpoint cleanup 2026-08-06

Purpose: remove redundant tau2 training-state checkpoints after reconciling the
checkpoint inventory with the recorded SFT/RL promotion and rejection decisions.

## Decision basis

- `tau2-rl-agent-user-boundary-v2-long100` selected long100 iter99 as the
  default Agent model (`win`).
- `tau2-rl-agent-user-boundary-v2-long200` kept the formal decision `reject`;
  its KL-waiver iter199 is diagnostic and remains available as HF weights.
- `tau2-sft-agent-user-boundary-v2` selected Contract + boundary SFT and
  rejected Contract-only.
- `tau2-rl-agent-user-boundary-v2-domain-experts` made Retail iter119 and
  Telecom iter129 OPD-eligible and rejected Airline. Because this experiment
  completed on the cleanup date, all three expert training roots were retained.
- The cluster listing contained no running/starting `fsh` tau2 job. The one
  suspended Retail forgetting job had already written and validated its final
  summary and was suspended only during SGLang cleanup.

## Retained

- Base-model torch-dist roots: `Qwen3-4B-Instruct-2507_slime` and
  `Qwen3-4B_slime`.
- Selected Agent RL: long100 `iter_0000099` and `iter_0000099_hf`.
- Selected Agent SFT: Contract + boundary `iter_0002413` and `final_hf`.
- User models: v1 `iter_0006899` + HF and v2 `iter_0006311` + HF.
- All three 2026-08-06 domain-expert torch-dist roots and all expert HF roots.
- Every converted HF artifact used by historical comparisons, checkpoint
  curves, mixed controls, and the long200 KL-waiver diagnostic.

## Removed

- Early torch-dist checkpoints from retained long100, selected SFT, and v1/v2
  User lineages.
- All torch-dist checkpoints from completed historical or rejected lineages:
  Pilot iter9, long200, local-first RL/SFT, stability LR sweep, Contract-only,
  multitool SFT, older strict SFT, and non-STOP User SFT. Their HF weights remain.
- Smoke/preflight roots, empty roots, and failed/incomplete pre-stability RL
  attempts that had no retained HF artifact.
- Stale `latest_checkpointed_iteration.txt` files in roots with no remaining
  torch-dist checkpoint.

## Result and verification

- Audited deletion manifest: 657 exact targets, estimated 34 TiB.
- `/mnt/afs` changed from 151,370 GiB used / 2,231 GiB available (99%) to
  117,228 GiB used / 36,373 GiB available (77%): 34,142 GiB additional free
  space, about 33.3 TiB by filesystem accounting.
- `checkpoints/` is 1.1 TiB after cleanup and contains 26 top-level roots,
  16 torch-dist checkpoint directories, and 31 HF directories.
- The cleanup script's post-run dry-run reports
  `targets=0 reclaimable=0B mode=--dry-run`.
- The selected Agent RL/SFT and v1/v2 User HF directories each retain both
  nonempty safetensors shards; their retained final torch-dist directories are
  each approximately 53 GiB.

The exact idempotent cleanup policy and safety checks are in
[`scripts/cleanup_tau2_checkpoints_20260806.sh`](../../scripts/cleanup_tau2_checkpoints_20260806.sh).
