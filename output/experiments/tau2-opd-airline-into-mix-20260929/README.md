# tau2-opd-airline-into-mix-20260929

Purpose: distill the Airline domain expert's capability into the progress-db-count-v1 mix RL student with pure on-policy distillation (OPD). Requested 2026-09-29 after the mixed evaluation showed the progress Mix RL student (iter139, overall pass@1 31.22%) trails the Airline expert (53.75%) on Airline.

## Design

- **Student**: progress-db-count-v1 mix RL, fresh optimizer via the finetune path (`REF_LOAD` + `REF_CKPT_STEP=129`, empty `LOAD_DIR`). The job loaded the 0927 credit-signal-v1 run's `iter_0000129` weights (never evaluated); the documented student baseline is the 0925 run's final `iter139` (officially evaluated, 31.22 / 48.73 / 13.71 overall).
- **Teacher**: Airline expert `iter_0000029_hf` (`tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-airline-train/checkpoints`), the best observed Airline checkpoint (53.75/75.00/35.00 seed300). Persistent 2-GPU TP2 sglang server, port 31001, publishes `teacher_endpoints.env`.
- **User**: persistent 4-GPU pool, two TP2 Qwen3.6-27B nonthinking replicas behind a router (`scripts/serve_tau2_user_pool.sh`), publishes `user_endpoint.env`.
- **No trajectory-level advantage**: `TAU2_OPD_PURE=1` sets every sample reward to 0.0, so GRPO advantages are identically zero; the only learning signal is the per-token OPD reverse-KL term (task reward kept as `raw_reward` diagnostic).
- **Distillation estimator**: validated per-token reverse KL (tinker-cookbook style, teacher logprob at sampled tokens via sglang `input_token_logprobs`), as in `tau2-opd-four-domain-20260923`. A top-k forward-KL variant was considered but declined: it needs new teacher top-logprobs plumbing plus a non-PG loss path, while this path is already exercised end-to-end.
- **Group size K=1**: with rewards pinned to zero there is no group baseline; larger groups only reduce prompt diversity per batch. `ROLLOUT_BATCH_SIZE=16`, `GLOBAL_BATCH_SIZE=16`.
- **Data**: Airline-only `airline_train.jsonl` (1148 rows), shuffled with replacement. 80 updates (~1.1 source passes), save every 10.
- **Recipe**: LR 2e-6, `opd_kl_coef` 1.0, KL/entropy 0, eps clip 0.2/0.2, TIS [0,2], async pool 32 / pending 64 / unlimited lag, 4 environment workers, Agent temp 1.0, max_steps 200, train token cap 16384, seeds 1234/42. Student 8 GPUs: Trainer2 + three TP2 generators.

## Launch

```bash
bash scripts/submit.sh --experiment tau2-opd-airline-into-mix-20260929 --gpus 4 \
  --name tau2-opd-mix-user-persistent output/experiments/tau2-opd-airline-into-mix-20260929/run_user.sh
bash scripts/submit.sh --experiment tau2-opd-airline-into-mix-20260929 --gpus 2 \
  --name tau2-opd-airline-expert-persistent output/experiments/tau2-opd-airline-into-mix-20260929/serve_airline_expert.sh
bash scripts/submit.sh --experiment tau2-opd-airline-into-mix-20260929 --gpus 8 \
  --name tau2-opd-airline-into-mix-iter129 output/experiments/tau2-opd-airline-into-mix-20260929/run_opd.sh
# After training completes (80 updates -> iter_0000079):
bash scripts/submit.sh --experiment tau2-opd-airline-into-mix-20260929 --gpus 8 \
  --name tau2-opd-airline-into-mix-iter79-eval output/experiments/tau2-opd-airline-into-mix-20260929/eval_final.sh
```

Validation: shell syntax checks and `run_opd.sh --dry-run` passed locally 2026-09-29. `examples/tau2-bench/opd/test_tau2_opd.py` and the rollout-logic preflight run inside the training job (`SKIP_PREFLIGHT=0`).

## Jobs

| Job | GPUs | Purpose / status | Log |
|---|---:|---|---|
| 24945 (`pt-lad9jle3`) | 4 | Persistent User pool; endpoint ready `http://10.119.108.47:30000/v1`; STOPPED 2026-09-29 01:53 CST after eval submission | [Run log](jobs/24945-tau2-opd-mix-user-persistent-0929-005900366/run_*.log) |
| 24946 (`pt-aviiza64`) | 2 | Persistent Airline expert iter29; health + input-logprob preflight passed, `http://10.119.108.145:31001/generate`; STOPPED 2026-09-29 01:53 CST after eval submission | [Run log](jobs/24946-tau2-opd-airline-expert-persistent-0929-005918855/run_*.log) |
| 24947 (`pt-54ebwhpq`) | 8 | Pure OPD student, mix iter129 init, 80 updates; SUCCEEDED 2026-09-29 01:48 CST, `iter_0000079` saved | [Run log](jobs/24947-tau2-opd-airline-into-mix-iter129-0929-010627033/run_*.log) |
| 24954 (`pt-bckq2d5m`) | 8 | iter79 official four-domain seed300 eval; SUCCEEDED 2026-09-29 02:33 CST, 788 sims, 0 infra errors | [Run log](jobs/24954-tau2-opd-airline-into-mix-iter79-eval-0929-015331004/run_*.log) |
| 25000 | 8 | iter49 official four-domain seed300 eval; SUCCEEDED 2026-09-29 ~13:40 CST, 788 sims, 0 infra errors | [Run log](jobs/25000-tau2-opd-airline-into-mix-iter49-eval-0929-123950777/run_*.log) |
| 25001 | 8 | iter59 official four-domain seed300 eval; 788 sims complete, summary written 2026-09-29 ~13:40 CST, 0 infra errors | [Run log](jobs/25001-tau2-opd-airline-into-mix-iter59-eval-0929-123953986/run_*.log) |
| 25002 | 8 | iter69 official four-domain seed300 eval; SUCCEEDED 2026-09-29 ~13:35 CST, 788 sims, 0 infra errors | [Run log](jobs/25002-tau2-opd-airline-into-mix-iter69-eval-0929-123954621/run_*.log) |

Training health: all 80 updates finite. Final-step metrics loss 0.0018, grad_norm 0.29, `opd_reverse_kl` 0.0016 (0.0039 at step 19, trending down), diagnostic `raw_reward` 0.5625→0.625 by the last rollouts. Checkpoints iter9–iter79 under `arms/async/20260929_airline-into-mix-iter129-opd/checkpoints/`.

## Evaluation

[eval_final.sh](eval_final.sh) converts `iter_0000079` to HF and runs the unchanged official four-domain seed300 protocol (197 tasks × 4 trials = 788 simulations, BM25, Agent temp 0.6, concurrency 1/2/2/4 global 9). Selection comparison: student (progress Mix iter139) 31.22/48.73/13.71 overall and 40.00/70.00/15.00 Airline; teacher 53.75/75.00/35.00 Airline. Submitted only after training completes.

Results (iter79, seed300, pass@1 / pass@4(any) / pass^4, from `eval/opd-airline-into-mix-iter79-four-domain/seed300_*_summary.json`):

| Domain | Raw Instruct | SFT4505 | Vanilla GRPO iter139 | Credit mix iter139 (student) | OPD iter79 | Teacher (Airline) |
|---|---|---|---|---|---|---|
| Airline | 33.75 / 65.00 / 10.00 | 42.50 / 65.00 / 30.00 | 42.50 / 60.00 / 25.00 | 40.00 / 70.00 / 15.00 | 53.75 / 65.00 / 35.00 | 53.75 / 75.00 / 35.00 |
| Retail | 52.50 / 72.50 / 30.00 | 58.75 / 77.50 / 35.00 | 61.25 / 87.50 / 37.50 | 63.75 / 85.00 / 40.00 | 61.25 / 82.50 / 37.50 | 59.38 / 90.00 / 30.00 |
| Telecom | 17.50 / 42.50 / 5.00 | 47.50 / 82.50 / 12.50 | 51.88 / 85.00 / 15.00 | 56.25 / 87.50 / 20.00 | 53.75 / 80.00 / 17.50 | 49.38 / 82.50 / 17.50 |
| banking_knowledge | 2.58 / 4.12 / 1.03 | 4.12 / 8.25 / 1.03 | 5.41 / 10.31 / 0.00 | 5.67 / 13.40 / 0.00 | 4.12 / 8.25 / 1.03 | 4.12 / 9.28 / 0.00 |
| Overall (197) | 18.91 / 31.98 / 8.63 | 27.92 / 43.15 / 13.20 | 29.95 / 46.19 / 13.20 | 31.22 / 48.73 / 13.71 | 30.84 / 43.65 / 15.23 | 29.57 / 47.21 / 13.20 |

Lineage sources, all the same seed300 protocol: Raw Instruct from `slime/output/experiments/tau2-banking-expert-sft/eval/raw-full-qwen36-bm25-fixed/`; SFT4505 from `tau2-sft-areal3-banking-simplified-full-20260920/eval/four-domain-sft-iter4505/`; vanilla iter139 and credit mix iter139 from `tau2-mix-rl-sft4505-b128-20260925/eval/{vanilla-grpo-iter139,mix-iter139}-four-domain/`; teacher from `tau2-domain-experts-sft4505-b128/eval/airline-iter29-four-domain/`.

Airline transfer works: pass@1 40.00→53.75% (+13.75pp, reaches the teacher exactly), pass^4 15→35% (+20pp, also teacher level); pass@4(any) 70→65% sits below teacher 75%. Overall pass@1 −0.38pp and pass@4(any) −4.57pp vs the student (single seed, so not established as regression). Diagnostics: action_accuracy 0.381, db_accuracy 0.256, 20 max_steps, 0 infra errors, 0 single-call protocol errors.

Checkpoint curve (same seed300 protocol; pass@1 / pass@4(any) / pass^4):

| Ckpt | Overall | Airline | Retail | Telecom | banking_knowledge |
|---|---|---|---|---|---|
| student (mix iter139) | 31.22 / 48.73 / 13.71 | 40.00 / 70.00 / 15.00 | 63.75 / 85.00 / 40.00 | 56.25 / 87.50 / 20.00 | 5.67 / 13.40 / 0.00 |
| iter49 | 30.96 / 46.19 / 14.21 | 45.00 / 75.00 / 20.00 | 65.62 / 85.00 / 40.00 | 52.50 / 82.50 / 17.50 | 4.90 / 9.28 / 1.03 |
| iter59 | 30.33 / 45.18 / 15.74 | 42.50 / 65.00 / 20.00 | 63.12 / 85.00 / 40.00 | 55.00 / 85.00 / 25.00 | 4.12 / 8.25 / 1.03 |
| iter69 | 29.95 / 43.65 / 15.23 | 47.50 / 70.00 / 25.00 | 64.38 / 85.00 / 42.50 | 49.38 / 77.50 / 17.50 | 4.12 / 7.22 / 1.03 |
| iter79 | 30.84 / 43.65 / 15.23 | 53.75 / 65.00 / 35.00 | 61.25 / 82.50 / 37.50 | 53.75 / 80.00 / 17.50 | 4.12 / 8.25 / 1.03 |

The Airline gain is a late-training phenomenon: pass@1 dips to 42.50 at iter59 before climbing 47.50→53.75, and pass^4 only takes off at iter79 (20→20→25→35). Airline pass@4(any) jumps early (70→75 at iter49) then settles 65-70. Overall is flat (30.0-31.0 / 43.7-46.2 / 13.7-15.7) across the whole curve, so the iter79 overall delta vs the student is inside the run's own noise band. Non-airline domains stay stable (retail 61-66, telecom 49-55, banking ~4-5 pass@1) — no monotone forgetting trend within this curve. Iter79 remains the selected checkpoint.
