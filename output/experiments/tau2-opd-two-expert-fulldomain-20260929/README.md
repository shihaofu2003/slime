# tau2-opd-two-expert-fulldomain-20260929

Purpose: keep the Airline gain of `tau2-opd-airline-into-mix-20260929` while removing its cross-domain cost. That arm's airline-only pure OPD lifted Airline pass@1 40.00→53.75% (teacher level) but overall fell 31.22→30.84%. Here the same mix RL student trains on ALL four domains with two teachers: the Airline expert teaches airline trajectories, and the mix RL model itself anchors retail/telecom/banking (anti-forgetting self-distillation). Requested 2026-09-29.

## Design

Correction 2026-09-29: this run used the wrong student/anchor checkpoint (0927 credit-signal-v1 iter129). Comparisons against the evaluated0925 mix iter139 do not establish OPD gains or retention over the actual initialization. The [corrected rerun](../tau2-opd-mix139-airline-20260929/README.md) uses mix139 for both roles and retains the user-selected historical buffer32 recipe. Existing scores and logs are retained as measurements of the wrong-initialization arm.

- **Student**: progress-db-count-v1 mix RL, fresh optimizer (`REF_LOAD` + `REF_CKPT_STEP=129`, empty `LOAD_DIR`) — identical to the airline-into-mix arm. The job loaded the 0927 credit-signal-v1 run's `iter_0000129` weights (never evaluated); the documented student baseline is the 0925 run's final `iter139` (officially evaluated, 31.22 / 48.73 / 13.71 overall).
- **Teacher 1 (airline)**: Airline expert `iter_0000029_hf` (53.75/75.00/35.00 seed300), TP2 on GPUs 0,1, port 31001.
- **Teacher 2 (anchor)**: mix RL `iter_0000129` converted to HF in-job (CPU), TP2 on GPUs 2,3, port 31002; serves `TAU2_OPD_RETAIL/TELECOM/BANKING_URL`. Anchoring at the exact student init means non-airline reverse-KL signal is zero at step 0 and only resists drift.
- **User**: persistent 4-GPU pool, two TP2 Qwen3.6-27B nonthinking replicas behind a router (`scripts/serve_tau2_user_pool.sh`), publishes `user_endpoint.env`.
- **No trajectory-level advantage**: `TAU2_OPD_PURE=1` pins every reward to 0.0; the only learning signal is the per-token OPD reverse-KL term (`opd_kl_coef` 1.0). Task reward kept as `raw_reward` diagnostic. Group size K=1.
- **Data**: `four_domain_train.jsonl` (2524 rows: airline 1148, retail 563, banking 542, telecom 271) from `tau2-opd-four-domain-20260923/data/`, shuffled with replacement. Per-sample `metadata.domain` drives teacher routing in `examples/tau2-bench/opd/tau2_opd.py`.
- **Budget**: 160 updates ≈ one source pass (ceil(2524/16)=158), giving airline ~1160 trajectories of exposure — matching the airline-only arm's ~1280 — while the other domains get their own exposure. Save every 10.
- **Recipe**: LR 2e-6, KL/entropy 0, eps clip 0.2/0.2, TIS [0,2], async pool 32 / pending 64 / unlimited lag, 4 environment workers, Agent temp 1.0, max_steps 200, train token cap 16384, seeds 1234/42. Student 8 GPUs: Trainer2 + three TP2 generators.

## Launch

```bash
bash scripts/submit.sh --experiment tau2-opd-two-expert-fulldomain-20260929 --gpus 4 \
  --name tau2-opd-mix-user-persistent output/experiments/tau2-opd-two-expert-fulldomain-20260929/run_user.sh
bash scripts/submit.sh --experiment tau2-opd-two-expert-fulldomain-20260929 --gpus 4 \
  --name tau2-opd-two-expert-persistent output/experiments/tau2-opd-two-expert-fulldomain-20260929/serve_two_experts.sh
bash scripts/submit.sh --experiment tau2-opd-two-expert-fulldomain-20260929 --gpus 8 \
  --name tau2-opd-two-expert-iter129 output/experiments/tau2-opd-two-expert-fulldomain-20260929/run_opd.sh
# After training completes (160 updates -> iter_0000159):
bash scripts/submit.sh --experiment tau2-opd-two-expert-fulldomain-20260929 --gpus 8 \
  --name tau2-opd-two-expert-iter159-eval output/experiments/tau2-opd-two-expert-fulldomain-20260929/eval_final.sh
```

Validation 2026-09-29: `bash -n` on all four scripts and `run_opd.sh --dry-run` passed. `examples/tau2-bench/opd/test_tau2_opd.py` and the rollout-logic preflight run inside the training job (`SKIP_PREFLIGHT=0`). The serving job also runs an input-token-logprob preflight on both teachers before publishing endpoints.

## Jobs

| Job | GPUs | Purpose / status | Log |
|---|---:|---|---|
| 24959 | 4 | Persistent User pool; STOPPED 2026-09-29 11:26 CST after eval submission | [Run log](jobs/fsh-tau2-opd-mix-user-persistent-0929-094702/run_*.log) |
| 24960 | 4 | Persistent two-expert service (airline iter29 + mix iter129 anchor; converted iter129 to HF in-job); STOPPED 2026-09-29 11:26 CST after eval submission | [Run log](jobs/fsh-tau2-opd-two-expert-persistent-0929-094718/run_*.log) |
| 24961 | 8 | Pure OPD student, mix iter129 init, four-domain data, 160 updates; all 160 updates complete 2026-09-29 11:05 CST, `iter_0000159` saved, all-finite (final loss 0.0011, `opd_reverse_kl` ~0.008 late-run range, grad_norm ≤2.2) | [Run log](jobs/fsh-tau2-opd-two-expert-iter129-0929-094744/run_*.log) |
| 24983 | 8 | iter159 official four-domain seed300 eval; 788 sims complete, summary written 2026-09-29 12:33 CST, 0 infra errors | [Run log](jobs/24983-tau2-opd-two-expert-iter159-eval-0929-112648993/run_*.log) |
| 25019 | 8 | iter39 official four-domain seed300 eval; SUCCEEDED 2026-09-29 ~15:15 CST, 788 sims, 0 infra errors | [Run log](jobs/25019-tau2-opd-two-expert-iter39-eval-0929-141220143/run_*.log) |
| 25020 | 8 | iter89 official four-domain seed300 eval; 788 sims complete, summary written 2026-09-29 ~15:15 CST, 0 infra errors | [Run log](jobs/25020-tau2-opd-two-expert-iter89-eval-0929-141221210/run_*.log) |
| 25021 | 8 | iter119 official four-domain seed300 eval; SUCCEEDED 2026-09-29 ~15:15 CST, 788 sims, 0 infra errors | [Run log](jobs/25021-tau2-opd-two-expert-iter119-eval-0929-141222242/run_*.log) |

## Evaluation

[eval_final.sh](eval_final.sh) converts `iter_0000159` to HF and runs the unchanged official four-domain seed300 protocol (197 tasks × 4 trials = 788 simulations, BM25, Agent temp 0.6, concurrency 1/2/2/4 global 9). Comparisons: student (mix iter139) 31.22/48.73/13.71 overall and 40.00/70.00/15.00 Airline; airline-only OPD iter79 30.84/43.65/15.23 overall and 53.75/65.00/35.00 Airline; Airline teacher 53.75/75.00/35.00. Success criterion: Airline stays at/near teacher level while overall pass@1/pass^4 no longer degrade vs the student. Submitted only after training completes.

Results (iter159, seed300, pass@1 / pass@4(any) / pass^4, from `eval/opd-two-expert-fulldomain-iter159-four-domain/seed300_*_summary.json`):

| Domain | Raw Instruct | SFT4505 | Vanilla GRPO iter139 | Credit mix iter139 (student) | Two-expert iter159 | Airline-only OPD iter79 | Teacher (Airline) |
|---|---|---|---|---|---|---|---|
| Airline | 33.75 / 65.00 / 10.00 | 42.50 / 65.00 / 30.00 | 42.50 / 60.00 / 25.00 | 40.00 / 70.00 / 15.00 | 47.50 / 75.00 / 15.00 | 53.75 / 65.00 / 35.00 | 53.75 / 75.00 / 35.00 |
| Retail | 52.50 / 72.50 / 30.00 | 58.75 / 77.50 / 35.00 | 61.25 / 87.50 / 37.50 | 63.75 / 85.00 / 40.00 | 62.50 / 90.00 / 35.00 | 61.25 / 82.50 / 37.50 | 59.38 / 90.00 / 30.00 |
| Telecom | 17.50 / 42.50 / 5.00 | 47.50 / 82.50 / 12.50 | 51.88 / 85.00 / 15.00 | 56.25 / 87.50 / 20.00 | 48.75 / 82.50 / 15.00 | 53.75 / 80.00 / 17.50 | 49.38 / 82.50 / 17.50 |
| banking_knowledge | 2.58 / 4.12 / 1.03 | 4.12 / 8.25 / 1.03 | 5.41 / 10.31 / 0.00 | 5.67 / 13.40 / 0.00 | 5.15 / 8.25 / 3.09 | 4.12 / 8.25 / 1.03 | 4.12 / 9.28 / 0.00 |
| Overall (197) | 18.91 / 31.98 / 8.63 | 27.92 / 43.15 / 13.20 | 29.95 / 46.19 / 13.20 | 31.22 / 48.73 / 13.71 | 29.95 / 46.70 / 13.20 | 30.84 / 43.65 / 15.23 | 29.57 / 47.21 / 13.20 |

Lineage sources, all the same seed300 protocol: Raw Instruct from `slime/output/experiments/tau2-banking-expert-sft/eval/raw-full-qwen36-bm25-fixed/`; SFT4505 from `tau2-sft-areal3-banking-simplified-full-20260920/eval/four-domain-sft-iter4505/`; vanilla iter139 and credit mix iter139 from `tau2-mix-rl-sft4505-b128-20260925/eval/{vanilla-grpo-iter139,mix-iter139}-four-domain/`; teacher from `tau2-domain-experts-sft4505-b128/eval/airline-iter29-four-domain/`.

Checkpoint curve (same seed300 protocol; pass@1 / pass@4(any) / pass^4):

| Ckpt | Overall | Airline | Retail | Telecom | banking_knowledge |
|---|---|---|---|---|---|
| student (mix iter139) | 31.22 / 48.73 / 13.71 | 40.00 / 70.00 / 15.00 | 63.75 / 85.00 / 40.00 | 56.25 / 87.50 / 20.00 | 5.67 / 13.40 / 0.00 |
| **iter39** | **30.20 / 44.16 / 15.74** | **53.75 / 75.00 / 35.00** | 63.12 / 87.50 / 40.00 | 47.50 / 77.50 / 12.50 | 4.64 / 6.19 / 3.09 |
| iter89 | 29.70 / 45.69 / 14.21 | 45.00 / 65.00 / 25.00 | 62.50 / 82.50 / 42.50 | 50.00 / 85.00 / 12.50 | 4.64 / 10.31 / 1.03 |
| iter119 | 28.05 / 44.16 / 11.68 | 47.50 / 60.00 / 30.00 | 58.75 / 85.00 / 32.50 | 46.25 / 82.50 / 10.00 | 3.87 / 8.25 / 0.00 |
| iter159 | 29.95 / 46.70 / 13.20 | 47.50 / 75.00 / 15.00 | 62.50 / 90.00 / 35.00 | 48.75 / 82.50 / 15.00 | 5.15 / 8.25 / 3.09 |

The curve changes the earlier "hypothesis not confirmed" reading of iter159. **iter39 achieves the design goal**: Airline reaches the teacher on all three metrics (53.75 / 75.00 / 35.00) while overall pass@1/pass^4 (30.20 / 15.74) stays within noise of the student and pass^4 is the run's best. Continued training then oscillates Airline (45.00→47.50→47.50 pass@1, 35→25→30→15 pass^4) and erodes overall pass^4 (15.74→11.68→13.20) — the same mid-training peak / late decay pattern as the parent RL runs. **iter39 is the selected checkpoint of this arm** (vs airline-only iter79: same Airline pass@1 53.75, higher Airline pass@4(any) 75 vs 65, higher overall pass^4 15.74 vs 15.23, lower Telecom 47.50 vs 53.75). Single seed throughout; the Airline oscillation means iter39's exact numbers should be re-confirmed on a second seed before final claims.
