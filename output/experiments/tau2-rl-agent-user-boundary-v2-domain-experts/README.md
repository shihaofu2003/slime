# tau2-rl-agent-user-boundary-v2-domain-experts

Purpose: compare three 30-update single-domain continuations from selected
long100 iter99 with matched mixed-RL prefixes as future OPD teacher candidates.
This experiment does not start distillation.

## Fixed design

- Airline, Retail, and Telecom each resume the selected long100 iter99 model,
  optimizer, RNG, dataset fingerprint, and all three sampler cursors into a
  separate checkpoint root. Their update-100–129 quotas are respectively
  `6/0/0`, `0/6/0`, and `0/0/6` in airline/retail/telecom order.
- The full three-domain training JSONL and `DomainQuotaDataSource` remain in use.
  LR `2e-6`, K2 `0.01`, K=8, field reward/penalties, `max_steps=60`, v1 User,
  signed Agent-owned Contract, and turn-credit-v1 match long100.
- Checkpoints are iter109/119/129. Scheduler horizon becomes 130 while restored
  optimizer state and consumed samples remain intact. Before iter109, a retry
  reloads iter99; afterward it reloads only the newest complete expert checkpoint.
- Three 8-GPU normal-priority jobs may run together (24-GPU peak). Available
  shared space before submission must be at least 750GB. The pre-submit check
  found 3469GB free; the final artifact audit still found 2237GB free.

## Checkpoint integrity

The existing mixed prefixes and all expert checkpoints were checked for the
correct source, domain quota, restored sampler cursors, complete save state, and
rollout/train alignment before conversion. OOM, NaN/Inf, corrupt recovery,
quota drift, or Contract/turn-credit misalignment would invalidate a run. KL is
reported as a training diagnostic and is not used to discard a checkpoint.

## Evaluation and selection

Phase 1 evaluates mixed iter109/119/129 on all 100 test tasks and each expert on
its target domain at seed300, four trials, temperature `0.6`, `max_steps=200`.
Reports compare equal updates and expert109 versus mixed129, where both consumed
60 task groups (480 trajectories) from that domain. Per expert, checkpoint
selection primarily uses pass@1, with pass@4(any), pass^4, and the earlier
iteration resolving ties.

Phase 2 evaluates only each selected expert and its same-iteration mixed control
at seed301. The selected expert also evaluates the other two domains at seed300
and runs namespace probes. The original plan required the Telecom probe for OPD
selection; it passed. Forgetting, behavior, DB/action, pass^4, KL, and namespace
results are reported as diagnostics alongside task success.

The stored decision files use the original `advantage`, `near-parity`, and
`reject` labels with a 5pp comparison bound. Those labels are retained for
reproducibility; the interpretation below also reports the continuous metrics
and does not treat a comparison loss as an invalid experiment.

All three training jobs completed updates 100-129 and all nine expert
checkpoints were complete. Seed300 selected Airline iter129, Retail iter119,
and Telecom iter129.
Their training raw-success mean/last-10 mean was respectively
`38.33/42.29%`, `34.58/37.71%`, and `11.60/12.08%`. The distributed selected
checkpoints are `iter_0000129`, `iter_0000119`, and `iter_0000129` under the
three domain-specific roots recorded with the jobs. The complete 9-point
expert curve, 9 matched mixed points, equal-domain-sample comparison, and all
seed300 paired task-bootstrap intervals are in
[`DOMAIN_EXPERT_CURVE.md`](DOMAIN_EXPERT_CURVE.md).

| Domain | Selected | seed300 expert | seed300 delta to comparator | seed301 expert | seed301 delta to comparator | Original label |
|---|---:|---:|---:|---:|---:|---|
| Airline | 129 | 37.50/55.00/25.00% | -2.50/-10.00pp | 32.50/55.00/15.00% | -8.75/-5.00pp | reject |
| Retail | 119 | 31.87/47.50/17.50% | +1.87/+0.00pp | 35.00/55.00/25.00% | +0.00/-2.50pp | advantage |
| Telecom | 129 | 28.12/67.50/0.00% | +0.00/+2.50pp | 31.25/65.00/2.50% | +0.62/+0.00pp | advantage |

Metric triplets are pass@1/pass@4(any)/pass^4; comparator deltas show pass@1 and
pass@4(any). All evaluations completed without training-integrity or protocol
errors, and Telecom's two-temperature namespace probe passed. Retail and
Telecom are useful OPD teacher candidates. Airline is not selected as the
target-domain teacher because it trails the mixed comparator on both seeds; the
result remains useful evidence that cross-domain training provides positive
transfer. No distillation was started.

The experiment is complete for these checkpoints. No longer-training
continuation is currently scheduled: Airline's curve was still rising, but its
two-seed comparison did not support selection; Retail pass@1 peaked at iter119;
Telecom pass@1 rose slowly while
training truncation increased from `52.71%` to `58.96%`, K2 KL from `0.0223` to
`0.0507`, and pass^4 fell from `5.00%` to `0.00%`. These signals do not support
treating extra pure-domain updates as an established fix. A future extension
could test that hypothesis independently; the current measurements remain the
result of this 30-update experiment.

| Selected expert | Forgetting domain | pass@1 | pass@4(any) | pass^4 |
|---|---|---:|---:|---:|
| Airline iter129 | Retail | 30.00% | 55.00% | 12.50% |
| Airline iter129 | Telecom | 28.75% | 75.00% | 0.00% |
| Retail iter119 | Airline | 41.25% | 60.00% | 20.00% |
| Retail iter119 | Telecom | 21.88% | 50.00% | 2.50% |
| Telecom iter129 | Airline | 40.00% | 60.00% | 20.00% |
| Telecom iter129 | Retail | 28.75% | 45.00% | 17.50% |

Selected-expert versus same-iteration mixed paired 95% task-bootstrap CIs for
pass@1/pass@4(any) were Airline seed300 `[-12.50,+7.50]/[-30.00,+10.00]pp`
and seed301 `[-7.50,+8.75]/[-20.00,+20.00]pp`; Retail seed300
`[-3.12,+8.12]/[-15.00,+15.00]pp` and seed301
`[-6.88,+7.50]/[-12.50,+12.50]pp`; Telecom seed300
`[-8.75,+8.12]/[-12.50,+20.00]pp` and seed301
`[-10.00,+11.88]/[-17.50,+17.50]pp`. Full intervals against both controls are
stored in each `*_EXPERT_DECISION.json`.

## Jobs

- `pt-5dj2gojx`, `fsh-boundary-v2-domain-expert-preflight-0806-013547`:
  1-GPU normal-priority rollout-logic preflight before the three training
  submissions, succeeded — [run log](jobs/fsh-boundary-v2-domain-expert-preflight-0806-013547/run_20260806_013547.log).
- `pt-zys9tg81` / `pt-5ej10k1a` / `pt-d20d1y2x`: first three submissions were
  deleted before startup/checkpoint creation because their shared User-server
  log path was detected during post-submit review. Submit evidence:
  [Airline](jobs/fsh-boundary-v2-airline-expert-0806-013843/submit_20260806_013843.log),
  [Retail](jobs/fsh-boundary-v2-retail-expert-0806-013843/submit_20260806_013843.log),
  [Telecom](jobs/fsh-boundary-v2-telecom-expert-0806-013843/submit_20260806_013843.log).
- `pt-mdvdzxaw`, `fsh-boundary-v2-airline-expert-r1-0806-013947`: corrected
  8-GPU normal-priority Airline updates 100-129 —
  [run log](jobs/fsh-boundary-v2-airline-expert-r1-0806-013947/run_20260806_013947.log).
  [`AIRLINE_ITER109_HEALTH_GATE.json`](AIRLINE_ITER109_HEALTH_GATE.json) passes.
- `pt-3r3d55v9`, `fsh-boundary-v2-retail-expert-r1-0806-013947`: corrected
  8-GPU normal-priority Retail updates 100-129 —
  [run log](jobs/fsh-boundary-v2-retail-expert-r1-0806-013947/run_20260806_013947.log).
  [`RETAIL_ITER109_HEALTH_GATE.json`](RETAIL_ITER109_HEALTH_GATE.json) passes.
- `pt-kl7jsboo`, `fsh-boundary-v2-telecom-expert-r1-0806-013947`: corrected
  8-GPU normal-priority Telecom updates 100-129 —
  [run log](jobs/fsh-boundary-v2-telecom-expert-r1-0806-013947/run_20260806_013947.log).
  Iter109 saved completely and its
  [`TELECOM_ITER109_HEALTH_GATE.json`](TELECOM_ITER109_HEALTH_GATE.json) passes;
  both KL checks are diagnostic-only. The Gate binds the Telecom Contract
  signature to the matching immutable iter99 domain signature.

- Mixed-control conversions: `pt-lxpkpaad` iter109
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-mixed-i109-0806-032832/run_20260806_032832.log)),
  `pt-swazegss` iter119
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-mixed-i119-0806-032833/run_20260806_032833.log)),
  and `pt-lfa1pqk2` iter129
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-mixed-i129-0806-032833/run_20260806_032833.log)).
- Airline conversions: `pt-3epnw6ei` iter109
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-airline-i109-0806-032834/run_20260806_032834.log)),
  `pt-sg38y60e` iter119
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-airline-i119-0806-032834/run_20260806_032834.log)),
  and `pt-3zq68znh` iter129
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-airline-i129-0806-032835/run_20260806_032835.log)).
- Retail conversions: `pt-2uwvboy7` iter109
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-retail-i109-0806-032835/run_20260806_032835.log)),
  `pt-yak7ha2g` iter119
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-retail-i119-0806-032836/run_20260806_032836.log)),
  and `pt-u2zvrrp6` iter129
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-retail-i129-0806-032836/run_20260806_032836.log)).
- Telecom conversions: `pt-ihoh1437` iter109
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-telecom-i109-0806-032836/run_20260806_032836.log)),
  `pt-2g9uap7c` iter119
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-telecom-i119-0806-032837/run_20260806_032837.log)),
  and `pt-afvxxzav` iter129
  ([log](jobs/fsh-boundary-v2-domain-expert-convert-telecom-i129-0806-032837/run_20260806_032837.log)).

- Mixed-control seed300 evaluations: `pt-a4p2g369` iter109
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-mixed-i109-s300-0806-033952/run_20260806_033952.log)),
  `pt-c8nza3pc` iter119
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-mixed-i119-s300-0806-033952/run_20260806_033952.log)),
  and `pt-2xldle81` iter129
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-mixed-i129-s300-0806-033953/run_20260806_033953.log)).
- Airline target-domain seed300 evaluations: `pt-7fiuuyiz` iter109
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-airline-i109-s300-0806-033953/run_20260806_033953.log)),
  `pt-nxfyhfon` iter119
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-airline-i119-s300-0806-033953/run_20260806_033954.log)),
  and `pt-8dprcbjh` iter129
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-airline-i129-s300-0806-033954/run_20260806_033954.log)).
- Retail target-domain seed300 evaluations: `pt-wkoec94m` iter109
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-retail-i109-s300-0806-033954/run_20260806_033954.log)),
  `pt-skdcs318` iter119
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-retail-i119-s300-0806-033955/run_20260806_033955.log)),
  and `pt-4zp8szlo` iter129
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-retail-i129-s300-0806-033955/run_20260806_033955.log)).
- Telecom target-domain seed300 evaluations: `pt-cpbldqjd` iter109
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-telecom-i109-s300-0806-033955/run_20260806_033956.log)),
  `pt-mqulm701` iter119
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-telecom-i119-s300-0806-033956/run_20260806_033956.log)),
  and `pt-h522drpk` iter129
  ([log](jobs/fsh-boundary-v2-domain-expert-eval-telecom-i129-s300-0806-033956/run_20260806_033956.log)).

- Phase-one selection completed with Airline iter129, Retail iter119, and
  Telecom iter129; see [`DOMAIN_EXPERT_CURVE.md`](DOMAIN_EXPERT_CURVE.md).
- Selected-expert seed301 evaluations: `pt-dgzp1vra` Airline iter129
  ([log](jobs/fsh-fsh-boundary-v2-domain-expert-eval-airline-i129-s301-0806-063202-0806-063202/run_20260806_063202.log)),
  `pt-75608sdb` Retail iter119
  ([log](jobs/fsh-fsh-boundary-v2-domain-expert-eval-retail-i119-s301-0806-063202-0806-063203/run_20260806_063203.log)),
  and `pt-m0u3xivl` Telecom iter129
  ([log](jobs/fsh-fsh-boundary-v2-domain-expert-eval-telecom-i129-s301-0806-063202-0806-063203/run_20260806_063203.log)).
- Deduplicated mixed-control seed301 evaluations: `pt-4f1x2zei` iter119
  ([log](jobs/fsh-fsh-boundary-v2-domain-expert-eval-mixed-i119-s301-0806-063202-0806-063203/run_20260806_063203.log))
  and `pt-77isp1et` iter129
  ([log](jobs/fsh-fsh-boundary-v2-domain-expert-eval-mixed-i129-s301-0806-063202-0806-063204/run_20260806_063204.log)).
- `pt-g9wn6vp1`, selected Airline iter129 seed300 forgetting diagnostic over
  Retail+Telecom
  ([log](jobs/fsh-boundary-v2-domain-expert-forgetting-airline-i129-s300-0806-063240-0806-063240/run_20260806_063240.log)).
- `pt-fqqmf28s`, selected Retail iter119 seed300 forgetting diagnostic over
  Airline+Telecom
  ([log](jobs/fsh-boundary-v2-domain-expert-forgetting-retail-i119-s300-0806-065820-0806-065820/run_20260806_065820.log)).
- `pt-3pjj301n`, selected Telecom iter129 seed300 forgetting diagnostic over
  Airline+Retail
  ([log](jobs/fsh-boundary-v2-domain-expert-forgetting-telecom-i129-s300-0806-071044-0806-071044/run_20260806_071044.log)).

The forgetting jobs were staggered behind the short target confirmations, so
the phase-two peak remained 12 GPUs. All phase-one and phase-two evaluation
summaries were written with complete task/trial coverage and zero infrastructure
errors. The final Retail forgetting job `pt-fqqmf28s` wrote and validated its
complete summary, then hung after `eval complete` while waiting for sglang
process cleanup. After a 30-minute cleanup-only window it was stopped exactly
and reached `SUSPENDED`; all result artifacts remain intact and all preceding
jobs reached `SUCCEEDED`. `run_eval.sh` now bounds each TERM wait to 30 seconds
and escalates to SIGKILL, preventing future cleanup hangs without changing eval
semantics.

Final submit-interface dry-runs exited 0 for the 8-GPU training, 1-GPU
conversion, and 2-GPU evaluation worker specs. `--dry-run` created no cluster
jobs or submit logs; its copied-script directories are respectively
`fsh-domain-expert-training-final-dryrun-0806-112735`,
`fsh-domain-expert-conversion-final-dryrun-0806-112746`, and
`fsh-domain-expert-eval-final-dryrun-0806-112751`.

## Outputs

- `DOMAIN_EXPERT_CURVE.json/.md`: seed300 curves, paired task-bootstrap CIs, and
  selected checkpoints.
- `AIRLINE_EXPERT_DECISION.json`, `RETAIL_EXPERT_DECISION.json`, and
  `TELECOM_EXPERT_DECISION.json`: two-seed comparator decisions and diagnostics.
- `DOMAIN_EXPERT_SUMMARY.json/.md`: final OPD eligibility only; no OPD launch.

All listed outputs are complete. The stored summary marks Retail and Telecom
eligible and Airline `reject` under the original rubric; in capability terms,
Airline is simply not selected as the domain teacher.

## Commands

```bash
bash examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_boundary_v2.sh domain-expert airline
bash examples/tau2-bench/analysis/finalize_domain_expert_health.sh airline 109 TRAINING_LOG
bash scripts/convert_tau2_agent_rl_boundary_v2_iter_to_hf.sh domain-expert airline 109
bash scripts/convert_tau2_agent_rl_boundary_v2_iter_to_hf.sh mixed-control 109
bash examples/tau2-bench/eval/official/models/run_full_tau2_agent_rl_boundary_v2_user_stop_parser.sh domain-expert airline 109 300 probe
bash examples/tau2-bench/eval/official/models/run_full_tau2_agent_rl_boundary_v2_user_stop_parser.sh domain-expert airline 129 300 no-probe forgetting
bash examples/tau2-bench/eval/official/models/run_full_tau2_agent_rl_boundary_v2_user_stop_parser.sh mixed-control 109 300 probe
bash examples/tau2-bench/analysis/finalize_domain_experts.sh curve
bash examples/tau2-bench/analysis/finalize_domain_experts.sh decision airline
bash examples/tau2-bench/analysis/finalize_domain_experts.sh summary
```
