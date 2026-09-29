# tau2-mix-rl-sft4505-b128-20260925

Purpose: compare asynchronous four-domain mix-RL with progress-db-count credit assignment against original outcome-only GRPO, starting from the same SFT4505 and sampling randomly from the same task pool.

## Configuration

Reference: [SFT4505 experts](../tau2-domain-experts-sft4505-b128/README.md). Initialize from `Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920/iter_0004505_hf`, reference step 4505, with a fresh RL optimizer and output directory.

LR 2e-6; batch 128 (16 groups × 8); progress-db-count-v1 with progress/format weights 1/1 and gamma 0.98; KL/entropy 0; train/rollout seeds 1234/42; pool 32, pending 64, unlimited policy lag; 4 environment workers; Agent/step concurrency 48/160. Keep uniform-outcome filtering, max_steps 200, Agent max_tokens 1200, training context cap 16384, and two over-cap retries. Agent request timeout: 60s for Airline/Retail/Telecom, 600s for Banking. Save every 10 updates and at completion; reward plots every 300s. [Launcher](run_mix.sh).

Source tasks: the unchanged expert training pools, Airline 1148, Retail 563, Telecom 271, Banking 542 (2524 total). Sample tasks uniformly with replacement from the merged pool using `random_tasks.RandomTaskDataSource`; expected domain proportions are 45.48%/22.31%/10.74%/21.47%. This replaces the experts' shuffled/retry sampler as requested. User-confirmed budget: 140 updates, matching the four experts' combined 60+30+20+30 updates.

User: Qwen3.6-27B nonthinking, two TP2 replicas on 4 GPUs, temperature 0, top_p 1, max_tokens 512. Training: 8 GPUs, Trainer 2 + three TP2 Agent generators on 6. Submit User first, then training, at normal priority. Training waits for this experiment's User endpoint.

## Jobs

| Job | GPUs | Purpose | Run log |
|---|---:|---|---|
| 23962 (`pt-tv6bpdt9`) | 4 | Persistent User, submitted 2026-09-25 09:40 CST | [Log](jobs/23962-tau2-mix-rl-user-persistent-0925-094053871/run_0_20260925_094053871.log) |
| 23965 (`pt-7mg24tx2`) | 8 | Four-domain mix-RL, 140 updates, submitted 2026-09-25 09:44 CST | [Log](jobs/23965-tau2-mix-rl-sft4505-b128-0925-094406560/run_0_20260925_094406560.log) |
| 23970 (`pt-tt0n84w7`) | 0 | Loss regression validation, 8 CPUs / 32 GB, submitted 10:17 CST | [Log](jobs/23970-tau2-mix-rl-loss-regression-0925-101744932/run_0_20260925_101744932.log) |
| 23971 (`pt-5429zudr`) | 8 | Fixed mix-RL retry, 140 updates, submitted 10:19 CST | [Log](jobs/23971-tau2-mix-rl-sft4505-b128-retry1-0925-101905793/run_0_20260925_101905793.log) |
| 24037 (`pt-6zoaqn3t`) | 8 | Update90 (iter89) four-domain official eval, submitted 15:54 CST | [Log](jobs/24037-tau2-mix-rl-update90-four-domain-eval-0925-155453183/run_0_20260925_155453183.log) |
| 24061 (`pt-gjvi6lim`) | 8 | Latest iter129 four-domain official eval, submitted 17:29 CST | [Log](jobs/24061-tau2-mix-rl-iter129-four-domain-eval-0925-172938008/run_0_20260925_172938008.log) |
| 24067 (`pt-xk33xcyk`) | 8 | Final iter139 four-domain official eval, submitted 18:10 CST | [Log](jobs/24067-tau2-mix-rl-iter139-four-domain-eval-0925-181001322/run_0_20260925_181001322.log) |
| 24077 (`pt-olqxeaov`) | 8 | Original GRPO control, fresh 140 updates, submitted 18:59 CST | [Log](jobs/24077-tau2-mix-rl-vanilla-grpo-b128-0925-185930950/run_0_20260925_185930950.log) |
| 24088 (`pt-84u0284s`) | 8 | GRPO control resumed after elastic reclaim, submitted 20:40 CST; reuses the original log directory | [Log](jobs/24077-tau2-mix-rl-vanilla-grpo-b128-0925-185930950/run_0_20260925_185930950.log) |
| 24120 | 8 | Iter59 four-domain official eval, submitted 21:59 CST; FAILED before cluster creation | No run log; [scheduler log](jobs/24120-tau2-mix-rl-iter59-four-domain-eval-0925-215922419/jobm.log) |
| 24121 | 8 | Unchanged iter59 eval retry, submitted 22:03 CST; FAILED before cluster creation | No run log; [scheduler log](jobs/24121-tau2-mix-rl-iter59-four-domain-eval-0925-220312234/jobm.log) |
| 24125 (`pt-fpvxm3op`) | 8 | Unchanged iter59 submission test, accepted 22:43 CST; STOPPED on request at 22:45 CST | No run log; [submission](jobs/24125-tau2-mix-rl-iter59-submit-test-0925-224305622/submit_20260925_224305622.log), [scheduler log](jobs/24125-tau2-mix-rl-iter59-submit-test-0925-224305622/jobm.log) |
| 24128 (`pt-38pxs4oo`) | 8 | Vanilla GRPO iter89 four-domain official eval, submitted 2026-09-25 23:04 CST; SUCCEEDED | [Log](jobs/24128-tau2-mix-rl-vanilla-iter89-four-domain-eval-0925-230401419/run_0_20260925_230401419.log) |
| 24142 (`pt-tf1k2v9i`) | 8 | Vanilla GRPO iter129 four-domain official eval, submitted 2026-09-26 07:51 CST; RUNNING | [Log](jobs/24142-tau2-mix-rl-vanilla-iter129-four-domain-eval-0926-075136763/run_0_20260926_075136763.log) |
| 24143 (`pt-oq8ythfq`) | 8 | Vanilla GRPO iter139 four-domain official eval, submitted 2026-09-26 07:51 CST; SUBMITTED | [Log](jobs/24143-tau2-mix-rl-vanilla-iter139-four-domain-eval-0926-075142078/run_0_20260926_075142078.log) |

Job 23965 FAILED at 09:58 CST before its first optimizer update. The September 23 OPD field addition made the ordinary RL batch contain `opd_reverse_kl=None`; the shared loss reporter checked only key presence and called `torch.cat(None)`. Credit assignment ran successfully on 128 trajectories / 1275 Assistant turns before the failure. No checkpoint was saved. User 23962 remains available at `http://10.119.99.170:30000/v1`.

Fix: report OPD metrics only when their value is present. The existing CPU test file now exercises the actual batch iterator and policy loss with per-token credit, both with and without OPD metrics and TIS, and checks gradients. These four cases are also included in the asynchronous launch preflight. CPU validation precedes the retry, which starts fresh from SFT4505 in `20260925_mix-retry1-train`, with the same 140-update configuration and existing User service.

CPU job 23970 SUCCEEDED: `python3 tests/test_tau2_opd_training.py`, 26 tests passed in 3.26s. The test file is already registered in the CPU CI matrix. Retry 23971 passed rollout preflight and 153 tests, then completed optimizer updates 0–2 by 10:37 CST. Losses: -0.030703/-0.067208/-0.050036; gradient norms: 0.642841/0.736419/0.962701. The original OPD metric failure did not recur. Two HTTP read errors entered the existing fresh-connection retry path and training continued. Training 23971 has now SUCCEEDED after all 140 updates; final iter139 was saved at 18:01:00 CST and the run exited with code 0.

At 10:55 CST, 9/140 optimizer updates are complete; all recorded loss/gradient norms are finite. Latest loss 0.00540744, gradient norm 0.56504245. The first scheduled checkpoint is after update 10.

Shell/Python syntax and whitespace checks passed; the merged pool exactly matches all expert rows. The Banking timeout override preserves its expert setting within the mixed process.

Failed run: [training log](arms/async/20260925_mix-train/run.log). Retry outputs: `arms/async/20260925_mix-retry1-train/` and `plots/20260925_mix-retry1-train/`. User readiness is published to `user_endpoint.env`.

## Update 90 official evaluation

User-requested iter90 evaluation uses the checkpoint after 90 optimizer updates: `arms/async/20260925_mix-retry1-train/checkpoints/iter_0000089` (zero-based IDs). Training logged its successful save at 15:00:19 CST. [Launcher](eval_update90.sh) converts this completed checkpoint to the adjacent `iter_0000089_hf` and runs the unchanged SFT4505 expert four-domain protocol.

All 197 official test tasks × 4 trials = 788 simulations; seed 300; BM25; max_steps 200, max_errors 10; Agent temperature 0.6, top_p 1, max_tokens 1200. User remains Qwen3.6-27B nonthinking, temperature 0, max_tokens 512. Domain concurrency 1/2/2/4, global 9, completed-domain slot borrowing enabled. Eight GPUs: two TP1 4B Agent replicas on GPUs 0/1, three TP2 User replicas on GPUs 2–7. Normal priority.

Job 24037 was submitted at 15:54 CST and has SUCCEEDED. HF conversion succeeded, all five model replicas and both routers became ready, and official evaluation started at 16:02 CST. All four saved run configurations confirm seed 300, four trials, max_steps 200, max_errors 10, and Agent temperature/top_p/max_tokens 0.6/1/1200.

Converter input checks and dry-run passed; a launch-environment comparison confirmed that all evaluation/deployment settings match the existing four-domain protocol, with only the Agent checkpoint changed. Evaluation outputs: `eval/mix-update90-iter89-four-domain/`.

## Iter129 official evaluation

Latest complete checkpoint at this request: `iter_0000129`, after 130 updates, saved successfully at 17:22:41 CST. [Launcher](eval_iter129.sh) uses the same conversion and evaluation workflow as job 24037. Only the checkpoint and output labels change; all four-domain evaluation and 8-GPU deployment parameters above are retained. Converter dry-run and resolved launch-parameter comparison passed.

Job 24061 was submitted at 17:29 CST with normal priority. The scheduler admitted it through elastic capacity while the 16-GPU quota had 12 GPUs in use; status is RUNNING at 18:08 CST. Outputs: `eval/mix-iter129-four-domain/`.

## Iter139 official evaluation

Final checkpoint `iter_0000139` follows all 140 optimizer updates and was saved successfully at 18:01:00 CST. [Launcher](eval_iter139.sh) preserves all iter129 evaluation/deployment settings, changing only checkpoint and output labels. Conversion input validation, dry-run, launch-parameter comparison, and shell syntax checks passed.

Job 24067 was submitted at 18:10 CST with normal priority and SUCCEEDED. Outputs: `eval/mix-iter139-four-domain/`.

## Iter59 official evaluation

Checkpoint `iter_0000059` follows 60 optimizer updates and was saved successfully at 13:18:26 CST. [Launcher](eval_iter59.sh) copies job 24067's submitted iter139 launcher, changing only the checkpoint, output labels, and checkpoint comment. All four-domain evaluation parameters and the 8-GPU deployment are unchanged. Shell syntax and conversion input dry-run passed. Planned outputs: `eval/mix-iter59-four-domain/`.

Job 24120 entered job-manager at 21:59:22 CST with normal priority, then FAILED at 21:59:26 before a cluster job was created: the scheduler's SCO process could not connect to its proxy at `127.0.0.1:7890` (`connection refused`). No conversion or evaluation ran; GPU time is zero. Per-job `--env` settings apply only inside the job container and cannot fix the scheduler's proxy.

User-requested retry 24121 reused the same launcher and 8-GPU submission at 22:03:12 CST. It FAILED at 22:03:17 with the same scheduler proxy connection refusal, before cluster creation; no conversion or evaluation ran.

User-requested submission test 24125 reused the identical iter59 launcher and 8-GPU configuration. The cluster accepted it as `pt-fpvxm3op` at 22:43:11 CST, with normal priority and elastic admission; the proxy error did not recur. It remained SUBMITTED and was stopped at 22:45:24 CST as requested; final status is STOPPED. No run log or iter59 HF conversion output was created, and recorded GPU time is zero. This verifies cluster submission, not evaluation execution. No proxy configuration was changed.

## Original GRPO control

[Launcher](run_vanilla_grpo.sh) starts a fresh 140-update run at `arms/async/20260925_vanilla-grpo-train/` from the same SFT4505 weights and fresh optimizer/RNG. Reuse User 23962 and the same 2524 tasks, uniform random sampling, seeds 1234/42, LR 2e-6, batch 128 (16×8), KL/entropy 0, pool 32, pending 64, unlimited policy lag, 4 environment workers, TIS, request limits, context cap/retries, and save interval. Training still uses Trainer2 + three TP2 Agent generators on 8 GPUs.

Only the training reward/advantage path changes: `TAU2_TURN_CREDIT_VERSION` is empty and reward shaping is disabled. No custom reward postprocessor or custom advantage function is registered. The official binary task rewards are normalized within each K=8 group as `(reward - mean) / (sample_std + 1e-6)` before segment expansion; the resulting scalar advantage is broadcast to each trajectory's response tokens and the existing loss mask selects Assistant tokens. DB-progress credit and local format penalties are absent. Uniform-outcome and unusable-group filtering remain identical to the credit arm. The progress/format weight exports inherited from the reference launcher are unused in this mode.

Job 24077 was submitted at 18:59 CST with normal priority and admitted through elastic capacity. The existing User endpoint returned HTTP 200. The copied task pool matches the credit run; shell checks and a resolved launch-environment comparison passed, with differences limited to run/output identities and the credit/reward-shaping switches. Planned held-out comparison uses the same official four-domain evaluation protocol above; no additional evaluation job is submitted here.

Startup verification: rollout preflight and 123 asynchronous/loss tests passed. Runtime arguments confirm `advantage_estimator=grpo`, reward/std normalization enabled, both custom reward/advantage hooks `None`, reward shaping 0, and OPD off. Fresh SFT4505 initialization completed with `finetune=True` and `no_load_optim=True`. Optimizer update 0 completed at 19:10:05 CST: loss 0.000145479, gradient norm 0.267733, LR 2e-6, batch 128.

Job 24077 was elastically reclaimed at 20:39 CST and requeued as 24088 at 20:40 CST. Job 24088 entered RUNNING at 20:41:28 with normal admission, reusing the original output/log directory. It has SUCCEEDED after 140 updates; iter89, iter129, and iter139 were saved successfully.

## Vanilla GRPO iter89 official evaluation

Checkpoint `arms/async/20260925_vanilla-grpo-train/checkpoints/iter_0000089` is the completed 90-update checkpoint from training job 24088. [Launcher](eval_vanilla_grpo_iter89.sh) copies the credit iter89 evaluation launcher submitted for job 24037, changing only the checkpoint root and model/output labels. It retains all 197 test tasks × 4 trials, seed 300, BM25, the Qwen3.6-27B User, sampling and termination settings, domain/global concurrency, and the same 8-GPU deployment. Shell syntax, conversion input dry-run, and launcher comparison passed.

Evaluation job 24128 (`pt-38pxs4oo`) was accepted by the cluster at 23:04:05 CST with normal priority and elastic admission, then entered RUNNING at 23:05:10 CST. It has SUCCEEDED, with results saved for all four domains. Outputs: `eval/vanilla-grpo-iter89-four-domain/`.

## Vanilla GRPO iter129 and iter139 official evaluation

[Iter129 launcher](eval_vanilla_grpo_iter129.sh) and [iter139 launcher](eval_vanilla_grpo_iter139.sh) copy job 24128's submitted script, changing only the completed vanilla GRPO checkpoint, model/output labels, and iteration comment. All four-domain evaluation parameters and the 8-GPU deployment remain unchanged. Shell syntax and conversion input dry-runs passed; script diffs confirm only those replacements.

Jobs 24142 (`pt-tf1k2v9i`, iter129) and 24143 (`pt-oq8ythfq`, iter139) were accepted by the cluster on 2026-09-26 at 07:51 CST, each with 8 GPUs and normal priority. Iter129 is RUNNING; iter139 is SUBMITTED. Admission is normal for iter129 and elastic for iter139. Outputs: `eval/vanilla-grpo-iter129-four-domain/` and `eval/vanilla-grpo-iter139-four-domain/`. Run logs appear after the containers start.
