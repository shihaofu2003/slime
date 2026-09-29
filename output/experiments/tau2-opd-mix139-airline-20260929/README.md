# tau2-opd-mix139-airline-20260929

Purpose: rerun four-domain pure OPD with the evaluated 2026-09-25 mix iter139 as both student initialization and fixed non-Airline teacher, plus Airline expert iter29. Replaces the wrong-initialization run evaluated by job24983. Training completed2026-09-29; User/expert services were stopped afterward on explicit user request.

## 最终 checkpoint

**用户于 2026-09-29 选定 OPD iter59（完成 60 次 OPD 更新）作为最终 checkpoint。**

HF 模型：[iter_0000059_hf](arms/async/20260929_mix139-airline-opd/checkpoints/iter_0000059_hf)；Megatron：[iter_0000059](arms/async/20260929_mix139-airline-opd/checkpoints/iter_0000059)。这两个目录属于本实验训练任务 25056。

```text
/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2/output/experiments/tau2-opd-mix139-airline-20260929/arms/async/20260929_mix139-airline-opd/checkpoints/iter_0000059_hf
```

iter59 总体 **31.98 / 48.22 / 14.21%**，相对实际初始化 mix139 为 **+0.76 / −0.51 / +0.51pp**（顺序均为 pass@1 / pass@4(any) / pass^4）。Airline 为 **51.25 / 75.00 / 20.00%**，较初始化提高 **+11.25 / +5.00 / +5.00pp**。非 Airline 三域按任务数加权的 pass@1 保留率为 **98.6%**（成功轨迹214/708→211/708，保留率211/214）。最终选择按用户决定记录，单 seed 结果不作显著性结论。

## 评测结果对比

评测数据快照：**2026-09-29 19:58 CST**。单元格为 **pass@1 / pass@4(any) / pass^4，单位 %**；pass@4(any) 为四次至少一次成功，pass^4 为四次全部成功。

统一口径：official-native、test、seed300、每任务 4 trials、BM25、Agent temperature0.6/top_p1/max_tokens1200，Qwen3.6-27B nonthinking User temperature0/max_tokens512，max_steps200/max_errors10，初始域并发1/2/2/4、总并发9及完成域槽位借用。完整评测各含 Airline20、Retail40、Telecom40、Banking97 个任务，共197任务/788条轨迹；总体按任务数加权。下表五个参照模型和六个完整 OPD 检查点的 **8,668 条原始 reward 已复算**，三个 pass 指标均与官方 summary 一致，且各任务均有4个不重复 trial。

**完整评测（模型名链接到原始 summary）**

| 模型 | 总体（197任务） | Airline | Retail | Telecom | Banking knowledge |
|---|---|---|---|---|---|
| [Raw Instruct](../../../../slime/output/experiments/tau2-banking-expert-sft/eval/raw-full-qwen36-bm25-fixed/seed300_0910_raw_fixed_summary.json) | 18.91 / 31.98 / 8.63 | 33.75 / 65.00 / 10.00 | 52.50 / 72.50 / 30.00 | 17.50 / 42.50 / 5.00 | 2.58 / 4.12 / 1.03 |
| [SFT4505](../tau2-sft-areal3-banking-simplified-full-20260920/eval/four-domain-sft-iter4505/seed300_20260920_sft_iter4505_four_domain_summary.json) | 27.92 / 43.15 / 13.20 | 42.50 / 65.00 / 30.00 | 58.75 / 77.50 / 35.00 | 47.50 / 82.50 / 12.50 | 4.12 / 8.25 / 1.03 |
| [Vanilla GRPO iter139](../tau2-mix-rl-sft4505-b128-20260925/eval/vanilla-grpo-iter139-four-domain/seed300_20260925_vanilla_grpo_iter139_seed300_summary.json) | 29.95 / 46.19 / 13.20 | 42.50 / 60.00 / 25.00 | 61.25 / 87.50 / 37.50 | 51.88 / 85.00 / 15.00 | 5.41 / 10.31 / 0.00 |
| [Mix RL iter139（学生初始化 / 非 Airline 教师）](../tau2-mix-rl-sft4505-b128-20260925/eval/mix-iter139-four-domain/seed300_20260925_mix_iter139_seed300_summary.json) | 31.22 / 48.73 / 13.71 | 40.00 / 70.00 / 15.00 | 63.75 / 85.00 / 40.00 | 56.25 / 87.50 / 20.00 | 5.67 / 13.40 / 0.00 |
| [Airline expert iter29（Airline 教师）](../tau2-domain-experts-sft4505-b128/eval/airline-iter29-four-domain/seed300_20260921_sft4505_airline_iter29_summary.json) | 29.57 / 47.21 / 13.20 | 53.75 / 75.00 / 35.00 | 59.38 / 90.00 / 30.00 | 49.38 / 82.50 / 17.50 | 4.12 / 9.28 / 0.00 |
| [OPD iter9](eval/opd-mix139-airline-iter9-four-domain/seed300_20260929_opd_mix139_airline_iter9_seed300_summary.json) | 32.99 / 46.19 / 15.23 | 50.00 / 70.00 / 15.00 | 63.13 / 82.50 / 37.50 | 60.00 / 85.00 / 25.00 | 5.93 / 10.31 / 2.06 |
| [OPD iter19](eval/opd-mix139-airline-iter19-four-domain/seed300_20260929_opd_mix139_airline_iter19_seed300_summary.json) | 32.99 / 45.18 / 17.77 | 51.25 / 75.00 / 20.00 | 66.88 / 82.50 / 45.00 | 56.25 / 82.50 / 27.50 | 5.67 / 8.25 / 2.06 |
| [OPD iter29](eval/opd-mix139-airline-iter29-four-domain/seed300_20260929_opd_mix139_airline_iter29_seed300_summary.json) | 30.84 / 45.69 / 15.74 | 48.75 / 70.00 / 25.00 | 60.00 / 82.50 / 37.50 | 53.13 / 82.50 / 25.00 | 5.93 / 10.31 / 1.03 |
| [OPD iter39](eval/opd-mix139-airline-iter39-four-domain/seed300_20260929_opd_mix139_airline_iter39_seed300_summary.json) | 31.98 / 46.70 / 17.77 | 45.00 / 65.00 / 25.00 | 63.13 / 90.00 / 42.50 | 58.75 / 82.50 / 30.00 | 5.41 / 10.31 / 1.03 |
| [OPD iter49](eval/opd-mix139-airline-iter49-four-domain/seed300_20260929_opd_mix139_airline_iter49_seed300_summary.json) | 31.73 / 47.21 / 15.74 | 52.50 / 70.00 / 30.00 | 62.50 / 87.50 / 40.00 | 56.25 / 87.50 / 22.50 | 4.64 / 9.28 / 0.00 |
| **[OPD iter59（最终选择）](eval/opd-mix139-airline-iter59-four-domain/seed300_20260929_opd_mix139_airline_iter59_seed300_summary.json)** | **31.98 / 48.22 / 14.21** | **51.25 / 75.00 / 20.00** | **65.00 / 85.00 / 47.50** | **54.38 / 87.50 / 12.50** | **5.15 / 11.34 / 0.00** |

Mix RL iter139 同时是学生初始化和 Retail/Telecom/Banking 固定教师。Airline expert iter29 行展示该单一教师模型的四域成绩；实际蒸馏仅在 Airline 使用它。错误初始化的旧 OPD 实验保留在各自文档中，不混入本次 checkpoint 对比。

**未完成或中止的评测**

以下为上述更新时间读取的已保存轨迹。只展示已完成整域全部4 trials 的指标；未完成域显示轨迹进度，均不计算四域总体成绩。iter69 仍在运行；iter149/159 已按用户要求停止，现有输出保留。

| Checkpoint | 状态 | 已保存轨迹 | Airline（80条） | Retail（160条） | Telecom（160条） | Banking（388条） |
|---|---|---:|---|---|---|---|
| [OPD iter69](eval/opd-mix139-airline-iter69-four-domain/trajectories) | RUNNING | 491/788 | 47.50 / 70.00 / 25.00 | 62.50 / 92.50 / 32.50 | 97/160，未完成 | 154/388，未完成 |
| [OPD iter149](eval/opd-mix139-airline-iter149-four-domain/trajectories) | STOPPED | 527/788 | 46.25 / 65.00 / 20.00 | 61.88 / 87.50 / 37.50 | 105/160，未完成 | 182/388，未完成 |
| [OPD iter159](eval/opd-mix139-airline-iter159-four-domain/trajectories) | STOPPED | 522/788 | 48.75 / 70.00 / 30.00 | 62.50 / 85.00 / 42.50 | 107/160，未完成 | 175/388，未完成 |

**完整评测的辅助诊断**

Action accuracy 统计 golden read/write 动作匹配；DB accuracy 的分母仅含实际完成 DB 检查的轨迹。max_steps 列是触及步数上限的终止条数，不是最长对话步数。这些诊断不替代官方任务成功率。

| 模型 | Action accuracy % | DB accuracy % | max_steps 终止 | 基础设施错误 |
|---|---:|---:|---:|---:|
| Raw Instruct | 27.97 | 20.18 | 14/788 | 0 |
| SFT4505 | 35.38 | 23.67 | 14/788 | 0 |
| Vanilla GRPO iter139 | 42.43 | 29.76 | 108/788 | 0 |
| Mix RL iter139 | 40.15 | 28.12 | 49/788 | 0 |
| Airline expert iter29 | 37.36 | 25.86 | 21/788 | 0 |
| OPD iter9 | 39.99 | 28.57 | 49/788 | 0 |
| OPD iter19 | 39.07 | 29.99 | 36/788 | 0 |
| OPD iter29 | 37.95 | 27.59 | 29/788 | 0 |
| OPD iter39 | 39.54 | 27.74 | 42/788 | 0 |
| OPD iter49 | 40.50 | 27.81 | 53/788 | 0 |
| **OPD iter59** | **39.33** | **29.50** | **43/788** | **0** |

## Historical OPD findings

- [Buffer A/B](../tau2-opd-current-buffer-ab/ANALYSIS_AND_NEXT.md): cap **generating + ready + training** groups at32. Pool32/pending64 alone does not cap the completed queue. Bounded A mean/max policy lag1.03/4 updates; unbounded B8.29/20. The task-score comparison did not establish an A/B winner; B also resumed after a cluster crash.
- [Matched LR study](../tau2-opd-bounded-lr-20260922/RESULTS.md): retain **2e-6**. Overall two-seed pass@1 was55.125% versus49.75% at5e-6, paired difference CI[+1.50,+9.375]pp. With buffer32, mean/max lag was1.022/5 and1.029/6; these were observed maxima, not configured step limits.
- [Gradient review](../tau2-opd-airline-retail-pilot/REVIEW_20260922.md): use **current-student** logprobs in the teacher-minus-student advantage, with behavior logprobs only in TIS[0,2]. Keep `OPD_USE_BEHAVIOR_LOGPROBS=0`; the historical behavior-logprob surrogate plus TIS changes the objective under stale sampling.
- Pure OPD keeps task reward0, K1, no advantage/std normalization, no outcome/zero-variance replacement. Official task success remains a diagnostic. Preserve raw on-policy tokens, Assistant-only loss masks and per-trajectory normalization across segments.
- PPO clip0.2 is not an update-size guarantee with one optimizer step per batch. Retain post-update diagnostics every10 updates; do not treat the sign of an unweighted sampled log-ratio as proof of an incorrect gradient.

User explicitly selected **historical buffer32, no fixed lag-step limit** for this rerun: `TAU2_MAX_POLICY_LAG=-1`, four environment workers. The prior run already had these settings; this rerun does not claim to fix a buffer-parameter mismatch.

## Models and fixed recipe

All paths below are relative to `output/experiments/`:

| Role | Checkpoint | GPU placement |
|---|---|---|
| Student | `tau2-mix-rl-sft4505-b128-20260925/arms/async/20260925_mix-retry1-train/checkpoints/iter_0000139` | 8-GPU training job: Trainer2 + three TP2 Agent generators |
| Retail/Telecom/Banking teacher | Same root, `iter_0000139_hf`; also student's `HF_CHECKPOINT` | Expert job GPUs2,3, TP2 |
| Airline teacher | `tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-airline-train/checkpoints/iter_0000029_hf` | Expert job GPUs0,1, TP2 |
| User | Workspace `models/Qwen3.6-27B`, nonthinking | Separate4-GPU job, two TP2 replicas plus router |

Student uses explicit `REF_CKPT_STEP=139`, a fresh optimizer and a new checkpoint directory under `arms/async/20260929_mix139-airline-opd/`. The wrong run loaded the **0927 credit-signal-v1 iter129**, not this **0925 progress-db-count iter139**. Historical OPD comparisons against mix139 therefore do not measure improvement over their actual initialization.

LR2e-6, OPD coefficient1, batch16/K1,160 updates (2560 consumed trajectories), save every10; train/rollout seeds1234/42 unchanged from the replaced run. Other KL/entropy coefficients0, PPO clip0.2/0.2, TIS[0,2]. Pool32/pending64/total buffer32; async sampling with4 environment workers. Agent temperature1/top_p1, max_tokens1200, max_steps200, max_errors10, context cap16384, two over-cap retries. User temperature0/top_p1/max_tokens512, thinking disabled.

Source: `../tau2-opd-four-domain-20260923/data/four_domain_train.jsonl`,2524 tasks (Airline1148/Retail563/Telecom271/Banking542), `ShuffledTaskDataSource`, no domain quota. The shuffled deck covers tasks before reshuffling; retries/over-cap exclusions and asynchronous ordering mean2560 training trajectories are only approximately one source pass. Report actual domain counts and coverage.

## Launch

```bash
bash scripts/submit.sh --experiment tau2-opd-mix139-airline-20260929 --gpus 4 \
  --name tau2-opd-mix139-user-persistent output/experiments/tau2-opd-mix139-airline-20260929/run_user.sh
bash scripts/submit.sh --experiment tau2-opd-mix139-airline-20260929 --gpus 4 \
  --name tau2-opd-mix139-experts-persistent output/experiments/tau2-opd-mix139-airline-20260929/serve_two_experts.sh
bash scripts/submit.sh --experiment tau2-opd-mix139-airline-20260929 --gpus 8 \
  --name tau2-opd-mix139-buffer32-train output/experiments/tau2-opd-mix139-airline-20260929/run_opd.sh
```

Training topology:16 GPUs total, normal priority. Both services publish experiment-local endpoint files and run until explicitly stopped. Training waits for ready services and runs the OPD, gradient and rollout preflights.

## Jobs and validation

Submitted2026-09-29 15:58 CST, normal priority:

| Job | GPUs | Task/status | Run log |
|---|---:|---|---|
|25054 (`pt-3nb4o2z5`)|4|Qwen3.6-27B User; STOPPED on request after training|[Log](jobs/25054-tau2-opd-mix139-user-persistent-0929-155851228/run_0_20260929_155851228.log)|
|25055 (`pt-k5mrgkur`)|4|mix139 + Airline29 teachers; STOPPED on request after training|[Log](jobs/25055-tau2-opd-mix139-experts-persistent-0929-155852468/run_0_20260929_155852468.log)|
|25056 (`pt-1p8he5ap`)|8|Fresh mix139 pure OPD; SUCCEEDED,160 updates, final iter159 saved, completed17:17:09 CST|[Log](jobs/25056-tau2-opd-mix139-buffer32-train-0929-155853697/run_0_20260929_155853697.log)|
|25061 (`pt-iqqhn7pd`)|8|OPD iter9 official four-domain seed300 evaluation; SUCCEEDED|[Log](jobs/25061-tau2-opd-mix139-iter9-eval-0929-162626504/run_0_20260929_162626504.log)|
|25062 (`pt-65bt4220`)|8|OPD iter19 official four-domain seed300 evaluation; SUCCEEDED|[Log](jobs/25062-tau2-opd-mix139-iter19-eval-0929-162627869/run_0_20260929_162627869.log)|
|25072 (`pt-b6ozusye`)|8|OPD iter149 official four-domain seed300 evaluation; STOPPED on request, existing outputs retained|[Log](jobs/25072-tau2-opd-mix139-iter149-eval-0929-172038347/run_0_20260929_172038347.log)|
|25073 (`pt-3t1nsk1m`)|8|OPD iter159 official four-domain seed300 evaluation; STOPPED on request, existing outputs retained|[Log](jobs/25073-tau2-opd-mix139-iter159-eval-0929-172039553/run_0_20260929_172039553.log)|
|25078 (`pt-7zssm941`)|8|OPD iter29 official four-domain seed300 evaluation; SUCCEEDED|[Log](jobs/25078-tau2-opd-mix139-iter29-eval-0929-180707511/run_0_20260929_180707511.log)|
|25079 (`pt-x87d27h2`)|8|OPD iter39 official four-domain seed300 evaluation; SUCCEEDED|[Log](jobs/25079-tau2-opd-mix139-iter39-eval-0929-180708705/run_0_20260929_180708705.log)|
|25080 (`pt-tstj1nj8`)|8|OPD iter49 official four-domain seed300 evaluation; SUCCEEDED|[Log](jobs/25080-tau2-opd-mix139-iter49-eval-0929-180709902/run_0_20260929_180709902.log)|
|25081 (`pt-h6nbjwve`)|8|OPD iter59 official four-domain seed300 evaluation; SUCCEEDED; final selected checkpoint|[Log](jobs/25081-tau2-opd-mix139-iter59-eval-0929-181133775/run_0_20260929_181133775.log)|
|25082 (`pt-51oqkull`)|8|OPD iter69 official four-domain seed300 evaluation; RUNNING, submitted18:12 CST|[Log](jobs/25082-tau2-opd-mix139-iter69-eval-0929-181211556/run_0_20260929_181211556.log)|

Local `bash -n` and `run_opd.sh --dry-run` passed. Both139 checkpoint formats, teacher29 HF shards and all2524 data rows are present. Cluster OPD routing/alignment20 tests, OPD training26 tests, continuous sampler119 tests and the runner's4 loss regression checks passed, along with rollout-logic preflight. Both teachers passed input-token-logprob probes; live `/get_model_info` confirms exact Airline29 and0925 mix139 HF paths.

Actual trainer arguments: `ckpt_step=139`, `finetune=True`, `no_load_optim=True`, `start_rollout_id=0`, `opd_use_behavior_logprobs=False`, buffer32, lag=-1. Checkpoint load succeeded from the0925 mix root. Updates0/1 completed at16:13:08/16:13:30 CST with finite loss0.007646/0.010232 and grad_norm0.4523/0.5778; LR2e-6 and batch16. Initial observed generating+ready+training groups≤32. The first three admitted batches have max policy lag0/1/2; there is no configured step cap. Zero-variance-group rate1.0 is expected for pure OPD with K1 and zero training reward.

## Evaluation reference

Matched student baseline: [mix139 official seed300](../tau2-mix-rl-sft4505-b128-20260925/eval/mix-iter139-four-domain/seed300_20260925_mix_iter139_seed300_summary.json), overall31.22/48.73/13.71% and Airline40.00/70.00/15.00% pass@1/pass@4(any)/pass^4. Airline teacher seed300 is53.75/75.00/35.00%. Evaluation retains197 tasks ×4 trials, four domains, BM25, Agent temperature0.6 and concurrency1/2/2/4 with global9.

Requested2026-09-29: independent8-GPU official seed300 evaluations of OPD iter9 and iter19. Saves completed at16:16:59 and16:21:40 CST respectively. [eval_checkpoint.sh](eval_checkpoint.sh) converts each selected checkpoint to its own HF directory, then uses2 TP1 Agent replicas and3 TP2 Qwen3.6-27B User replicas. Each evaluation has788 simulations and a separate `eval/opd-mix139-airline-iter<N>-four-domain/` output directory.

Shell syntax and iteration-specific configuration dry-runs passed before submission. After training completed, the user requested stopping both persistent services and evaluating the last two checkpoints. Iter149/159 saves completed at17:13:12/17:17:00 CST; tracker=159 and the training job exited0. Services25055/25054 were stopped first, then8-GPU evaluations25072/25073 were submitted at17:20 CST with the same protocol.

At18:07 CST, requested iter29/39/49 evaluations were submitted as25078/25079/25080, each8 GPUs and normal priority. All three checkpoint saves and configuration dry-runs were checked; HF conversion runs inside each evaluation job. Protocol and output naming follow the same launcher. At submission, iter9/19 evaluations had SUCCEEDED and iter149/159 evaluations were RUNNING.

On the next request, iter149/159 evaluations25072/25073 were STOPPED and their existing outputs retained. Iter59/69 checkpoints and dry-run configurations were checked, then the same8-GPU four-domain protocol was submitted as25081/25082 at18:11/18:12 CST. Iter69's initial admission retry waited for the24-GPU queued+SUBMITTED cap to clear; it was accepted when iter49 started running. No duplicate iter69 job was created.

```bash
bash scripts/submit.sh --experiment tau2-opd-mix139-airline-20260929 --gpus 8 \
  --name tau2-opd-mix139-iter9-eval --env ITER=9 \
  output/experiments/tau2-opd-mix139-airline-20260929/eval_checkpoint.sh
bash scripts/submit.sh --experiment tau2-opd-mix139-airline-20260929 --gpus 8 \
  --name tau2-opd-mix139-iter19-eval --env ITER=19 \
  output/experiments/tau2-opd-mix139-airline-20260929/eval_checkpoint.sh
bash scripts/submit.sh --experiment tau2-opd-mix139-airline-20260929 --gpus 8 \
  --name tau2-opd-mix139-iter149-eval --env ITER=149 \
  output/experiments/tau2-opd-mix139-airline-20260929/eval_checkpoint.sh
bash scripts/submit.sh --experiment tau2-opd-mix139-airline-20260929 --gpus 8 \
  --name tau2-opd-mix139-iter159-eval --env ITER=159 \
  output/experiments/tau2-opd-mix139-airline-20260929/eval_checkpoint.sh
for iteration in 29 39 49 59 69; do
  bash scripts/submit.sh --experiment tau2-opd-mix139-airline-20260929 --gpus 8 \
    --name "tau2-opd-mix139-iter${iteration}-eval" --env "ITER=${iteration}" \
    output/experiments/tau2-opd-mix139-airline-20260929/eval_checkpoint.sh
done
```
