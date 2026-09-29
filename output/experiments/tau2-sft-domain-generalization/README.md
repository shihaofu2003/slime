# tau2-sft-domain-generalization

Purpose: compare four-domain mixed SFT with independently initialized domain
experts, including target-domain gains and cross-domain performance.
Name: `tau2-sft-domain-generalization`. Batch: `20260912`.

## Recipe

All five arms initialize from raw Qwen3-4B-Instruct-2507: two epochs, eight GPUs,
batch 16, LR 1e-5 cosine to 1e-6, 10% warmup, CP=1, qwen3_full, 16384 tokens/GPU.
Synchronous train.py, training/data seeds 1234, start rollout ID 0.
Final checkpoints only; no checkpoint selection on evaluation.

| Arm | Rows | Updates |
|---|---:|---:|
| mixed | 36048 | 4506 |
| airline | 13783 | 1722 |
| retail | 13614 | 1700 |
| telecom | 2979 | 372 |
| banking | 5672 | 708 |

Sources are shared `tau2-sft-official-native-expanded/data/agent_official_native_expanded.jsonl`
and `tau2-banking-simplify-qwen38-v1/data/full/banking_simplified_sft.jsonl`.
No pilot append. JSONL rows retain messages, tools, metadata and target masks.
`20260912/data/token_stats.json` records tokenizer totals and continuous-batch
coverage: all first-pass rows, followed by a partial reshuffled second pass.

## Execution

Entrypoint: `examples/tau2-bench/domain_generalization/run.sh` with
`prepare`, `train ARM smoke|full`, `convert ARM smoke|full`, or
`eval ARM smoke|full SEED`. `pipeline ARM full` runs train → convert → seed300
→ seed301; `convert-eval ARM smoke 300` continues a completed training smoke.
Run only inside project job containers. Full training writes `train.log` and
`training_report.json` under its arm/mode directory.
Submit with `scripts/submit.sh --project-root "$PWD" --env PROJECT_ROOT="$PWD"`
plus `--experiment tau2-sft-domain-generalization`; pass entrypoint arguments
following `--`. Keep normal priority, check quota before submitting, and use
at most two eight-GPU jobs. Do not stop unrelated jobs.

Order: tokenizer validation; longest eight rows/domain training smoke (two
updates), conversion, 3 tasks/domain × 1 trial eval; mixed full train/convert/
seeds 300 and 301; four experts with identical evaluation; raw baselines.
Artifacts remain in this worktree under the batch/arm/mode/seed directories.

## Evaluation and status

Official-native Agent, Qwen3.6-27B non-thinking User, Banking BM25; quotas
20/40/40/97; seeds 300 and 301, four trials each. Agent temperature 0.6,
top-p 1, max_tokens 1200, max_steps 200, max_errors 10, retries 3, no timeout.
Use a private benchmark data copy and actual result paths. Report pass@1,
pass@4(any), pass^4 per seed and their means, task-weighted overall and domain
macro means, paired task bootstrap differences stratified by domain.
Report action/DB accuracy, truncation, termination and infrastructure errors;
missing diagnostics are N/A. Incomplete runs are not valid comparisons.

CPU source preparation and seven new unit tests passed. Container job 17741
passed all 39 official-eval regression tests. Tokenizer validation job 17740
passed every source row, including full target-span and EOS supervision, with
no over-cap rows. Training smoke 17751 completed two updates in 193 seconds with finite losses
1.19494 and 1.43313, no CUDA OOM, and final checkpoint iter_0000001 saved.
[Training report](20260912/mixed/smoke/training_report.json). HF conversion and
12-simulation evaluation smoke passed in job 17754: 3 unique tasks/domain,
one trial each, no infrastructure errors. Agent/User configuration and private
result paths match the intended protocol. Success counts are Airline 1/3,
Retail 2/3, Telecom 1/3, Banking 0/3; these are smoke checks, not formal scores.
One Telecom trajectory hit max_steps; this diagnostic does not prevent training.
Mixed job 17782 completed all 4506 updates in 23651 seconds (6.57 hours),
with finite losses (first 1.02122, final 0.12738) and final checkpoint
iter_0004505. All 36048 rows were consumed twice.
[Training report](20260912/mixed/full/training_report.json). Final HF conversion
completed. Full seed300 evaluation finished with 197 tasks × 4 trials,
no duplicate/missing trials and no infrastructure failures. Seed301 also
completed all 788 simulations without missing/duplicate trials or infrastructure
failures; job 17782 succeeded. [Mixed results](MIXED_RESULTS.md).
[Seed300 summary](20260912/mixed/full/eval/seed300/summary.json).
Actual startup arguments match the fixed recipe and raw-model initialization.

Airline job 17900 completed 1722 updates in 9295 seconds (2.58 hours),
with finite losses and final checkpoint iter_0001721. Coverage: 13769 rows
twice and 14 rows once; 27552 samples consumed. HF conversion and both
197-task × 4-trial evaluations completed without missing/duplicate trials or
infrastructure failures. [Training report](20260912/airline/full/training_report.json),
[seed300](20260912/airline/full/eval/seed300/summary.json),
[seed301](20260912/airline/full/eval/seed301/summary.json).
Retail job 17902 was stopped at user request during training and will not
automatically restart. Telecom and Banking experts have not been run.

The corrected historical raw seed300 summary `seed300_0902_021944_summary.json`
loads `slime_sglang_agent`, not official-native `llm_agent` (verified in actual
Airline results.info.agent_info.implementation). It cannot serve as this
experiment's raw baseline. Later official-native raw seed300 runs
`seed300_0902_044845_summary.json` and `seed300_0902_060330_summary.json`
were found and their task definitions, trial coverage and generation settings
checked against the current results. These are two repeats of seed300, not
two evaluation seeds. They are included in the [raw comparison](AIRLINE_VS_RAW.md);
raw seed301 remains outstanding.

Summarize completed full evaluations with
`python3 examples/tau2-bench/domain_generalization/summarize.py output/experiments/tau2-sft-domain-generalization/20260912`.
This writes `RESULTS.md` and `results.json` only after complete task/trial
coverage, binary rewards, matching protocol and no infrastructure failures.

## 当前结果与结论（2026-09-12）

已完成 Mixed 和 Airline 专家。下表统一使用 seed300 的 pass@1（%），
原始模型范围来自两次 seed300 运行；完整三项指标见
[原始模型、Mixed 与 Airline 对比](AIRLINE_VS_RAW.md)。

| 评测域 | 原始 Qwen3-4B 两次运行 | Mixed SFT | Airline 单域 SFT |
|---|---:|---:|---:|
| Airline | 32.50–37.50 | 41.25 | 36.25 |
| Retail | 53.75–59.38 | 59.38 | 33.75 |
| Telecom | 16.25–18.75 | 48.75 | 26.88 |
| Banking | 3.35–3.61 | 3.87 | 1.55 |
| 任务加权 Overall | 19.67–20.94 | 28.05 | 16.75 |

Mixed 与 Airline 的两 seed 平均如下，每格为
pass@1 / pass@4(any) / pass^4（%）。每个 seed 分别计算四次指标后取均值，
不将两个 seed 拼成八次试验。

| 评测域 | Mixed SFT | Airline 单域 SFT |
|---|---:|---:|
| Airline | 44.38 / 62.50 / 25.00 | 42.50 / 62.50 / 25.00 |
| Retail | 60.62 / 87.50 / 30.00 | 36.88 / 67.50 / 12.50 |
| Telecom | 49.69 / 86.25 / 15.00 | 25.62 / 53.75 / 7.50 |
| Banking | 5.28 / 10.82 / 0.00 | 1.68 / 4.12 / 0.52 |

本次 Airline-only 配方出现明显的跨域退化，目标域收益不足以抵消损失。
seed300 的目标域 pass@1 落在原始模型两次运行之间；目标域 pass^4
从原始模型的 20% 提高至 25%，但 Retail 从 30% 降至 5%，Overall
从 10.15% 降至 6.60%。Telecom 相对原始模型有所提升，不能概括为
所有未训练领域均退化。与 Mixed 比较，两 seed 平均的目标域 pass@1
低 1.88 个百分点，pass@4(any) 与 pass^4 持平，未观察到专家优势。

对于需要同时保留四域能力的模型，当前优先采用 Mixed，暂不采用这套
Airline-only SFT 配方。这是针对已完成配方的实验判断，不推广为所有
单域 SFT 均不适用：仅完成一个专家域、一个训练 seed，训练计算量不等，
raw seed301 与配对 bootstrap 95% CI 尚未补齐，以上均为点估计。
Banking 训练检索分布与 BM25 评测不同。Qwen3.5 的外部参考结果见
[强模型对比](REFERENCE_SEED300.md)，其 max_tokens=8192，当前模型为
1200，不属于等生成预算比较。

## Limitations

Compute differs across arms; only one training seed. Banking training retrieval
mix differs from BM25 evaluation. Point estimates alone do not establish
improved generalization. No RL, OPD, or extra ablations in this experiment.

## Jobs

- 17739: CPU tokenizer submission rejected before launch: unsupported 8 CPU / 64 GB combination. [Submission log](jobs/17739-sft-domain-tokenize-0912-012958838/submit_20260912_012958838.log). No run log was created.
- 17740: tokenizer validation passed, 16 CPU / 64 GB, normal priority. [Run log](jobs/17740-sft-domain-tokenize-0912-013035205/run_0_20260912_013035205.log).
- 17741: CPU regression tests passed in the training container (five new tests at execution time, 39 official-eval tests; the sixth new report test also passed locally). [Run log](jobs/17741-sft-domain-cpu-tests-0912-014200173/run_0_20260912_014200173.log).
- 17751: eight-GPU, two-update training smoke on the longest eight rows per domain. [Run log](jobs/17751-sft-domain-smoke-0912-015011536/run_0_20260912_015011536.log).
- 17754: smoke HF conversion and four-domain evaluation (3 tasks/domain, 1 trial, seed300), eight GPUs. [Run log](jobs/17754-sft-domain-smoke-eval-0912-015628423/run_0_20260912_015628423.log).
- 17782: mixed full training → final HF conversion → full four-domain seeds300/301, eight GPUs, normal priority. [Run log](jobs/17782-sft-domain-mixed-full-0912-073603008/run_0_20260912_073603008.log).
- 17900: airline expert full training → HF conversion → four-domain seeds300/301, eight GPUs, normal priority. [Run log](jobs/17900-sft-domain-airline-full-0912-161618025/run_0_20260912_161618025.log).
- 17902: retail expert pipeline stopped at user request during training (STOPPED); no automatic restart. Eight GPUs, normal priority. [Run log](jobs/17902-sft-domain-retail-full-0912-161632984/run_0_20260912_161632984.log).
