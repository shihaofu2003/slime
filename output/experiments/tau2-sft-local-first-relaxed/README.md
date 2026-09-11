# tau2 SFT local-first relaxed filter

实验名：`tau2-sft-local-first-relaxed`。目的：将已由 Qwen3.6-27B 与
`gpt-5.6-luna` 对照校准的质量标准迁移为 local-first 全量筛选，按 target prefix
挽救干净训练轮，生成比 69-dialog strict seed 大得多的 Agent SFT 数据；随后从原始
Qwen3-4B-Instruct-2507 训练，并与 raw model、旧 multitool SFT 做同配置 official tau2
评估。全量筛选不发起新 API 请求；既有 Luna 覆盖继续作为保守否决证据。

## Tasks

- `fsh-tau2-sft-local-first-filter-0803-011956`（job `pt-fe8v4pyi`）：运行
  `prepare → Qwen3.6-27B local review`；239 条旧 cache 全部命中并新增 14 条后，发现 build
  还需对改写后的 system prompt 重验 8192-token cap，故提前停止，253 条有效 review 留在
  共享 cache 中；[run log](jobs/fsh-tau2-sft-local-first-filter-0803-011956/run_20260803_011956.log)。
- `fsh-tau2-sft-local-first-filter-r2-0803-013332`（job `pt-jvrvfzs5`）：加入真实
  Qwen3-4B tokenizer 的改写后 token-cap/loss-mask preflight，确认单卡服务正常但吞吐约
  19 dialogs/5 min；为缩短全量筛选等待时间，在保留 409 条有效 cache 后主动停止；
  [run log](jobs/fsh-tau2-sft-local-first-filter-r2-0803-013332/run_20260803_013332.log)。
- `fsh-tau2-sft-local-first-filter-8way-0803-015318`（job `pt-l7jt6t2n`）：8 GPU 各运行
  一个独立 Qwen3.6-27B，本地 hash 分片负载为每卡 241–275 dialogs；合并前强制验证
  2,046-dialog exact coverage，随后构建 target-only canonical JSONL 与 19,318-row budget
  JSONL；0 review failures，任务成功，仍无 fallback/API 配置；
  [run log](jobs/fsh-tau2-sft-local-first-filter-8way-0803-015318/run_20260803_015318.log)。
- `fsh-eval-raw-dependency-safe-multi-0803-015848`（job `pt-tw0vepd2`）：原始
  Qwen3-4B-Instruct-2507 的对照评测；400 simulations、0 infra errors，任务成功；
  [run log](jobs/fsh-eval-raw-dependency-safe-multi-0803-015848/run_20260803_015848.log)。
- `fsh-eval-old-sft-dependency-safe-multi-0803-015848`（job `pt-myyiv9ri`）：旧 multitool
  SFT 的对照评测；400 simulations、0 infra errors，任务成功；
  [run log](jobs/fsh-eval-old-sft-dependency-safe-multi-0803-015848/run_20260803_015848.log)。
- `fsh-train-local-first-smoke-0803-023110`（job `pt-xlyvn9rv`）：从原始
  Qwen3-4B-Instruct-2507 torch-dist 权重初始化的 32-row、8-GPU 训练 smoke；
  iteration 0→1，loss 1.0973、grad norm 31.41，checkpoint 保存成功，任务成功；
  [run log](jobs/fsh-train-local-first-smoke-0803-023110/run_20260803_023110.log)。
- `fsh-train-local-first-full-0803-024227`（job `pt-5vwcnhoy`）：同一 raw init、19,318
  rows × 2 epochs、batch 16、LR 1e-5→1e-6 的完整 8-GPU SFT；iteration 0→2,413，
  最终 loss 0.1934、grad norm 2.55，最终 checkpoint 保存成功，任务成功；
  [run log](jobs/fsh-train-local-first-full-0803-024227/run_20260803_024227.log)。
- `fsh-fsh-convert-local-first-final-0803-035134`（job `pt-ler5iev3`）：将最终
  iteration 2,413 torch-dist checkpoint 转为 `final_hf`；两份 safetensors、索引、配置与
  tokenizer 文件写入成功，任务成功；
  [run log](jobs/fsh-fsh-convert-local-first-final-0803-035134/run_20260803_035134.log)。
- `fsh-eval-new-sft-dependency-safe-multi-0803-040224`（job `pt-88y4lkdy`）：新 local-first
  SFT 的同协议 official tau2 对照评测；400 simulations、0 infra errors，任务成功；
  [run log](jobs/fsh-eval-new-sft-dependency-safe-multi-0803-040224/run_20260803_040224.log)。

## Filter result

- consensus-success 16,811 targets / 2,046 dialogs；最终 canonical 15,807 targets /
  1,863 dialogs，其中 2,554 个 target 含 dependency-safe multi-tool turn。
- canonical 领域分布为 airline 7,277、retail 8,243、telecom 287；19,318-row budget
  文件为 8,873 / 10,092 / 353，保持相近分布。
- success target 保留率为 airline 96.1%、retail 96.7%、telecom 40.1%；因此 telecom
  训练占比从旧文件的 10.0% 降至 1.83%。不回填被拒 target，分域 eval 单独检验该影响。
- canonical SHA256 `2891b80b59e5e7696ef075bcf4e34417c017374103157501af405986585ed9f8`；
  budget SHA256 `729e33bb286075478a6d9bba004cfb8f766dd11a09b35d98124e202eea295577`。
- 改写后实测最大 8,190 tokens；结构、target-only mask、调度索引、唯一 provenance 与
  summary SHA256 的独立 preflight 通过。

## Filter protocol

- 仅自动保留 reward/correct 双成功；可靠 local judge 的 `keep`/`review` 可进入候选，
  `drop`、核心维度低于 3、支持维度低于 2 均拒绝。
- 无效工具/schema/namespace、tool+text 同轮只阻断包含该问题的 target prefix；晚期问题
  不再丢弃此前干净 target。judge 的 critical turns 保留为诊断，由 verdict 和维度门槛
  约束，不再叠加一个未经校准的硬拒绝条件。
- target dependency prefix 中每个 multi turn 都必须是独立、参数已知、必要且写入已授权；
  已有 Luna review 覆盖的 dialog 必须同时通过同一门槛。
- 历史 assistant turn 使用 `step_loss_mask=0`，只监督获准 final target。训练 prompt 与
  review prompt 统一为 dependency-safe multi，并移除 airline/telecom 的 single-call 冲突。

## Comparison protocol

- 新 SFT 从原始 `Qwen3-4B-Instruct-2507` 初始化，不从旧 SFT 或 RL checkpoint 继续训练。
- 训练 row/update budget 与旧 max8192 SFT 对齐；固定 seed 并记录 canonical 与 budget
  文件的精确 SHA256。
- official tau2 eval 对 raw、新 SFT、旧 SFT 使用同一 v1 STOP User、100 tasks × 4 trials、
  seed 300、Agent temperature 0.6、max tokens 8192，并报告 pass@1、pass@4、pass^4 与
  paired confidence interval。

## Result and decision

| Model | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| raw | 20.50% | 43.00% | 6.00% |
| old SFT | 21.25% | 41.00% | 10.00% |
| local-first SFT | 27.00% | 54.00% | 4.00% |

新 SFT 相对 raw/旧 SFT 的 pass@1 分别提升 `+6.50pp [1.00, 12.00]` 和
`+5.75pp [0.50, 11.00]`；pass@4(any) 分别提升 `+11.00pp [2.00, 21.00]` 和
`+13.00pp [2.00, 24.00]`。但 pass^4 相对旧 SFT 下降
`-6.00pp [-12.00, -1.00]`。因此保留 dependency-safe multi 与 local-first broad tier，
但当前数据配方不直接升级为唯一默认；下一版增加分域 consistency anchor，并以 pass^4
non-inferiority 为发布门槛。完整统计见 [EVAL_COMPARISON.md](EVAL_COMPARISON.md)，最终
标准与下一步见 [FINAL_RECOMMENDATION.md](FINAL_RECOMMENDATION.md)。
