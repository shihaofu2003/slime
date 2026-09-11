# tau2 SFT quality audit

实验名：`tau2-sft-quality-audit`。目的：审计当前 Agent SFT checkpoint 实际使用的
max-8192 AReaL 数据，先重建成功轨迹并提炼高质量模式，再用确定性规则和校准后的
Qwen3.6-27B 盲审做 `keep/review/drop`；只有本地 judge 不可靠或样本不确定时才调用
API，最终输出可追溯的筛选 manifest 与训练 JSONL。

状态：本实验保留为历史 single-policy baseline。后续审计已证明 multi 数量本身不是质量
代理，并已完成 local-first 数据构建、raw-init SFT 与三模型 official eval；最终采用标准见
[tau2-sft-local-first-relaxed](../tau2-sft-local-first-relaxed/FINAL_RECOMMENDATION.md)。

## Tasks

- `tau2-sft-quality-qwen36-27b`：job `pt-8e71ysbm`，本地模型成功加载，但 fallback
  检测把占位 OpenRouter key 误认为可用配置，在校准前退出；无 API 调用。
  [run log](jobs/fsh-tau2-sft-quality-qwen36-27b-0802-002543/run_20260802_002543.log)。
- `tau2-sft-quality-qwen36-27b-r2`：job `pt-oy3gv9yv`，修正为验证 key 可用性并在
  OpenRouter 不可用时采用已有 OpenAI-compatible endpoint。校准前 7 条显示显式
  thinking 挤占输出预算并造成结构化 JSON 截断，因此主动终止，未启动全量审查；
  部分结果保存在 `calibration_reviews_thinking_partial_r2.jsonl`。
  [run log](jobs/fsh-tau2-sft-quality-qwen36-27b-r2-0802-003400/run_20260802_003400.log)。
- `tau2-sft-quality-qwen36-27b-r3`：提交因成员 quota 已满而失败，没有创建 job。
  [submit log](jobs/fsh-tau2-sft-quality-qwen36-27b-r3-0802-005132/submit_20260802_005132.log)。
- `tau2-sft-quality-qwen36-27b-r3-spot`：job `pt-099ntln5`，NORMAL spot；关闭显式
  thinking、使用 v2 防注入 prompt。本地双审校准通过；全量进行到 78 条时发现
  Gemini 会把 confidence 错写成维度量表的整数 `4`，因此主动终止。诊断记录保存在
  `judge_reviews_confidence4_partial_r3.jsonl`，未用于最终筛选。
  [run log](jobs/fsh-tau2-sft-quality-qwen36-27b-r3-spot-0802-005924/run_20260802_005924.log)。
- `tau2-sft-quality-qwen36-27b-r4-spot`：job `pt-sphfwbnr`，NORMAL spot；环境依赖安装
  时遇到镜像域名解析失败，未进入模型审查，随后删除。
  [run log](jobs/fsh-tau2-sft-quality-qwen36-27b-r4-spot-0802-012459/run_20260802_012459.log)。
- `tau2-sft-quality-qwen36-27b-r5`：job `pt-tsrhtg9y`，NORMAL reserved；复用已通过的
  本地校准，API 对非法 confidence 会收到具体值、0–1 合法范围和禁止复用 0–4
  维度量表的修复指令；全量 2,496 个 dialog 完成，0 error，job `SUCCEEDED`。
  [run log](jobs/fsh-tau2-sft-quality-qwen36-27b-r5-0802-012816/run_20260802_012816.log)。

## Historical deterministic result

- 19,318 条实际训练 prefix 对应 2,496 个 dialog；source 有 2,499 个 dialog。
- 445 个 dialog 的 `correct=0`，333 个 `reward=0`；两种标签存在 122 个冲突 dialog，
  实际训练文件共有 2,507/19,318 行来自非正或冲突标签 dialog。
- 在历史 single-policy 下，1,883/2,496 (75.44%) 个训练视图包含同轮多调用；SFT loss
  中该模式被 prefix
  重复暴露 16,722 次，12,258/19,318 个训练 prefix 已含该模式，而 target 本身有
  3,014 个属于同轮多调用。
- 历史协议冲突高度集中于 airline：994/999 (99.50%) 个 dialog 含同轮多调用，
  875 个正标签 dialog 中只有 5 个未命中确定性硬错误；retail 为 856/1,000，
  telecom 为 33/497；这些数量不再作为当前质量判断。
- 只有 315 个 dialog 同时为 `correct=reward=1` 且未命中确定性硬错误；对应
  airline/retail/telecom 仅 31/742/547 条训练行，且 telecom 没有规则干净的写入轨迹。

## Historical model result

- Qwen3.6-27B 在 72 条分层样本上的双审 verdict 一致率 94.44%，维度平均绝对差
  0.118，硬错误 false-keep 为 0，median confidence 0.95，校准通过。
- 全量 `keep/review/drop` 为 `235/78/2,183` dialogs；严格保留 998/19,318 行
  (5.17%)，分领域为 airline 31、retail 666、telecom 301 行。
- 规则干净候选中，airline `5/5`、retail `132/144`、telecom `98/166` 最终 keep；
  模型额外识别了漏掉第二意图、状态记忆失败、错误恢复和假成功声明等语义问题。
- 783 条本地不确定样本调用 API；114 条 judge 结论因与精确机械事实冲突而被保守
  覆盖。配对集合的 verdict 一致率只有 61.43%，因此模型分数用于筛选和定位，不能
  替代可执行 evaluator 或匹配重训。

## Historical held-out cross-check

[Raw→SFT→RL 轨迹分析](../tau2-traj-pattern/README.md)中，同轮 multi 从
`0.75% → 65.50% → 75.50%`，而需要写 DB 的成功数为 `42 → 37 → 37`；SFT/RL
提高了调用覆盖，没有同步提高事务成功。结合本审计中的 prompt 冲突和 loss 暴露，
SFT 是 multi 行为的强候选传播路径；但 action/DB 能力缺口还受轨迹语义质量、截断、
RL credit assignment 与优化稳定性影响，仍需匹配重训才能做因果归因。

## Artifacts

- [`DETERMINISTIC_REPORT.md`](DETERMINISTIC_REPORT.md)：标签、规则和 loss 暴露。
- `success_sample.jsonl` / `calibration_sample.jsonl`：成功画像与校准样本。
- `judge_reviews.jsonl` / `calibration_report.json`：模型判断及本地可靠性门槛。
- `REPORT.md` / `SUCCESS_CASES.md` / `selection_manifest.jsonl`：最终结论、代表正反例
  与每个 dialog 的筛选证据；`judge_adjudication_overrides.jsonl` 保留机械裁决。

## Follow-up completed

- 协议中立审计将“轨迹含 multi”“决定性错误在 multi turn”和“batching 导致错误”分开，
  确认调用数量不能作为 drop reason；结果见
  [COMPARISON_REPORT.md](../tau2-sft-multitool-quality-audit/COMPARISON_REPORT.md)和
  [FAILURE_ATTRIBUTION.md](../tau2-sft-multitool-quality-audit/FAILURE_ATTRIBUTION.md)。
- local-first 筛选从 consensus-success 中保留 1,863 dialogs / 15,807 canonical targets，
  只监督安全 target prefix，并将 system 与领域 policy 统一为 `dependency-safe-multi`。
- 同配置 official eval 中，新 SFT 为 `27.00%` pass@1、`54.00%` pass@4(any)、`4.00%`
  pass^4；前两项相对 raw 和旧 SFT 显著提升，pass^4 相对旧 SFT 显著下降 6pp。
- 最终决策是保留 broad tier、增加分域 consistency anchor，并以 pass^4 non-inferiority
  作为下一版发布门槛；不使用本实验的 998-row strict 输出直接重训。
