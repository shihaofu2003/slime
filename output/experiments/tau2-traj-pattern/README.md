# Raw→SFT→RL 轨迹模式分析

实验名：`tau2-traj-pattern`。目的：在相同的 1,200 条官方 tau2 test 轨迹上，对比 Raw Instruct、multitool SFT 与 RL `2e-6 iter99` 的成功和失败行为模式。

## 范围与判定

- 每阶段 400 条轨迹，覆盖 airline、retail、telecom；同一 task/trial 进行 Raw→SFT→RL 配对比较。
- Judge 为 `openai/gemini-2.5-flash`，prompt 为 `tau2-pattern-v1`；54 条校准轨迹与 18 条重复审查的一致率为 92.328%，JSON 解析率为 100%。
- 精确可判定的多工具、schema、namespace 与工具存在性事实优先于 Gemini 的冲突标签；所有覆盖记录在 `adjudication_overrides.csv`。
- 报告明确区分官方 reward、确定性规则事实、Gemini 语义判断和配对推断；reference action/DB check 是证据，不是唯一合法轨迹。

## 总体结果

| 模型阶段 | 官方 reward 成功 | 行为有效成功 | Reward-positive 中行为无效 |
|---|---:|---:|---:|
| Raw 原始模型 | 94/400 (23.50%) | 77/400 (19.25%) | 18.09% |
| Multitool SFT | 93/400 (23.25%) | 25/400 (6.25%) | 73.12% |
| RL 2e-6 iter99 | 102/400 (25.50%) | 14/400 (3.50%) | 86.27% |

“行为有效成功”要求：官方 reward=1、没有 critical Agent error，且经过确定性裁决后所有适用的关键好模式均为 pass/not_applicable。

## Tau2-bench 好模式

- **意图持续跟踪**（`intent_tracking`）：正确保持用户目标、话题切换和多个子任务。
- **行动前澄清**（`clarify_before_act`）：行动前收齐身份、实体、选项、原因和支付信息。
- **先读取、再规划、后行动**（`read_plan_act`）：先读取真实状态，再依据状态与 policy 选择流程。
- **原子化提交**（`atomic_commit`）：列明变更、费用和影响，获得最新确认后只写入一次。
- **渐进式诊断**（`progressive_diagnosis`）：Telecom 按 service→data→MMS 依赖顺序诊断并复验。
- **基于工具结果恢复**（`grounded_recovery`）：理解工具错误后改变方案，不重复碰撞。
- **基于事实收尾**（`grounded_closeout`）：仅在工具结果支持时宣称成功，并完成全部子任务。
- **领域 policy 合规**（`domain_policy_compliance`）：遵守领域 policy 和所有必需流程。

## Raw 原始模型：成功与失败模式

- **成功轨迹**：官方成功 94/400，行为有效 77/400；成功轨迹中适用好模式 pass 率最高的是 意图持续跟踪 100.0%、先读取、再规划、后行动 100.0%、渐进式诊断 100.0%、基于事实收尾 100.0%。
- **失败轨迹**：主要 failure mode 为 未识别到主要 Agent 错误（`none`）61.1%、工具命名空间混淆（`tool_namespace_confusion`）15.0%、参数错误（`wrong_arguments`）6.5%；高频失败 pattern 为 领域 policy 合规 26.8%、Agent/User 工具命名空间混淆 18.6%、参数错误 14.7%、工具或流程错误 13.7%。
- **事务能力**：无需 Agent 写 DB 的成功数为 52，需要 Agent 写 DB 的成功数为 42；失败但调用过预期写工具名的轨迹为 117。
- **Policy 风险**：0.75% 的轨迹含单轮多工具调用；reward-positive 中行为无效率为 18.09%。
- **阶段判断**：Raw 已有较强的基础意图跟踪和收尾能力；主要问题是 telecom 的 Agent/User 工具混淆、错误参数和事务流程不完整。部分 reward=0 轨迹未识别到主要 Agent 错误，说明失败还包含 evaluator、action/DB mismatch 或 User simulator 因素。

## Multitool SFT：成功与失败模式

- **成功轨迹**：官方成功 93/400，行为有效 25/400；成功轨迹中适用好模式 pass 率最高的是 意图持续跟踪 100.0%、行动前澄清 100.0%、先读取、再规划、后行动 100.0%、渐进式诊断 100.0%。
- **失败轨迹**：主要 failure mode 为 Policy 违规（`policy_violation`）51.1%、未识别到主要 Agent 错误（`none`）15.0%、工具命名空间混淆（`tool_namespace_confusion`）9.8%；高频失败 pattern 为 领域 policy 合规 81.1%、单轮多工具违规 65.5%、重复调用或扇出搜索 21.2%、工具或流程错误 18.2%。
- **事务能力**：无需 Agent 写 DB 的成功数为 56，需要 Agent 写 DB 的成功数为 37；失败但调用过预期写工具名的轨迹为 120。
- **Policy 风险**：65.50% 的轨迹含单轮多工具调用；reward-positive 中行为无效率为 73.12%。
- **阶段判断**：SFT 没有提高总体 reward，却显著学会同轮批量提交调用；这在评测 system prompt 中被允许，但违反三个 domain policy 的单轮单工具约束。runner 实际按顺序执行这些调用，并非并发执行。它提高了调用覆盖，却没有提高核心事务写入正确性。

## RL 2e-6 iter99：成功与失败模式

- **成功轨迹**：官方成功 102/400，行为有效 14/400；成功轨迹中适用好模式 pass 率最高的是 意图持续跟踪 100.0%、行动前澄清 100.0%、先读取、再规划、后行动 99.0%、基于事实收尾 99.0%。
- **失败轨迹**：主要 failure mode 为 Policy 违规（`policy_violation`）61.4%、未识别到主要 Agent 错误（`none`）10.7%、流程错误（`wrong_workflow`）10.1%；高频失败 pattern 为 领域 policy 合规 88.9%、单轮多工具违规 73.8%、重复调用或扇出搜索 25.5%、任务未完成 19.1%。
- **事务能力**：无需 Agent 写 DB 的成功数为 65，需要 Agent 写 DB 的成功数为 37；失败但调用过预期写工具名的轨迹为 138。
- **Policy 风险**：75.50% 的轨迹含单轮多工具调用；reward-positive 中行为无效率为 86.27%。
- **阶段判断**：RL 的 reward 增益来自无需写 DB 的任务；需要写 DB 的成功数与 SFT 相同。RL 更常调用看似正确的工具名，但参数、流程和最终状态仍不正确，同时进一步放大了同轮批量调用倾向。

## 阶段变化与配对结论

- 无需 Agent 写 DB 的成功数：`52 → 56 → 65`；需要 Agent 写 DB 的成功数：`42 → 37 → 37`。RL 没有改善核心事务执行能力。
- 失败但调用过预期写工具名：`117 → 120 → 138`。工具名覆盖增加没有转化为正确参数、流程或 DB 结果。
- 单轮多工具调用率：`0.75% → 65.50% → 75.50%`。同轮批量提交既符合评测 system prompt，又违反领域 policy，因此不能直接当作正向效率能力。
- Reward-positive 反例 `sft|airline|6|0`：官方 reward=1，但包含单轮多工具和 5 次写调用，其中一次带保险的新预订失败；Gemini 将其保险成功声明判为 `fail`，证据轮次 `[31]`。

关键好模式的配对 pass-rate 变化（按 task 聚类 bootstrap）：
- `raw->sft` `domain_policy_compliance`：-56.25%, 95% CI [-63.75%, -48.50%]。
- `raw->sft` `grounded_recovery`：-8.76%, 95% CI [-14.35%, -3.44%]。
- `raw->sft` `grounded_closeout`：-7.17%, 95% CI [-10.75%, -3.83%]。
- `sft->rl2e6` `domain_policy_compliance`：-9.00%, 95% CI [-13.75%, -4.50%]。
- `sft->rl2e6` `atomic_commit`：-4.70%, 95% CI [-8.55%, -1.18%]。
- `sft->rl2e6` `grounded_recovery`：+1.35%, 95% CI [-3.79%, +6.73%]。

配对阶段比较基于相同 task/trial 的变化，不把未配对均值解释为因果效应。`paired_transitions.csv` 包含全部 400 组转换。

## 代表案例与产物

- [`MULTI_TOOL_STRATEGY_REPORT.md`](MULTI_TOOL_STRATEGY_REPORT.md)：Tau2 /
  VitaBench、真实系统、理论边界、项目创新性与四臂实验的跨 benchmark 决策。
- [`SINGLE_CALL_REPORT.md`](SINGLE_CALL_REPORT.md)：同轮批量工具调用问题的完整证据、
  真实轨迹、数据占比、判断与整改方案。
- `representative_cases.md`：好模式、主要失败模式、改善、退化、持续成功和持续失败的证据案例。
- `features.jsonl`：1,200 条确定性特征。
- `judge_calibration.jsonl` / `judge_reviews.jsonl`：校准与全量语义审查。
- `pattern_summary.csv`：模型×domain×reward outcome 的裁决后模式率。
- `paired_transitions.csv` / `pattern_deltas.csv`：配对变化及 task-cluster 95% CI。
- `failure_modes.csv`：主要失败模式分布。
- `adjudication_overrides.csv`：规则事实覆盖 judge 标签的完整审计。
- `diagnostics/`：已被替代的格式错误或低一致率校准尝试。

`3e-6/5e-6` 仅保留既有确定性退化统计，未进入本轮 Gemini 全量审查。

## 复现与验证

依次运行 `analyze_trajectories.py`、`judge_trajectories.py --mode calibration`、校准验证、`--mode full`，最后运行 `summarize_trajectory_patterns.py`。脚本参数见各自的 `--help`。
