# tau2 SFT 筛选结论与下一步

## Decision

- 删除“每轮最多一个 tool call”作为全局 keep/drop 条件；统一使用
  `dependency-safe-multi`。multi 只有在存在 observation dependency、参数无依据、写入未授权、
  顺序相关或无必要调用时才拒绝。
- `local-first-target-prefix-v1` 可作为大规模候选层：它把 69-dialog strict seed 扩为
  1,863 dialogs / 15,807 canonical targets，并在同预算 SFT 后显著提升 pass@1 和
  pass@4(any)。
- 当前 checkpoint 不能直接替代旧 SFT 作为唯一默认模型：它的 pass^4 相对旧 SFT 显著下降。
  筛选协议可以保留，训练集还需要一致性锚点和领域分层。
- 后续语义复审继续使用本地 Qwen3.6-27B；已有 Luna 结果只作保守校准/否决，不需要新增
  外部 API 调用。

## Evidence

相同 raw init、19,318 rows、2 epochs、2,413 updates 后，用完全相同的 official tau2
协议评测 100 tasks × 4 trials：

| Model | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| raw | 20.50% | 43.00% | 6.00% |
| old SFT | 21.25% | 41.00% | 10.00% |
| local-first SFT | 27.00% | 54.00% | 4.00% |

- 相对 raw：pass@1 `+6.50pp [1.00, 12.00]`，pass@4(any)
  `+11.00pp [2.00, 21.00]`，pass^4 `-2.00pp [-8.00, 4.00]`。
- 相对旧 SFT：pass@1 `+5.75pp [0.50, 11.00]`，pass@4(any)
  `+13.00pp [2.00, 24.00]`，pass^4 `-6.00pp [-12.00, -1.00]`。
- 新 SFT 的 `too_many_errors / max_steps` 为 `21 / 12`，优于旧 SFT 的 `36 / 17`，
  但差于 raw 的 `3 / 14`；pass^4 回退不能简单归因为更多终止错误。

训练数据审计也不支持按调用数量筛选：成功轨迹 matched 总质量差（multi − single）为
`+0.005 [-0.034, +0.045]`；唯一有足够两类暴露的 telecom 中，multi 与 non-multi 的
consensus-failure 风险差为 `+0.94pp [-7.95, +9.82]`。450 条非一致成功轨迹中，决定性
错误位于 multi turn 111 条、single-call turn 200 条、non-tool 52 条、environment/user
7 条、unknown 80 条；这是失败构成，不是调用模式的因果失败率。

held-out 轨迹同样只能作观察性诊断：新 SFT 的 multi-exposed 为 311/400，成功率 26.37%；
non-multi-exposed 为 89/400，成功率 29.21%。旧 SFT 对应 271/400、23.99% 与
129/400、15.50%。任务和领域混杂使这些比例不能作为 single/multi 的因果比较。

## Final screening standard

1. **Label gate**：仅 `reward=1 AND correct=1` 自动进入正向 SFT；其余进入 failure/review
   池，不因任一单标签成功自动保留。
2. **Mechanical gate**：tool/schema/namespace 必须有效；禁止 tool+text 同 turn；原始消息、
   target fingerprint、conversion hash、loss mask 和 tokenizer cap 必须通过确定性校验。
3. **Target-prefix gate**：以监督 target 及其依赖前缀为决策单元；晚期错误不得删除此前干净
   target，历史 assistant turn 全部 `step_loss_mask=0`。
4. **Semantic gate**：本地 judge 必须 reliable；tool choice、argument grounding、business
   policy、authorization、task completion 均至少 3/4，communication、recovery、necessity
   均至少 2/4；`drop` 拒绝，已有 Luna 覆盖按同门槛保守否决。
5. **Multi gate**：每个同轮调用必须在调用前已有独立依据、互不依赖结果、写入已明确授权且
   order-independent，并且每个调用都有必要；否则拆为单调用等待 observation。
6. **Two-tier release**：通过上述门槛的样本进入 broad tier；双评审一致 `keep`、所有维度和
   multi-turn 判定一致的样本另标 anchor tier。anchor 用于训练采样权重，不再把 broad tier
   全部丢弃。
7. **Coverage gate**：按 domain/task family 分层统计和采样；telecom 缺少高质量数据时应继续
   本地复审或补充生成，不回填已确认的低质量 target，也不靠少量样本高倍重复伪造覆盖。

## Acceptance gate

下一版候选必须在预先固定的第二 eval seed 上复现 pass@1、pass@4(any) 增益；pass^4 相对
旧 SFT 的 paired CI 下界应不低于预设的 `-2pp` non-inferiority margin，且每个 domain 的
pass@1 不得比最佳现有基线低超过 5pp。`too_many_errors`、`max_steps`、无效工具和重复调用
作为独立安全指标报告，不得用总体 reward 掩盖。

## Next experiment

1. 用本地 Qwen3.6-27B 对 broad tier 中的低置信、business-policy/communication 边界样本和
   telecom 样本做第二次独立复审，形成分域 anchor，而不是调用外部 judge。
2. 保持 raw init、19,318 rows、2,413 updates 和训练超参不变，对比当前 broad baseline 与
   domain-stratified anchor-mixture；先试 20% anchor，再根据 canonical 去重后的有效覆盖调整。
3. 用至少两个固定 eval seed 跑同一 100 tasks × 4 trials 协议；只有 breadth 指标保持增益且
   pass^4 通过 non-inferiority gate，才将新数据配方升级为默认。

完整指标见 [EVAL_COMPARISON.md](EVAL_COMPARISON.md)，训练数据单/多工具审计见
[COMPARISON_REPORT.md](../tau2-sft-multitool-quality-audit/COMPARISON_REPORT.md)，失败归因见
[FAILURE_ATTRIBUTION.md](../tau2-sft-multitool-quality-audit/FAILURE_ATTRIBUTION.md)。
