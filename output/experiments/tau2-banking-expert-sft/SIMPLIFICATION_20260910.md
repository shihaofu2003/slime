# Banking SFT 轨迹简化试验

目的：删除无关或冗余搜索，缩短完整专家对话，同时保住检索证据、任务结果和最后的 Assistant 回答。
日期：2026-09-10。只处理现有训练轨迹；未修改原数据、AReaL 数据或评测配置，未启动训练。

## 已完成的逐条审查试验

对 8 条“按标题查询政策文档 ID”的 BM25 专家轨迹，逐条核对了用户请求、所有 query、最终回答和对应文档正文。每条保留一次能独立从用户请求构造、且实际命中所需文档的原始搜索；删除其他搜索及“再试一次搜索”等过渡叙述。原始 query、保留工具结果和最终回答不改写。保留的完整搜索结果仍含原 BM25 返回的其他文档，未偷偷缩短工具返回内容。

| task 后缀 | 搜索次数前→后 | 完整对话 tokens 前→后 |
| --- | ---: | ---: |
| 000074 | 6→1 | 19,643→5,580 |
| 000232 | 8→1 | 24,990→5,698 |
| 000311 | 3→1 | 11,627→6,381 |
| 000374 | 8→1 | 34,989→7,102 |
| 000496 | 10→1 | 31,707→6,221 |
| 000533 | 4→1 | 12,465→5,800 |
| 000611 | 3→1 | 14,750→8,065 |
| 000917 | 4→1 | 16,314→6,046 |

合计搜索 46→8，tokens 166,485→50,893（减少 69.4%）。4 条原本超过 16,384 tokens、丢失结尾监督的轨迹现可完整保留。生成 16 条逐 Assistant turn 的 SFT 样本：8 个搜索目标、8 个回答目标。

审查依据：000074/000533 回答中的存款和转账限额直接来自命中文档；000232/000496 的重试与失败处理说明来自 scheduled-payments 文档；000374 的额度、利息、费用和提额条件来自 Diamond Elite 文档；000917 的零年费说明来自 Business Bronze 文档。000311/000611 只需报告所查文档的标题和 ID。保留 query 未使用原先猜测出的文档 ID；没有将隐藏答案作为新 query 写入训练。

产物：[SFT 数据](data/reviewed-lookup-v1/banking_simplified_sft.jsonl)、[完整轨迹](data/reviewed-lookup-v1/simplified_trajectories.jsonl)、[长度与删除调用对照](data/reviewed-lookup-v1/comparisons.json)、[回放结果](data/reviewed-lookup-v1/replay_results.json)。选择输入为 [reviewed_lookup_searches.json](data/reviewed_lookup_searches.json)，仅适用于这 8 条已审查轨迹。

## 全量保守处理结果

另对全部 1,914 条源轨迹执行了自动保守处理：同一连续 Agent 工具调用段内，只有搜索返回的每一篇文档正文都已在此前保留结果中出现，才删除该搜索。正文比较忽略检索排名分数和计时后缀，保留输出仍原样保存。未按 task.required_documents 自动删除其余文档，避免删掉答案或工具依赖的额外证据。

- 1,499 条无工具错误的轨迹进入简化。其余 415 条排除（4 条没有合格原始 SFT runtime，411 条含工具错误）；没有强删错误调用后留下失去上下文的纠错回答。
- 148 条发生变化，删去 223 次搜索。总 tokens 19,190,342→18,481,080，减少约 3.7%。
- 按完整轨迹 16K 上限保留 1,085 条（326 个任务）：Golden 862、BM25 223，生成 2,976 条目标样本。414 条仍超长，整条排除，未再次保留孤立前缀。
- 自动精简只额外救回 4 条完整轨迹，最终入选的 1,085 条中也只有这 4 条发生删调用。这一版本主要是无工具错误、完整结尾的数据筛选结果，不能把它描述为全量语义精简成功。
- 保留专家原来的多工具调用消息结构，2,976 个目标中有 67 个多工具目标。工具调用与正文混合的消息保留其可见正文；不复制原始 reasoning 字段。

产物：[保守候选 SFT](data/simplified-v1/banking_simplified_sft.jsonl)、[统计](data/simplified-v1/stats.json)、[回放结果](data/simplified-v1/replay_results.json)。该文件与 8 条审查试验集分开保存，不应直接拼接，因为部分 source trajectory 重叠。

## 验证与结论边界

1. 所有输出目标使用现有 `qwen3_full` 重建 token 和 loss mask，逐条确认监督片段精确对应最后一个 Assistant 目标，包含 EOS。最大样本长度：保守候选 16,316，审查试验 8,065 tokens。全量扫描未发现 think 标签或原始 reasoning/raw_data 残留。
2. 对保守候选 1,085/1,085 条、审查试验 8/8 条重放原始及简化后的状态变更，Agent/User 最终 DB 内容完全相同；ACTION/COMMUNICATE 分数前后相同，任务要求的这些分项仍为 1。这里比较的是与源轨迹的一致性，没有调用 LLM 重新生成对话，也没有另改官方成功定义。
3. 审查试验中 8 个保留 query 还实际重新执行了 BM25，返回的文档 ID/正文与保存结果全部相同。因此只保留该次搜索时，最终答案所用证据仍能真实取得。

结论：逐条判断证据与回答依赖后，可以显著压缩这类检索轨迹；仅做机械重复删除收益很小。8 条试验尚不覆盖验证身份、读写账户、解锁工具等完整业务流程，也未验证重训后的评测收益。全量保守候选仍偏 Golden，不能据其规模或回放成功就宣布解决了原 BM25 监督分布问题。

## 复现

实现：[simplify_banking_expert.py](../../../examples/tau2-bench/sft/simplify_banking_expert.py)、[replay_simplified_banking.py](../../../examples/tau2-bench/sft/replay_simplified_banking.py)。复用现有工具序列化、Qwen 模板/loss mask 和 Banking 环境；不改原转换器。Python 编译检查已通过。

在 repo 根目录执行；转换环境需要 transformers，回放环境需要本仓库匹配的 tau2 依赖及 rank-bm25 0.2.2：

```bash
PYTHONPATH=. python examples/tau2-bench/sft/simplify_banking_expert.py \
  --reviewed-searches output/experiments/tau2-banking-expert-sft/data/reviewed_lookup_searches.json \
  --output-dir /tmp/banking-reviewed-rebuild
PYTHONPATH=.:../tau2-bench/src python examples/tau2-bench/sft/replay_simplified_banking.py \
  --data-dir /tmp/banking-reviewed-rebuild --check-search-results
```

不指定 `--reviewed-searches` 时，执行全量保守冗余删除。本次 token 构建使用 BEACON-verl-agent 环境；回放使用 fsh-tau2 环境，并将 rank-bm25 单独安装到 `/tmp/banking-simplify-deps`，未改共享环境。
