# Banking SFT 数据与评测变慢诊断

目的：解释 Banking 单独 SFT 和混合 AReaL SFT 的评测异常耗时。
检查时间：2026-09-10，使用现存数据、训练日志和已保存的官方评测轨迹；未重训或更改评测协议。

## 任务处置与进度

| Job | 检查点 | 已完成其他三域 | Banking 完成 | 最近约一小时新增 | Banking 剩余估计 | 处置 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 17090 | iter3999，之后原计划 iter4673 | 400/400 | 87/388 | 16 | 约 19 小时 | STOPPED |
| 16762 | iter4399 | 400/400 | 93/388 | 23 | 约 13 小时 | STOPPED |

两者均满足用户指定的“剩余超过一小时立即停止”。17090 使用原 16763 的日志目录；尚未开始 iter4673。停止后再次查询确认两个 Job 均为 STOPPED，requeued_as=0。停止保留了已有日志、结果和检查点。

## 评测中实际发生了什么

以下为 Banking 域，SFT 仅统计停止前保存的轨迹，包含失败记录；不能当作完整 held-out 成绩。Agent turn 数含框架的初始问候。Prompt 均值按 Assistant 消息统计，初始无 usage 的问候计零。

| Agent | 保存轨迹数 | 平均耗时（秒/轨迹） | 平均 Agent turns | KB_search（次/轨迹） | 平均 Agent prompt tokens | max_steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始模型，16566 | 388 | 25.43 | 13.70 | 1.24 | 11,101 | 8 |
| Banking-only，fixed 评测 | 95 | 1,425.42 | 62.49 | 29.60 | 120,947 | 34 |
| 混合 iter3999，17090 | 87 | 1,063.72 | 50.08 | 26.26 | 107,078 | 18 |
| 混合 iter4399，16762 | 93 | 1,215.53 | 49.03 | 26.18 | 115,281 | 24 |

按相同 task_id + trial 匹配原始模型控制：iter3999 的 87 对为 1,063.72 vs 22.41 秒、26.26 vs 1.33 次检索；iter4399 的 93 对为 1,215.53 vs 24.12 秒、26.18 vs 1.38 次检索。耗时约放大 47–50 倍，检索次数约放大 19–20 倍。这仍是已完成子集，不包含停止时未保存的在途对话，也不能消除服务负载差异。

单次生成并没有普遍变长：iter3999/4399 的 completion tokens 均值为 38/44，中位数都是 29。问题是连续搜索、读取长结果、继续搜索，累积到超过 10 万 token；最大 prompt 分别达 258,847/259,163。部分后期输出出现错误工具格式或参数错误，长上下文和重试进一步放大耗时。

例：iter3999 的 task_066 连续搜索账户关闭、储蓄利率、账户对比，prompt 从 3,757 增至 9,198、13,746、19,270，末期达到 258,847，并输出无法解析的 KB_search 文本。这里多为不断改写 query，不能只用“完全相同调用重复次数”衡量检索循环。

“整个混合模型数值崩溃”不符合现有证据：混合训练 4,674 条 loss/grad_norm 记录均有限；已完整完成的 Telecom，raw/iter3999/iter4399 分别成功 28/75/77 次（各 160 次）。Banking-only 存在跨域退化，但混合模型当前主要是 Banking 行为退化拖住总评测。

## 主问题：BM25 训练保留搜索前缀，丢掉后续完成动作

输入为 1,914 条成功源轨迹。四条因转换错误被排除，其余 1,910 条生成 13,804 个目标；16,384-token 上限排除 6,788 个目标，保留 7,016 个。按检索模式拆开后：

| 数据 | 源轨迹 | 候选目标 | 保留目标 | 保留文本目标 | KB_search 目标 | 保住最后一个 Agent 目标的轨迹 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Golden retrieval | 925 | 2,095 | 2,095 | 1,165（55.6%） | 0 | 925（100%） |
| BM25 | 985 | 11,709 | 4,921 | 365（7.4%） | 3,927（79.8%） | 240（24.4%） |
| AReaL processed，实际混合来源 | — | — | 30,376 | 13,979（46.0%） | 不适用 | 未统计 |

全部 6,788 个超长目标来自 BM25。其原有 1,693 个文本目标只剩 365（21.6%），原有 520 个 give_discoverable_user_tool 目标只剩 57（11.0%）。694/985 条 BM25 轨迹没有任何文本回答目标留下；298 个 BM25 task 中只有 129 个保有文本目标。

问题发生在 [build_banking_expert.py](../../../examples/tau2-bench/sft/build_banking_expert.py) 的逐目标长度过滤（659–679 行）：早期检索前缀合格就保留，后续越过上限的回答或执行目标单独丢弃。并非将单条 token 序列截断，也不是丢失 EOS；它改变了监督动作的分布。

Golden 的所需文档直接放在 system prompt 中，BM25 则需要检索；二者的 prompt 和工具集合不同。Golden 中充足的回答目标不能替代“检索结果之后应如何回答”的监督。BM25 保留数据中，工具结果后接 KB_search 有 2,922 条，接文本只有 351 条。

这支持的主要因果解释是：SFT 强化了检索场景中的“继续查”，缺少同一场景下的正确执行和收尾监督；推理时不断累积长检索结果，离开 16K 训练分布后进一步退化。各因素的独立贡献仍需清洗数据后的受控重训验证。

## 另外两个具体数据问题

1. **源轨迹已有大量绕路检索，拆分又改变了决策条件。** 985 条 BM25 源轨迹共 7,873 次 KB_search，694 条至少搜索五次。保留的 3,927 个搜索目标中，1,823 个来自原始多调用 turn，793 个是同一批中第二个及之后的调用。转换器 301–320 行将原始并行调用拆开，并在下一个调用前插入上一个结果；这些后续 query 原本在看见结果前已确定，却被训练成“已经看见结果，仍决定继续搜索”。这是监督条件偏差，不代表所有多调用拆分都错误。
2. **轨迹总 reward=1 掩盖了错误动作。** 源轨迹共有 491 条工具错误结果，其中 351 条为不存在的 request_human_agent_transfer。追踪回训练目标后，仍有 102 个实际执行失败的 Agent 动作被作为正向 SFT 目标保留；其中 give_discoverable_user_tool 有 93 个，占这一类全部 173 个目标的 53.8%。例如 task banking_syn_v1_train_000018_l3_single_action 的前两个目标分别 give/unlock 不存在的 request_human_agent_transfer。另有 401 条源轨迹含 action_match=false；其中一些最终 DB 成功仍成立，因此不能仅据 action mismatch 否认官方成功，但也不能把整条成功轨迹的每个动作视为优质监督。

Golden 中另有 269 个 Yes/No 短回答目标，多数来自明确要求 Yes/No 的合成问答。这些不是角色写反，但与真实多轮检索服务任务差异较大。将两种场景汇总成“1,530 条文本回答”会掩盖 BM25 完成样本的短缺。

## 已检查的替代解释与建议

- Banking 全量消息未发现字面量 im_start/im_end、think 标签、tool_call 标签或 STOP 标记污染。结构化工具 arguments 顶层与 schema 的类型检查没有发现普遍错误，只发现一条 log_verification 缺 address。
- Banking/AReaL 各随机抽取 40 条，以本地原模型 tokenizer 和当前 qwen3_full 重建训练 mask：80/80 个监督片段均精确对应最后一个 Assistant 目标，80/80 均训练 im_end。此项为抽样检查，不代表全量 token 验证。
- Banking-only final_hf、混合 iter3999、iter4399 的 chat_template 均与原模型一致，结束 token 为 im_end，generation_config EOS IDs 为 [151645,151643]。
- 训练脚本启用 rollout-shuffle；AReaL 后拼接 Banking 的文件顺序本身不能证明训练末尾连续只见 Banking。

建议先重建 Banking SFT，而非继续延长现有检查点评测：以能完整保留有效执行和结束响应的 BM25 轨迹为单位构建首个对照集，删除已知失败动作的监督，筛掉无效搜索过程；单独统计 Golden/BM25 的监督分布。超长成功轨迹优先重新生成更紧凑的解决过程；若压缩检索结果，必须保留完成任务所需事实，并检查与推理输入的匹配。不要简单把上限继续调大，或仅重复采样孤立的结尾行。

随后固定原模型初始化、AReaL 数据、学习率、训练预算和 User，做清洗前后的 Banking 小规模对照；先看检索次数、成功执行、完成对话和上下文长度，再按同一官方协议评估。本次完成数据诊断和任务停止，未执行重训。

## 数据来源与复查位置

- 原始成功数据：[successful_trajectories.jsonl](../tau2-banking-independent-synthetic-v1/final-minimal/successful_trajectories.jsonl)。
- 实际 Banking 训练数据：[banking_expert_sft.jsonl](data/banking_expert_sft.jsonl)；AReaL 数据：[agent_official_native_expanded.jsonl](../tau2-sft-official-native-expanded/data/agent_official_native_expanded.jsonl)。
- 混合数据来源：[agent_banking_mixed_sft.stats.json](../tau2-sft-full-domain/data/agent_banking_mixed_sft.stats.json)。
- 原始模型控制：[16566 summary](eval/raw-full-qwen36-bm25-fixed/seed300_0910_raw_fixed_summary.json)。本报告使用用户指定的 16566，不复用 README 中旧控制的数值。
- 16762：[run log](../tau2-sft-full-domain-eval/jobs/16762-full-domain-eval-0004399-r2-0909-184039134/run_0_20260909_184039134.log)。17090：[复用的 16763 run log](../tau2-sft-full-domain-eval/jobs/16763-full-domain-eval-0003999-0004673-r2-0909-184040172/run_0_20260909_184040172.log)。
- 官方逐轨迹结果位于 `/mnt/afs/users/fush/projects/ServiceAgent/tau2-bench/data/simulations/`：混合 iter4399 使用 `seed300_0910_070824`，iter3999 使用 `seed300_0910_090401`，Banking-only fixed 使用 `seed300_0910_fixed`，raw fixed 使用 `seed300_0910_raw_fixed`，各目录下 `results.json`。
