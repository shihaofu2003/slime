# Raw-all 与 Processed vanilla GRPO 行为分析

## 结论

- Raw-all RL 在 iter39 仍正常且略有提升，但 rollout 逐步重建把真正的行为
  转折进一步缩小到了 **iter39--iter49**：`rollout_id=46`（update 45 后的
  weight version 47；若逐步保存即 iter45）首次出现持续跃升，紧随 iter49 的
  rollout 已有 40/40 条
  base trajectory 出现 schema-copy；iter59 是第一个可以确认 schema-copy 已占
  多数 Assistant turn 的已保存 checkpoint。
- Raw-all 退化后形成了非常具体的生成模式：把工具定义中的
  `description/parameters/properties` 当成调用内容，而不是生成
  `arguments`；解析后参数变成 `{}`，工具报缺少必填参数，模型随后原样重复，
  最终以 `too_many_errors` 或长对话结束。
- 新发现的关键机制是 **raw generation 与模型可见/训练序列不一致**。合法 JSON
  的 schema-copy 会被 parser 接受，未知的 `parameters` 被忽略、缺失的
  `arguments` 默认为 `{}`。`AssistantMessage` 同时保存规范化 `tool_calls` 和原始
  `raw_data.text`，但下一轮 prompt 与训练样本只读取前者，将其渲染成
  `{"name": ..., "arguments": {}}`。因此失败轨迹的负 advantage 通常没有
  直接作用到模型实际生成的 `description/parameters` token，而成功轨迹有时又会
  给原始 schema-copy 文本或其空参数代理正 advantage。这是比“Raw 数据较脏”更
  直接的保留机制。
- Processed RL 没有发生格式坍塌。它维持约 `75%` 的训练 reference-action
  exact tier、低空参数率和很小的 K2 漂移，并逐渐形成“更少调用、较少工具错误”
  的保守策略。
- 但 Processed RL **相对自己的 SFT 没有总体能力提升**。iter99 两 seed
  pass@1 为 `50.75%`，比 SFT 的 `55.75%` 低 `5.00 pp`，task-paired
  95% CI 为 `[-9.38, -0.62] pp`。它在 Retail 有点估计改进，在 Telecom
  明显退化；因此只能说部分局部行为变干净，不能说最终任务能力变强。

## 证据与口径

分析使用 [最终评测结果](RESULTS.md)、[Raw-all 训练诊断](arms/raw-all/TRAINING_DIAGNOSTICS.json)、
[Processed 训练诊断](arms/processed/TRAINING_DIAGNOSTICS.json)、两臂的
[Raw-all rollout dump](arms/raw-all/trajectories/train100.jsonl) 与
[Processed rollout dump](arms/processed/trajectories/train100.jsonl)，并复用
[SFT 数据质量审计](../tau2-sft-raw-all-max16384/QUALITY_AUDIT.md)。

- 训练阶段表中的 reward、K2 和组构成来自每步最终接受的 5 个 prompt
  group；`有 GRPO 信号`指 K=8 内同时有成功和失败的组。
- 生成模式表扫描非 replacement 的原始候选轨迹。严格 `schema-copy` 定义为：
  可解析 JSON 的 `<tool_call>` 顶层含 `parameters/properties`，或顶层含
  `description` 而没有 `arguments`；对于已经不合法的 JSON，则要求同时出现
  `description` 与 `parameters/properties`，或出现完整 `properties` schema。
  这样不会把合法 `arguments` 内部恰好名为 `parameters` 的业务字段误计为
  schema-copy。所有 turn 率的分母只含模型实际采样的 Agent Assistant turn，不含
  固定的首句问候。用同一口径扫描 Processed 全部 base rollout，计数为 0。
- 官方行为指标在完全相同的 seed-300、100 tasks × 4 trials 上计算；最终
  SFT/RL 能力结论使用预先约定的 seed-300/301 两 seed 聚合。
- 下文将“观测到的时间顺序和共现”与“机制推断”分开。当前只有每个初始化
  一条训练 seed，不能证明某一个因素是唯一原因。

## Raw-all RL：退化从何时开始

接受组的训练曲线给出了清晰的相变。update 10--39 仍有一半以上的组提供
GRPO 对比信号；update 40 以后 K2 漂移和失败组同时上升，update 70--99
已有 80% 的接受组全失败。

| Updates | Raw reward | 有 GRPO 信号 | 全失败组 | K2 均值 | 阶段末 K2 | Grad norm 均值 |
|---|---:|---:|---:|---:|---:|---:|
| 0--9 | 0.3225 | 22/50 (44.00%) | 23/50 (46.00%) | 0.000335 | 0.000677 | 0.3230 |
| 10--39 | 0.4375 | 80/150 (53.33%) | 47/150 (31.33%) | 0.024909 | 0.049570 | 0.3742 |
| 40--69 | 0.3058 | 55/150 (36.67%) | 78/150 (52.00%) | 14.087525 | 72.971790 | 0.8084 |
| 70--99 | 0.1175 | 24/150 (16.00%) | 120/150 (80.00%) | 107.515793 | 58.641052 | 1.2264 |

退化也完整迁移到了 held-out official-native 评测，而不是训练 parser 的单独
统计现象：

| Raw-all checkpoint（seed 300） | pass@1 | Action accuracy | DB accuracy | 有参数 schema 错误的轨迹 | 空参数调用 | 有重复调用的轨迹 | `too_many_errors` / `max_steps` |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT | 53.25% | 73.89% | 45.99% | 1.25% | 0.21% | 12.00% | 16 / 10 |
| RL iter39 | 56.25% | 77.35% | 48.14% | 1.00% | 0.15% | 17.25% | 18 / 6 |
| RL iter69 | 25.50% | 70.63% | 38.00% | 86.00% | 46.68% | 62.50% | 89 / 61 |
| RL iter99 | 21.00% | 49.59% | 29.39% | 77.50% | 51.63% | 46.00% | 89 / 15 |

因此 iter39 是退化前的最后一个正式观察点，iter40--69 是坍塌区间；iter99
不是偶然评测波动，因为两 seed pass@1 都很低（`21.00%`、`24.50%`）。

## Raw-all schema-copy：精确转折点与奖励路径

每个 rollout 的 Agent `raw_data.text` 都保留了实际生成文本，且
`weight_version = rollout_id + 1`。训练先 rollout、再做同编号 update，因此
`rollout_id=50 / weight_version=51` 使用的正是 update 49 后保存的 iter49。
按这一关系可以直接定位 checkpoint，而不需要用 iter69 的评测结果倒推。

| 生成模型状态 | 下一次 rollout | strict schema-copy trajectory | schema-copy / sampled Agent turn | 含义 |
|---|---:|---:|---:|---|
| 初始 SFT | 0 | 0/40 | 0.00% | 本次 40 条中未观测到 |
| update 0 后 | 1 | 2/40 | 0.34% | 第一次出现；所在组全失败，advantage=0 |
| update 2 后 | 3 | 1/40 | 0.19% | 第一次 reward=1；所在组全成功，advantage=0 |
| update 8 后 | 9 | 4/40 | 0.89% | 第一次进入 mixed group 并取得正 advantage；随后写入 iter9 |
| **iter39** | 40 | 11/40 | 2.03% | 仍是低频模式，和 iter39 official eval 正常一致 |
| update 45 后（未保存 iter45） | **46** | **33/40** | **21.56%** | 第一次持续跃升，转折点落在 iter39--iter49 |
| **iter49** | 50 | **40/40** | 25.09% | 第一个保存后立即表现为全 trajectory 受影响的 checkpoint |
| update 56 后 | 57 | 40/40 | 45.99% | 快速扩张段开始 |
| update 57 后 | 58 | 40/40 | 68.90% | 首次超过一半采样 Agent turn |
| **iter59** | 60 | **40/40** | **66.78%** | 第一个可确认已由 schema-copy 主导的保存 checkpoint |
| **iter69** | 70 | 32/40 | 65.04% | 第一个接受 official eval 的坍塌 checkpoint |

固定为三段、以采样 Agent turn 为伯努利观测做穷举最大似然分段时，两个断点
正是 `46/57`：

| Rollout IDs | schema-copy / Assistant turn | 解释 |
|---|---:|---|
| 0--45 | 2.873% | 偶发探索与低速积累 |
| **46--56** | **21.422%** | 持续放大 |
| **57--99** | **67.093%** | 主导模式 |

rollout 46 的跃升也不是单域任务重复造成的：Airline 为 8/8、Retail 为
11/16、Telecom 为 14/16 条 trajectory 命中 schema-copy，三域同时出现。

这给出了三个不同但不冲突的“开始”：第一次观测是 rollout 1；第一次产生实际
正 GRPO advantage 是 rollout 9；真正的分布断点是 rollout 46，即 update 45 后的
weight version 47。若每步都保存，它对应 iter45；在现有每 10 步 checkpoint 中，
它落在 iter39--iter49 之间。现有证据能确定的第一个受影响保存点是 iter49，第一
个主导保存点是 iter59。iter49/59 没有做 official eval，因此这里判断的是它们
紧接着生成的 on-policy rollout 行为，不冒充 held-out 分数。

为什么它能“得到奖励”也可以从接受组中直接重建。本次实际进入训练的 4,000 条
trajectory 中，2,500 条含 strict schema-copy；其中 510 条最终 reward=1。
组归一化以后：

| schema-copy trajectory 的训练信号 | 数量 |
|---|---:|
| mixed group 成功，正 advantage | **370** |
| mixed group 失败，负 advantage | 384 |
| 全成或全败组，零 advantage | **1,746** |

370 条正-advantage trajectory 中，354 条（95.7%）出现过 tool execution
error，185 条（50.0%）出现 malformed JSON，130 条（35.1%）有 repetition，
但 366 条（98.9%）最终 DB match；145 条（39.2%）没有匹配全部 action check。
原因是本实验关闭了 shaping/行为惩罚，field error 只记录不进 reward，而许多任务的
二值 reward 由最终 DB/communication 决定。一次失败的读调用只要后来恢复并完成
写操作，就仍然可以得到 1；只要同组还有失败轨迹，它就进一步得到正 advantage。

最早的正 advantage 出现在 rollout 9 的 Airline group 51，组 reward 为
`[1,1,0,1,0,1,0,0]`。其中 `sample_index=414` 先生成
`description/parameters` 形式的 `get_user_details`，执行失败后改用正确格式并完成
任务，最终 reward=1、标准化 advantage=`+0.9354`。该轨迹仍有 execution error，
且 4 个 action check 只匹配 2 个，但最终 DB/communication 都成功。

更关键的是实际进入 loss 的 token。对可解析 schema-copy，parser 使用
`parsed.get("arguments") or {}`，所以真实生成：

```json
{"name":"get_user_details","description":"...","parameters":{"user_id":"chen_gonzalez_5516"}}
```

在 simulation/state 的 `AssistantMessage` 主字段中变成 `arguments={}`；原文仍在
同一对象的 `raw_data.text` 中。下一轮 Agent prompt 和训练 chat 都从主字段重建，
于是实际渲染为：

```json
{"name":"get_user_details","arguments":{}}
```

也就是说，原文没有从轨迹文件中消失，但在**模型可见历史和 `sample.tokens`** 中，
rollout 采样的 schema-copy token 被替换了。以 `sample_index=414` 为例，下一轮
真正送入模型的相邻历史片段是：

```text
<|im_start|>assistant
<tool_call>
{"name": "get_user_details", "arguments": {}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Error: AirlineTools.get_user_details() missing 1 required positional argument: 'user_id'
</tool_response><|im_end|>
```

其中不再出现原始 `description/parameters`。370 条正-advantage
trajectory 中，258 条只含这种规范化 schema-copy，102 条同时含规范化和原文保留
的 schema-copy，另有 10 条只含原文保留；合计 112 条的原始 schema-copy 文本
确实直接取得正 advantage，另外 258 条则把正 advantage 施加到了空参数代理上。

第一次“原始 schema-copy token 本身”取得正 advantage 是 rollout 21 的
`sample_index=987`。它把解释文本和 schema-copy `<tool_call>` 混在一个 Assistant
message 中，parser 将整段保留为 content；之后模型恢复并完成写操作，reward=1，
所在组为 `[1,1,1,1,0,1,0,0]`，advantage=`+0.7246`。因此这里不是理论上的
“可能得到奖励”，而是保存轨迹中实际发生过的正向更新。

离转折点最近、信号最强的直接样本是 rollout 42 的 `sample_index=1944`。它是
rollout 40--45 中唯一一条“原始 schema-copy 文本保留且为正 advantage”的
trajectory，但单条轨迹就有 8 个这种 raw-preserved turn。它出现 9 次 tool
execution error、8 次 malformed JSON、4 次 repetition，最后正确调用任务唯一要求
的 `transfer_to_human_agents`，于是 `ENV_ASSERTION=1`、`ACTION=1`，整个 episode
仍是 reward=1；所在组为 `[0,0,1,0,1,0,1,0]`，advantage=`+1.2076`。该样本没有
被截断或移除，3,363 个 Agent token 全部参与训练。它在 update 42 被正向更新，
rollout 46 在三次后续更新后发生分布跃升。这是一条具体且时间上紧邻的奖励泄漏
路径，但单次训练 seed 仍不足以断言它独自造成了整个相变。

同时也不能把 update 45 误写成“直接奖励了 schema-copy 原文”。产生 rollout 46
之前的 update 45 批次中有 18 条 schema-copy trajectory：5 条正 advantage、3 条
负 advantage、10 条零 advantage；这 18 条的 schema-copy 原文全部被规范化成
`arguments={}` 后才进入 loss。因此更准确的解释是：rollout 42 已经直接强化过一
条高密度 schema-copy 轨迹，随后多批成功恢复轨迹又强化了规范化空参数代理，而大
量零 advantage 和有损序列重建没有提供稳定的反向纠正；update 45 的参数更新触发
了可观测跃升，但不能仅凭相邻批次把它解释为一次直接的 schema-token 奖励。

这里还要区分对话 turn 和 optimizer update。simulation 确实逐 turn 调用 Agent，
但中途不更新模型；完整 episode 结束以后，`_training_messages` 遍历整条消息历史，
`qwen3_full` 一次渲染完整 transcript，并把每一个由 RL Agent 生成的 Assistant span
标成 `loss_mask=1`。`response_length` 从第一个可训练 Assistant token 一直延伸到
episode 末尾，中间的 User/tool token 仍在上下文中但 mask 为 0。actor 随后对完整
序列做 teacher-forcing forward，分别得到每个位置的 logits；因果 attention 保证
早期 turn 只能看到它之前的 prefix，并不会看到未来 turn。也就是说，它是在 episode
结束后**同时更新所有 Assistant turn**，不是只更新最后一轮，也不是拿最后一轮的
logits 代替前面各轮。

对“最终成功是否把错误 turn 奖励成 schema-copy”需要再区分两种目标。本次没有
turn credit；标准 GRPO 把一个组归一化后的 episode advantage 广播到该 trajectory
的全部可训练 Assistant token。设模型实际采样的调用为 `a_raw`，parser/serializer
为 `N`。对可解析的纯 schema-copy，rollout 采样的是 `a_raw`，但实际优化项近似为
`A(trajectory) * log pi(N(a_raw) | history)`，而不是
`A(trajectory) * log pi(a_raw | history)`；这里 `N(a_raw)` 正是空参数调用。只有
混合 prose 导致 parse error、使 `N` 不改写原文时，用户所说的“错误 schema token
随最终成功得到正 advantage”才逐字成立。

这条直接路径在转折点附近并非可以忽略。`sample_index=1944` 占 update 42 全批
可训练 token 的 `3,363 / 65,108 = 5.17%`；以
`advantage * trainable-token-count` 作为 loss 权重代理，它贡献 `4,061`，约占该批
全部正向权重的 `19.7%`。它在时间和权重上都足以成为局部触发候选。可是现有证据
仍不支持把“成功轨迹直接奖励原始 schema token”当成唯一根因：rollout 0--45 中，
原文保留的 schema-copy 正向样本只有 6 条 trajectory / 16 个 turn，负向却有
16 条 / 46 个 turn；紧邻跃升的 update 45 又没有任何原始 schema-copy token 进入
loss。

因此更准确的根因判断是：**episode-level credit leakage 与 many-to-one 有损
规范化共同构成了 schema-copy 的核心保留机制**。前者无法把成功轨迹里的错误 turn
单独判负，后者又把模型实际动作和受优化动作错配，使多数原始 schema-copy 得不到
对应的负梯度，并把正负信号施加到空参数代理；少数原文保留的成功轨迹则直接放大
schema-copy。它很好地解释了错误模式为何能留存并扩张，但还不能单独解释模型最初
为何选择 `description/parameters` 这一具体表面形式；后者仍需要 Raw-all 初始化的
潜在分布、prompt 中可复制的 tool schema 和 temperature=1 探索共同提供种子。

## Raw-all RL：为什么退化

1. **初始化对高温 on-policy rollout 已经脆弱。** Raw-all SFT 数据中
   `25.89%` 是 failure-labeled row，`33.23%` 的 tool-call target 含多调用，
   `15.54%` 混合文本和调用，`62.21%` 的 row prefix 含多调用；Processed
   对应项均为 0，且全部是 double-success、atomic target。Raw-all 源调用本身
   仍有 `99.92%` 通过当前 namespace/schema，所以问题不是“训练数据中的工具名
   普遍无效”，而是失败 outcome 与非原子上下文留下了更脆弱的生成分布。证据是
   temperature=1 的最初 10 个 update 中，Raw-all 已有 `56.0%` 候选轨迹受
   malformed JSON 影响，而 Processed 只有 `2.3%`；在 temperature=0.6 的
   SFT official eval 中两者又都基本正常。

2. **parser/训练序列的有损规范化是直接的保留通道。**
   `_parse_tool_call_content` 只要求 `name` 合法，把缺失 `arguments` 默认为
   `{}`，不会因为存在未知 `description/parameters` 而拒绝。随后
   `native_agent_message_to_chat` 从已解析 `ToolCall` 重建模型可见和训练 message。
   它没有删除 `raw_data.text`，而是根本不读取这个旁路字段。因此 state/dump 中保留
   了原文，下一轮 prompt 与 `sample.tokens` 却没有保留模型实际采样的 token；这
   破坏了该 turn 的严格 on-policy 对应关系，也使负 reward 很难直接压低原始
   schema-copy 序列。混合 prose 与 `<tool_call>` 的 parse-error message 是例外：
   它的原文进入 `content` 主字段，因此后续 prompt 和训练都能看到原始 schema-copy。

3. **二值 episode reward 无法给错误发生的 turn 定向 credit。** 本实验没有
   reward shaping 或 turn credit，同一轨迹所有 Agent token 只继承最终 0/1
   outcome 的组归一化 advantage。Raw-all 全程 500 个接受组中有 319 个
   零方差组（268 个全失败、51 个全成功），即 `63.80%` 的组没有更新信号；
   同时，一条最终失败轨迹中此前做对的动作也会在 mixed group 中得到负 advantage。
   update 59、60、61 更直接地显示了锁死状态：三步都是 5/5 零方差组，raw reward
   分别为 `0/0.2/0`，grad norm 都是 0；update 60 只是一个全成功组加四个全失败组。

4. **schema-copy 不但缺少稳定惩罚，确实多次获得正 advantage。** 但正/负
   trajectory 数量是 370/384，不能简化成“reward 总体偏爱 schema-copy”。更准确
   的说法是：有损规范化让许多负 advantage 作用在错误的 token 序列上；恢复成功
   又让 370 条带错误轨迹得到正 advantage；其余 1,746 条没有任何信号。

5. **没有 KL 锚点阻止模式进一步放大。** `KL_COEF=0`、
   `KL_LOSS_COEF=0`；K2 只记录、不进 loss。时间上 schema-copy 先起飞：
   rollout 46 已到 `21.56%`，当步 K2 仍只有 `0.0799`；随后 K2 在 step
   55/57/58/59 依次升到 `0.50/1.97/5.20/10.58`，iter69 达 `72.97`，
   全程最大 `204.62`。所以大 K2 不是初次触发条件，而是无锚点放大已经出现的
   token/serialization 错配的结果。成功率继续下降后，全失败组产生零 advantage，
   最终形成“格式错配 → 漂移 → 失败 → 无恢复梯度”的吸收状态。

排除项也很明确：训练完成 100 个有限 update，没有 NaN/OOM；每步 domain quota
始终是 `1/2/2`；训练 truncation 均值仅 `0.25%`、单步最大 `5%`。因此资源、
配额和超长轨迹不是这次退化的主因。

## Raw-all RL：退化后的生成模式

非 replacement 候选轨迹展示了模式如何从少量偏差演化成主导输出：

| Updates | 轨迹数 | schema-copy / sampled Agent turn | 空参数 / call | 工具执行错误轨迹 | 重复调用轨迹 | exact reference action | unmatched action | `too_many_errors` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0--9 | 400 | 0.28% | 2.55% | 34.25% | 11.00% | 63.6% | 23.5% | 14 |
| 10--39 | 1,200 | 3.44% | 8.02% | 40.92% | 13.83% | 71.7% | 18.7% | 36 |
| 40--69 | 1,200 | 39.85% | 57.89% | 83.33% | 56.83% | 42.1% | 41.9% | 497 |
| 70--99 | 1,191 | 66.17% | 100.00% | 84.05% | 78.34% | 0.0% | 75.8% | 563 |

典型错误链条是：

1. 模型知道应调用哪个工具，也从用户消息中拿到了值，但把工具 schema 的键复制
   到输出中。
2. official-native parser 能保留工具名，却找不到顶层 `arguments`，于是把参数解析
   为 `{}`。
3. backend 返回缺少必填参数；模型没有根据错误修正格式，而是重复同一调用。
4. 轨迹以 `too_many_errors` 结束、reward=0；当 K=8 都进入相同模式时整组
   advantage=0，无法把它拉回来。

例如 Raw-all `rollout_id=70`、`task=airline_498`、`sample_index=3120`
先生成：

```json
{"name":"get_reservation_details","description":"Get the details of a reservation.","parameters":{"reservation_id":"XAZ3C0"}}
```

结构化历史中它变成 `get_reservation_details(arguments={})`，环境连续返回
`missing 1 required positional argument: reservation_id`。Agent 原样调用 10 次，
产生 10 个 execution error、9 次 repetition，最终 `too_many_errors`。这说明
错误不是“不知道 reservation id”，而是生成格式模式错位后又失去了 error
recovery。

## Processed RL：训练中形成了什么模式

Processed 的训练没有相变。它的 K2 缓慢增至约 `0.01`，reference-action exact
tier 始终约 `75%`，严格 schema-copy 为零，空参数调用始终低于 `0.24%`。

| Updates | Raw reward | 有 GRPO 信号 | K2 均值 | schema-copy / Assistant turn | 空参数 / call | exact reference action | 工具执行错误轨迹 | 重复调用轨迹 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0--9 | 0.3525 | 36.00% | 0.000295 | 0.00% | 0.00% | 74.1% | 26.07% | 11.53% |
| 10--39 | 0.4525 | 62.00% | 0.001748 | 0.00% | 0.06% | 75.8% | 17.00% | 7.17% |
| 40--69 | 0.4875 | 52.67% | 0.006323 | 0.00% | 0.23% | 75.0% | 19.18% | 7.09% |
| 70--99 | 0.4358 | 45.33% | 0.010029 | 0.00% | 0.08% | 75.9% | 20.03% | 7.18% |

对完整训练集的 action-identity 审计也显示，Processed 没有 Raw-all 那种系统性
重写。按恰好 8 条、全组未移除的 group 恢复出实际训练的 500 组 / 4,000 条
trajectory，共有 58,679 个 RL Agent Assistant turn 和 26,509 个结构化工具调用。
其中 26,508 个（99.996%）的模型原始 `name/arguments` 与 parser 后用于执行和训练
的动作语义完全一致；唯一例外是 rollout 27、`sample_index=1188`：模型把 `dob` 和
`full_name` 放在调用顶层、没有 `arguments`，所以被改写为 `arguments={}`。该样本
所在 K=8 group 全部 reward=0，标准化 advantage 为 0，因此这次语义错配没有产生
由该样本驱动的 policy-gradient 更新。

去掉固定的 Assistant header/stop/separator framing 后，按完整
`<tool_call>...</tool_call>` 文本比较，26,485 / 26,509（99.909%）逐字一致；其余
24 个 turn 中，20 个只是 `\uXXXX` 与对应 Unicode 字符之间的 JSON
canonicalization，2 个是空白/JSON 排版变化，1 个删除了重复 JSON key，最后 1 个
就是上述空参数语义错配。仅有 2 个文本差异取得正 advantage，二者都只是 Unicode
转义变化、工具名和参数值未变。此外有 132 个含 malformed/mixed tool markup 的
turn 被当作普通 content 原文保留，并未规范化；另有 3 个只生成 stop token 的空
输出被替换成固定澄清句，其中 1 个为负 advantage、2 个为零 advantage。这说明
Processed 的工具调用内容在严格 token 层面仍有极少量 serializer/fallback 差异，
但不存在能解释训练趋势的系统性 sampled-action / trained-action 语义错配。
`qwen3_full` 还会把 chat template 在 `<|im_end|>` 后追加的换行纳入 Assistant span
mask；若把固定分隔符也算作 action token，则两臂都有同样的边界级差异。这里的
“一致”只指模型实际采样动作与训练动作相同，并不表示每个调用都符合任务或工具
schema。

在 seed-300 official eval 上，SFT → iter99 的平均 Agent tool call 从
`7.57` 降至 `6.49`，有 tool execution error 的轨迹从 `28.00%` 降至
`22.75%`，error/traj 从 `0.735` 降至 `0.395`，参数 schema error 从
`0.75%` 降至 0。与此同时，golden-action match 从 `73.06%` 缓慢降至
`71.02%`，全动作 exact 的轨迹从 `47.75%` 降至 `42.00%`。因此它学到的是
**格式稳定、较少出手、较少执行错误**，而不是更完整地完成所有必要动作。

## Processed RL：相对 SFT 是否改进

总体答案是否定的；局部答案是“Retail 和错误率有改善，但被 Telecom 的
under-execution 抵消”。

| 两 seed 总体 | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---:|---:|---:|---:|---:|
| Processed SFT | 55.75% | 81.00% | 28.00% | 74.76% | 44.11% |
| Processed RL iter99 | 50.75% | 80.00% | 22.50% | 73.73% | 46.21% |
| RL - SFT | **-5.00 pp** | -1.00 pp | -5.50 pp | -1.02 pp | +2.10 pp |

pass@1 的两次评测都下降（seed 300 `-4.75 pp`、seed 301
`-5.25 pp`），paired CI 排除 0。DB accuracy 的小幅上升不能替代二值任务
成功，因为它只覆盖可检查的 DB 状态，而且没有要求整条 action/NL outcome 同时正确。

| Domain | SFT pass@1 | RL99 pass@1 | Δ pass@1 | Δ Action accuracy | Δ DB accuracy |
|---|---:|---:|---:|---:|---:|
| Airline | 48.75% | 45.62% | -3.13 pp | +3.95 pp | -1.88 pp |
| Retail | 58.12% | 62.19% | +4.06 pp | +1.25 pp | +3.14 pp |
| Telecom | 56.88% | 41.88% | **-15.00 pp** | -5.71 pp | +2.47 pp |

Retail 的 `+4.06 pp` 是点估计改进，但 pass@1 CI
`[-1.25, +9.38] pp` 仍包含 0；Telecom 的 `-15.00 pp` CI 为
`[-22.19, -8.13] pp`，排除 0。Retail 中较明显的 action-check 改进包括
`exchange_delivered_order_items`（`71.2% → 81.2%`）、
`modify_pending_order_items`（`50.7% → 61.8%`）和
`return_delivered_order_items`（`77.8% → 82.9%`）。例如 seed300 Retail
task 9/trial 0 中，SFT 读完订单和商品后没有执行 exchange；RL99 在用户最终
确认后正确调用 exchange 并得到 reward 1。

Telecom 则出现相反模式。两 seed 中“Agent 服务端工具调用为 0”的轨迹从
`15.94%` 增至 `23.75%`；在需要状态写工具的任务里，期望写工具未调用率从
`28.29%` 增至 `40.13%`。关键 action-check match 也下降：
`send_payment_request` 为 `83.3% → 33.3%`，`resume_line` 为
`83.3% → 58.3%`，`refuel_data` 为 `65.6% → 53.1%`，
`transfer_to_human_agents` 为 `63.7% → 41.2%`。

一个直接例子是 seed300 Telecom
`[service_issue]break_apn_settings|lock_sim_card_pin[PERSONA:None]`、trial 1。
RL99 正确诊断 SIM PIN locked，也在文本中告诉用户“需要 transfer to a human
agent”，但它没有实际调用 `transfer_to_human_agents`；User 随即输出
`###TRANSFER###` 并停止，最终 reward 0。这个“在语言中描述应做的动作，却不执行
对应服务端工具”的模式解释了为什么对话看起来合理、工具错误更少，严格任务成功率
却下降。

## 结论边界与后续含义

这些结果只代表一次配对训练 seed。时间顺序、轨迹内容和两臂差异共同支持上述
机制，但没有单独消融 Raw-all 数据噪声、serialization 修复、turn credit 和 KL，
因此不能把任何一项表述为已经独立证明的唯一原因。不同的是，raw generation 被
规范化成另一条训练序列不是相关性推断，而是轨迹与代码共同确认的管线事实。

本实验能下的结论是：Raw-all 初始化在这套无 KL、纯二值 episode-level GRPO
下不可稳定训练；Processed 初始化解决了格式和漂移稳定性，却没有把训练 reward
转化为总体 held-out 能力增益。要验证因果，最小的后续对照应从同一个 Raw-all
SFT、同一训练 seed 重启，只修复“实际 generation、后续 structured history、
进入 loss 的 token”三者不一致，并补做 iter49/59 的 official eval。turn credit、
更细 action-state reward 与小 KL 锚点应分别做后续受控比较，继续以二值 official
eval 选配方；本实验不启动 OPD。
