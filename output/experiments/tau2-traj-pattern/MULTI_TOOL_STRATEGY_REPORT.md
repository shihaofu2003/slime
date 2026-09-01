# 单轮多工具调用：Tau2 / VitaBench 跨 Benchmark 决策报告

本文回答的不是“Agent 要不要使用多个工具”，而是一个更窄的问题：
**模型是否应该在尚未看到任何工具结果时，在同一个 assistant turn 中一次提交
多个 tool call。** 证据来自官方 Tau2/VitaBench 代码与 prompt、AReaL 论文及
数据、主流工具调用接口、相关研究，以及我们 1,200 条 held-out 轨迹。统计口径
覆盖 Airline、Retail、Telecom；完整 Tau2 案例见
[Tau2 专项报告](SINGLE_CALL_REPORT.md)。

## 一页结论

之前“所有场景都应改成单轮单调用”的结论范围过大，需要修正；但当前
**不加区分的 multi-call 数据确实有问题**。最终决策不是二选一，而是两个协议：

| 场景 | 决策 | 原因 |
|---|---|---|
| Tau2 官方训练与评测 | `tau2_official_single`：每轮至多一个调用 | 三个 domain policy 明确这样要求 |
| VitaBench 的独立读取/纯计算 | `safe_batch`：允许有上限的同轮 batch | 官方示例这样做，且没有结果依赖 |
| 读后写、前一步结果决定后一步参数 | 必须逐轮执行 | 后一步需要先看到真实 observation |
| 支付、取消、修改、同一对象多写、需用户确认 | 默认逐轮执行 | 防止越权、部分提交和状态冲突 |
| 工具图中的环 | 跨轮观察与重规划 | 环不是并行机会；可并行的是当前可立即执行集合（ready frontier） |

直接回答几个核心问题：

- **单轮多调用是否天然更好？** 不是。它在正确性上不优于串行；优势主要是减少
  LLM 往返，在真正并发的执行器里还可能降低工具等待时间。
- **是否天然更差？** 也不是。参数已经全部知道、相互独立的只读查询可以安全
  batch。我们也有成功的真实轨迹。
- **我们当前学到的 multi-call 是否更差？** 在 Tau2 协议层面全部不合规；这
  不等于实证上每一种 multi 的任务成功率都更低。含写、有依赖、重复和过量扇出
  （一次铺开许多调用，fanout）的 batch 明显更危险；对所有纯读取 batch，现有
  观察数据不足以证明其 final-state success 更低。
- **是否需要筛数据？** 需要，但不能把 multi 全删掉。应把数据分成“安全 batch、
  必须等待、高风险写、重复/错误”四类，并为 Tau2 和 Vita 生成不同协议视图。
- **能否成为项目亮点？** “支持多个 tool call”本身不新。真正有价值的方向是：
  **让 4B 模型根据 benchmark policy、当前状态和工具依赖，学习何时不能并行，
  再由运行时安全检查（guard）验证边界。**

当前四个直接阻塞是：Tau2/Vita 的协议互相污染；reward 不识别依赖与事务安全；
两个 runner 都串行执行且没有 batch 回滚；Vita 缺少可直接训练的 gold
trajectory。

一句话项目主张：

> 我们不是让 4B 模型“多叫工具”，而是让它在有状态服务环境中学习可验证的调用
> 宽度：独立读可 batch，依赖和写串行，协议要求时自动退化为单调用。

## 先把三个“多工具”分开

很多争论来自术语混用。

| 能力 | 例子 | 是否必要 |
|---|---|---|
| 一条完整轨迹使用多个工具 | 查用户→查订单→查规则→修改订单 | 服务 Agent 必需 |
| 一次用户请求触发多步工具循环 | 模型→工具→观察→模型→工具 | 服务 Agent 必需 |
| 一次 assistant 输出多个调用 | 同时提交查订单 A、B、C | 只在安全子集有价值 |

Tau2 policy 反对的是第三种，不是让模型“一辈子只会调用一个工具”。一个严格
single-call Agent 仍然可以在 20 个 turn 中调用 20 次不同工具，完成复杂事务。

本文把第三种称为**同轮 batch**。只有执行器真的并发执行时才称为“并行调用”。
Tau2 和 VitaBench 当前 runner 都用普通 `for` 循环顺序执行 batch，因此现阶段
最多减少 LLM turn，不能宣称获得了工具 wall-clock 并行加速。

## 理论上什么时候 batch 才更好

把当前已经满足全部前置条件、参数也已经知道的动作集合记为“当前可立即执行集合”
（`ready frontier`）。合法 batch 必须是这个集合的子集，并至少满足：

1. batch 中调用的参数都只依赖当前历史，不能依赖同 batch 另一个调用的结果；
2. 调用之间没有读后写、写后读或写后写冲突；
3. 调换执行顺序不改变结果，或者它们都是只读/纯计算；
4. 用户授权和 benchmark policy 允许；
5. 调用数量有上限，没有重复和无目的 fanout。

```text
当前信息已知
   ├─ read(order A) ─┐
   ├─ read(order B) ─┼─→ 一起观察结果 → 再决定下一步写操作
   └─ calc(distance) ┘

read(order) → 得到 status/order_id → cancel(order_id)
              上面这条依赖链不能放在同一 batch
```

**为什么单调用更稳。** 在当前 benchmark 的顺序执行和相同 observation 语义下，
串行策略通常可以实现与安全 batch 相同的任务结果，同时在每次结果后改变计划；
其代价是更多 LLM 往返。batch 预先锁定全部动作，失去了中途停止、改参、换工具
和恢复错误的机会。因此 batch 不会在正确性上严格支配串行。

**为什么选择性 batch 仍可能有价值。** 在上述语义下，如果 scheduler 对安全性
判断完全正确，它可以在不改变任务结果的前提下减少模型生成次数；若工具真正异步
执行，还可把独立调用的总耗时从“相加”降为“取最大值”。所以理论上最优策略是
动态选择 `k∈{0,1,2,...}`，而不是永远 `k=1` 或永远 `k>1`。现实中的 4B 会
误判依赖，因此必须同时训练模型选择器（gate）和设置运行时安全检查（guard）。

**有环为什么不等于适合 batch。** 若某个具体动作 `A→B` 表示 B 依赖 A 的输出，
A、B 本来就应隔着 observation。Vita Figure 3 则是工具**类型**图，不是某条任务
可直接拓扑调度的 action DAG；环说明同类工具可能跨状态、跨实体实例重新进入。
真正的 batch 机会必须在每个决策点结合当前参数、状态和 precondition，实例化
具体 action 后再计算 ready set。强连通分量（SCC）不能一次展开成整圈调用。

## 真实系统和已有研究怎么做

主流接口普遍同时支持同轮多调用与跨轮组合调用；OpenAI、Anthropic 还提供显式
关闭 parallel 的开关，其他系统也可由应用层 validator/executor 强制单调用。

| 证据 | 实际含义 |
|---|---|
| [OpenAI Function Calling](https://developers.openai.com/api/docs/guides/function-calling) | 模型可在一轮返回多个调用；`parallel_tool_calls=false` 可保证零或一个 |
| [Anthropic Parallel Tool Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) | 支持同轮多个调用，并明确建议只 batch 相互独立的调用 |
| [Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling) | 分开支持 independent parallel calling 和 compositional sequential calling |
| [BFCL](https://gorilla.cs.berkeley.edu/leaderboard) | 把 Simple、Parallel、Multiple Parallel 和 stateful multi-turn 分开评测 |

这说明真实服务 Agent 的主流不是“单”或“多”二选一，而是：

- 多步骤、有状态链路继续逐步观察；
- 已知参数的独立读取可以 batch；
- 部署方仍需有关闭或拦截 parallel call 的协议控制。

“让模型输出多个调用”本身已经不是创新点。[LLMCompiler](https://proceedings.mlr.press/v235/kim24y.html)
已经用依赖计划和执行器调度可并行节点；[DTA-Llama](https://arxiv.org/abs/2501.12432)
已经用 DAG 数据训练 7B 模型；[Plan-over-Graph](https://arxiv.org/abs/2502.14563)
训练模型生成可并行任务图；[W&D](https://arxiv.org/abs/2602.07359) 也研究了并行
宽度调度。它们证明并行有价值，同时也意味着我们的贡献必须落在更难的部分：
有状态服务、写副作用、用户确认、动态环、4B 小模型和跨 benchmark 协议切换。

**AReaL 到底在做什么。** [论文](https://arxiv.org/abs/2601.22607)明确主张的
核心是 multi-turn interaction、multi-step tool execution、自演化合成数据、
可执行 verifier、User SFT 和 GRPO，而不是同轮并行。公开论文和
[Tau2 代码说明](https://github.com/areal-project/AReaL/tree/main/examples/tau2)
中没有找到：

- 为什么 Tau2 应同轮多调用的设计论证；
- single 与 multi 的消融；
- multi-call policy compliance 指标；
- 对官方“one tool call at a time”冲突的说明。

因此不能说作者“没发现”，也不能说作者“有意证明 multi 更好”；公开证据只能
支持：他们没有把这个冲突作为论文问题讨论。AReaL 原始 schema 允许
`tool_calls` 列表，公开数据实际包含 multi，但公开材料没有解释 SEA 为什么生成
这些 batch。我们的本地 `strict` 转换器只过滤“自然语言+工具同轮”，仍会保留
“多个工具同轮”；Tau2 runner 与 final-state verifier 也不会拦截或直接惩罚它们。
我们实际审计还发现 source dialog 的整体 `reward=1` 会复制到各 turn，不能把
它当作每个 batch 都安全的标签。

## Tau2：实现能接收，不等于官方想要

本文所说的官方 Tau2 就是本地
[`/mnt/afs/users/fush/projects/ServiceAgent/tau2-bench`](../../../../tau2-bench)
这套仓库。答案很明确：**Tau2 正式模式要的是每个 assistant turn 至多一个
工具调用。**

- [Airline policy](../../../../tau2-bench/data/tau2/domains/airline/policy.md)
  明确写着 `only make one tool call at a time`；
  [Retail policy](../../../../tau2-bench/data/tau2/domains/retail/policy.md) 和
  [Telecom policy](../../../../tau2-bench/data/tau2/domains/telecom/main_policy.md)
  也有同样约束。
- 数据结构和 parser 会保留一个 assistant message 中的全部调用。
- [orchestrator](../../../../tau2-bench/src/tau2/orchestrator/orchestrator.py)
  用 `for` 循环逐个执行，并在全部结束后才把结果返回给模型；不是并发，也没有
  batch 事务和回滚。
- [evaluator](../../../../tau2-bench/src/tau2/evaluator/evaluator.py) 主要检查
  最终 DB、action、communication 等条件，没有直接把 `tool_calls>1` 判零分。

所以这里同时存在两个不同事实：

| 问题 | 答案 |
|---|---|
| Tau2 的接口能否容忍同轮多个调用 | 能 |
| Tau2 的 domain policy 是否允许 | 不允许 |
| runner 是否真正并发 | 否，串行执行 |
| reward 是否可靠发现这项违规 | 否 |
| 我们的 official score 是否应使用 multi | 不应 |

这也是 AReaL 数据看起来“能工作”的原因：**宽松实现让它跑得通，final-state
reward 让它可能得分，但二者都不能把 policy 违规变成官方期望行为。**

我们的 held-out eval 使用了这套仓库的 tasks、runner 和 evaluator，所以任务与
最终状态评分实现是一致的；但 Agent 输出 prompt 明确允许 `one or more tool
calls`，parser/runtime 又没有强制 `len(tool_calls)<=1`，所以**行为协议并不完全
一致**。现有 pass@1 可用于 checkpoint 间同配置比较，却不能证明模型满足官方
single-call policy；修正 prompt 和 runtime 后应重新评测。

Tau2 主线应固定为 `tau2_official_single`。如果研究 parallel Tau2，只能作为
非官方诊断环境单独报告，不能和 official score 混在一起。

## VitaBench：允许安全 batch，但环仍要跨轮走

VitaBench 与 Tau2 不同：

- [Agent prompt](../../../../vitabench/src/vita/prompts/agent_system_prompt.yaml)
  要求依据 Precondition/Postcondition 完成任务，没有规定每轮只能一个，也没有
  要求尽量多调。
- `AssistantMessage.tool_calls` 是列表，并定义了 `MultiToolMessage`；
  [orchestrator](../../../../vitabench/src/vita/orchestrator/orchestrator.py)
  会执行同一消息中的全部调用。
- evaluator 会看到全部调用，但没有“多调用加分”或独立的 parallel 指标。
- 当前 orchestrator 同样是串行 `for` 循环，所以只减少 LLM 往返。

[VitaBench 论文](https://arxiv.org/abs/2509.26490) Appendix C 的真实官方示例
很有代表性。整条轨迹有 17 个 assistant 工具 turn、22 次调用，但只有 2 个
multi-call turn，共 7 个调用：

1. 同时搜索手杖、成人纸尿裤和相关商店；
2. 已知全部坐标后，同时计算四家商店到餐厅的距离。

这 7 个调用全部是独立读取或纯计算。相反，官方示例中的依赖链全部等待了结果：

```text
地址 → 得到坐标 → 搜索附近商户
创建外卖订单 → 得到 order_id → 支付
搜索车次 → 查看详情 → 创建订单 → 得到 order_id → 支付
预约餐厅 → 得到 book_id → 支付
```

Figure 3 的确包含
`addr2coord → get_nearby → store_info → addr2coord` 一类环，但其合理含义是
根据新发现的位置再次转换坐标和探索，不是把整个环放在一个 turn。仓库中也没有
把该图编译为并行计划的 scheduler；Precondition/Postcondition 主要作为工具描述
交给模型理解。

因此对“Vita 是否鼓励同轮多调用”最准确的回答是：

| 判断 | 结论 |
|---|---|
| 运行时支持 | 是 |
| Prompt 禁止 | 否 |
| 官方示例采用 | 是，只用于独立读/计算 |
| 强制或单独奖励 | 否 |
| 鼓励任意 batch | 否 |

Vita 是训练 `safe_batch` 的合适试验场，但公开的 400 个任务没有 gold message
trajectory；Appendix C 只是一个示例。我们仍需自行生成、回放和验证训练标签。

## 我们的轨迹到底说明了什么

先看每个模型同一批 400 条 held-out 轨迹的变化：Airline 20 tasks、Retail 40
tasks、Telecom 40 tasks，每个 task 4 trials，共三个模型 1,200 条。

| 模型 | 至少出现一次同轮 multi | pass@1 |
|---|---:|---:|
| Raw Instruct | 3/400 = 0.75% | 23.50% |
| Multitool SFT | 262/400 = 65.50% | 23.25% |
| RL `2e-6` iter99 | 302/400 = 75.50% | 25.50% |

SFT/RL 强烈学会了“把调用放在一起”，但 SFT 没提高 pass@1；RL 相对 SFT 的
`+2.25pp` 之 95% 置信区间（CI）为 `[-2.84,+7.34]`，尚不能建立 RL>SFT。
在同一 held-out 集按 tool-call turn 统计时，SFT→RL 的 multi-turn 比例从
`18.8%` 增至 `24.1%`，而有参考动作标签的轨迹，其逐轨迹宏平均动作命中率只从
`48.4%` 增至 `49.9%`。这更像 style 被放大，而不是已经学会可靠调度。

但不能把相关性说成因果：

- 纯读取 multi 的 success 为 SFT `60/243=24.7%`、RL `80/267=30.0%`；
  无 multi 分别为 `32/138=23.2%`、`20/98=20.4%`。任务难度和模型选择存在
  混杂，不能据此说纯读取 multi 更好，也不能说更差。
- batch 一旦含写，SFT 只有 `1/19=5.3%` 成功，RL 只有 `2/35=5.7%`。
  它也受任务难度混杂，但结合下面的部分提交案例，已经是强工程风险信号。
- 实际 SFT 的 3,014 个 multi target 中，2,640 个是“暂未发现已知写工具”的
  batch，374 个含写；前者仍需检查依赖、重复和 fanout，不能直接都标成安全。

**真实案例 1：安全读取 batch。**
用户想把一个家具订单的地址从 Suite 640 改到 Suite 641；Agent 从账户中得到
三个订单 ID，需要先找出目标订单。`SFT | retail | task 17 | trial 0`，
simulation `ada30f56-93e2-4d7c-aa4a-cf7a7d003e82`，reward=1、参考动作
6/6、DB match。

```text
已知 order A、B、C
→ 同轮读取三个订单
→ 三个读取均成功
→ 后续向用户确认
→ 单独执行一次地址修改
```

三个 ID 事先已知，读取相互独立。这说明安全 batch 确实存在；在 Vita parallel
模式可以保留，在 Tau2 official 视图则应使用真实结果顺序化。

**真实案例 2：没看读取结果就提前写。**
用户要取消 5 月 22 日 JFK→MCO 的预订并退款，但最初只给了描述而没有真实
reservation ID。`SFT | airline | task 35 | trial 1`，simulation
`85042109-7295-4b5c-92a9-f28bf74c5a0e`，reward=0。

```text
同一 turn:
  get_reservation_details(fake_id)
  cancel_reservation(fake_id)

结果:
  reservation not found
  cancellation failed
```

模型在看到“订单不存在”前已经决定取消。正确的观察屏障（barrier）应在第一次
结果之后。该 turn 是依赖机制的反例，不把整条轨迹的 reward=0 单独归因于它。

**真实案例 3：部分提交后谎报全部成功。**
用户要把 EWR→ORD 预订的乘客改成自己、升级舱位并加 3 件托运行李。
`SFT | airline | task 22 | trial 3`，simulation
`2f8b4239-4869-470c-8f9f-d58da796d960`，reward=0。

```text
同一 turn:
  update passengers → success
  update flights    → error: missing reservation_id

后续:
  Agent 声称 passenger、cabin、baggage 全部完成
```

runner 不回滚，实际 cabin 未修改。这不是格式小问题，而是事务安全和 grounded
closeout 同时失败。

**真实案例 4：纯读取也会退化成 fanout。**
用户先要修改旧预订，再预订 5 月 20–25 日纽约往返西海岸的低价航班。
`SFT | airline | task 24 | trial 0`，simulation
`13d8320a-1d4c-4677-bec1-39f4893dd40a`，reward=0。模型一轮搜索 12 条航线，
整条轨迹共 47 次工具调用、18 次非相邻重复，最后仍然 DB mismatch。读取虽无
副作用，但无界 batch 会增加上下文、重复、截断和错误候选选择。这个案例展示
fanout 的放大机制，不把整条轨迹失败单独归因于该次搜索。

**真实案例 5：我们真正希望保留的单调用多步能力。**
用户要把 pending order 中的白色耳机换成蓝色，中途又改变续航和防水偏好。
`Raw | retail | task 60 | trial 0`，simulation
`687cdc83-d372-41d1-80a1-f7489ec17dfe`，reward=1、参考动作 4/4、DB match。

```text
验证身份 → 读取订单 → 读取商品库存
→ 用户改变偏好 → 更新方案并重新确认
→ 一次原子写入 → 根据工具结果宣布成功
```

关键不是“只用了一个工具”，而是每次只提交一个动作、持续使用 observation，
最终完成了多工具、多步骤事务。

所以对用户问题“multi 是否比 single 更差”的严谨答案是：

| 范围 | 判断 |
|---|---|
| 当前 Tau2 official 协议 | 更差，因为 multi 全部不合规 |
| 当前无约束 AReaL 式 multi 训练信号 | 不能未经分类整体作为正例；调用覆盖增长快于正确规划 |
| 含写、有依赖、重复和 fanout | 风险显著高于逐步调用 |
| 独立、参数已知、有限宽度的纯读取 | 没有证据更差；可能更高效，需受控实验验证 |

## 这个方向怎样才算项目亮点

如果项目名称只是“Qwen3-4B 支持 parallel tool calling”，答辩时很容易被
OpenAI/Claude/Gemini、BFCL、LLMCompiler 和 DTA-Llama 直接覆盖。它可以是工程
Demo，但研究辨识度不高。

更强的定位是：

> **Learning When NOT to Parallelize: protocol- and state-conditioned
> selective width for a 4B stateful service agent**

它的贡献应至少包括：

1. **动态安全标签。** 为每个决策点标注 `ASK / FINAL / SINGLE / SAFE_BATCH /
   MUST_WAIT`，并记录 read/write、资源 key、幂等性、前后条件、用户确认和协议。
2. **模型真的做选择。** 4B 原生提出 ready set；guard 只验证和拦截，不能替模型
   完成全部规划。同时报告 guard 前 proposal 和 guard 后 execution。
3. **有状态安全。** 不只做搜索 API；覆盖共享订单、写副作用、失败恢复、动态用户
   信息和图中环。
4. **跨协议泛化。** 同一模型看到 `tau2_official_single` 时稳定输出宽度 1，
   看到 `vita_safe_batch` 时才在真实机会处扩大宽度。
5. **真实效率。** 实现异步 executor 或明确只报告 saved LLM turns；不能把串行
   `for` 循环包装成 latency speedup。
6. **与强基线比较。** 除 always-single 和 unrestricted-multi 外，还要比较规则
   read-only scheduler 与 LLMCompiler 类调度器。

对于实习项目，这个版本有清晰问题、真实失败案例、训练方法、系统 guard 和可量化
收益，足以形成有辨识度的完整故事；如果只展示 multi-call 格式和调用数量增长，
则不会是亮点。

## 下一步数据、训练与四臂实验

**第一步：不要全删，建立双视图数据。**

| 原始同轮行为 | Tau2 official 视图 | Vita safe-batch 视图 |
|---|---|---|
| 独立、参数已知、只读/纯计算 | 插入真实结果，顺序化 | 保留，初期宽度上限 2 |
| 后一步依赖前一步结果 | 顺序化或重生成 | 顺序化或重生成 |
| 写、支付、取消、确认、共享对象 | 顺序化；不机械拆 JSON | 默认顺序化 |
| 重复、冲突、超大 fanout、错误调用 | 删除或做 preference 负例 | 同左 |

对 2,640 个 pure-read multi target 继续标注：参数是否预先知道、调用是否重复、
是否读相同可变资源、结果是否决定后续选择。374 个含写 target 默认拆分或重生成。
不能仅按 target 删除，因为 `12,258/19,318=63.45%` 的训练行在历史或 target
中暴露过 multi。clean 主数据应先按 source dialog 切分 train/dev/test，再完整
重建违规 turn 及其下游 observation；无法可靠重建的对话段删除或重新 rollout。
只对违规历史 turn 关闭 loss 仍会把它作为上下文影响后续预测，只能用作快速消融，
不能叫 clean 数据。

数据中加入成对 hard negatives：

- 三个独立订单读取 vs `get_order→cancel_order`；
- 两个只读查询 vs read+write；
- 正确 ready set vs 正确调用再加多余调用；
- 有确认的单次写 vs 未确认的批量写；
- 宽度 2 的必要查询 vs 12 路重复 fanout。

4B 先做课程训练：单调用和参数准确 → 安全宽度 2 → 少量宽度 3 → 协议条件切换。
不要追求全局 multi rate；优化的是
`P(batch合法 | 模型选择batch)` 和 `P(模型选择batch | 确有机会)`。

**第二步：运行时做保守 guard。**

v1 只放行“参数已知、非重复、只读/纯计算、无资源冲突”的 batch；所有 write、
general、ask-user 和未知类型都降级为 single。对可 clone 的 simulator 状态，
可用不同调用顺序和失败注入 replay 检查结果是否不变。后续再研究作用于不同
resource key 的可交换写，第一版不要冒险。

guard 的规则不能同时充当评测真值，否则会循环论证。先冻结一个独立的
state-level 安全集，由人工复核参数可得性、前后条件、资源冲突、授权、顺序交换
和失败注入；labeler/guard 开发集与调度测试集按 task、tool-pair、graph motif
隔离。还要先 census Vita 400 个任务中 `ready width≥2` 的真实覆盖率，机会太少
时不应强行追求全局加速。

**第三步：做能回答因果问题的四臂实验。**

| Arm | SFT | RL 目的 |
|---|---|---|
| A `Single` | 全部安全顺序化 | task reward；合规基线 |
| B `Unrestricted` | 保留现有 AReaL 式 multi | task reward；负对照 |
| C `Safe-TaskRL` | 只保留验证过的 read batch | 仅 task reward；隔离数据筛选收益 |
| D `Selective-RL` | 与 C 相同 | 学习宽度；成功且安全后才给效率奖励 |

预注册的主对比是：`B-A` 测无约束 multi 的代价，`C-A` 测安全数据表示，
`D-C` 测选择性宽度奖励的增量。再把同一个 D checkpoint 在推理时强制
`width=1`，用 `D vs D@width1` 隔离“执行 batch 本身”的收益。C/D 必须使用
完全相同的 prompt、guard 和 executor；另加 `C@width1`、规则 read-only
scheduler 和 LLMCompiler 类 scheduler 作为 inference-only 对照。完整矩阵因此
是“四个训练臂 + 若干固定权重的执行对照”，不是把所有差异塞进四个模型。

四臂使用同一 Qwen3-4B 起点、source dialog/task 分布、RL updates、rollout 数、
解码、User、任务和 seed，并报告实际 optimizer tokens。tool-call 和上下文/token
上限一致；**不强行匹配 turn 上限**，因为 single 天然需要更多 turn。正确性主
实验采用 benchmark 标准配置，再加一个 max-turn 足够大、几乎不截断的敏感性
实验。RL 沿用已验证稳定的 `2e-6 + k2 KL` 起点，不再使用“每多叫一个正确工具
名就继续加分”的 field reward。奖励依据 guard **修正前**的 raw proposal 按
门控顺序计算，guard 干预本身也要处罚：

```text
协议/依赖/授权违规 → 硬罚
任务失败           → 效率奖励为 0
任务成功且 batch 合法 → 才给截断后的 saved-turn / critical-path 奖励
```

Tau2 和 Vita 回答不同问题：Tau2 测任务成功、guard 前的协议切换能力和执行
合规；Vita 才测 safe-batch 的正确性、安全性与效率。B 的 Tau2 结果只能标为
protocol-violation diagnostic，不能列作 official score。

Tau2 用当前 100 tasks（Airline 20、Retail 40、Telecom 40）×4 paired trials；
Vita 优先做 100 个 cross-domain×4，预算允许再做 300 个 single-domain×4。
按 task 做 clustered bootstrap，而不是把同任务四次 trial 当四个独立样本；
正式运行前对 `-2.5pp` 非劣界做 power analysis。增加同一任务的 trial 不能替代
独立 task；若功效不足，就增加独立任务、给出有业务依据的更宽 margin，或只把
Tau2 定位为协议合规验证。按 ready width、依赖深度、write、环/SCC 分层报告。

核心指标包括：

- 任务：Tau2 pass 指标、Vita Avg@4、pass@4（4 次中至少一次）、pass^4（4 次
  全部成功）、DB/rubric、action/argument precision；
- 调度：guard 前/后的 batch-valid precision、机会 recall、exact ready set、
  guard intervention rate、依赖违规；
- 安全：policy 违规、unsafe write proposal/execution、partial commit、未授权写；
- 稳定：重复、fanout p50/p95/max、tool error、恢复失败；
- 效率：LLM turns、tokens/cost、tool calls、p50/p95 latency 和 critical path。

效率主分析使用实验前冻结的 oracle-opportunity task/state 集合；全任务按
intention-to-treat 报告，失败按 timeout/最大成本计，并先通过 success 非劣门槛
再比较效率。最干净的主对比是 `D vs D@width1`；“两个 arm 都成功”的
matched-success 子集存在事后选择偏差，只作为辅助诊断。当前 runner 若不改为
异步，Go/No-Go 只使用 LLM turn、token 和成本；校准延迟的 critical-path replay
只能标为 simulated upper bound。真正声称 p50/p95 latency speedup 前必须实现
异步 executor 并实测。

预注册 Go/No-Go 门槛：

- Tau2 official 的 guard 后同轮 `>1` 必须为 0，同时报告 guard 前 proposal
  合规率和 intervention rate；
- D 相对 A 的 task-success 95% CI 下界不能低于 `-2.5pp`；
- guard 后 unsafe batch 必须为 0，并用零事件时约 `3/N` 的 95% 单侧上界报告
  尚未观察到的风险；guard 前 batch-valid precision 至少 99%，且在冻结机会集上
  opportunity recall 默认至少 50%，最终阈值在 Vita opportunity census 后预注册；
- 在 oracle 确认存在宽度≥2机会的固定任务集上，D 相对 `D@width1` 的 LLM turn
  至少降低 15%，且全任务 success 先满足非劣，否则实际价值不足；
- D 相对 C 必须在 success 非劣后达到预注册效率增益，否则选择性宽度奖励没有
  额外贡献；D 不优于规则 scheduler/LLMCompiler 的 Pareto frontier，说明研究
  创新不足；
- 每任务 tool calls 或重复比 A 增加超过 5%、收益在匹配 tool-call 上限后消失，
  判为 reward hacking。

最先应该做的不是继续 RL，而是完成冻结的 state-level 安全集、Vita 并行机会
census 和四臂中的 A/C 小规模 SFT preflight。只有 Tau2 的 guard 前 proposal
达到 100% 协议合规、Vita safe-batch precision 达到 99% 且 recall 过预注册
门槛、任务成功不劣于 single baseline 后，再进入 Selective-RL。
