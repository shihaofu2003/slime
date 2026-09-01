# Tau2 单轮批量工具调用问题：证据、案例与整改决策

本文整理 Qwen3-4B-Instruct-2507 在 tau2 上经过 SFT 和 100 步 GRPO RL 后的
结果，重点回答一个问题：同一个 assistant turn 内提交多个工具调用，是否是我们
希望模型学习的能力。报告中的“成功”默认指 tau2 final-state binary reward；
“合规”单独判断，二者不能混为一谈。

**适用范围说明（2026-07-31）：** 本报告的整改决策针对
`tau2_official_single` 协议，不应解读为对所有 benchmark 的 multi-call 全面
禁用。VitaBench、真实系统、安全 batch 与跨 benchmark 方案见
[跨 Benchmark 决策报告](MULTI_TOOL_STRATEGY_REPORT.md)。

## 摘要：结论与决策

官方 tau2 允许一个任务跨多轮调用很多次工具，但要求每个 assistant turn 最多
调用一个工具。我们当前的 SFT 数据、SFT/RL/eval prompt、runner 和 reward
共同形成了相反的学习信号：

```text
AReaL SFT 中存在同轮多调用
→ 我们原样保留，并在 prompt 中明确允许 one or more tool calls
→ SFT 学会并放大该行为
→ RL/eval 继续允许
→ runner 顺序执行所有调用
→ final-state evaluator 不直接处罚这项 policy 违规
```

核心事实如下。

- Raw→SFT→RL `2e-6` 的同轮多调用轨迹比例为
  `3/400=0.75% → 262/400=65.50% → 302/400=75.50%`。
- 描述上 SFT 相对 Raw 没有提高 pass@1；RL `2e-6` 相对 SFT 只提高
  `2.25pp`，95% CI 包含 0，尚不能证明 RL>SFT。
- 现有观察数据不能证明所有纯读取 multi-call 的 final-state reward 更差。
- 只要同轮 batch 中包含写操作，SFT/RL 的成功率分别只有
  `1/19=5.3%` 和 `2/35=5.7%`；这有任务难度混杂，但已经是强工程风险信号。
- 对官方协议而言，所有同轮多调用都不合规；对当前 4B agent 的稳定性而言，
  含写、前后依赖、重复和大规模 fanout 的 batch 风险最高。

因此对 Tau2 official 主线采用以下默认决策：

| 数据或行为 | 处理 |
|---|---|
| 每轮零或一个调用 | 保留 |
| 相互独立、参数已知的纯读取 batch | 用真实工具结果顺序化为多个 turn |
| batch 中包含写操作 | 删除或重新生成，不机械拆分 |
| 读后写、同实体连续写、调用错误 | 删除或重新生成 |
| 重复调用或大规模 fanout | 删除或重新规划 |
| RL task specification | 不删任务；约束 rollout、parser 和 reward |

本报告在 Tau2 范围内的目标不是“整条轨迹只能调用一次工具”，而是“每轮最多
一个调用、能够跨轮完成复杂多步任务的 agent”。跨 benchmark 的统一模型仍可在
明确的 Vita safe-batch 协议下学习独立读取 batch。

## 我们做了什么

| 阶段 | 已完成工作 | 主要产物 |
|---|---|---|
| Raw | 评测原始 Qwen3-4B-Instruct-2507 | 与 SFT/RL 相同 User、task、seed、temperature、top-p；但 Agent `max_tokens=1200`，SFT/RL 为 8192 |
| SFT | 使用 AReaL tau2 数据训练 Agent；实际数据为 strict、无 thinking、最大 8192 token | `iter_0002413_hf`，实际训练文件 19,318 行 |
| RL | 以 SFT 为初始化，在 airline 上进行 100 步 GRPO；完成 `2e-6/3e-6/5e-6` 稳定性扫描 | 三个 iter99 checkpoint，`2e-6` 为最佳稳定 arm |
| Held-out eval | 使用官方 tau2 task/evaluator 的本地评测，100 tasks×4 trials，seed 300，Agent temperature 0.6 | Raw/SFT/RL 可比的 400 条轨迹 |
| 轨迹分析 | 对相同 task/trial 的 Raw、SFT、RL `2e-6` 共 1,200 条轨迹做确定性特征和校准 judge 分析 | `features.jsonl`、配对变化、失败模式和代表案例 |

因此 SFT↔RL 是相同评测配置，Raw→SFT 则是近似对照，不是只替换 checkpoint 的
严格 ablation；Raw 与后两者的差异不能全部归因于 SFT。

RL 稳定性修复本身是有效的：`k2` KL、较小学习率、field reward、失败处罚和
动态 group replacement 使三个 arm 都完成 100 步且没有 NaN。问题是“训练能够
稳定运行”不等于“模型能力已经显著提高”。`5e-6` 仍出现明显行为退化，
`2e-6` 虽最稳定，但 held-out 结果没有建立 RL>SFT。

需要特别澄清：AReaL 的 RL split 主要是 task specification，不是带固定答案的
RL 轨迹。因此 RL 阶段不能靠“筛 RL 数据中的 multi-call answer”解决问题；
multi-call 行为来自 SFT 初始化、rollout prompt、parser/runner 和 reward。

## 结果怎么样

相同 User、tasks、seed、temperature 和 top-p，100 tasks×4 trials 下：

| Agent | pass@1/pass^1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| Raw Instruct | 23.50% | 45.00% | 6.00% |
| Multitool SFT | 23.25% | 46.00% | 8.00% |
| RL `2e-6` iter99 | 25.50% | 49.00% | 7.00% |
| RL `3e-6` iter99 | 23.50% | 47.00% | 5.00% |
| RL `5e-6` iter99 | 18.50% | 39.00% | 3.00% |

这些结果说明：

- 描述上 SFT−Raw 的 pass@1 为 `-0.25pp`、pass^4 为 `+2pp`；由于 Raw
  的 `max_tokens` 不同，不能把差异全部归因于 SFT。
- RL `2e-6` 相对 SFT 的 pass@1 为 `+2.25pp`，95% CI
  `[-2.84,+7.34]`；pass@4(any) 为 `+3.00pp`，95% CI
  `[-8.09,+14.09]`；两个区间都包含 0。
- RL `2e-6` 的 pass^4 从 SFT 的 8% 降到 7%，没有提高四次都成功的稳定性。
- 三个 RL arm 虽都无 NaN 地跑完，最终指标却随学习率增大而单调变差；
  `5e-6` 的 pass@1 比 SFT 低 `4.75pp`，说明数值稳定不等于行为稳定。
- 需要 Agent 写 DB 的成功轨迹数为 `42→37→37`；无需 Agent 写 DB 的成功数
  为 `52→56→65`。RL 的收益集中在不需要核心事务写入的任务。
- “失败但已经调用过预期写工具名”的轨迹为 `117→120→138`。模型越来越会
  覆盖看似正确的工具名，但没有同步提高参数、流程和最终 DB 状态的正确性。

同轮多调用和 final-state reward 的关系不能用未配对均值直接解释：

| 轨迹类型 | SFT success | RL `2e-6` success |
|---|---:|---:|
| 没有 Agent 同轮多调用 | 32/138 = 23.2% | 20/98 = 20.4% |
| 只有纯读取 multi，没有 batched write | 60/243 = 24.7% | 80/267 = 30.0% |
| 至少一个 multi turn 含写工具 | 1/19 = 5.3% | 2/35 = 5.7% |

在 model×task 固定效应的观察性分析中，即只在同一模型阶段、同一 task 的四次
trial 内比较，并按 task 聚类 bootstrap，any-multi 与重复调用 `+16.80pp`
（95% CI `[+4.44,+29.17]`）、至少一个 Agent 工具错误 `+24.80pp`
（`[+10.66,+38.58]`）、namespace confusion `+26.00pp`
（`[+14.40,+37.61]`）相关。success 为 `+8.00pp`
（`[-0.82,+16.47]`）、DB match 为 `+4.40pp`（`[-4.41,+13.27]`）、
exact action match 为 `+6.80pp`（`[-2.60,+15.73]`），区间都跨 0。
该分析仍是生成后行为关联，不是随机因果实验，也没有单独识别纯读取 multi 的
效果。

## 问题如何进入流水线

先统一术语：

```text
官方允许的跨轮多步调用：
call A → result A → 根据 A 决定 call B → result B

本文讨论的同轮批量调用：
[call A, call B] → runner 再依次执行 A、B
```

三个官方 domain policy 都明确要求一次最多一个工具调用：
[Airline policy](../../../../tau2-bench/data/tau2/domains/airline/policy.md)、
[Retail policy](../../../../tau2-bench/data/tau2/domains/retail/policy.md)、
[Telecom policy](../../../../tau2-bench/data/tau2/domains/telecom/main_policy.md)。
但实现存在三个不同层面的事实。

| 层面 | 当前行为 |
|---|---|
| Policy | 同轮多调用被禁止 |
| Runner | 接受一个 tool-call list，并用普通 `for` 循环顺序执行全部调用 |
| Evaluator | 主要检查 DB、ENV、ACTION、NL、COMMUNICATE，没有独立的同轮多调用处罚 |

AReaL 并不是完全没有看到官方规则。发布数据只有三个 system-prompt 版本：

- Airline 12,842 行和 Telecom 9,294 行都明确写了每轮只允许一个工具调用，
  但仍分别有 2,217 和 441 个 multi-call target；
- 这 2,658 个“明示单调用却输出 multi”的 target，占全部 4,147 个 multi
  target 的 64.09%；
- Retail 11,395 行的 source prompt 未包含 tau2 初始 release
  `37199f3` 已存在、当前仍保留的“at most one tool call”规则，并产生
  1,489 个 multi target；
- 三个 source prompt 中没有任何一个明确允许 multi，但 4,147 个 multi target
  中仍有 3,760 个带 `metadata.reward=1`。

例如 [AReaL source row 19](../../../../datasets/AReaL-tau2-data/tau2_sft_train.jsonl)
的 Airline prompt 两次强调 `exactly ONE`，target 却同轮搜索
`ORD→MIA` 和 `MDW→MIA`，且 `correct=1, reward=1`。这直接说明约束没有进入
数据验收闭环。

因此不能把这些数据解释成一个经过验证的“批量调用设计”。论文强调的是
multi-turn、multi-step interaction，不等于同轮 multi-call；论文附录中鼓励
multiple function calls 的段落服务于任务合成/评审等 meta-agent，也没有给出
目标客服 agent 使用 batch 的收益消融。当前开源 RL 示例实例化 tau2 fork 的
`LLMAgent` 并读取 `env.get_policy()`；该 fork 的三个 domain policy 也都明确
要求单调用。这说明当前 RL 示例会加载这项约束，但不能反推 SFT 合成当时使用了
完全相同的 prompt 或 policy snapshot。

论文附录的 `Generated Trajectory Example` 确实写出了同一 assistant 段落调用
`get_user_details` 和 `get_reservation_details`。它能证明作者发布的目标轨迹中
存在 batch，不能证明作者做过“batch 优于单调用”的实验或设计论证。

最符合现有证据的解释是：生成系统和通用 `tool_calls` 列表能够产出 batch，
parser/runner 接受并顺序执行它们，轨迹验收又主要看最终状态，因而没有拦住这些
违规示范；Retail 还存在额外的 prompt 放宽。是否出于吞吐考虑，论文没有说明，
不能替作者推断。

我们的实现进一步放大了这个不一致：

- [SFT 转换 prompt](../../../examples/tau2-bench/sft/prepare_sft_data.py)
  明确写着 `one or more tool calls`。
- [RL Agent prompt](../../../examples/tau2-bench/rl/agent.py)和
  [eval Agent prompt](../../../examples/tau2-bench/eval/official/sglang_agent.py)
  也明确允许多个 `<tool_call>` block。
- `strict` 只去除了“同轮自然语言+工具调用”，没有去除同轮多个工具调用。
- [官方 runner](../../../../tau2-bench/src/tau2/orchestrator/orchestrator.py)
  会顺序执行全部调用；它不是实际并行执行。
- [官方 evaluator](../../../../tau2-bench/src/tau2/evaluator/evaluator.py)
  没有把该 policy 违规纳入 reward。

于是模型看到的是互相矛盾的指令：domain policy 要求一个，输出格式却允许一个
或多个。实际训练集有 `10,723/19,318=55.51%` 行同时包含我们加入的
`one or more` 和 source policy 的 `only/exactly one`：Airline 8,783 行、
Telecom 1,940 行。Retail 的 8,595 行则因 source prompt 遗漏官方单调用句，只
呈现允许 multiple 的一侧。SFT 正向监督 multi-call，RL 又没有足够的负反馈将
它推回单调用。

## 问题有多大

不同数字的统计单位不同，不能混用。

| 数据层 | 统计单位 | 同轮多调用 |
|---|---|---:|
| AReaL 原始 SFT | target turn | 4,147/33,531 = 12.37% |
| AReaL 原始工具调用 target | tool-call target | 4,147/12,497 = 33.18% |
| 我们实际 SFT | training row 的 target | 3,014/19,318 = 15.60% |
| 我们实际工具调用 target | tool-call training row | 3,014/8,384 = 35.95% |
| Held-out Raw | 至少出现一次 multi-call 的 trajectory | 3/400 = 0.75% |
| Held-out SFT | 同上 | 262/400 = 65.50% |
| Held-out RL `2e-6` | 同上 | 302/400 = 75.50% |

AReaL 原始 SFT 的 domain 分布：

| Domain | multi target/全部 target | multi/工具调用 target |
|---|---:|---:|
| Airline | 2,217/12,842 = 17.26% | 44.38% |
| Retail | 1,489/11,395 = 13.07% | 28.88% |
| Telecom | 441/9,294 = 4.74% | 18.80% |

我们实际训练集的 3,014 个 multi target 可进一步分为：

| 类型 | 数量 | 占 multi target |
|---|---:|---:|
| 纯读取 batch | 2,640 | 87.59% |
| 一个写调用与其他调用同轮 | 9 | 0.30% |
| 同轮至少两个写调用 | 365 | 12.11% |
| 所有含写 batch | 374 | 12.41% |

374 个含写 multi target 中，372 个 metadata 仍为 `reward=1`。这不能证明它们
安全，因为 source `correct/reward` 是整个 source dialog 级标签，在同一对话
的所有 turn 中保持不变，也没有检查单调用 policy。

只过滤 3,014 个 target 仍然不够：

- `12,258/19,318=63.45%` 的训练行在历史或 target 中至少出现过一次
  multi-call。
- 当前多轮 SFT loss 会监督所有 assistant turns，而不只监督最后一个 target。
- 全部被监督的工具调用 assistant occurrence 中，
  `16,722/(16,722+31,520)=34.66%` 是 multi-call。

若直接删除任何历史或 target 含 multi-call 的整行，只剩 7,060 行，数据损失
过大。因此最终方案必须顺序化完整对话，或者对历史违规 turn 关闭 loss，而不是
只做 target-level 过滤。还要注意，19,318 行来自 2,496 个 source dialog；
每一行是不同 target turn 的历史前缀快照，不是 19,318 条独立完整对话。因此
63.45% 表示训练行的暴露率，也受相同历史在多个前缀中重复出现影响。按 dialog
统计，`1,883/2,496` 个保留 source dialog 至少出现过一次 multi。

## 为什么这个问题严重

**第一，后续调用没有利用前一调用的结果。** 模型在看到 `result A` 之前已经
决定 `call B`。只要 B 依赖 A 的实体、状态或错误信息，动作链就不再可靠。

**第二，写操作不是事务。** runner 顺序执行 batch，不自动回滚。A 成功、B
失败时会留下 partial commit；模型又容易把整组调用误判为全部成功。

**第三，确认和权限边界被模糊。** 多个写调用可能对应不同金额、实体或副作用。
把它们合并在一个 assistant turn 中，使“用户确认了什么”更难审计。

**第四，multi-call 容易退化成 fanout。** 当前 multi 轨迹平均约 9.3–9.5
次 Agent 工具调用，没有 multi 的轨迹约 3.1–3.2 次。在 model×task 固定效应
分析中，any-multi 与工具调用数约 `+6.67`（95% CI `[+5.41,+8.00]`）相关，
而消息数的 CI 跨 0。当前没有直接统计端到端时延或成本，且 runner 是顺序执行，
因此不能声称 batch 更高效。

**第五，轨迹级 reward 的 credit assignment 更差。** 一个正确调用和多个
无效调用共享最终 reward。当前 field reward 对 tool name 的平均 credit
明显高于精确 DB success，模型更容易学到“多覆盖工具名”，而不是“选对工具、
参数和时机”。

**第六，reward=1 不再等价于好轨迹。** evaluator 主要看最终状态，因此行为
违规、无关读取、失败写入甚至错误声明都可能与 reward-positive 同时出现。

**第七，对 4B 模型和训练稳定性不友好。** 大 batch 会产生更多 JSON、更多工具
返回和更长上下文，放大遗漏参数、重复、截断和显存风险。已有 OOM 根因就是单条
过长轨迹形成超大 microbatch；无约束 fanout 会重新增加这一风险。

## 真实轨迹

以下案例均来自 held-out `results.json`，不是构造示例。完整轨迹可从
[Raw Airline results](../../../../tau2-bench/data/simulations/tau2_official_Qwen3-4B-Instruct-2507_user_stop_parser_0720_053557_airline_test_4trials/results.json)、
[Raw Retail results](../../../../tau2-bench/data/simulations/tau2_official_Qwen3-4B-Instruct-2507_user_stop_parser_0720_053557_retail_test_4trials/results.json)、
[SFT Airline results](../../../../tau2-bench/data/simulations/tau2_official_Qwen3-4B-tau2-agent-sft-multitool-iter0002413_user_stop_parser_0720_130353_airline_test_4trials/results.json)
和
[SFT Retail results](../../../../tau2-bench/data/simulations/tau2_official_Qwen3-4B-tau2-agent-sft-multitool-iter0002413_user_stop_parser_0720_130353_retail_test_4trials/results.json)
按 task、trial、simulation ID 定位。

**案例 A：正确的单调用多步事务。** `Raw | retail | task 60 | trial 0`，
simulation `687cdc83-d372-41d1-80a1-f7489ec17dfe`，reward=1、DB match、
golden action 4/4。用户要把 pending order 中的白色耳机换成蓝色耳机：

```text
find_user_id_by_name_zip
→ result：身份验证成功
get_order_details(#W5061109)
→ result：订单为 pending，原耳机 $256.67
get_product_details(9924732112)
→ result：得到各个蓝色 variant 的真实库存和价格
向用户展示候选
→ 用户把偏好改为“8 小时、不要防水”
重新展示准确 item、价格 $242.92、退款 $13.75
→ 用户再次明确确认
modify_pending_order_items(...6077640618...)
→ result：订单中的 item 已真实更新
基于工具返回告知成功
```

模型每一步都使用了上一步结果；用户中途改变偏好后，它没有沿用旧确认，而是更新
方案并重新确认，最后只提交一次写操作。这是我们希望 SFT/RL 学习的基本模式：
身份验证→读取→判断→说明→最新确认→原子写入→基于结果收尾。

**案例 B：纯读取 batch 可以成功，但不必原样保留。**
`SFT | retail | task 17 | trial 0`，
simulation `ada30f56-93e2-4d7c-aa4a-cf7a7d003e82`，reward=1、
golden action 6/6、DB match：

```text
turn 12:
  get_order_details(order A)
  get_order_details(order B)
  get_order_details(order C)

results:
  三个读取均成功

later:
  用户确认
  modify_pending_order_address(...)  # 单次写入
```

三个订单 ID 在提交前已经知道，读取之间没有状态依赖。这个案例说明，不能声称
“所有 multi-call 都降低 final-state reward”。但它可以无损顺序化：

```text
get order A → result A → get order B → result B → get order C → result C
```

**案例 C：读后写被提前提交。**
`SFT | airline | task 35 | trial 1`，
simulation `85042109-7295-4b5c-92a9-f28bf74c5a0e`，reward=0：

```text
同一 turn:
  get_reservation_details(fake_id)
  cancel_reservation(fake_id)

result 1: reservation not found
result 2: cancellation failed
```

取消动作在模型看到“reservation 不存在”之前已经决定。单调用流程会在第一步
停止，避免无效写入。整条轨迹后续还有字段和支付不匹配，因此这里只把该 turn
作为“依赖未解析就提前写入”的机制证据，不把最终 reward=0 单独归因于它。

**案例 D：部分写入后仍宣称全部成功。**
`SFT | airline | task 22 | trial 3`，
simulation `2f8b4239-4869-470c-8f9f-d58da796d960`，reward=0：

```text
同一 turn:
  update_reservation_passengers(...)  → success
  update_reservation_flights(...)     → error: missing reservation_id

later:
  baggage update → success
  Agent 声称 passenger、cabin、baggage 全部更新成功
```

实际 cabin 没有修改。这是典型的 partial commit + false success。

**案例 E：同一实体的第二个写操作被第一个写操作破坏。**
`SFT | retail | task 5 | trial 1`，
simulation `ce9bac2f-8306-4343-980b-c5d1ba9fdd73`，reward=0：

```text
同一 turn:
  exchange item A → success，订单状态变为 exchange requested
  exchange item B → error: Non-delivered order cannot be exchanged
```

第二个动作依赖第一个动作之后的订单状态。这两个 item 应合并到一次原子
exchange，或者执行第一步后重新读取状态，不能同轮提交两个写调用。

**案例 F：纯读取也可能退化为暴力 fanout。**
`SFT | airline | task 24 | trial 0`，
simulation `13d8320a-1d4c-4677-bec1-39f4893dd40a`，reward=0：

- turn 14 一次搜索 12 条航线；
- 后续多轮继续一次搜索 6 条；
- 总计 47 次工具调用、82 条消息；
- 18 次非相邻重复；
- 第一次 booking 因支付金额 `162 != 163` 失败；
- 第二次 booking 虽然成功写入，但行程和支付方式都不是 reference 要求的方案，
  最终 DB mismatch、reward=0。

问题不是“读取有副作用”，而是 batch 鼓励模型先穷举、后判断，导致上下文膨胀、
重复，并且没有把候选信息收敛成正确方案。这个案例不能证明 fan-out 是 reward=0
的唯一原因，但清楚展示了它如何放大搜索和决策负担。

**案例 G：reward-positive 仍可能是坏轨迹。**
`SFT | airline | task 6 | trial 0`，
simulation `b6071750-81a7-40db-9543-3dec11deac44`，官方 reward=1、
DB match，但轨迹同时包含四路 reservation batch、五次写调用、一次失败的新
预订和重复 update。模型一度错误声称保险已添加，之后才承认 DB 中仍为
`insurance=no`。最终任务的 DB/communication 条件通过，并不能把中间行为
变成合规、精确或值得学习的示范。

## 当前判断

不能用一个笼统的“更好/更差”覆盖三个不同目标：

| 判断维度 | 结论 |
|---|---|
| 官方 tau2 policy | 同轮 multi-call 明确更差：全部违规 |
| 当前 final-state binary reward | 尚未证明所有纯读取 multi-call 更差 |
| 事务安全与 4B 训练可靠性 | 含写、前后依赖、重复和 fanout 明显风险更高 |

AReaL 的论文主结果来自 Qwen3-30B-A3B 和 235B-A22B，而不是 4B；附录还显示，
较小的 30B 模型做三域混合 SFT 已比单域训练明显下降。这意味着“同一份合成数据
在大模型上有效”不能直接外推到我们的 4B。值得借鉴的是 User SFT、大 batch
GRPO、dynamic sampling、可执行 verifier 和单域训练；不应无条件继承的是没有
通过 policy 门禁的 multi-call 示范。

纯读取 multi 的未配对成功率较高，不能解释成因果收益：模型更可能在特定任务上
选择 multi，AReaL 的 airline/retail 对话又几乎都含 multi，没有干净对照。
在同一模型、同一 task 的四次 trial 内比较时，success 的 95% CI 仍跨 0。

反过来，也不需要等到“所有 multi 都显著降低 pass@1”才整改。官方 policy 已经
给出行为约束，runner 的顺序执行和真实 partial-commit 案例又给出了明确失败
机制。对 Tau2 official 目标，单调用是更可控、可审计且更适合 credit
assignment 的协议；独立读取的覆盖能力在 Tau2 数据中应通过顺序化保留，也可在
隔离的 Vita safe-batch 数据视图中作为正例。

本报告不使用“行为有效成功率下降”作为 multi-call 有害的独立证据，因为该指标
定义中已经把 policy violation 作为 critical error；用它再次证明违规有害会形成
循环论证。

## 后续方案与验收

**P0：先统一协议。**

- 将 SFT、RL、eval prompt 中的 `one or more tool calls` 改为
  `at most one tool call per assistant turn`，并明确每轮只能二选一：
  plain-text response，或 exactly one tool call。
- parser/runtime 检测 `len(tool_calls)>1`，不得静默只执行第一个调用；静默截断
  会造成模型输出、环境状态和 on-policy token 不一致。
- eval 单独报告 single-call compliance，并把违规与 final-state reward 分开。

**P1：建立两个 clean-SFT 版本。**

快速对照版用于尽快验证方向：

- 删除 3,014 个 multi target，保留约 16,304 行；
- 对历史中的 multi assistant turn 设置 `step_loss_mask=0`，避免它继续进入
  SFT loss；当前
  [Qwen3 loss mask](../../../slime/utils/mask_utils.py)
  已支持 message-level `step_loss_mask`；
- 修正 system prompt；
- 该版本只清除直接监督信号，历史上下文仍可能出现 multi，因此是实验基线，
  不是最终数据。

正式版用于后续训练：

- 按 `source_dialog_id + turn_index` 重建完整对话；source 中没有可普遍依赖的
  稳定 `tool_call_id`，需依据同轮调用列表与随后 tool result 的存储顺序和工具名
  配对，并用数量、名称、顺序和可回放结果做门禁；
- 对 2,640 个不含已知写工具的 multi target 继续做依赖和重复审计；仅将确认
  相互独立的调用按真实 tool result 顺序化，其余删除或重新生成；
- 374 个含写 batch，以及读后写、同实体多写、调用错误样本删除或重新生成；
- 重复和大规模 fanout 轨迹删除或重新规划；
- 完成后重新做 8192-token 长度过滤和 domain/工具覆盖统计。

**P2：约束 RL rollout，而不是筛 RL task。**

- multi-call 违规应得到明确 behavior penalty 或终止信号，使 GRPO 能学习负例；
- prompt/runtime 修正后再对残留违规 sample 做 mask/filter；
- 保持 `list[list[Sample]]` group 结构，优先标记 sample 而不是直接破坏 group；
- 当前 RL 轨迹中 75.5% 含 multi，不能在旧 policy 下直接删除大多数样本，否则
  有效 batch、组内方差和数据分布都会失真；
- 含写任务增加 action/argument/DB 的强 credit，不再以“多调用正确工具名”代替
  “执行了正确事务”。

**P3：做能够回答因果问题的 A/B。**

至少比较：

| Arm | 数据与协议 |
|---|---|
| Current | 当前 multitool SFT checkpoint |
| Clean-filter | 修正 prompt；multi target 删除；历史 multi loss-mask |
| Clean-sequential | 独立读取顺序化；高风险 batch 删除或重生成 |

三组使用相同 base model、训练 token/step budget、User、100 tasks×4 trials、
seed 和 decoding。验收同时报告：

- pass@1、pass@4(any)、pass^4 及 paired 95% CI；
- single-call compliance；
- exact action match、DB match 和含写任务 success；
- Agent 工具错误、参数错误、重复和 fanout；
- 平均工具调用数、消息数、生成 token 和截断率；
- Raw→SFT 之后，再决定是否进入新的 RL；不要用 RL 掩盖 SFT 数据协议问题。

数据门禁应满足：

- 被监督的 assistant turn 中 `tool_calls>1` 为 0；
- text+tool 同轮为 0；
- prompt 中不再出现允许多个调用的语句；
- 所有顺序化样本的 call/result 配对、数量、工具名、顺序、参数和结果可回放；
- 含写示范具备必要前置读取、最新用户确认、一次写入和基于真实结果的收尾；
- 预检覆盖 parser、loss mask、reward 和 group filter，不只检查 JSON 可解析。

完成 clean SFT A/B 后，才有证据回答“单调用是否提高最终能力”。如果 clean SFT
提高合规和事务正确性但 pass@1 持平，它仍是更好的 RL 初始化；如果 pass@1
明显下降，则应检查顺序化质量、训练 token budget 和调用轮数，而不是直接恢复
含写 batch。

**数据来源与统计边界。**

主要证据：

- [本实验 README](README.md)：1,200 条配对轨迹的总体模式。
- [确定性特征](features.jsonl)：reward、调用数、multi、multi 中含写、错误、
  重复、DB/action match 等。
- [配对变化](paired_transitions.csv)与
  [pattern delta](pattern_deltas.csv)：同 task/trial 的阶段变化和 CI。
- [代表案例](representative_cases.md)：Raw→SFT→RL 的成功、失败、改善和退化。
- [AReaL source SFT](../../../../datasets/AReaL-tau2-data/tau2_sft_train.jsonl)。
- [实际 SFT 文件](../../datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking_max8192.jsonl)
  与[长度统计](../../datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking_max8192.stats.json)。
- [最终 held-out eval](../tau2-rl-stability-k2-fieldreward-final-eval/README.md)。
- [RL 稳定性扫描](../tau2-rl-stability-k2-fieldreward-lr-sweep/README.md)。
- [AReaL-SEA 论文](https://arxiv.org/pdf/2601.22607)、
  [AReaL tau2 示例](https://github.com/areal-project/AReaL/tree/main/examples/tau2)。

统计边界：

- `has_multicall` 是模型生成后的行为，不是随机处理；未配对均值不能解释因果。
- model×task 固定效应由 77 个在四次 trial 内同时出现 single/multi 的单元识别：
  SFT 42 个、RL 35 个，对应 51 个唯一 domain×task，其中 26 个在两个模型阶段
  都有组内变异。95% CI 按 100 个唯一 domain×task 聚类 bootstrap 20,000 次，
  seed 为 20260731；样本量仍有限。
- “multi turn 含写”集中于更难的 state-changing task；约 5% 的成功率是强风险
  信号，不是最终因果估计。
- AReaL metadata 的 `correct/reward` 是 dialog-level 标签，不是当前 target turn
  的独立质量标签。
- Gemini judge 只用于需要语义理解的模式；工具数量、schema、namespace、
  reward 和 DB/action 等事实以确定性脚本为准。
- 本地评测使用官方 tau2 task/evaluator，但 Agent prompt 被我们修改过，故不能
  将当前结果描述为完全遵循官方 single-call protocol 的评测。
