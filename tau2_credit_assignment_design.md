# Tau2 Turn-Level Credit Assignment 实现设计

本文规定第一版可实现的算法和数据接口，供后续编码使用。当前仅完成设计审阅，
尚未实现状态采集、运行训练或验证效果；不包含消融矩阵、超参数搜索和 OPD。
范围是 `TASK.md` 指定的 AReaL 三域数据、固定 User 和现有同步 GRPO 路径。

## 1. 已核实的输入与方案边界

2026-09-13 只读扫描
[`areal_tasks_slime.jsonl`](../datasets/tau2/rl/areal_tasks_slime.jsonl)，并核对本地
[`EvaluationCriteria`](../tau2-bench/src/tau2/data_model/tasks.py) 的默认值：

| 域 | 任务数 | 运行时 reward_basis | 本版状态信号 |
|---|---:|---|---|
| Airline | 1,148 | DB + COMMUNICATE | 到参考 DB 终态的字段距离 |
| Retail | 563 | DB + COMMUNICATE | 到参考 DB 终态的字段距离 |
| Telecom | 205 | DB + COMMUNICATE | Agent 与 User 两侧 DB 的字段距离 |
| Telecom | 50 | ENV_ASSERTION | 已满足环境断言的比例 |
| Telecom | 16 | ACTION + ENV_ASSERTION | 已满足环境断言的比例；ACTION 仍参与终局判定 |

未显式填写 `reward_basis` 的任务会使用 `[DB, COMMUNICATE]`，不能按空列表解释。
271 条 Telecom 任务都有 `env_assertions`，但只有上表的 66 条将其作为成功条件。
训练信号按实际生效的条件取值，不自动把其余断言加入奖励。

全部 1,982 条任务的 `communicate_info` 都为空，因此删去本版的 COMM progress 和
系数 α。现有 [`CommunicateEvaluator`](../tau2-bench/src/tau2/evaluator/evaluator_communicate.py)
只做忽略大小写、去逗号后的子串匹配，也不能据此声称验证了自然语言表达的正确性。

保留的算法结构是：**官方终局优势 + 状态进展 reward-to-go + 局部协议惩罚**。
这是新增辅助训练目标，不是仅把原有轨迹优势守恒地重新分配。它不直接判断查询、
确认或工具参数是否正确，也不提供反事实因果归因。

`actions` 对 DB 任务用于构造参考终态，不要求 Agent 逐条模仿；若 ACTION 在
`reward_basis` 中，则仍须保留官方动作匹配判定。本版不新增工具名、调用次数、
逐条 golden-action 匹配或 read-only 调用的正奖励。

## 2. 当前代码实际计算什么

[`reward.py`](examples/tau2-bench/rl/reward.py) 的
`evaluate_simulation_with_constructor` 使用任务对应的 DB 构造环境，返回官方语义的
binary reward：正常 `agent_stop` / `user_stop` 时，将 `reward_basis` 中的分量相乘；
`max_steps` 等提前终止返回 0。额外诊断分量不会自动进入该乘积。

[`reward_postprocess.py`](examples/tau2-bench/rl/reward_postprocess.py) 有三条不同路径：

| 路径 | 轨迹标量 | 回合修正 |
|---|---|---|
| 未设 turn-credit 版本的旧 shaping | 官方 reward + 分域加权 partial score，再扣轨迹行为惩罚 | 无 |
| `turn-credit-v1`，selected boundary 训练使用 | 官方 reward + 分域加权 partial score | 归一化之后，在对应 Assistant span 的每个 token 上减去该回合惩罚 |
| `turn-credit-v2` | 官方 binary reward；field score 只作诊断 | 根据精确参考动作与规则错误构造固定预算、零和 modifier，再按 token 数缩放 |

V1 的 partial score 包含工具名、参数字段、最终 DB、环境断言和通信匹配。
默认非 Telecom 权重依次为 `0.25/0.35/0.25/0.10/0.05`，按存在分量重新归一化；
Telecom 权重为 `0.20/0.25/0.20/0.15/0.20`。
默认 partial 系数在 Airline/Retail/Telecom 分别为 `0.25/0.20/0.40`。
这些是已有机制，不是本版继续叠加的奖励。

完整路径为：

```text
同一 prompt 的 K 个 Sample → 各自生成完整交互 → 整组过滤
→ tau2_reward_post_process：在完整 group 上归一化
→ Sample.train_metadata → Ray 分发 / DP 分批
→ turn_aware_grpo_advantage：构造或读取 response-token 优势，再按 CP 切片
→ Megatron policy loss：逐 token 概率比、clipping、loss mask、轨迹内 token 平均
```

`_group_normalize` 使用样本标准差（`correction=1`）和 `epsilon=1e-6`；
无效 sample 不计入统计，完整有效组（K≥2）等价于 `(R - mean) / (std + 1e-6)`。
V1 在后端减去局部惩罚；V2 在 postprocess 内完成优势组合，后端只读取和切片。
纯 GRPO 的 `get_grpo_returns` 则只广播标量。
不能在 DP 分发后用某个 rank 收到的局部样本重新组成 group。

[`filters.py`](examples/tau2-bench/rl/filters.py) 当前默认
`TAU2_REPLACE_ZERO_SIGNAL_GROUPS=1`：V1 同时看 global score 方差和局部惩罚，
V2 看组合后的优势是否有信号。超长或移除的 sample 会导致整组替换；使用
`DomainQuotaDataSource` 时替换来自同域的新 prompt。
[`EXP_QA.md`](output/doc/EXP_QA.md) 中的旧开关和历史公式不替代当前代码。

## 3. 回合、环境状态与 token 的对应关系

一条完整交互保留为一个 `Sample`。同一次 prompt 抽样的 K 条轨迹组成一个 group；
使用 `group_index` 保持抽样边界，并在 reward postprocess 入口核对 task/domain 和 K。
相同 task 在其他批次再次出现，不合并为更大的 group；多个 Assistant 回合也不拆成 K。

第 t 回合只指本次策略实际生成的第 t 条 Assistant 消息，包含纯文本或一批工具调用。
用 `raw_data.tau2_rl_agent` 识别，再通过现有 `assistant_source_indices` 对应到
`simulation.messages` 下标。历史 Assistant、默认首句、User 与 Tool 消息均不计为
可训练回合；原消息下标不能直接当作连续的 t。

状态边界固定如下：

- `S[i,0]`：第一次可训练 Assistant 生成前的环境状态。
- `S[i,t]`（t < T_i）：下一条可训练 Assistant 生成前的状态。
- `S[i,T_i]`：最后一次实际执行的 orchestrator step 完成后的状态。

因此一个回合的状态变化包含其工具执行、后续 User 工具操作以及框架原有的
`sync_tools()` 效果，直到下一次 Agent 决策。Telecom 用户按指示操作设备所产生的
变化可以归入这个交互区间，但这只是时间上的归属。首个 Agent 回合之前的用户操作
已进入 `S[i,0]`，不反向奖励后续 Agent。

多工具消息对应一个 Assistant span，使用该批调用及后续交互的净状态变化；
不在同一个生成消息内伪造逐工具 advantage。失败调用也按真实状态观察，不根据
`ToolMessage.error` 假定数据库一定未变。延迟变化落入实际发生的区间，RTG 再向前累计。

采集应在本 worktree 的新模块 `examples/tau2-bench/rl/progress.py` 中实现一个小型
`Orchestrator` 子类：在 `step()` 即将生成 Assistant 前读取状态评分，调用原有
`step()` 执行交互；结束时追加最后一个评分，并记录新 Assistant 的原消息下标。
只保存每个边界的评分及必要的距离/断言诊断，不保存每回合整库副本。
不额外调用 `sync_tools()`，不改写共享 tau2 源码，也不逐前缀重新运行完整 evaluator。

## 4. 可程序化取得的状态进展

**DB 任务：构造目标后比较字段。** 终局 `reward_info.db_check` 只有整体匹配结果，
没有 `D*` 或逐回合字段奖励；这些需要新增采集逻辑。

对每条任务，以 `envs.make_environment_constructor` 创建独立的内存环境，执行
`initial_state.initialization_data`、`initialization_actions` 和 `message_history`。
随后按 [`EnvironmentEvaluator.calculate_reward`](../tau2-bench/src/tau2/evaluator/evaluator_env.py)
的 gold 分支，依次通过 `make_tool_call(name, requestor, **arguments)` 执行参考 actions，
得到 `D*`。保留其 Agent/User 所属关系和现有同步时机，不能擅自换成同步行为不同的
`get_response`。目标只提供给训练评分器，不进入 Agent/User prompt。

`D` 包含 `env.tools.db` 与存在时的 `env.user_tools.db`，分别置于 Assistant/User
命名空间；通过 `model_dump(mode="json")` 得到值。字段比较规则固定为：

1. 递归展开字典，路径使用 key 元组；列表作为一个完整值，保留顺序和重复元素。
2. 空字典和其他叶值保留为比较单元。以当前库和目标库的路径并集进行比较，
   缺失值与 `null` 区分，因此新增、删除、改坏无关字段也会计入距离。
3. 同路径值精确相等记 0，不等记 1；不自行加入模糊匹配、列表排序或浮点容差。

令这些不匹配单元的数量为 `d(D,D*)`，定义：

$$
\Phi_{\mathrm{DB}}(S)=-d(D(S),D^*).
$$

无需按整库字段数平均：未修改且正确的字段不贡献距离，后续在同 task 内归一化。
这衡量的是序列化字段差异，列表内部部分正确没有部分分，字段数量也不代表业务重要性。
参考 actions 为空时，目标是初始化后的 DB；正确保持不变不会得到额外进展奖励。

目标构造每条 rollout 做一次即可，先不增加持久化缓存。若参考调用报错，不把异常后
残留的库当作可信目标：该 task 的辅助状态项不可用，同组统一置零并记录原因，
保留官方终局项和协议项，不因辅助信号缺失另抽容易任务。官方 evaluator 的行为不改。

**ENV_ASSERTION 任务：直接读取已配置的断言。** 对当前数据中 66 条此类任务，
不再加 DB 距离，调用
`env.run_env_assertion(assertion, raise_assertion_error=False)` 得到各断言是否满足：

$$
\Phi_{\mathrm{ENV}}(S)=\frac{1}{m}\sum_{j=1}^{m}\mathbf{1}[\text{assertion}_j(S)\text{ 成立}].
$$

本地 Telecom 的相关断言检查 MMS、服务状态、移动数据、网速、充值量和欠费状态。
它们返回布尔值，已有容差由断言本身负责。本版接入前需用小型环境样例验证重复检查
不会改变状态。断言返回 False 是有效信号；调用抛异常属于辅助信号不可用，按同组
统一降级处理，不能把两者混成 0 分。

按第 1 节表格选择一个 Φ，不混合未参与成功条件的状态指标。新数据若出现不同的
reward basis 组合，需另行定义信号；当前任务无需为尚不存在的组合增加评分规则。

## 5. Reward-to-go 与优势的精确定义

状态进展及其累计值为：

$$
r^{\mathrm{prog}}_{i,t}=\Phi(S_{i,t})-\Phi(S_{i,t-1}),\qquad
G_{i,t}=\sum_{l=t}^{T_i}r^{\mathrm{prog}}_{i,l}
       =\Phi(S_{i,T_i})-\Phi(S_{i,t-1}).
$$

第一版固定 γ=1。保留正负差值，不只统计“首次成功”，也不把负进展裁为 0。
状态未变的 read-only 回合即时 reward 为 0，未来有净进展时才可能获得 RTG。
提前终止的轨迹使用最后实际状态，不补造剩余交互或成功终态。

定义 `GN` 为样本标准差归一化：有 n≥2 个值且 std>1e-6 时，
`GN(x_i)=(x_i-mean)/(std+1e-6)`；否则全部返回 0。
这延续现有 all-valid GRPO 的均值/std 口径，并显式处理单元素及近零方差。

官方终局优势在同一 group 的 K 条完整、有效轨迹上计算：

$$
A_i^{\mathrm{out}}=GN(\{R_j\}_{j=1}^{K})_i.
$$

保留原设计的按位置比较，但准确称为**相同序号下的相对未来进展**。
令 `I_t={i:T_i≥t}`，只用实际存在的回合计算：

$$
A_{i,t}^{\mathrm{prog}}=GN(\{G_{j,t}\}_{j\in I_t})_i,\qquad
A_{i,t}=A_i^{\mathrm{out}}+\lambda A_{i,t}^{\mathrm{prog}}+\beta f_{i,t}.
$$

不为已经结束的轨迹补 0 参与统计。尾部只剩一条轨迹时，progress advantage 为 0，
仍保留该轨迹的 outcome 和局部协议项。辅助状态不可用的 group，其 progress advantage
整体为 0，不能只排除某条轨迹后用缩小的组计算。

后续实现可用 λ=0.3、β=0.1 作为初始默认值，以便 smoke 有确定配置；这两个数尚无
效果证据，也不保证 outcome 一定压过局部项。两项均不再经过额外 std 缩放。

**需要保留的数值例子和限制：**

| 即时进展 r | RTG G | 能说明的性质 |
|---|---|---|
| `[0,0,1]` | `[1,1,1]` | 有净进展时，信号覆盖前缀，也覆盖其中无用的绕路 |
| `[0,0,0]` | `[0,0,0]` | 最终写入失败且从未产生进展时，不能辨认前面的正确查询/确认 |
| `[1,-1]` | `[0,-1]` | 先改对再破坏，早期正进展不会永久保留 |
| `[1,-1,1]` | `[1,0,1]` | 修复可获正进展；总进展不累加循环次数，但 token 梯度仍可能受循环影响 |

`G>0` 不保证归一化后的 progress advantage 为正，更不保证总 advantage 为正。
相同序号的两条轨迹可能处在不同任务阶段：例如两个成功轨迹的 r 分别为 `[1,0]`
和 `[0,1]`，第 2 回合的 G 为 0 和 1，GN 后约为 `-0.7071/+0.7071`。
它会给较早完成者的后续回合负修正，给较晚完成者正修正；不能解释为同状态下的
动作优劣。因此本估计量只是待验证的训练启发式，不声称保持最优策略不变或解决了
因果 credit assignment。

## 6. 局部协议项与零信号组

`f[i,t]` 每回合最多取一次 -1，否则为 0，只使用当前能够稳定取得的两类证据：

- `raw_data.tau2_tool_parse_error`：现有解析器识别到的协议错误，包括所选协议下
  不允许的文本/工具混排、损坏的 tool-call 块等。
- 结构化调用或现有 attempted-call 记录中的工具名不在 Agent 工具集合内，
  包括误用 User 工具命名空间。

缺少业务参数、参数类型错误、合法工具返回“订单不存在”等情况，目前不能仅凭
通用 `ToolMessage.error` 准确分类，本版不把它们扩展为 format penalty。
与 golden 参数不同、重复调用、达到 `max_steps` 也不进入此局部项；保留为诊断。
合法调用没有正 format reward。协议项仅施加于发出该消息的 Assistant span，
不经过 RTG，避免把解析失败的固定惩罚传播给所有前置回合。

本版固定 `TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0`。组内全对或全错时 outcome advantage
精确为 0；有相对进展差异或协议错误仍可能产生训练信号。若所有可训练 token 的
最终 advantage 都为 0，保留该组，policy-gradient 项为 0，K2 KL loss 仍可能非零。
“全错仍可学习”只在确有辅助信号时成立。

超过 16,384 总训练 token 的轨迹仍按现有逻辑最多额外重采样两次；最终超长、
无可训练 token 或 rollout 基础设施失败的组继续整组替换，记录原因和额外采样次数。
`max_steps` 导致的有效失败轨迹本身不是基础设施错误，不因此删除。
分别统计 binary 零方差组率与最终 advantage 全零组率，不能以 reward 方差代替后者。

## 7. 接入位置与 loss 语义

暂定新 recipe 名 `progress-rtg-v1`，当前代码尚不识别。后续变更集中在以下位置：

| 位置 | 所需实现 |
|---|---|
| 新增 `examples/tau2-bench/rl/progress.py` | DB 目标、字段距离、断言评分、决策边界采集 |
| [`rollout.py`](examples/tau2-bench/rl/rollout.py) | 使用采集子类；每次重试重建采集状态；将评分序列与 Assistant 源下标、span 对齐 |
| [`reward_postprocess.py`](examples/tau2-bench/rl/reward_postprocess.py) | 注册新 recipe；完整 group 内计算 outcome/progress GN，组合协议项并写入 token 向量 |
| [`filters.py`](examples/tau2-bench/rl/filters.py) | 识别新 recipe 的有效样本，保留零信号组和现有整组超长处理 |
| 同步训练入口 | 显式选择新 recipe、reward postprocess、custom advantage 和下述固定配置 |

复用 `qwen3_full` 的 `return_assistant_spans=True`，设
`response_start=len(tokens)-response_length`，则 response span 为
`[token_start-response_start, token_end-response_start)`。
response suffix 中仍穿插 User/Tool 文本，`response_length` 不等于可训练 token 数。
每个可训练 token 只归属一个新生成 Assistant 回合；沿用现有 mask 对结束标记的处理。

在完整 group 的 postprocess 中生成长度恰为 `response_length` 的优势数组：对应
Assistant span 内广播 `A[i,t]`，mask=0 的位置填 0；放入
`sample.train_metadata={"turn_credit_version": "progress-rtg-v1", "token_advantages": ...}`。
普通 metadata 只需记录源下标、response span、Φ/r/G、三个优势分量及信号不可用原因。
`_dump_trajectory` 当前早于 group postprocess，若要保存最终 GN/advantage，需在
postprocess 后写到独立诊断目录，不能假设现有 trajectory dump 已包含这些值。

postprocess 仍返回 `(raw_binary_rewards, normalized_outcome_rewards)`。
扩展现有 `turn_aware_grpo_advantage` 的向量读取分支，使用
`slice_log_prob_with_cp` 与 log-prob 一致地切片，设置 `advantages` 和 `returns`；
读取最终 token 向量后不再加一次 scalar reward 或旧 V1/V2 修正。
无需为此改写 PPO clipping 或新增 value model。

配置要求为：`advantage_estimator=grpo`、`rewards_normalization=True`、
`grpo_std_normalization=True`、`normalize_advantages=False`，保留现有按轨迹平均的
loss reducer，不启用 `calculate_per_token_loss` 或额外 turn-mean reducer。
若第 t 回合有 n_it 个 trainable token、整条轨迹有 N_i 个，则该回合按 n_it/N_i
参与轨迹 loss。这不是回合等权，也不做 V2 的 N_i/n_it 预算缩放；GN 的位置均值为零
不等于整条轨迹或整个 batch 的 token 加权优势和为零。

概率比仍为每个 Agent token 的
`exp(logp_current-logp_old)`，其中 old log-prob 由训练后端在更新前计算。
Tau2 rollout 当前写入的 `rollout_log_probs` 是全零占位，不能开启
`use_rollout_logprobs` 或依赖它的 TIS。保留完整交互上下文、现有生成协议和 tokenization；
完整模板重编码本身不证明与原始采样 token 逐个一致，后续需检查 raw output 经
解析/重渲染后的实际对齐，不能以 mask 检查替代该验证。

沿用 `kl_coef=0`，通过单独 `use_kl_loss` 加 K2、系数 0.01，避免把 KL 重复加进
优势。新方法不新增 judge、value 或额外生成模型；现有 KL reference model 仍保留。
附加开销是一次参考动作执行和每个决策边界的状态检查：DB 路径约 O(T×F)，
F 为序列化 DB 大小；断言路径约 O(T×m)。需要测量 CPU 时间，不声称零开销。

## 8. 后续实现的最小验收

本轮只完成文档、代码核对、输入统计和公式算例检查；以下是后续实现的验收要求，
不是已经完成的训练验证：

1. **状态信号**：三域各取小型真实环境样例验证目标构造；覆盖正确修改、无关字段
   改坏、列表/新增/删除、无状态变化、破坏再修复，以及 Telecom 的 User 工具与同步。
   对正常结束样例核对 DB 距离为零与官方 DB check、最后断言结果与 evaluator 一致。
   若不一致，先解决采集/目标语义，不能通过放宽相等条件掩盖差异。
2. **数值定义**：验证第 5 节 RTG 算例、sample std、不同 T、单轨迹尾部、全对/全错、
   近零方差和辅助信号不可用；验证不同 group/task 不混合。保留阶段错位算例，
   避免测试把该启发式误写成“提前完成者所有回合都得更高分”。
3. **训练张量**：覆盖历史上下文、纯文本、多工具、失败工具与终止回合；检查源消息
   与 span 的唯一对应、mask 外优势为零、CP 切片一致，确认协议惩罚只出现一次。
   用实际 tokenizer 核对生成前缀和 Assistant 输出，不丢弃开头上下文来满足长度上限。
4. **最小训练 smoke**：实现后再运行既有 `test_rollout_logic.py` 及新增必要用例，
   在作业容器内做最小训练 smoke，检查非有限值、有效梯度、K2、零信号组保留行为与
   额外 CPU/采样成本。本轮不提交训练或消融作业。

后续记录官方 binary success、action/DB accuracy、KL、截断、两类零信号组率及
progress 缺失率。新状态信号缺失不应当变成静默丢组或填入伪造正确性标签。
训练 reward 或 smoke 正常不能证明方法优于基线；以后选择配方仍需固定 SFT init、
User、域配额、LR/KL、K、rollout 预算和 held-out 协议，报告 pass@1、pass@4(any)、
pass^4 与同 task/seed 的配对不确定性。本阶段不安排该对照或消融。

实现、日志和输出均留在当前 worktree；后续输出使用
`output/experiments/tau2-credit-assignment/` 下独立运行目录和新的 checkpoint 根目录。
共享数据、模型、历史 checkpoint 与外部 tau2 源码只读；作业路径及日志索引遵循
[`TASK.md`](TASK.md)，只在本分支提交，不合并 main 或推送远程。
