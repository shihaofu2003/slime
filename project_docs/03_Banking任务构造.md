# ServiceAgent 第三项：Banking 任务构造

本文对应简历第三点，说明 Banking 训练任务从哪里来、如何构造和验证、过程中为什么调整，以及模型试验和后续训练分别得到什么结果。核对日期为 2026-09-22，依据两个实现仓库的代码及已有实验产物。当前项目称 tau3-benchmark，历史目录仍使用 `tau2-bench`，Banking 的运行时领域名为 `banking_knowledge`，下文沿用这些可定位的名称。

这项工作解决 Banking 缺少可用训练示范的问题。先将长程业务拆成检索、判断、工具操作和用户协作等能力，再按业务政策、初始状态和动作依赖构造不同复杂度的新场景。程序确定业务事实、实体、参考动作和判分条件，Qwen3.8 改写用户开场；任务验证后，再用模型自主交互生成示范。

最终构造集覆盖 **15 类业务、647 个不同场景**，按训练/开发/挑战划分为 **542/75/30 个任务**。真实环境参考回放 647/647 通过；训练集采集到 1,914 条 Qwen3.8 成功轨迹。为保留长轨迹中的业务执行和最终回复，先删除冗余步骤，经语义复核和环境回放后再导出，最终得到 **1,651 条完整轨迹、5,672 行 SFT 样本**，覆盖 530 个训练任务。

用这批数据对原始 Qwen3-4B-Instruct-2507 做 Banking SFT 后，在官方 97 题 × 4 trials、同一评测 seed 下，**pass@1 从 2.58% 升至 4.38%，pass@4(any) 从 4.12% 升至 9.28%，pass^4 保持 1.03%**。这些是单 seed 点估计；第八节给出协议和对照，第九节提供简历表述。Qwen3.8 的合成任务验证结果另列于第七节。

## 1. 为什么需要单独构造 Banking 任务

[第二点][doc02]处理的是已有 AReaL Airline、Retail、Telecom 数据，Banking 没有对应的上游 SFT 数据。直接用官方复杂任务采样，又很难获得足够的成功示范，也容易让 GRPO 的同组轨迹全部失败、缺少组内差异。

对官方运行时任务文件的盘点如下。本次重新统计了任务和文档文件，文档数、参考动作数与历史盘点一致。[任务盘点与分析][inventory]

| 维度 | 官方 Banking 任务的实际结构 |
|---|---|
| 任务与知识库 | 97 个任务，698 篇文档；任务合计引用 218 篇不同文档 |
| 单任务所需文档 | 平均 9.89 篇，中位数 10 篇，最多 30 篇 |
| 单任务参考动作 | 平均 9.56 个，中位数 8 个，最长 33 个 |
| 身份验证与工具发现 | 81 个任务涉及身份验证，69 个涉及动态 Agent 工具 |
| User 协作 | 21 个任务要求向 User 提供动态工具，其中 20 个要求 User 执行 |
| 判分方式 | 主要看动作和最终 DB 状态，原始任务的 `communicate_info` 均为空 |

因此，一条 Banking 任务往往同时要求“找政策—读事实—确认实体—判断资格—验证身份—发现工具—协同操作—核对状态”。遗漏其中一个必要操作，整条轨迹仍可能得到 0 分。

历史诊断中，Qwen3-4B 两次完整评测分别成功 13/388、14/388，Qwen3.5-4B non-thinking 为 15/388；参考动作超过 16 个的任务，两种模型都没有成功。Qwen3.5 已产生不少正确局部操作，最终成功仍很少。这支持先拆解能力、降低组合难度的选择；这些是早期诊断结果，不与后文另一组固定协议的 Raw/SFT 对照混算。

这一步交付的是可运行任务：用户需求、初始数据库、允许使用的工具、参考动作及成功条件都需要明确。仅生成一段银行客服对话，无法保证它能在环境中执行或被可靠评分。

## 2. 最初的尝试：拆解官方任务，用于诊断

第一阶段将源任务拆成六部分：业务目标、证据文档、实体和初始状态、判断条件、动作及执行方、成功检查与最终状态。动作依赖用于区分“事实没找对”“动作参数错误”“用户侧步骤未完成”和“整个工作流没有完成”。

先从 4 个代表任务构造 24 个探针，再扩展到 94 个可回放源任务，每个源任务生成六种形态，共 564 个诊断任务。另 3 个源任务保留在官方清单中，但未用于扩展：`task_051` 的写请求与已有状态冲突，`task_077` 引用当前不可用工具，`task_083` 的争议调用缺必需参数。这里没有修改官方任务来凑齐数量。[第一批构造与验证][diagnostic-readme]、[验证报告][diagnostic-validation]

Qwen3.8-27B 同时承担 Agent 和 User，第一批结果如下。16 个生成或上下文协议问题经过定向恢复后，得到表中成绩；保守首次尝试成功数为 315/564，恢复后为 329/564。[第一批模型结果][diagnostic-eval]

| 任务形态 | 主要隔离的能力 | 成功数 / 任务数 |
|---|---|---:|
| evidence-given | 给定文档，抽取指定事实 | 58/94，61.70% |
| retrieval-only | 检索目标文档 | 93/94，98.94% |
| decision-only | 给定证据和状态，完成判断 | 84/94，89.36% |
| single-action | 给定正确前置状态，执行一个动作 | 48/94，51.06% |
| two-skill-composition | 组合两个能力 | 14/94，14.89% |
| full-composition | 执行完整工作流 | 32/94，34.04% |

第二批 Scale V2 继续细化到逐事实、逐动作节点：933 个事实题、933 个检索题、874 个单动作题，共 2,740 个；参考回放全部通过，Qwen3.8 成功 2,163 个，成功率 78.94%。其中单动作题为 653/874（74.71%），Assistant 与 User 发起的参考动作检查通过率分别为 90.50% 和 42.16%，暴露出用户工具协作的困难。[第二批验证][scale-validation]、[第二批模型结果][scale-eval]

两批合计 **3,304 个诊断任务、2,492 条成功轨迹**。第二批更原子化，任务组成也不同，不能把 58.33% → 78.94% 解释为同一测试集上的模型能力提升；模型没有在这两批之间接受相应训练。

**随后调整了训练来源。**2026-09-06 确认官方 97 个任务全部用于 benchmark 后，所有官方派生任务及成功轨迹统一改为 `benchmark_derived_diagnostic`、`training_eligible=false`，退出 SFT、GRPO、OPD 候选池。按源任务 ID 切分也不能解决问题，因为所有源任务本身都属于评测集。[用途修正记录][diagnostic-star]、[第一批用途清单][diagnostic-pools]、[Scale V2 用途清单][scale-pools]

这些数据保留的价值是能力诊断和构造工具验证。后续训练集重新生成场景，没有把这 2,492 条成功轨迹改名后继续训练。

## 3. 独立场景如何构造并接入真实环境

独立构造使用已有 Banking 业务接口、工具 schema 和文档，不把官方逐题请求、实体、初始状态及 gold 动作作为场景实例输入。官方任务此前确实被用于盘点和诊断，因此“独立”指训练实例与生成输入的隔离，不能表述成从未分析过 benchmark。

**运行环境。**从 698 篇知识库文档中排除官方任务引用的 218 篇，留下 480 篇已有政策文档；并非新写了 480 篇文档。构建时复制这 480 篇文档和运行提示，基础 DB 置空，再由每个新任务初始化自己的客户、账户、卡片、交易等实体。`isolation.py` 负责生成文档白名单及生成后的实例比较；生成器只读取白名单文档和合成规格。[文档隔离实现][isolation-code]、[环境装配][documents-code]

**一条任务的构造过程。**业务模板同时定义客户要解决的问题、环境当前有什么，以及哪些结果算完成。

| 步骤 | 具体做法 | 输出 |
|---|---|---|
| 选择业务和复杂度 | 从 15 类业务中选定意图，组合课程形态与正常处理、拒绝、用户工具等情形，并选择许可文档和政策事实 | 业务目标、证据、能力标签 |
| 生成实体与初始状态 | 新建客户、账户、卡片、交易等实例，按业务前提设置状态；单动作题可预置已验证身份或已解锁工具，工作流题保留相应操作 | `ScenarioSpec` 中的初始 DB 和前置条件 |
| 定义执行和完成条件 | 指定每个参考动作的工具、参数、Agent/User 执行方及依赖，配置所需的 DB、动作或沟通判分；DB 目标状态由参考动作在环境中执行得到 | 参考动作链及 evaluation criteria |
| 装配交互任务 | 将公开开场、User 私有信息、可用工具和初始状态转为 Tau2 `Task`，由 Qwen3.8 改写公开措辞；同时导出供验证和分组使用的 `TaskContract` | 可直接运行的任务及配套规格 |

动作节点包含执行方 `requestor`、工具名、参数和 `depends_on`。当前模板的 `_nodes` 将参考动作按顺序连接成依赖链，便于逐步回放和检查前置状态。[数据结构及任务转换][models-code]、[场景模板][templates-code]

**参考流程与实际评分。**参考动作链给出一种可行执行路径，自主轨迹的最终成功由 `Task.evaluation_criteria.reward_basis` 决定。最终 647 个任务的评分依据如下。[评分依据生成逻辑][models-code]

| 评分依据 | 任务数 | 实际检查内容 |
|---|---:|---|
| `DB` | 423 | 最终数据库状态是否与参考执行后的状态一致 |
| `ACTION` | 47 | 是否完成指定动作，并匹配要求比较的参数 |
| `COMMUNICATE` | 150 | 回复是否包含指定沟通内容 |
| `ACTION` + `COMMUNICATE` | 27 | 同时满足动作与沟通要求 |

仅以 `DB` 判分的任务中，检索等不改变数据库的读取动作不单独计分；其他执行路径只要产生相同的最终状态，也可通过。因此，参考链的长度和步骤不能直接解释为评分器逐步强制检查的动作清单。

**两种检索条件承担不同课程目标。**`golden_retrieval` 将所需文档直接提供给 Agent，用来单独检查理解、判断和操作；`bm25` 不直接给定所需文档，Agent 通过 `KB_search` 检索。它们都存在于最终合成集，不能将混合结果称为纯 BM25 端到端成绩。

**模型负责改写表达，程序负责业务事实。**模板先确定实体、产品、事实、合法动作和成功条件，再用 Qwen3.8 将公开用户开场改得自然。改写时保护金额、日期、实体、业务目标、工具线索及输出要求，恢复这些字段后再检查；多次改写不合格时保留原句并记录。最终 r18 的 1,500 个 pilot 开场全部完成改写，没有回退。[自然语言改写实现][naturalize-code]、[pilot 改写结果][pilot-naturalization]

**Agent 与 User 信息边界。**公开需求作为固定首条 User 消息，保证 Agent 能看到必要线索。身份资料主要放在 User 私有指令中，用户工具的执行时机也由私有指令规定；部分公开工具参数同时包含身份资料。逐题检查最终 647 个任务，其中 23 个任务的公开开场包含该客户的邮箱、电话和地址，因此不能概括为身份字段全部私有。用户工具按依赖执行，完成后才停止，不能让 Agent 代替用户操作。[User 协作协议][models-code]、[公开调用线索生成逻辑][templates-code]

有一项明确的难度控制：部分动态工具名无法从 480 篇许可文档中查到，因此公开开场提供相应工具名、参数，必要时包含 wrapper 的准确调用方式。这些线索降低了动作选择与参数构造的难度，对独立发现工具调用签名的考察也有限；任务最终按上述评分依据检查结果。

## 4. 课程设计与一条真实任务示例

最终集覆盖信用卡选择、身份验证、推荐奖励、返现、争议处理、销户、调额、账户管理、ATM 费用、卡片遗失、借记卡争议、PIN、储蓄利息等 15 类业务，同时加入正常处理、拒绝处理、User 工具、动态 Agent 工具、多实体五类情形。

L1–L6 表示任务结构，不是模型实测难度排名。最终有七种形态，因为 L3 分为判断题与单动作题。本次按最终 `TaskContract` 重新统计如下。[最终训练规格][train-contracts]、[开发规格][dev-contracts]、[挑战规格][challenge-contracts]

| 形态 | 实际要求 | 训练 | 开发 | 挑战 |
|---|---|---:|---:|---:|
| L1 evidence | 给定文档，按开头定位词引用完整事实 | 54 | 0 | 0 |
| L2 retrieval | 按指定政策标题检索，返回文档 ID | 27 | 0 | 0 |
| L3 decision | 判断给定文档是否支持一项政策陈述 | 81 | 15 | 0 |
| L3 single-action | 已给定必要前置状态，完成一个动作 | 109 | 15 | 0 |
| L4 two-skill | 组合检索、参数构造或用户协作等能力 | 163 | 15 | 0 |
| L5 medium-workflow | 中等业务流程，参考链设计为 5–8 个动作 | 81 | 15 | 15 |
| L6 full-workflow | 多步骤或多实体流程，参考链设计为 8–14 个动作 | 27 | 15 | 15 |
| 合计 |  | **542** | **75** | **30** |

每个开发集业务类别有五种形态各一题；每个挑战集业务类别有 L5、L6 各一题。最终数据中的参考链最长为 9 个动作，14 是模板允许的上限，不能说最终样本已覆盖 14 步。

当前 L3 decision 模板从文档抽取真实陈述，参考答案统一为 `Yes`，它主要是证据支持判断和输出格式训练；没有构成均衡的 Yes/No 决策集。最终训练集的 97 个拒绝场景位于动作和工作流形态，应与判断题区分。[课程常量][constants-code]、[实际模板分支][templates-code]

检索条件的分布为：

| 集合 | Golden retrieval | BM25 | 合计 |
|---|---:|---:|---:|
| 训练 | 244 | 298 | 542 |
| 开发 | 30 | 45 | 75 |
| 挑战 | 0 | 30 | 30 |

例如训练任务 `banking_syn_v1_train_000744_l5_medium_workflow` 要求临时冻结一张遗失的借记卡。客户、卡片和状态均为合成实例，公开开场给出目标卡片及动态操作线索，参考流程为：

```mermaid
flowchart LR
    A[检索卡片政策] --> B[查询客户信息]
    B --> C[取得当前时间]
    C --> D[核对身份并记录验证]
    D --> E[解锁冻结卡片工具]
    E --> F[调用工具冻结目标卡片]
```

该任务以 `DB` 为评分依据，将自主轨迹的最终数据库与参考执行后的数据库比较；参考执行涉及 `verification_history`、`agent_discoverable_tools`、`debit_cards` 的状态变化。只回复“已经冻结”不能替代真实操作。上述动作图是内部参考，不会把整条参考执行序列直接交给自主 rollout 的 Agent；公开工具调用线索则确实存在，应保留这一条件说明。[示例所在任务文件][train-tasks]、[动作与预期状态][train-contracts]

## 5. 构造过程中的问题、尝试与修改

早期 1,500 题已经通过 schema、参考回放和检索检查，但模型交互仍暴露业务语义及评分问题。原因是：一条预先写好的参考链能够执行，不代表产品、政策、用户意图一致，也不代表模型从可见信息中能完成任务。

下面按问题归纳 r5–r18 的主要修改；版本号用于定位历史记录，不表示每版都做了同等规模的模型对照。[独立构造实验记录][synthetic-readme]

| 发现的问题 | 采取的修改 | 为什么有必要 |
|---|---|---|
| 同类文档与初始化产品、请求操作不一致 | r5–r7 将操作绑定到具体政策依据，统一产品、关键事实、状态及参数 | 类别相同不等于描述同一个可执行业务案例 |
| 知识题可以抄标题；选中的“事实”可能是导语或表头 | r6 去掉抄标题捷径；r9 排除不完整事实 | 保证监督对应政策内容 |
| 宽泛业务类别选出无关事实；同一文档多个事实都可能回答问题 | r17 改为选完整且与类别相关的事实；r18 加短开头定位词 | 让题目唯一指向目标，且不直接复制完整答案 |
| 正确同义表述、货币格式仍被字符串判分拒绝 | r8/r11 调整为简短沟通要求和数值锚点；r16 明示 Yes/No 格式 | 减少表达形式与成功条件不一致 |
| 动态工具名无法从许可语料中查到 | r12 提供公开工具标签和参数；r13 明示正确 wrapper | 修复缺信息导致的不可解任务 |
| User 未执行自己的工具，或提前停止 | r11 增加用户侧执行时序；r14 将必要公开线索固定为首条 User 消息 | 仅写在 User 私有提示里，不能保证模型会转述或执行 |
| 合理的重复验证增加日志，或转人工理由不同，造成误判 | r13 为适用 L3/L4 写任务预置且公开已完成验证；忽略转人工的非关键理由文本 | 让成功条件围绕所需业务变化 |
| 正确拒绝被要求严格复现无关读取动作 | r15 对 L4–L6 拒绝流程按用户侧转人工请求的最终 DB 状态判分 | 允许正确业务结果采用不同检索路径 |
| 资料修改请求缺少新邮箱 | r16 将目标邮箱同时写入公开需求和 User 已知信息 | 请求、参数来源和预期状态必须一致 |

除 1,500 题 pilot 外，还构造了 **15 类业务 × 5 类情形 = 75 题**模板检查集，参考回放 75/75、参考检索 120/120 通过；r18 的 pilot 加该检查集合计 1,575 次参考执行、1,190 次检索检查通过。30 题 smoke 覆盖全部业务类别、七种形态和五类情形，再进入规模化生成。

这些修改解决的是任务是否自洽、信息是否足够、评分是否对应所需行为。各版 smoke 的题目和定义发生过变化，不能把它们的成功率串成同一基准上的单因素提升曲线。

## 6. 最终采用了多少数据，怎样验证和选择

历史规模与最终采用量需要分开看：

| 阶段 | 规模 | 最终用途 |
|---|---:|---|
| 官方派生诊断 | 564 + 2,740 = 3,304 个任务 | 全部仅用于诊断，不进入训练 |
| 独立 pilot | 1,500 个任务，每类业务 100 个 | 校准模板并提供候选 |
| 独立扩容候选 | 10,000 train + 500 dev + 500 challenge | 已构建候选及验证记录，不等于全部采用或完成四次试验 |
| 中途缩减方案 | 2,000 / 500 / 500 | 保留为历史候选选择记录 |
| **最终构造集** | **542 / 75 / 30** | `final-minimal`，本步骤采用的交付 |

542 的目标规模在实验记录中设为 AReaL Telecom 271 个不同源任务的两倍。最终训练任务 ID 全部位于独立 pilot 的场景编号范围；扩容候选没有因此被全部纳入训练。部分早期 `PLAN.md` 和包内 README 仍保留 3,000 题计划，最终数量以 [acceptance.json][acceptance]、[assembly_report.json][assembly] 和 [constants.py][constants-code] 为准。

选择时同时满足业务类别与任务形态配额，在每个配额内优先保留已有四次有效试验、成功与失败混合的任务。最终 647 个任务对应 647 个不同场景，train/dev/challenge 没有复用同一场景。[选择实现][assemble-code]

**验证同时检查任务能否执行、模型能否理解，以及结果能否正确评分。**

1. 静态检查任务 schema、日期金额、实体引用和工具归属，排除参数、状态或执行方不一致的问题。
2. 将参考链逐步交给真实 Banking 环境执行，再运行任务所需的 ACTION、COMMUNICATE、DB evaluator；BM25 任务还逐次检查参考查询能否命中目标文档。参考回放的最后文本按评分事实构造，证明的是预定解法与判分可执行。
3. 让 Qwen3.8 分别承担 Agent 和 User 自主交互，检查模型从实际可见信息出发能否完成任务。工具名缺失、User 提前停止、正确回答被字符串判分拒绝等问题，都是在这一步暴露并推动模板修订的。冻结任务后保留业务失败记录，并将协议补跑单独统计；成功示范只从 train 中采集。

三类验证的结果分别保留：参考回放通过率用于说明任务可执行，自主交互用于检查任务定义和采集示范，后续 4B 官方评测用于衡量 SFT 效果。[回放实现][validate-code]、[模板修订记录][synthetic-readme]

| 最终构造集检查 | 结果 |
|---|---:|
| Task schema | 647/647 通过 |
| 参考链在真实环境执行且 reward=1 | 647/647 |
| 参考 BM25 查询的目标文档 top-10 命中 | 530/530 |
| 运行时许可文档 | 480 篇 |
| 与官方实例的实体、请求、文档集合、初始记录复用检查 | 各项发现数均为 0 |
| 场景数 / 任务数 | 647/647 |

530 是参考执行中发生的检索检查次数。按三个 split 的参考动作统计，共有 530 次 `KB_search` 调用，查询文本按完全相同内容去重后为 **63 条**；它不是 530 个任务或 530 条不同查询组成的数据集。参考 query 来自目标文档正文片段，作用是证明文档可被检索；不能据此宣称 Agent 自己生成的查询命中率为 100%。复用检查的结论限定于已检查字段，业务规则、模板与工具接口仍然共用。[参考验证结果][reference-validation]、[实例隔离结果][isolation-report]

模型试验之后形成的训练候选如下。池之间可以重叠，不能把数量直接相加。[候选池清单][pool-manifest]、[分池实现][pools-code]

| 产物 | 数量 | 判定含义 |
|---|---:|---|
| SFT candidate | 542 个任务 | 每个训练任务至少有一条成功轨迹 |
| GRPO frontier | 174 个任务 | Qwen3.8 的四次有效试验中成功 1–3 次 |
| Behavior anchor | 368 个任务 | Qwen3.8 的四次有效试验全部成功 |
| Hard SFT only | 0 个任务 | 最终没有采用仅靠节点教师补出的任务 |
| 成功训练轨迹 | 1,914 条 | 仅来自 train，reward=1，均为自主生成 |

本次逐条重新统计 1,914 条源轨迹：轨迹 ID 均不同，恰好覆盖 542 个训练任务；成功次数分布为 21 个任务各成功一次、38 个各两次、115 个各三次、368 个各四次。所有行均标记为 `independent_synthetic` 和可用于训练。

这些 frontier/anchor 是 **Qwen3.8 下的观测分类**，不是已经证明适合 4B 学生模型的 GRPO 难度分层。曾实现逐动作节点指导的 teacher 路径并完成 smoke，但最终 `teacher_success_trajectories=0`，不能把 1,914 条描述为节点教师纠错生成。

## 7. 合成留出集评测：首次尝试与恢复后结果

合成任务评测中，Agent 和 User 都是本地 `/mnt/afs/models/Qwen3.8-27B`。Agent 开启 xhigh thinking，temperature=1.0、top-p=0.95、单次输出上限 16,384，服务 context 为 262,144；User 关闭 thinking，temperature=0、输出上限 1,024。每个任务计划 seeds 300–303 四次试验，最多 60 步。缺少最终回复、输出截断、服务错误或非正常终止进入恢复流程，恢复时 Agent 输出上限升至 32,768。[评测脚本][synthetic-eval-code]、[有效试验定义][protocol-code]

75 个 dev 与 30 个 challenge 共 105 个任务，其中 30 个直接给定文档、75 个使用 BM25。这是内部合成留出集，不是官方 97 个任务。

train/dev/challenge 按场景实例划分，模板和政策允许复用。按 `TaskContract.template_id` 比较，dev 中 **60/75** 个任务、challenge 中 **14/30** 个任务使用训练集中已有的模板。因此，下述成绩反映这些合成留出场景上的表现，不能解释为对未见模板或新政策的泛化结果。[训练规格][train-contracts]、[开发规格][dev-contracts]、[挑战规格][challenge-contracts]

| 统计对象 | 尝试数 | 有效试验 | 成功试验 | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|---:|---:|
| 原计划四次尝试 | 420 | 364 | 301 | 82.69% | 91.30% | 73.91% |
| 仅恢复部分 | 90 | 56 | 45 | 80.36% | 不作整体对比 | 不作整体对比 |
| **合并有效试验后** | **510** | **420** | **346** | **82.38%** | **91.43%** | **71.43%** |

口径说明：

- pass@1 是有效试验成功比例。最初为 301/364；若将原计划 420 次中的所有非成功都计 0，则为 **301/420 = 71.67%**。
- 最初只有 69 个任务凑齐四次有效试验，因此前一行 pass@4(any) 和 pass^4 的分母是 69，分别为 63/69、51/69。
- 恢复后全部 105 个任务各有四次有效试验：pass@4(any)=96/105，pass^4=75/105。前者表示四次中至少成功一次，后者表示四次全部成功。
- 原计划 420 次中有 56 次协议失败，比例 **13.33%**，含 52 次 infrastructure error 和 4 次 max_steps。恢复又发生 34 次失败；“补齐所有有效试验”不代表首次或全程零失败。
- 本实验把 max_steps 也归为协议失败并补跑。这是该数据构造实验的统计约定，max_steps 仍可能反映模型长程行为问题；恢复结果不能当作相同预算下的首次部署成功率。

恢复只补无效试验，保留有效的业务失败，并非每题重试到成功。但输出预算增加，且汇总以有效试验为条件，必须同时保留上述首次统计。[最终机器可读结果][synthetic-metrics]、[汇总逻辑][summarize-code]

恢复后的 DB / ACTION / COMMUNICATE 分项均值为 79.41% / 80.00% / 100.00%；分母是相应分项实际有评分的有效试验，不能将 COMMUNICATE=100% 理解为所有工作流都成功。

`final-minimal/metrics.md` 只展示首次统计；简历引用的 82.38% / 91.43% / 71.43% 来自 `metrics.json` 的 `all_attempts`，该对象中的 pass 指标仍按有效试验计算。两组数字不同并非训练前后提升。

## 8. 成功轨迹如何变成 SFT 数据，训练结果如何

任务构造完成后，已有后续实验验证了这些轨迹的训练用途。这部分与第二点使用相同的原生消息、工具 schema 和目标轮 loss mask 思路，但 Banking 暴露出额外的长检索问题。

**首次转换的问题。**1,914 条成功轨迹中，4 条因格式或未知工具问题被排除；其余生成 13,804 个 Assistant 监督目标。按 16,384-token 上限逐目标过滤后，去掉 6,788 个超长目标，留下 7,016 行，覆盖 542 个任务。[首次转换与训练][expert-readme]

问题不只是数据变少：6,788 个被过滤目标全部来自 BM25 轨迹。985 条 BM25 源轨迹中，只有 240 条（24.4%）保住最后一个 Agent 目标；保留的 4,921 个 BM25 目标有 3,927 个是 `KB_search`，占 79.8%。监督分布偏向“看到检索结果后继续搜索”，执行与收尾监督不足。源轨迹整体成功也没有保证局部动作正确，旧导出仍保留了 102 个实际执行失败的 Agent 动作作为监督。[数据与行为诊断][expert-diagnosis]

后续评测确实出现了大量检索循环和长上下文：已保存的 Banking-only 固定协议子集平均每条搜索 29.60 次，原始模型为 1.24 次。该 SFT 子集只有 95 条，不是完整基准；不能用它报完整成功率或严格加速比。一次 60 秒超时版跑完全部 388 条 Banking 轨迹，成功为 0，但它改变了运行预算，也不应与无超时结果直接作严格训练收益对照。

**修改为完整轨迹精简后再导出。**用 Qwen3.8 提议删除冗余 Agent 调用及配对结果、无用文本，再由独立评审检查证据与语义连贯性，在环境中按保留顺序回放读写、比较工具结果和最终 DB、重新计算所需 reward。保留 User 事件、最终回复及原有多调用批次，不再把批次后续调用改造成“看到前一个结果后新决定的动作”。只接受完整 Agent 可见轨迹能放进 16,384 tokens 的样本，随后展开目标轮监督。[精简方法][simplify-readme]

它删除不必要步骤，不改写保留工具返回的正文，也不把模型评审或 thinking 写入训练样本。对需要保留的错误恢复过程，仍按完整上下文和回放结果判断，不能称导出数据为“所有中间动作零错误”。

| 全量精简处理 | 轨迹数 |
|---|---:|
| 输入成功源轨迹 | 1,914 |
| 最终保留并导出 | **1,651** |
| 评审或验证拒绝 | 237 |
| 精简后仍超长 | 22 |
| 处理错误 | 3 |
| 导出错误 | 1 |

最终输出 5,672 行，覆盖 530 个训练任务，包含 3,296 个工具调用目标和 2,376 个文本目标。565 条原本超长的轨迹通过精简恢复可用；对同一批最终保留的 1,651 条源轨迹比较，完整轨迹 tokens 总量由 25,953,260 降至 10,713,590，减少 **58.72%**。这是相同轨迹 ID 下的压缩量，不把删掉其他样本的数量算作压缩收益。[全量统计][simplify-stats]

**4B 学生模型的官方任务结果。**以下均为已有结果：97 个官方 Banking 任务 × 4 trials，seed=300、BM25，User 为 Qwen3.6-27B non-thinking；Agent temperature=0.6、每次输出上限 1,200、max_steps=200，无单轨迹超时。它们与第七节的 Qwen3.8 合成留出评测不同。

| 模型及训练数据 | Banking 成功次数 | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|---:|
| 原始 Qwen3-4B-Instruct-2507 | 10/388 | 2.58% | 4.12% | 1.03% |
| 部分精简数据 SFT：1,782 行、207 个任务 | 16/388 | 4.12% | 9.28% | 0.00% |
| 全量精简数据 SFT：5,672 行、530 个任务 | 17/388 | 4.38% | 9.28% | 1.03% |
| AReaL 30,376 行 + Banking 5,672 行混合 SFT | 16/388 | 4.12% | 8.25% | 1.03% |

部分和全量 Banking SFT 均从原始 4B 初始化，使用两轮训练、batch=16、学习率 1e-5 cosine 降至 1e-6、训练 seed=1234。各数据集行数不同，更新数也不同；四领域混合还改变了领域配比。全量相对 Raw 的 pass@1 点估计增加 **1.80 pp**，pass@4(any) 增加 **5.15 pp**，pass^4 不变。仅有一个评测 seed，绝对成功率仍低，不能称已证明显著或充分的官方任务泛化提升。精简还改变了样本选择和监督分布，不能将差异唯一归因于缩短轨迹。[部分重训][partial-sft]、[全量重训][full-sft]、[Raw 原始结果][raw-eval]、[全量 SFT 原始结果][full-eval]

9 月 20 日的四领域混合数据总计 36,048 行，已完成 4,506 updates 和全部 788 条四领域评测。Banking 共发生 1,170 次搜索，平均 3.02 次/轨迹，检索循环较早期长链问题缓解；但官方任务成功率仍只有 4.12%。本步骤产物已真正接入训练，数据可用性改善与最终业务成功率提升仍需分别评价。[四领域混合训练和评测][mixed-sft]

## 9. 简历表述与证据入口

第三点命名为“Agentic 任务构造”，以“针对 tau3-benchmark Banking 领域”明确场景。任务合成展开说明能力拆解、业务模板与实体状态生成、任务复杂度控制，以及用户需求、工具动作和成功条件如何配套构造。程序确定业务事实与执行规则，模型改写用户表达。随后说明如何在 Banking 环境中执行预设解法验证任务，再从自主交互的成功轨迹中删除冗余步骤、保留完整监督，最后给出 SFT 的官方任务对照。

可用于简历的条目：

> **Agentic 任务构造：**针对 tau3-benchmark Banking 领域，将复杂客服流程拆解为事实抽取、知识检索、规则判断、工具执行和用户协作等能力，设计由单项能力、能力组合到完整工作流的 7 种任务形态。依据政策文档和工具接口编写 15 类业务模板，程序化生成新的客户、账户、交易及初始数据库，同步配置用户需求、工具参数、Agent/User 执行分工、动作依赖与成功条件；通过调整是否给定文档、是否预先完成身份验证或工具解锁，以及操作链长度，构造不同复杂度的场景，并覆盖正常办理、拒绝办理、动态工具和多实体等情形。使用 Qwen3.8-27B 在保留实体、金额和业务目标的前提下改写用户开场，最终构建 647 个独立场景（训练集 542 题）。
>
> 在 Banking 环境中逐题执行预设解法，全部达到任务设定的成功条件；再使用 Qwen3.8-27B 分别模拟客服 Agent 与用户采集成功轨迹，删除冗余工具调用及对应返回、无用回复，保留必要证据与最终答复，经语义检查和环境重放产出 5,672 条 SFT 样本。训练 Qwen3-4B-Instruct-2507，在官方 97 题、每题 4 次试验中，pass@1 由 2.58% 升至 4.38%，pass@4 由 4.12% 升至 9.28%。

这条表述对应第八节的 **Banking-only 全量精简数据 SFT**。其 pass@1 / pass@4 分别增加 1.80 / 5.15 pp，pass^4 保持 1.03%；成功轨迹数由 10/388 增至 17/388，四次中至少成功一次的任务由 4/97 增至 9/97。现有结果说明这套数据已完成训练验证，官方任务上的提升仍有限，不能归因为某一项单独的构造或精简措施。四领域混合 SFT 的 Banking pass@1 为 4.12%，应作为另一组结果说明。

版面取舍上，优先保留构造方法、环境回放、5,672 条训练样本及官方 SFT 对照。1,651 条保留轨迹的 token 总量减少 58.72%，可在介绍长轨迹处理时补充，口径为同一批轨迹压缩前后比较。480 篇文档、530 次参考检索检查和 Qwen3.8 的合成留出集成绩放在实验材料中即可；82.38% / 91.43% / 71.43% 是补齐有效试验后的 Qwen3.8 结果，不能替代学生模型的训练结果。

计数时保留以下区别：647 包含 train/dev/challenge 三个 split；1,914 条成功示范只来自 542 个训练任务，精简后的 SFT 数据覆盖其中 530 个。参考检索的 530 是检查次数，查询文本去重后为 63 条；若简历需要保留该项，应写“530 次参考 BM25 检索检查全部命中 top-10”。合成留出集按场景实例划分，仍可复用训练集模板和政策。

| 入口 | 可核对内容 |
|---|---|
| [任务盘点与分解方案][inventory] | 原始任务复杂度、能力分类、早期失败模式 |
| [第一批验证][diagnostic-validation]、[第二批验证][scale-validation] | 官方派生任务数量、参考链错误和扩展检查 |
| [独立构造实验 README][synthetic-readme] | r5–r18 调整、规模变化、历史任务及日志 |
| [场景模板][templates-code]、[任务转换][models-code] | 实体状态、公开与私有信息、动作图、reward basis |
| [最终构造目录][final-dir] | 三个 split 的 Task、ScenarioSpec、TaskContract |
| [最终采用结果][acceptance]、[参考回放][reference-validation]、[实例检查][isolation-report] | 647 个任务及 530 次参考检索的完成证据 |
| [模型评测 JSON][synthetic-metrics]、[训练候选池][pool-manifest] | 首次与恢复口径、1,914 条轨迹和任务分层 |
| [Banking SFT 数据诊断][expert-diagnosis] | 长度过滤为何丢掉执行和收尾监督 |
| [精简统计][simplify-stats]、[重训结果][full-sft]、[混合 SFT][mixed-sft] | 5,672 行实际数据及官方任务表现 |

本次重新统计了官方任务结构、最终三个 split、1,914 条成功训练轨迹，并核对原始评测 summary 与精简统计；两个仓库的场景模板、任务结构、课程常量、分池、回放和评测汇总代码内容一致。本文整理已有工作，没有新增任务、修改训练配方或启动训练评测。

[doc02]: 02_AReaL三领域SFT数据质量治理.md
[inventory]: ../docs/tau2-banking-task-decomposition-curriculum.md
[diagnostic-star]: ../docs/03-banking-task-synthesis-star.md
[diagnostic-readme]: ../output/experiments/tau2-banking-task-curriculum/README.md
[diagnostic-validation]: ../output/experiments/tau2-banking-task-curriculum/expanded_validation_report.json
[diagnostic-eval]: ../output/experiments/tau2-banking-curriculum-qwen38-eval/all_recovered.md
[diagnostic-pools]: ../output/experiments/tau2-banking-curriculum-qwen38-eval/training_pool_manifest.json
[scale-validation]: ../output/experiments/tau2-banking-task-curriculum-scale-v2/scaled_validation_report.json
[scale-eval]: ../output/experiments/tau2-banking-scale-v2-qwen38-eval/all_results.md
[scale-pools]: ../output/experiments/tau2-banking-scale-v2-qwen38-eval/training_pool_manifest.json
[synthetic-readme]: ../output/experiments/tau2-banking-independent-synthetic-v1/README.md
[models-code]: ../examples/tau2-bench/analysis/banking_synthetic/models.py
[templates-code]: ../examples/tau2-bench/analysis/banking_synthetic/templates.py
[constants-code]: ../examples/tau2-bench/analysis/banking_synthetic/constants.py
[isolation-code]: ../examples/tau2-bench/analysis/banking_synthetic/isolation.py
[documents-code]: ../examples/tau2-bench/analysis/banking_synthetic/documents.py
[naturalize-code]: ../examples/tau2-bench/analysis/banking_synthetic/naturalize.py
[assemble-code]: ../examples/tau2-bench/analysis/banking_synthetic/assemble.py
[validate-code]: ../examples/tau2-bench/analysis/banking_synthetic/validate.py
[pools-code]: ../examples/tau2-bench/analysis/banking_synthetic/pools.py
[summarize-code]: ../examples/tau2-bench/analysis/banking_synthetic/summarize.py
[protocol-code]: ../examples/tau2-bench/analysis/banking_synthetic/protocol.py
[synthetic-eval-code]: ../examples/tau2-bench/analysis/banking_synthetic/run_qwen38_eval.sh
[pilot-naturalization]: ../output/experiments/tau2-banking-independent-synthetic-v1/pilot/naturalization_report.json
[final-dir]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal
[train-contracts]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/train_contracts.jsonl
[train-tasks]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/train_tasks.json
[dev-contracts]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/dev_contracts.jsonl
[challenge-contracts]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/challenge_contracts.jsonl
[acceptance]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/acceptance.json
[assembly]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/assembly_report.json
[reference-validation]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/reference_validation.json
[isolation-report]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/isolation_audit.json
[pool-manifest]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/pool_manifest.json
[synthetic-metrics]: ../output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/metrics.json
[expert-readme]: ../output/experiments/tau2-banking-expert-sft/README.md
[expert-diagnosis]: ../output/experiments/tau2-banking-expert-sft/DATA_DIAGNOSIS_20260910.md
[simplify-readme]: ../output/experiments/tau2-banking-simplify-qwen38-v1/README.md
[simplify-stats]: ../output/experiments/tau2-banking-simplify-qwen38-v1/data/full/stats.json
[partial-sft]: ../output/experiments/tau2-banking-simplified-sft-test/README.md
[full-sft]: ../output/experiments/tau2-banking-simplified-full-sft/README.md
[raw-eval]: ../output/experiments/tau2-banking-expert-sft/eval/raw-full-qwen36-bm25-fixed/seed300_0910_raw_fixed_summary.json
[full-eval]: ../output/experiments/tau2-banking-simplified-full-sft/eval/simplified-sft-banking/seed300_0911_142847_summary.json
[mixed-sft]: ../output/experiments/tau2-sft-areal3-banking-simplified-full-20260920/README.md
