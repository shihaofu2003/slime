# tau2-mix-rl-sft4505-goal-v4-20260928

Purpose：把任务含义、证据可见性、终局条件及训练进度对齐，覆盖v3发现的目标漂移、Banking困难子任务无信号、多个商品完成程度不可区分等问题。独立数据、代码及入口已准备；本目录的两组混合训练尚未提交，原v1/v2/v3实现和历史任务数据保留。

2026-09-28按用户要求先运行[Retail单领域expert](../tau2-retail-expert-sft4505-goal-v4-20260928/README.md)：4-GPU User24813、8-GPU训练24814，采用本版goal_progress配方及Retail子集。

## 问题抽象与实现

| 共性问题 | 已发现的情况 | v4处理 |
|---|---|---|
| 用户需求、真实对象和评分条件不一致 | 描述smartwatch但DB是耳机；用户选择礼品卡，参考目标固定为信用卡；接受Agent错误替代品后目标漂移 | 从真实DB和已有任务目标构造逐订单的业务需求，明确商品配置、数量、地址和各操作付款偏好；不给Agent参考工具调用或新SKU ID |
| 任务声明的证据条件没有被运行入口继承 | 任务要求引用“给定材料”，RL实际未提供材料，统一要求自行BM25检索 | 将现有任务定义中的retrieval_variant写入metadata，并在环境构造中读取；该给材料的提供材料，该检索的仍要求检索 |
| 进度量没有覆盖所有成功条件，或粒度过粗 | COMMUNICATE/ACTION没有DB势函数；商品列表只计为一个不同字段，改对1件与2件无法区分 | DB比较深入列表字段；ACTION记录已成功完成且满足参数/角色的要求；COMM记录已表达的目标事实，相关文档中的目标事实提供有限中间信号 |
| 组内比较混合了不同完成阶段和相关重复观测 | 同一个DB变化计数包含不同进度；一条长轨迹的多次停留会重复进入归一化 | 按目标势函数值分阶段，每条轨迹在同一阶段贡献一个平均return；至少两条轨迹才能产生比较信号 |

这是一版可运行的最小方案：使用已有任务条件、数据库、工具结果和轨迹，不增加在线LLM评分器、额外模型服务或复杂状态机。付款方式、真实金额、商品数量和顺序仍参与评分，没有通过忽略这些字段提高分数。

## 新发现：Banking的输入条件被改变了

原始 `datasets/tau2/rl/banking_independent_synthetic/contracts/train.jsonl` 为542个任务声明了两种条件：244个provided-document任务（代码名golden_retrieval），298个BM25任务。其中全部54个L1证据任务、81个L3决策任务和109个L3单动作任务都属于前者。v3的 `envs.make_environment_constructor` 却统一传入 `retrieval_variant="bm25"`。

因此v3的L1低分不能全部归因于“模型不愿检索”：原任务考的是阅读给定资料，运行时变成了需要先检索的另一种条件。v4恢复声明的条件，不把244个任务全部改成“会检索”，也不改变另外298个检索任务。未声明此字段的旧metadata仍默认BM25，官方测试不因此获得额外材料。

这是训练输入条件的调整。直接给定材料后可能出现的分数上升，不能解释成模型已经学会检索；需使用同条件SFT/旧RL作对照，并继续在正式BM25评测中验证迁移能力。

## 数据与代码

- [task_goals_v4.py](../../../examples/tau2-bench/rl/task_goals_v4.py)：生成 [data/train.jsonl](data/train.jsonl)。2,524个任务及原reward_basis/评价目标保留；522个Retail任务重建User需求；22个涉及转人工、14个无业务写入、5个要求比较选择的任务保留原文。选择类任务不能直接把参考选项变成用户已知偏好，否则会提前透露需要Agent推导的答案。Airline/Telecom任务输入不改。统计见 [task_preparation.json](task_preparation.json)。
- [goal_progress_v4.py](../../../examples/tau2-bench/rl/goal_progress_v4.py)：读取任务级证据模式，组合DB/ENV/ACTION/COMM进度，沿用v3的环境修复。ACTION在两组对照中都要求存在成功工具响应、必需参数和正确执行方；DB/ENV及COMM终局评分沿用v3。
- [goal_credit_v4.py](../../../examples/tau2-bench/rl/goal_credit_v4.py)：同一任务组内按目标势函数值比较阶段，先对每条轨迹该阶段的return取均值，再在轨迹间按样本标准差归一化。结果写入原有token-advantage接口，保持分段mask/对齐检查；记录 `credit_recipe=goal-progress-v4`。
- [train_tau2_goal_v4.py](../../../train_tau2_goal_v4.py)：通过独立Ray worker启动钩子接入；原源文件无需修改。

COMM进度按现有COMMUNICATE字面判定累计已表达事实；只在正确的required-document结果中确实出现目标事实时给0.25的中间完成度，表达该事实后为1。反复检索同一材料不叠加分数，只有工具名或无关结果得不到该信号。这没有解决任意自然语言事实判断，也没有伪造全错组的优势。

## 验证结果

- 16项CPU检查通过，涵盖数据目标不变、商品绑定/重复数量、付款偏好、选择任务不提前给出参考选项、证据模式、列表进度、事实/检索信号、失败调用不算完成、同阶段轨迹归一化、token mask、真实Orchestrator边界和Ray父子worker继承。最终完整检查于2026-09-28 14:57 CST完成；检查由新启动脚本显式运行，为保留原文件，未修改固定CI矩阵。
- 542个Banking任务的参考动作回放全部reward=1、最终势函数=0，244个任务应提供的原文均实际出现在policy中。[回放结果](banking_reference_checks.jsonl)、[汇总](reference_summary.json)。
- 使用真实Agent的official-native system prompt、工具schema和SFT4505 tokenizer测量：Banking初始上下文最大4,723 tokens，没有初始上下文超过16K。[逐任务长度](initial_prompt_tokens.json)。完整对话仍受原16K cap和重采样逻辑约束。
- 在真实 `retail_79` 中，分别正确修改0/1/2/3个商品：旧势函数为−3/−2/−2/0，新势函数为−17/−9/−6/0；新度量能区分只完成一件和两件，且全部完成仍为0。数值尺度变化会再经过组内归一化，不等于按该倍数放大梯度。
- 使用现有User服务24664做3个情境的v3/v4配对，以及两个后续对话，共8次User生成。`retail_265`从“全部退礼品卡”变为第一单Visa、第二单礼品卡，并在两单完成后停止；`retail_541`不再提出不存在的smartwatch目标；`retail_347`拒绝错误的5x替代品，重申原3x需求。[完整回复](user_probe_results.json)。这是少量User探针，不是Agent端到端成功率。
- 对旧v3的219个Banking无DB/ENV任务组重算credit：35组从无信号变为有信号，其中L1的31个被丢弃全错组仅恢复4组；L1仍有32组无信号（包含全对组）。1组中的1条原成功轨迹因ACTION执行条件不满足改判失败。[旧轨迹反事实结果](communication_credit_replay.json)。没有重新训练或为旧轨迹补入新文档，不能把恢复组数换算成模型提分。

## 可执行的对照

| 对照 | 任务/User与证据模式 | 终局评分 | credit |
|---|---|---|---|
| `task_consistent` | 新的同一份v4数据 | v3环境/COMM + ACTION有效执行检查 | 原progress-db-count-v1 |
| `goal_progress` | 与上一组相同 | 与上一组相同 | 覆盖所有目标、深入列表、按目标阶段归一化 |

两组均从SFT4505、新优化器开始；保持User、任务数、种子、LR2e-6、KL/entropy0、K8、batch128、140 updates、pool32、pending64和max_policy_lag=-1一致。目录按arm分开，不覆盖v3。4卡User、8卡mix RL配置见 [run_user.sh](run_user.sh)、[run_mix.sh](run_mix.sh)。本轮没有提交这两组训练。

```bash
TAU2_V4_ARM=task_consistent bash output/experiments/tau2-mix-rl-sft4505-goal-v4-20260928/run_mix.sh --dry-run
TAU2_V4_ARM=goal_progress bash output/experiments/tau2-mix-rl-sft4505-goal-v4-20260928/run_mix.sh --dry-run
```

应先把输入/目标一致性修复作为共同条件，再比较credit；同时报告实际生成轨迹数、token与GPU时间，因为过滤比例变化时140 updates不代表相同生成预算。最终配方仍由相同官方评测协议的pass@1/pass@4(any)/pass^4决定。

## 边界与复现

Retail生成器选择已有结构化评价目标作为需求重建依据，形成新的、一致的训练任务版本；它没有证明原始参考答案本身一定正确，也不能保证与原始自然语言任务的全部含义相同。41个保留原文的任务仍需单独核查；当前识别出的选择规则为最便宜/最贵等明确表述，其他复杂语言约束仍需抽查。

新User提示能减少目标漂移，8次探针不能保证所有未来对话都遵守。Goal stage目前用势函数值近似，不是完整业务状态或严格因果匹配；相同分数可能对应不同剩余目标。全组完全没有正确进展时仍无有效比较信号，不能仅靠改归一化解决探索问题。

候选可能改变学习信号和任务难度，尚无新训练/held-out提升结论。原官方评测脚本与运行中的作业未改。

复现顺序：运行 [task_goals_v4.py](../../../examples/tau2-bench/rl/task_goals_v4.py) 生成新数据；执行 [test_goal_v4.py](../../../examples/tau2-bench/rl/test_goal_v4.py)；再运行 [verify_goals.py](verify_goals.py)、[measure_prompts.py](measure_prompts.py)、[replay_communication_credit.py](replay_communication_credit.py)。[check_user_requests.py](check_user_requests.py)需要现有User服务，包含实际推理调用。
