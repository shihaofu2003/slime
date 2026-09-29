# Airline：Mix RL 与 expert 的差距诊断

目的：解释 progress-db-count-v1 Mix RL、Vanilla Mix 的 Airline 表现为何低于选出的 Airline expert。实验：`tau2-mix-rl-sft4505-b128-20260925`。分析日期：2026-09-29。只读取已有产物，没有改训练、提交评测或停止任务。[算法与完整结果表](../../../../../project_docs/05_Credit_assignment.md)

## 1. 结论和证据强度

直接观察到的主要退化是：**模型更容易执行本不该执行的业务操作，丢失了 SFT 已有的“不修改 DB”能力，而非普遍不会调用 Airline 工具。** iter129 相对 expert 少成功的6条/8条轨迹中，分别有4条/6条净差来自此类任务。需要写入的任务上，两个 Mix 都是14/52成功，接近 expert 的16/52，且高于 SFT 的7/52。

最有证据的解释是：训练任务偏重写入，拒绝/只读任务稀少且场景重复；uniform-outcome 过滤进一步减少它们的入训比例；随机放回采样降低相同组数下的任务覆盖；较长、KL系数为0的训练伴随业务边界能力退化。专家自身继续训练也下降，不能把全部差距归为混合训练独有的负迁移。

上述行为差异和采样计数已确认；采样方式、训练时长、跨域梯度各自造成多少分损失，现有实验无法单独识别。没有发现 Airline 缺少总训练量、progress 信号整体失效或数值崩溃的证据。

## 2. 同题比较：差距主要落在哪些任务

逐题汇总 SFT、Airline expert 的 iter9/19/29/39/49/59，以及两种 Mix 的 iter129/139，共880条 Airline 评测轨迹。原始任务、task/trial/seed 集合一致；主要五个对照的环境 policy/tool definitions、User 和 Agent 请求参数一致，Airline 基础设施错误均为0。

按参考动作是否包含 DB 写工具，把20道 Airline 题分成7道“不写 DB”和13道“需要写入”。前者为 task `2/6/13/26/31/45/48`；它们要求拒绝违规请求或保持原状态，不代表所有措辞和过程都已被官方 reward 检验。

| 模型 | 不写 DB：成功/28 | 需要写入：成功/52 | 总成功/80 | pass@1 |
|---|---:|---:|---:|---:|
| SFT4505 | 27 | 7 | 34 | 42.50% |
| Airline expert iter29 | 27 | 16 | 43 | 53.75% |
| progress Mix iter129 | 23 | 14 | 37 | 46.25% |
| Vanilla Mix iter129 | 21 | 14 | 35 | 43.75% |
| progress Mix iter139 | 20 | 12 | 32 | 40.00% |
| Vanilla Mix iter139 | 22 | 12 | 34 | 42.50% |

因此 expert iter29 的特点是改善写入能力的同时，保住27/28的原有边界表现。Mix iter129 也学会了更多写入任务，但这部分收益被边界失分抵消。progress139、Vanilla139 相对 expert 的11条/9条净差中，不写 DB 类分别贡献7条/5条。

## 3. 具体轨迹中的错误

下表 turn 使用原始消息的 `turn_idx`。依据任务参考动作、已返回的 DB 信息、实际调用参数及官方 DB 检查判断，不只看工具名匹配率。

| 任务与轨迹 | 直接观察 | 对照 |
|---|---|---|
| task45，progress129 trial0/1 | basic economy 且用户不接受升舱，却调用 `update_reservation_flights` 改到5月22日；trial0 turn20 | expert29 四次均不改 DB，4/4；progress129 为2/4 |
| task48，progress129 trial2 | DB 的 `created_at` 是5月2日，当前时间5月15日；仍采信用户“10小时前订票”，turn12 取消 `3RK2T9` | expert29 4/4；progress129 3/4 |
| task6，Vanilla129 trial1/3 | 为给已有订单补保险，尝试取消/重订或另建带保险订单 | expert29 4/4；Vanilla129 2/4 |
| task37，两种 Mix iter129 的 trial0 | 只根据 business cabin 判断可取消，取消已飞过的 `NQNU5R`，虽同时完成另一订单的升舱，终态仍错 | expert29 trial2 只升级 `M20IZO`，不取消已飞订单；两种 Mix 该题均0/4 |
| task22，progress129 trial1/3、Vanilla129 trial3 | Gold + Economy 应有3件免费行李，却把 `nonfree_baggages` 写成1，多收费 | expert29 4/4，正确写0；progress129 2/4、Vanilla129 3/4 |
| task30，两种 Mix iter129 trial0 | 改成直飞时保留了旧的 PHX→IAH 航段，提交了不应共存的航段组合 | expert29 trial3 正确保留新去程和原返程 |

这些多为业务规则、条件组合和参数决策错误。Airline policy 明确要求 Agent 自行判断 basic-economy 改签和取消资格，工具 API 不替 Agent 完整执行这些限制，所以“工具调用成功”并不代表业务正确。[policy][policy]、[工具实现][tools]

也核对了一个实际入训样例：`airline_150`，group200，sample1600–1607。7条保持 DB 不变的轨迹 reward=1；sample1607 在拒绝取消后额外执行了 `update_reservation_passengers`，reward=0。turn16 的 progress reward 为−1，最终 advantage 约−5.103。这说明该样例的 credit 正确压低了偏离参考目标的额外写入，不能把问题概括成“progress 奖励任何写 DB”。训练对话没有执行错误取消。[原始训练轨迹][train-dump]、[credit][credit-dump]

## 4. 训练数据、过滤与覆盖率

Airline 训练池的1148个任务中，只有53个（4.62%）的参考动作不写 DB；其中31个围绕同一个订单 `VAAOXJ`，53个任务只涉及22种订单集合。官方测试的不写 DB 类占7/20（35%）。训练和评测使用不同任务，这里的问题是业务类型比例与多样性不匹配，不是要求训练覆盖测试题本身。

progress Mix 的完整候选 dump 中，该类共有104个完成的 K8 组：52个全成功、9个全失败、43个混合 outcome。前两类被现有过滤移除，43组实际入训，占 Airline 入训组的3.55%（43/1211）。需要写入的完成组则有1173个混合 outcome，其中1168组被训练消费；剩余是停止时未消费的完成组。候选保留数与实际训练消费数不能混用。

专家也使用同样的过滤：iter29 的480个入训组只有17个不写 DB（3.54%）。所以稀缺性是共享问题，并非只有 Mix 才过滤了边界题；较短训练、不同任务覆盖与 checkpoint 选择共同影响了最终表现。

| 训练范围 | Airline 入训组 | 不同 Airline task | 说明 |
|---|---:|---:|---|
| expert 到iter29 | 480 | 480 | shuffled/retry |
| progress Mix 的前480个 Airline组 | 480 | 368 | 到Mix iter57，随机放回 |
| expert 到iter59 | 960 | 882 | 完整60 updates |
| progress Mix 的前960个 Airline组 | 960 | 609 | 到Mix iter112 |
| progress Mix 到iter139 | 1211 | 677 | 完整140 updates |

progress Mix 实际入训领域组数为 Airline/Retail/Telecom/Banking=`1211/585/142/302`，Airline 占54.06%，高于原始池比例45.48%。因过滤率和异步完成顺序不同，随机抽样的领域比例不等于训练比例。不能把问题解释成“其他领域抢走了 Airline 总量”，也没有证据支持直接增加 Airline 总配额就能解决。

## 5. 长训练的退化与信号检查

Airline expert 的 pass@1 随 checkpoint 为：iter9 **42.50%**、iter19 **40.00%**、iter29 **53.75%**、iter39 **50.00%**、iter49 **46.25%**、iter59 **43.75%**。不写 DB 类也由iter29的27/28、iter39的28/28下降到iter49的22/28、iter59的24/28。即使不混其他领域，晚期也出现同类退化。

对比的是专家筛选后的早期峰值与 Mix 的晚期 checkpoint，不是“相同训练阶段的专用模型必然更强”。现有 Mix 的iter89/129/139 Airline pass@1 为43.75/46.25/40.00%，也不是持续上升。

训练使用KL系数0。progress 的参考 KL 诊断从前30步均值0.00129增至最后10步0.01918；Vanilla 保留日志中最后10步为0.01715。该指标是跨域批次均值，只能说明模型偏离参考的程度增加，不能单独证明 Airline 的因果来源。expert 前30步/后30步为0.00141/0.00620。

progress 的140步、expert 的60步及现存 Vanilla 的100步日志中，loss/grad norm/KL 均有限。Vanilla 被抢占恢复后的 `run.log` 从update40开始；没有把这100步冒充全140步的训练分域统计。progress 的9688条 Airline 入训轨迹没有 progress-unavailable；125883个 turn 中118774个有非零 progress advantage（94.35%），并非 Airline credit 通道没有工作。

评测中重复相同工具名和参数的轨迹从 expert29 的5/80，增至 progress129/139 的11/15，Vanilla129/139 的19/23；max_steps 分别为0、0/2、2/5。这是行为退化诊断，不等于数值崩溃，也不应设为自动停训硬条件。

## 6. 为什么其他领域仍可能更好

Airline 的“不行动才正确”题占35%，而 Retail 的参考动作不写 DB 题只有4/40（10%）。同类退化在 Retail iter129 也能看到：expert9 在不写 DB 的16条 trial 中成功14条，progress129只有11条；但需要写入的144条 trial 从83条成功升至86条，二者抵消，所以总体 pass@1 正好持平60.625%。progress139 的两类分别为14/16和88/144，总体升至63.75%。这不是 Airline 独有的失败类型，而是它在 Airline 总分中占更大权重。

Telecom 的40道题都包含 ENV_ASSERTION，主要要求把服务/设备状态修复到目标；没有参考 Agent 写动作也不意味着应保持环境不变，因为还可能需要指导 User 操作。不能直接把 Airline 的拒绝/不变状态分类套到 Telecom。Banking 97题中93题的参考动作含写工具，但绝对成功率仍只有约5–6%，目前收益只是少数轨迹。

此外，所选 Retail/Telecom/Banking expert 都是iter9，各自160个入训组；progress Mix 最终对应585/142/302组，训练经历并不匹配。Telecom 在更少本域组数下仍能更好，但采样器与任务组成也变了，不能仅据此证明跨域正迁移。现有结果支持选择 progress Mix，不足以把“其他所有领域均显著超过 expert”当作统一结论。

## 7. 尚未确定的因素与下一步

Airline 只有20题。按任务配对、保留每题4次 trial 的均值差做20000次 bootstrap，progress129−expert29 的pass@1差为−7.50 pp，95%区间约[−17.50,+2.50] pp；Vanilla129为−10.00 pp，区间[−20.00,0.00] pp。iter139 两者区间分别约[−23.75,−5.00]和[−20.00,−1.25] pp。区间没有校正事后挑选 expert checkpoint，也未覆盖训练 seed 不确定性，仅作探索性描述。

下一步最有信息量的是在同初始化、User、过滤及预算下，只对比随机放回和覆盖导向采样，并检查相同 Airline 入训组数的 checkpoint；另做独立的业务类型分层或小KL对照。边界训练题应由独立训练场景补充，不能拿这7道测试题训练后再报提升。只保留全成功组并不能自动产生学习信号：若 outcome相同且状态全程不变，GRPO和progress advantage仍均为0。

跨域梯度干扰、KL约束和采样器谁是主因，需要这些受控对照才能进一步区分。本次没有实施这些改动，也没有重跑评测。

## 8. 来源与复算

原始评测：[SFT][sft-eval]、[expert29][expert-eval]、[progress129][p129]、[progress139][p139]、[Vanilla129][v129]、[Vanilla139][v139]。报告中的具体案例可用 `task_id`、`trial`、`turn_idx` 定位。

训练日志：[progress][train-log]、[Vanilla][vanilla-log]、[expert][expert-log]。入训组由 `tau2_pool_batch` 对齐实际 optimizer update；任务身份与逐轮 credit 来自 `credit.jsonl`，候选组 outcome 来自原始 rollout dump。训练池见 [train.jsonl](../data/train.jsonl)。

[analyze.py](analyze.py) 只读现有文件，标准库即可运行：

```bash
python3 output/experiments/tau2-mix-rl-sft4505-b128-20260925/airline_diagnosis/analyze.py
# 可选：额外顺序扫描三个大型训练轨迹文件，复算过滤前组分布。
python3 output/experiments/tau2-mix-rl-sft4505-b128-20260925/airline_diagnosis/analyze.py --scan-training
```

[policy]: ../../../../../tau2-bench/data/tau2/domains/airline/policy.md
[tools]: ../../../../../tau2-bench/src/tau2/domains/airline/tools.py
[sft-eval]: ../../tau2-sft-areal3-banking-simplified-full-20260920/eval/four-domain-sft-iter4505/trajectories/seed300_20260920_sft_iter4505_four_domain_airline_test_4trials/results.json
[expert-eval]: ../../tau2-domain-experts-sft4505-b128/eval/airline-iter29-four-domain/trajectories/seed300_20260921_sft4505_airline_iter29_airline_test_4trials/results.json
[p129]: ../eval/mix-iter129-four-domain/trajectories/seed300_20260925_mix_iter129_seed300_airline_test_4trials/results.json
[p139]: ../eval/mix-iter139-four-domain/trajectories/seed300_20260925_mix_iter139_seed300_airline_test_4trials/results.json
[v129]: ../eval/vanilla-grpo-iter129-four-domain/trajectories/seed300_20260925_vanilla_grpo_iter129_seed300_airline_test_4trials/results.json
[v139]: ../eval/vanilla-grpo-iter139-four-domain/trajectories/seed300_20260925_vanilla_grpo_iter139_seed300_airline_test_4trials/results.json
[train-log]: ../arms/async/20260925_mix-retry1-train/run.log
[vanilla-log]: ../arms/async/20260925_vanilla-grpo-train/run.log
[expert-log]: ../../tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-airline-train/run.log
[train-dump]: ../arms/async/20260925_mix-retry1-train/trajectories/train.jsonl
[credit-dump]: ../arms/async/20260925_mix-retry1-train/credit.jsonl
