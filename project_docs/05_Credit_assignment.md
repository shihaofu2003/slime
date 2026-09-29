# ServiceAgent 第五项：Credit assignment

当前最终项目主线采用本文的 **2026-09-25 Mix RL iter139** 作为 OPD 学生初始化和非 Airline 教师。同预算算法对照取 iter139；iter129 保留作 checkpoint 观察。后续双教师 OPD 最终选择 iter59，见 [核心项目导读](README.md) 与 [最终实验](../output/experiments/tau2-opd-mix139-airline-20260929/README.md)。

本文重点说明 `progress-db-count-v1`，以及同一 SFT4505 初始化下的四领域 Mix RL、Vanilla GRPO 和 domain experts 对照。资料核对日期：2026-09-29。历史 turn-credit 版本不再展开；后续 goal-progress 配方也不与本算法混用。

以总体 pass@1 为主要指标，`progress-db-count-v1 Mix RL iter129` 是下表最好的单模型，达到31.60%。同预算 iter139 的 progress Mix 在三项总体指标上均高于 Vanilla，但 iter129 并非如此。现有结果支持将 progress Mix 作为后续实验基线，不代表所有 checkpoint、领域和指标均领先。

## 1. 要解决的问题

[第四项](04_异步Agentic_RL.md)处理异步采样与完整轨迹训练；这里处理多轮工具调用中的训练信号。官方评测仍使用任务的二元成功结果，训练中的 partial progress 不算任务成功。

普通 GRPO 在同一任务的 K 条 rollout 内标准化 outcome，再将一个轨迹级 advantage 广播到全部可训 Assistant token。失败轨迹里的正确查询、有效写入和最后一次错误操作因此共享训练方向；同组全部成功或全部失败时，outcome advantage 为零。

`progress-db-count-v1` 保留 outcome 项，同时计算环境状态距离目标的变化，把后续进度归到相应 Assistant turn。反馈来自环境和参考任务，不依赖 LLM judge 逐轮评分。工具名称正确本身不产生 progress 奖励。

## 2. progress-db-count-v1 算法

DB 任务从相同初态重放参考动作，构造目标状态；读取 Agent 侧及存在时的 User 侧 DB。定义势函数：

$$
\Phi(s)=-d(s,s^*).
$$

`d` 是递归字段差异数：字典展开，list 按整体值比较。ENV_ASSERTION 任务改用当前满足的断言比例。只有 ACTION/COMMUNICATE 等、无法构造势函数的任务没有 progress 项；同组任一轨迹的 progress 不可用时，该组回退到 outcome/format。[状态实现][progress]

每次 Agent 决策前取状态，末轮后再取终局状态。对第 t 个可训 Assistant turn：

$$
r^{\mathrm{prog}}_{i,t}=\Phi(s_{i,t+1})-\Phi(s_{i,t}),\qquad
H_{i,t}=\sum_{u\ge t}\gamma^{u-t}r^{\mathrm{prog}}_{i,u},\qquad\gamma=0.98.
$$

间隔包括 Assistant 输出后的工具执行和 User 响应，所以指导 User 操作也能得到反馈。这是决策间隔的观察归因，不是反事实因果估计。实现是“势函数差，再折扣累积”，不是 `gamma*Phi(next)−Phi(now)`，不据此声称最优策略保持不变。

DB-count 决定比较对象。记 `c(i,t)` 为决策前 DB 相对初态发生变化的记录数；增加、删除、修改均计入，恢复初值后不再计入。Agent 侧按业务记录，Banking 两侧按 `table.data` 记录，其他领域的 User 侧按展开字段计数。它不是累计写调用次数，也不是到目标的距离。

同一任务 K 条轨迹内，所有 count 相同的 turn 构成桶：

$$
\mathcal B_c=\{(i,t):c(i,t)=c\},\qquad
Q_{i,t}=\frac{H_{i,t}-\operatorname{mean}_{\mathcal B_c}H}{\operatorname{std}_{\mathcal B_c}H+10^{-6}}.
$$

采用样本标准差，单元素或近零方差桶返回0。桶可包含同一轨迹的多个 turn，不要求 turn 序号相同。例如两条轨迹都尚未改变 DB，即使处于不同轮次，也比较各自后续取得的进度。同样的 count 不保证业务状态相同，长轨迹也会提供更多比较点。

最终 advantage 为：

$$
B_i=\frac{R_i-\overline R}{\operatorname{std}(R)+10^{-6}},\qquad
A_{i,t}=B_i+w_{\mathrm{prog}}Q_{i,t}+w_{\mathrm{fmt}}F_{i,t}.
$$

正式配方使用 `w_prog=1、w_fmt=1`。本轮出现 malformed JSON、nonexistent tool 或 wrong namespace 时 `F=-0.1`，否则为0；同轮不重复累加。该版本不直接叠加旧版 execution error、repetition 或参数不匹配 penalty。[advantage 合成][credit]

状态边界通过 simulation 消息索引关联到真实生成的 Assistant token span，将同一 turn advantage 写入其可训 token。System、User、工具返回的 loss mask 为0。组内计算在数据并行拆分前完成，再由 Trainer 消费，不在各卡重新标准化。本算法没有旧版固定零和预算，也没有按 turn 长度做逆比例补偿；实际贡献还受 token 数、loss reducer、importance weighting 和 clipping 影响。[rollout.py][rollout]、[loss.py][loss]

## 3. 本次实验协议

| 项目 | progress Mix / Vanilla Mix | 4 domain experts |
|---|---|---|
| 初始化 | Qwen3-4B-Instruct-2507 SFT iter4505，fresh optimizer | 相同初始化，每域独立 fresh optimizer |
| 数据 | Airline1148、Retail563、Telecom271、Banking542，共2524 tasks | 相同数据池的领域子集 |
| 采样 | `RandomTaskDataSource`，均匀随机放回 | `ShuffledTaskDataSource`，打乱任务牌堆；all-zero task 延迟2次 policy update 重试 |
| 预算 | 各140 updates，每10步保存 checkpoint | Airline60、Retail30、Telecom20、Banking30 updates |
| 共同配置 | LR2e-6；16 groups × K8 = 128 trajectories/update；KL/entropy系数0；train/rollout seed1234/42 | 相同 |
| 异步配置 | pool32、pending64、unlimited policy lag；4环境worker；8 GPUs（Trainer2 + Generator6） | 相同 |
| User | Qwen3.6-27B nonthinking，temperature0，max_tokens512 | 相同模型与请求参数，不同部署实例 |
| rollout 限制 | max_steps200，Agent max_tokens1200，训练上下文cap16384，超长最多重试2次 | 相同 |

三者均打开 `TAU2_DROP_UNIFORM_OUTCOME_GROUPS=1`：完整 K8 组的 outcome 全0或全1时丢弃并补抽，超长等无效组也移除。虽然本算法能在某些全失败组产生 progress 差异，这批实验没有保留这些组训练。[过滤][filter]、[采样器][sampler]

Vanilla 关闭 custom reward postprocess / advantage hooks 和 reward shaping，只使用官方二元 outcome 的标准 GRPO，不包含 progress 或局部 format penalty。因此对照检验的是整套 credit 配方，没有单独分离 DB-count 分桶或 format 项的贡献。[Mix 记录][mix]、[专家记录][experts]

官方评测固定 seed300，Airline/Retail/Telecom/Banking 分别20/40/40/97个 test tasks，每题4 trials，共788条。Agent temperature0.6、top_p1、max_tokens1200，max_steps200、max_errors10，Banking BM25，使用相同 Qwen3.6-27B User。

`pass@1` 为 trial 成功率，`pass@4(any)` 为每题四次至少成功一次，`pass^4` 为每题四次全部成功。Overall 按197个任务加权，不是四领域宏平均。Airline 只有20题，单条轨迹对应1.25 pp；单题的 pass@4(any)/pass^4 对应5 pp。

## 4. 官方结果

每格依次为 **pass@1 / pass@4(any) / pass^4，单位 %**。数值已与原始 summary 核对。

| 方案 | Iter | Airline | Retail | Telecom | Banking | Overall |
|---|---:|---|---|---|---|---|
| progress-db-count-v1 Mix RL | 129 | 46.25 / 60.00 / 30.00 | 60.63 / 90.00 / 32.50 | 56.88 / 82.50 / 22.50 | 6.19 / 11.34 / 0.00 | 31.60 / 46.70 / 14.21 |
| Vanilla GRPO Mix RL | 129 | 43.75 / 75.00 / 15.00 | 64.38 / 85.00 / 37.50 | 56.88 / 82.50 / 30.00 | 4.90 / 13.40 / 0.00 | 31.47 / 48.22 / 15.23 |
| progress-db-count-v1 Mix RL | 139 | 40.00 / 70.00 / 15.00 | 63.75 / 85.00 / 40.00 | 56.25 / 87.50 / 20.00 | 5.67 / 13.40 / 0.00 | 31.22 / 48.73 / 13.71 |
| Vanilla GRPO Mix RL | 139 | 42.50 / 60.00 / 25.00 | 61.25 / 87.50 / 37.50 | 51.88 / 85.00 / 15.00 | 5.41 / 10.31 / 0.00 | 29.95 / 46.19 / 13.20 |
| progress-db-count-v1 4 Experts RL* | — | 53.75 / 75.00 / 35.00 | 60.63 / 87.50 / 30.00 | 51.88 / 90.00 / 15.00 | 4.64 / 11.34 / 0.00 | 30.58 / 49.24 / 12.69 |

*4 Experts 是分别选择 checkpoint 后按领域路由的组合：Airline iter29、Retail iter9、Telecom iter9、Banking iter9。它不是一个通用模型，也没有统一的 iter129/iter139。Airline 完整训练60步，表中选的是30步 checkpoint。Retail iter29 为60.63/85.00/32.50%，不是这里使用的 Retail baseline。*

来源：[progress129][p129]、[progress139][p139]、[Vanilla129][v129]、[Vanilla139][v139]；专家：[Airline29][ea]、[Retail9][er]、[Telecom9][et]、[Banking9][eb]。Vanilla129 的 Banking 有1条基础设施错误，按失败计入；这五个被比较 checkpoint 的 Airline 基础设施错误均为0。

## 5. 如何解读结果

按总体 pass@1，progress Mix iter129 的31.60%高于 Vanilla iter129 的31.47%和4 Experts 的30.58%。但它只比 Vanilla 多成功1条轨迹（249/788 对248/788），不能把该差距写成充分的显著性证据。

同为最终 iter139，progress 相比 Vanilla 三项总体指标提高 **+1.27 / +2.54 / +0.51 pp**，对应多10条成功轨迹、多5道至少成功一次的题、多1道四次全成功的题。结果支持新 credit 配方的有效性；目前只有一个训练 run 和一个评测 seed，尚不能声称统计显著或跨随机种子稳定领先。

总体 pass@4(any) 最高的是4 Experts 的49.24%；只比较 Mix 时，最高的是 progress iter139 的48.73%。总体 pass^4 最高的是 Vanilla iter129 的15.23%。应固定 checkpoint 和主要指标后比较，不把不同迭代的最好指标拼成一个模型成绩。

progress iter129 相对对应 expert 的分域 pass@1 为 Airline −7.50 pp、Retail 持平、Telecom +5.00 pp、Banking +1.55 pp；iter139 为 −13.75/+3.13/+4.38/+1.03 pp。“大部分领域更好”主要成立于 pass@1，并非各域覆盖率和四次全成功率都更好；Banking 的 pass^4 仍为0。

Mix 与 experts 还改变了采样器、实际入训领域比例和更新日程，且 expert 选用早期 checkpoint。比较说明了当前模型选择结果，不能单独证明跨域正迁移或负迁移。Airline expert 自身也从 iter29 的53.75%回落到 iter59 的43.75%，需要结合入训任务和逐题轨迹解释差距。

## 6. Airline 为什么落后于选出的 expert

逐题检查发现，主要差距来自“不应修改 DB”的业务边界。7道此类题共28条 trial，SFT4505与expert29均成功27条，progress129/139降至23/20条，Vanilla129/139为21/22条。其余13道需要写入的题中，SFT成功7/52，expert29为16/52，两个Mix iter129均为14/52。Mix改善了执行能力，却损失了部分原有的拒绝/只读能力。

具体错误包括：basic economy不允许改签时仍修改航班；采信用户“10小时前订票”而忽略DB时间并取消订单；取消已经飞过的business航班；把Gold+Economy的3件免费行李错误计成2件。

训练池中只有53/1148个Airline任务的参考动作不写DB，其中31个围绕同一订单；测试中这类题却占7/20。progress Mix实际入训的此类组只有43/1211。Airline总量并不少：占全部入训组54.06%；但随机放回的前480个Airline组只覆盖368个任务，expert前480组覆盖480个。

这些证据更支持“业务类型覆盖不足，加上较长RL训练中边界能力退化”的解释，而非简单的Airline配额不足。专家晚期同样下降，两种Mix又同时存在问题，因此不能单独归因于credit算法或宣称已证明跨域负迁移。一个实际训练样例中，偏离参考目标的额外乘客修改获得了−5.10的turn advantage，credit方向正确。

完整的task/trial案例、训练过滤计数、checkpoint变化、比较限制及复算脚本见[Airline诊断报告](../output/experiments/tau2-mix-rl-sft4505-b128-20260925/airline_diagnosis/README.md)。

## 7. 实现和实验入口

| 内容 | 入口 |
|---|---|
| 目标状态、势函数、discounted return、DB-count 分桶 | [progress.py][progress] |
| outcome/progress/format 合成和 token span 写入 | [reward_postprocess.py][credit] |
| 过滤与采样 | [filters.py][filter]、[random_tasks.py][sampler] |
| 完成组入队、每批领域计数 | [continuous.py][continuous] |
| 两个 Mix 的训练、评测及原始日志 | [Mix README][mix] |
| 四个 experts 的训练与各 checkpoint 评测 | [Experts README][experts] |

[progress]: ../examples/tau2-bench/rl/progress.py
[credit]: ../examples/tau2-bench/rl/reward_postprocess.py
[rollout]: ../examples/tau2-bench/rl/rollout.py
[loss]: ../backends/megatron_utils/loss.py
[filter]: ../examples/tau2-bench/rl/filters.py
[sampler]: ../examples/tau2-bench/rl/random_tasks.py
[continuous]: ../examples/tau2-bench/rl/continuous.py
[mix]: ../output/experiments/tau2-mix-rl-sft4505-b128-20260925/README.md
[experts]: ../output/experiments/tau2-domain-experts-sft4505-b128/README.md
[p129]: ../output/experiments/tau2-mix-rl-sft4505-b128-20260925/eval/mix-iter129-four-domain/seed300_20260925_mix_iter129_seed300_summary.json
[p139]: ../output/experiments/tau2-mix-rl-sft4505-b128-20260925/eval/mix-iter139-four-domain/seed300_20260925_mix_iter139_seed300_summary.json
[v129]: ../output/experiments/tau2-mix-rl-sft4505-b128-20260925/eval/vanilla-grpo-iter129-four-domain/seed300_20260925_vanilla_grpo_iter129_seed300_summary.json
[v139]: ../output/experiments/tau2-mix-rl-sft4505-b128-20260925/eval/vanilla-grpo-iter139-four-domain/seed300_20260925_vanilla_grpo_iter139_seed300_summary.json
[ea]: ../output/experiments/tau2-domain-experts-sft4505-b128/eval/airline-iter29-four-domain/seed300_20260921_sft4505_airline_iter29_summary.json
[er]: ../output/experiments/tau2-domain-experts-sft4505-b128/eval/retail-iter9-four-domain/seed300_20260921_sft4505_retail_iter9_summary.json
[et]: ../output/experiments/tau2-domain-experts-sft4505-b128/eval/telecom-iter9-four-domain/seed300_20260921_sft4505_telecom_iter9_summary.json
[eb]: ../output/experiments/tau2-domain-experts-sft4505-b128/eval/banking-iter9-four-domain/seed300_20260921_sft4505_banking_iter9_summary.json
