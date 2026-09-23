# Tau2 三领域逐轮并发与异步 GRPO

## 目标与范围

**当前执行阶段：单个8卡任务内串行完成同步100更新、异步100更新，以及两臂iter99官方双seed评测。**
两臂均从同一processed SFT、全新optimizer开始，沿用已验证20步配置。每臂500组／4000条接受轨迹，每10更新保存。
顺序为sync训练→async训练→sync转换及seeds300/301评测→async转换及seeds300/301评测→汇总；常规外部巡检每30分钟一次。
20步及双seed评测已完成；不从iter19续训，不调整GPU分配或最早策略滞后≤1。

在 `/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2` 实现并验证
Tau2 airline、retail、telecom 的异步 GRPO。本文件按最新讨论更新，取代旧计划中
固定领域配额、预先划分采样 batch、按 batch 编号顺序消费的要求。

训练目标是 **Agent**；Trainer 更新 Agent 参数，generator 负责生成 Agent 动作。
**User** 是固定参数的用户模拟器，不参与优化。本任务不加入 Banking、SFT 或 OPD，
不借异步调度同时更换奖励和 advantage 配方。

区分两个层面，不能把它们混为一谈：

- **逐轮并发采样**：不同对话独立推进。某个对话等待 User、工具或环境执行时，
  其他就绪对话可以请求模型；单条对话内部仍严格遵循消息与环境状态的因果顺序。
- **训练与采样异步重叠**：Trainer 更新时，采样侧继续推进其他对话。
  同步对照也必须具备逐轮并发采样，只在训练与采样是否重叠上作对照。

参考 [Granite 4.2](https://huggingface.co/blog/ibm-granite/granite-4-2) 的持续采样、
共享就绪缓冲区、有界策略滞后和 TIS 思路。本任务明确接受换权重时全局暂停，
沿用本项目的 GRPO advantage 配方；不声称完整复现 Granite 的无暂停刷新、
旧 KV cache 复用或 leave-one-out baseline。

## 数据、模型与固定训练配方

- 直接读取共享 JSONL：
  `/mnt/afs/users/fush/projects/ServiceAgent/datasets/tau2/rl/areal_tasks_slime.jsonl`。
  共 1,982 个任务：airline 1,148、retail 563、telecom 271。
  数据说明见同目录的 `README.md`；JSONL 引用的数据库绝对路径须保持可读。
  跳过旧启动脚本中会重新生成或覆盖 prepared data 的步骤。
- 两臂均从同一 processed SFT 和全新 optimizer 开始：
  `checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903`，
  HF tokenizer / generator 初始化使用其中的 `final_hf`。
- 固定 User：`models/Qwen3.6-27B`。上述模型相对路径从
  `/mnt/afs/users/fush/projects/ServiceAgent` 算起。
- K=8：每个 task 生成 8 条独立的完整轨迹，组成一个 GRPO group。
  global batch=40：每次 optimizer 更新使用 5 个完整 group，不能改成一组一更新。
- reward 为官方 binary task success；同一 task 的 8 条完整轨迹按现有 mean/std
  配方计算 advantage，再分配到对应 Agent turn。全错、全对等零方差组保留，
  advantage 为零，仍计入该 batch 的 group 数及轨迹数。
- LR=`2e-6`，KL / entropy 系数均为零，PPO clip=`0.2/0.2`；
  train seed=1234，rollout seed=42。不得通过删掉零方差组改变训练分布或有效预算。
- official-native 协议；Agent 训练 temperature=1、top_p=1、每 turn 最多 1,200 token，
  max_steps=200、max_errors=10。User temperature=0、top_p=1、最多 512 token，
  保持现有 non-thinking 配置和终止协议。
- 沿用 16,384-token 训练长度上限及两次超长重采样，不截掉历史消息。

## 按 task 采样、按就绪 group 组装 batch

1. 从全部训练 task 中逐个随机、有放回抽样，默认对 JSONL 中的任务均匀抽样。
   **取消每更新 `1/2/2` 的领域配额**，不另行强制领域均衡；记录实际逐域抽样和训练数量。
   两臂使用相同数据、抽样分布和随机种子，不要求异步完成顺序逐条复现同步顺序。
2. 每抽到一个 task，启动该 task 的 K=8 条独立对话。以一次 Agent turn 为推理请求单位，
   在不同对话间交错执行；等待 User 或工具的对话不应独占 Agent 生成槽位。
   同一 task 可以在后续再次抽到，每次抽样均使用新的对话与环境。
3. 一个 task 的 8 条完整轨迹都结束后，计算该组 advantage，并将完整 group 放入公共
   就绪池。不能拿任意 8 条跨 task 轨迹凑 group，也不能把 8 个 turn 当成 K=8。
4. Trainer 从就绪池中取得 **5 个完整且满足策略滞后条件的 group**，组成 40 条轨迹的
   batch 后更新。不在采样前绑定 batch 归属；后启动的 task 可以先进入训练，
   不因某个预定 task 或旧 batch 未结束而强制等待。
5. 采样容量释放后就继续抽取新 task，不把“上一 batch 完成或换权重结束”作为唯一补充
   采样的时点。容量同时计算正在生成、已就绪但未消费的 group，以及尚未完成训练的
   已取出 group；初始上限为 10 个 group / 80 条对话，避免无限预取。
   具体何时允许补充采样还须服从策略滞后限制。
6. **全错 task 始终保留再次采样机会**，不因全错从任务池移除。
   有放回抽样即可满足该要求，不额外提高全错 task 的抽样权重，也不强制立即重试。
   全错 group 本次的 advantage 为零，在零 KL / entropy 系数下不会产生有效策略梯度；
   未来再次采样可能出现组内 reward 差异，从而获得学习信号。
7. 超长等不可训练轨迹沿用既有重采样规则。若最终导致 group 不可用，则重新随机抽取
   task 补充，不再按旧领域配额替换；额外采样和重试全部计入耗时。
   不能仅因任务慢而丢弃；已完成但未消费的 group 保留。

## 权重边界、概率校正与恢复

每个 Agent turn 记录完整 `prompt_token_ids`、实际 `output_token_ids`、对应行为
logprob、策略版本及结束原因。直接向 SGLang 发送 token IDs 并启用 `return_logprob`。
工具仍按 official-native 协议解析执行，训练目标使用原始生成 token，包括实际采样到的
终止 token；User、工具及框架补入文本不承担 loss。

只有实际 token 前缀完全一致的相邻 turn 才能合并训练片段；否则保留独立完整上下文。
长度检查覆盖完整对话与实际训练片段，不将重复前缀累计后当作轨迹长度。
先按完整轨迹计算组内 advantage，再扩展片段；使用 `rollout_id`、
`rollout_mask_sums` 保持一条轨迹的总 loss 权重，统计按轨迹去重。

采用解耦的 PPO/TIS：`r = πθ / πprox`，`w = min(2, πprox / μ)`。
其中 μ 是每个 token 实际生成时的行为策略，πprox 是本次更新前的 Trainer。
用停止梯度的 w 乘现有 PPO clipped loss，再按原轨迹归一化方式聚合。
两臂均开启 `--use-tis --tis-clip 2 --tis-clip-low 0`，
不同时开启互斥的 `--use-rollout-logprobs`。

允许每次 optimizer 更新后全局暂停新的 Agent 请求，等待已发出的 Agent turn 结束，
通过现有分布式权重同步更新所有 Agent generator、清空旧 KV cache 后恢复。
User 和工具执行可继续。单个 Agent turn 使用固定策略版本，同一对话的相邻 turn
可以跨版本；不要求对话整体结束后才换权重。

保持**最早 Agent turn 在进入训练时最多落后一次更新**，不能改用最后一个 turn 的版本。
取消 batch 顺序约束不等于取消这个限制：就绪池选择及 Trainer 推进需要考虑未完成组的
最早版本，优先处理即将到达滞后上限的组，必要时等待或限制新采样，不能不断消费快组
直到慢组过期。不得为了避免等待而丢弃慢任务、放宽滞后，或给旧轨迹重新标记版本。
逐轮并发并不承诺 Trainer 永不等待；没有 5 个可用完整组、换权重、容量或滞后限制
都可能产生等待，应分别记录原因。

checkpoint 保存 Trainer、optimizer、更新计数，以及 datasource 的抽样 RNG / 游标和
未消费 group 输入描述。恢复时使用恢复后的策略重新生成未消费组，不保存活环境或 KV
cache；已训练 group 不重复消费，未消费 task 不静默跳过，不要求逐 token 重现。
末尾只补充完成剩余更新预算所需的组，消费完成后正常关闭采样池。
权重更新失败或行为概率缺失时停止运行。

## 8 卡资源与公平对照

单个模型实例支持多个并发请求，**多副本不是异步的必要条件**。
TP2 表示一个实例由两张 GPU 共同执行，不是两个模型副本。
等待 User / 工具应只阻塞该条对话的下一轮，不能阻塞其他就绪对话。

初始部署沿用已验证配置，两臂一致：

| 角色 | GPU | 部署与并发 |
|---|---|---|
| Trainer | 0–1 | TP2 |
| Agent generator | 2–5 | 两个 TP2 实例，每实例最多 16 个请求 |
| 固定 User | 6–7 | 一个 TP2 实例，最多 32 个请求 |

显式设置对话线程池容量，复用 HTTP 连接和只读 tokenizer。不能将“多部署几个模型”
等同于“实现异步”，也不能把 User 排队增加误判为吞吐提升。
依据实测检查 Agent / User 请求、服务排队、工具执行、就绪池等待、Trainer、权重同步、
checkpoint 和 GPU 利用率；如优化服务设置，两臂共同采用，并在正式比较前冻结。
不扩大到 8 卡之外，也不擅自放宽策略滞后。

同步臂同样使用随机 task 抽样、逐轮并发、K=8、就绪池组 batch、原始 token 和 TIS。
同步臂在采样与训练之间保留阶段屏障，不与 optimizer 更新重叠；异步臂让其他对话在
训练期间继续推进。两臂使用相同每步训练量、采样容量上限及服务配置。
分别验证逐轮并发的基础收益与训练/采样重叠的增量收益，不预设 8 卡必然值得做后者。

## 验证、实验与验收

CPU 测试设置 `NUM_GPUS=0` 并按技能登记 CI；运行现有 rollout preflight 及受影响的
调度、loss 测试。覆盖对话/重试环境隔离、原始 token/logprob 对齐、JSON 重排、终止
token、超长处理、不等 turn 数的 K=8 归一化、零方差组和片段合并的 loss/gradient 等价性。
新增调度覆盖随机有放回抽样、全错 task 再次入选、后启动组先完成、5 个就绪组动态成批、
容量释放后补任务、全局换权重边界、最早版本滞后、尾部预算和未消费组恢复。

使用正式模型运行至少 3 更新的 GPU smoke，并实际验证跨 turn 换权重、训练/采样重叠
及 checkpoint 恢复。同步、异步做同资源短测，保留旧版同步短测用于解释 token 修正开销；
轨迹长度和替换次数不同会影响耗时，不能将所有差异都归因于实现。
新的就绪池调度必须重新验证，不能直接沿用旧顺序 batch 实现的加速结论。

正式两臂从同一 SFT、全新 optimizer 开始，各训练 100 更新，即各接受 500 个 group、
4,000 条轨迹；每10更新保存，最终比较iter99。两臂在同一个8卡计算任务中串行运行，使用同一节点、GPU型号、CPU和内存，分别记录阶段区间。
端到端训练计入 rollout、训练、概率校正、权重同步、checkpoint、
重试和替换，排除环境安装、模型转换和官方评测；同时记录初始化及循环耗时，明确计时边界。

iter99 使用官方 binary evaluator：100 个测试任务，每任务 4 trials，seeds 300/301，
固定同一 User，Agent temperature=0.6，保持原有终止协议。报告 pass@1、pass@4(any)、
pass^4、逐域结果、action/DB accuracy、配对置信区间，以及 KL、TIS 裁剪率/有效样本量、
策略滞后、截断、零方差组率和完整耗时。KL 系数为零时仍需报告真实的参考策略漂移，
不能用恒零的 KL reward 项代替。

保留原定正式验收目标：异步端到端耗时不超过新同步的一半；三项成功率相对新同步及
历史 processed RL 分别下降不超过 5 个百分点。历史 RL 为
`50.75 / 80.00 / 22.50%`，同时报告距 processed SFT
`55.75 / 81.00 / 28.00%` 的差距。历史双 seed 逐任务评测可用于配对比较，位置见实验 README。
只有速度和效果共同达标才判定实验成功；置信区间不足以支持统计非劣时明确说明。
2 倍是待检验目标，不是已保证的收益；普通行为诊断不增加停止门槛。

## 当前状态、工作区与交付

已完成原始token/TIS、随机任务公共就绪池、逐轮并发、换权重暂停和checkpoint恢复。
20更新同步／异步端到端3614／3408秒；双seed iter19评测54.00/79.50/24.00%与57.875/84.00/29.50%。
本阶段新增串行编排、阶段进程清理、独立日志及自动最终汇总。旧固定配额／顺序batch和失败作业只作历史诊断。

只修改本 worktree，共享 datasets、models、已有 checkpoint 作为只读输入。
外部 `tau2-bench/` 为共享代码，若确需修改，须在独立工作树处理，避免影响其他任务。
**按用户要求，修改代码和实验过程中先不做 Git 管理；实验完成后再统一整理提交到
`feature/async-tau2`，不合并 main、不推送。**

所有新 checkpoint、转换和评测输出放在
`output/experiments/tau2-areal-async-rl/` 的独立运行目录，旧结果保留作诊断。
启动脚本显式使用本 worktree 的 `PROJECT_ROOT` 和提交器 `--project-root`：

```bash
bash scripts/submit.sh --experiment tau2-areal-async-rl --gpus 8 --cpus 64 --memory 1024 \
  --project-root /mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2 \
  -e PROJECT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2 \
  <本任务运行脚本>
```

提交前检查 quota/queue，使用 normal priority，不停止其他任务。
每次提交维护实验 README、相对 run-log 链接和 `output/doc/INDEX.md`。
遵循本工作树 `AGENTS.md` 和相关技能，文档保持简洁且不超过 10 个标题。
交付新调度代码、必要测试、启动命令和实际实验结果，明确未验证项及不达标项。
