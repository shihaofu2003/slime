# tau2-retail-expert-sft4505-goal-v4-20260928

Purpose：使用v4任务一致性与goal-progress配方及新版credit assignment，从SFT4505训练Retail单领域expert。训练采用 shuffled task deck 保证 domain coverage，保留当前 credit-aware group filter；原代码、混合训练配置和历史实验保留。

## 数据与配方

训练集 [data/retail_train.jsonl](data/retail_train.jsonl)仅含563个不同Retail任务，从[已验证的v4数据](../tau2-mix-rl-sft4505-goal-v4-20260928/data/train.jsonl)按domain筛选，行内容不再改写：522个任务需求已与DB/评价目标对齐，22个转人工、14个无业务写入、5个比较选择任务保留原文。[数据统计](data/summary.json)。

使用 `train_tau2_goal_v4.py`、`TAU2_V4_ARM=goal_progress`，在新Ray worker中启用v3环境修复、列表字段进度、按目标阶段的轨迹间归一化和credit-signal筛选。方案与验证证据见[v4说明](../tau2-mix-rl-sft4505-goal-v4-20260928/README.md)。

## 运行配置

| 项目 | 配置 |
|---|---|
| 初始化 | `Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920/iter_0004505_hf`，新优化器/RNG，rollout0 |
| 训练资源 | 8 GPU：2卡trainer + 6卡rollout（三个TP2副本） |
| User | 4 GPU，Qwen3.6-27B nonthinking，两个TP2副本；temperature0 |
| 训练长度 | 30 updates，每10步保存，最终iter29 |
| Batch | 16任务组 × K8 = 128轨迹 |
| 优化 | LR2e-6，KL/entropy0，progress/format权重1，gamma0.98 |
| 采样 | Retail-only文件；`ShuffledTaskDataSource` shuffled deck，train/rollout seed1234/42；pool32、pending64、max_policy_lag=-1 |
| 并发与长度 | 4环境worker，Agent/step并发48/160，max_steps200，Agent每轮1200 tokens；16K训练cap、超长重采样2次 |
| 优先级 | normal |

[run_user.sh](run_user.sh)启动常驻User并发布本实验 `user_endpoint.env`。[run_expert.sh](run_expert.sh)先执行23项组过滤检查、16项v4检查及原有启动检查，等待User接口就绪后训练。单领域由独立数据文件保证，通用底层脚本清空domain环境变量不影响该数据范围。

训练产物：`arms/async/20260928_retail-expert-goal-v4-train/`；奖励图：`plots/20260928_retail-expert-goal-v4-train/`。本实验的训练曲线不是官方held-out评测。

## 作业

| 任务 | Job | GPU | 状态与运行日志 |
|---|---|---:|---|
| User常驻服务 | 24813 (`pt-4639nbms`) | 4 | 2026-09-28 15:05:04 CST提交，RUNNING，正在加载模型；normal priority；[运行日志](jobs/24813-tau2-retail-goal-v4-user-persistent-0928-150504366/run_0_20260928_150504366.log) |
| Retail expert（旧配置，已停止） | 24814 (`pt-zzdkhjw2`) | 8 | 2026-09-28 15:05:32 CST提交，因采样协议不符已停止；140 updates/RandomTaskDataSource；[运行日志](jobs/24814-tau2-retail-goal-v4-expert-sft4505-0928-150532494/run_0_20260928_150532494.log) |
| Retail expert（shuffled30，新配置） | 24828 (`pt-8h1psh0m`) | 8 | SUCCEEDED，30 updates/`ShuffledTaskDataSource`，最终iter29；[运行日志](jobs/24828-tau2-retail-goal-v4-shuffled30-expert-0928-0928-160147811/run_0_20260928_160147811.log) |
| Retail iter29 official eval（首提交，参数冲突失败） | 24881 (`pt-59577ngm`) | 8 | exit_code=2：单域 retail 不适用 `borrow_completed_domain_slots`；[评测脚本](eval_retail_shuffled30_iter29.sh)，[运行日志](jobs/24881-retail-goal-v4-shuffled30-iter29-eval-0928-181121025/run_*_20260928_181121025.log) |
| Retail iter29 official eval（修复后重提） | 24885 (`pt-h1dqhg2k`) | 8 | 已重新提交，删除单域冲突参数，其余评测参数不变；[评测脚本](eval_retail_shuffled30_iter29.sh)，[运行日志](jobs/24885-retail-goal-v4-shuffled30-iter29-eval-rerun-0928-182520058/run_*_20260928_182520058.log) |
| 三域 iter29 official eval | 24887 (`pt-zj9km0l2`) | 8 | `airline,retail,telecom` 全量 test×4 trials，seed300，400 simulations，SUCCEEDED；overall pass@1/pass@4(any)/pass^4=`51.00/83.00/22.00%`；[评测脚本](eval_retail_shuffled30_iter29.sh)，[运行日志](jobs/24887-three-domain-expert-shuffled30-iter29-eval-0928-184014819/run_*_20260928_184014819.log) |
| 三域 iter9 official eval | 24894 (`pt-ox024h2f`) | 8 | `airline,retail,telecom` 全量 test×4 trials，seed300，当前 `SUBMITTED`；[评测脚本](eval_three_domain_iter9.sh)，[运行日志](jobs/24894-three-domain-expert-iter9-eval-0928-193944419/run_*_20260928_193944419.log) |
| 三域 iter19 official eval | 24895 (`pt-w2jhxu49`) | 8 | `airline,retail,telecom` 全量 test×4 trials，seed300，400 simulations，SUCCEEDED；overall pass@1/pass@4(any)/pass^4=`53.50/83.00/25.00%`；[评测脚本](eval_three_domain_iter19.sh)，[运行日志](jobs/24895-three-domain-expert-iter19-eval-0928-193948435/run_*_20260928_193948435.log) |
| 三域 iter9 official eval | 24894 (`pt-ox024h2f`) | 8 | `airline,retail,telecom` 全量 test×4 trials，seed300，400 simulations，SUCCEEDED；overall pass@1/pass@4(any)/pass^4=`48.00/76.00/25.00%`；[评测脚本](eval_three_domain_iter9.sh)，[运行日志](jobs/24894-three-domain-expert-iter9-eval-0928-193944419/run_*_20260928_193944419.log) |
