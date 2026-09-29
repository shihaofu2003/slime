# tau2-mix-rl-sft4505-reward-v3-20260927

Purpose：从 SFT4505 重新训练四领域 mix RL，使用共同的数值参数规范化和 JSON ACTION 比较，修复等价表示导致的奖励误判。v3 为独立入口；原代码、v1/v2 实现和旧实验保留。

[训练完成后的四领域诊断](domain_diagnosis_20260928/README.md)：Airline首末完整窗口+12.35 pp；Banking高均值掩盖L1证据引用19.08%的弱项及零信号过滤；Retail商品修改任务进步不足，并确认用户付款选择与固定参考目标冲突的实例。完整统计、重复任务校正趋势及原始对话见报告。

## 修复范围

- [reward_v3.py](../../../examples/tau2-bench/rl/reward_v3.py)按工具声明的 int/float（含 Optional）规范化 21 个实际工具方法的等值数值，覆盖直接调用和 Banking 嵌套工具调用；整数参数只接受等值转换，不将小数截断为整数，也不强转字符串和布尔值。
- 对明确声明为 JSON 编码参数的三个 Banking 工具入口，ACTION 比较解析后的结构，忽略 JSON 空格、对象键顺序和等值数值格式；保留原 compare_args 选择、字符串 ID、列表顺序、重复项及真实数值差异。格式错误的 JSON 按动作不匹配处理。
- 复用 v2 的参考状态逐动作同步、结构化 DB 比较、Telecom 充值文本评分快照和 Retail 多商品对应关系修复。reward 与 progress 共用参考构造及评分快照，各任务 reward_basis 保持原定义。
- [train_tau2_reward_v3.py](../../../train_tau2_reward_v3.py)在新 Ray worker 启动时安装修复，继续使用 credit-signal-v1 过滤和既有 advantage/TIS/token-mask 路径。

工具方法仍暴露原有 schema。数值相关工具的实际返回文本和交易 ID 可能与旧版本不同；本次从新环境生成轨迹，历史轨迹保留原版本复现。历史官方评测脚本与结果未改，尚无 v3 官方提分结论。

## 验证

[26 项 CPU 回归测试](../../../examples/tau2-bench/rl/test_reward_v3.py)通过，覆盖四领域 reward/progress、真实金额差异、JSON 解析边界、交易 ID、Retail 多商品/重复项、提前终止、ACTION/COMMUNICATE 约束、全部数值工具 schema 和真实 Ray 父子 worker 接入。测试由本实验训练启动脚本显式执行，未加入固定 CI 矩阵，以保留旧文件。

已单独复核 v2 新发现的全部 33 个误判任务：54 个等价变换均保持 DB、progress 和 reward 一致，reward 均为 1。[定向复核结果](banking_regression_verification.json)。

[全任务验证脚本](verify_semantic_rewards.py)已在 User 作业中以 8 个 CPU 进程完成：Airline1,148、Telecom271、Retail563、Banking542，共 2,524 个任务的参考动作 reward 均为 1。906 组等价变换全部通过：直接数值参数608、JSON 排版209、嵌套 JSON 数值89；DB、各动作后的 progress 和最终 reward 均保持一致，ACTION 全部匹配。无可用 progress 的任务保持原来的不可用状态。耗时409.3秒，失败0；[逐任务结果](semantic_verification.jsonl)、[汇总](semantic_verification_summary.json)。

此前 v2 的 [298 组等价输入检查](../tau2-mix-rl-sft4505-reward-v2-20260927/SEMANTIC_EQUIVALENCE.md)定位了 33 个不同任务的评分误判；此数不是 rollout 错误率或预期训练收益。正确性通过也不保证奖励曲线单调上升。

## 训练配置

4-GPU User：Qwen3.6-27B nonthinking，两个 TP2 副本。8-GPU 训练：2 卡训练、6 卡 rollout（三个 TP2 副本）。SFT 初始化为 `Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920/iter_0004505_hf`，新优化器，140 updates，16 任务组 × K8，batch128，LR2e-6，KL/entropy0，progress/format1/1，gamma0.98，seeds1234/42。沿用随机任务池、pool32、pending64、max_policy_lag=-1、四个环境 worker、max_steps200、16K 训练 token cap。

[run_user.sh](run_user.sh)先完成 CPU 全任务回放，再启动常驻 User；[run_mix.sh](run_mix.sh)运行回归检查并等待本实验 `user_endpoint.env` 就绪后训练。所有作业使用 normal priority。

## 作业

| 任务 | Job | GPU | 状态与日志 |
|---|---|---:|---|
| 首次全任务检查 | 24660 (`pt-1b71xckd`) | 4 | FAILED：检查脚本未兼容无参考动作时的空 ACTION 结果；已修复并单独通过全部 136 个无参考动作任务。此前完成 720 个任务，无等价性失败；[运行日志](jobs/24660-tau2-reward-v3-user-persistent-0927-190755545/run_0_20260927_190755545.log) |
| 全任务 CPU 回放 + 常驻 User 重试 | 24664 (`pt-26nkrc2s`) | 4 | RUNNING，完整 CPU 回放通过，两个 TP2 User 副本及 router 已就绪；normal priority；[运行日志](jobs/24664-tau2-reward-v3-user-persistent-retry-0927-191444693/run_0_20260927_191444693.log) |
| SFT4505 起点 mix RL | 24668 (`pt-kt8cehks`) | 8 | 2026-09-27 19:26 CST 提交，现已 SUCCEEDED，140 updates 完成并保存 iter139；23 项过滤检查、26项 v3检查及原有 rollout/progress/continuous 检查全部通过，已连接 User并进入模型初始化；normal priority；[运行日志](jobs/24668-tau2-reward-v3-mix-sft4505-0927-192610879/run_0_20260927_192610879.log) |

User API：`http://10.119.98.94:30000/v1`，`/models` 已验证返回 `Qwen3.6-27B-tau2-user-nonthinking`；训练日志已记录 `MIX_RL_USER_READY` 和 `MIX_RL_START recipe=reward-v3-credit-signal-v1`。服务日志位于 `../tau2-external-user-pool/service/20260927_112321/`。

训练实际参数已核对：`load` 指向 SFT 根目录，`ckpt_step=4505`，HF 为 `iter_0004505_hf`，`finetune=True`，`no_load_optim=True`，`no_load_rng=True`，`start_rollout_id=0`，`num_rollout=140`，`global_batch_size=128`，`lr=2e-6`。本次为 SFT 新起点；尚未据此报告优化器更新或模型能力收益。


## 最终 checkpoint 四域评测

[评测脚本](eval_iter139.sh)使用训练 24668 的最终 `iter_0000139`，先转换到相邻 `iter_0000139_hf`。复制评测 24720 的实际提交脚本，仅替换模型、实验和输出标识；shell 语法与转换输入 dry-run 通过。沿用历史官方评测入口及评分口径。

8 卡：两个 TP1 Agent 副本、三个 TP2 Qwen3.6-27B nonthinking User 副本。Airline/Retail/Telecom/Banking Knowledge 全量 197 个 test 任务 × 4 trials，seed300，BM25，max_steps200，max_errors10；Agent temperature/top_p/max_tokens 为 0.6/1/1200，User temperature/max_tokens 为 0/512；域并发 1/2/2/4，全局 9，启用完成域空闲槽借用。

| Job | GPU | 状态与日志 |
|---|---:|---|
| 24757 (`pt-g4ex7d5a`) | 8 | 2026-09-28 09:44 CST 提交，SUBMITTED，normal priority；[运行日志](jobs/24757-tau2-reward-v3-iter139-four-domain-eval-0928-094426093/run_0_20260928_094426093.log) |

输出目录：`eval/reward-v3-iter139-four-domain/`。运行日志在容器启动后生成。
