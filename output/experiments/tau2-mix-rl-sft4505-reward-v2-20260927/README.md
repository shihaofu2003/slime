# tau2-mix-rl-sft4505-reward-v2-20260927

Purpose：在独立版本中修复四领域的奖励目标和环境状态错误，使 terminal reward 与 progress 使用一致的参考状态。原实现、credit-signal-v1 实验及历史结果保留。当前状态：前一轮修复和离线验证完成，扩大等价性检查后发现尚未修复的嵌套 JSON 参数问题；未提交本版本的 GPU 训练或评测。

[跨领域共性与新增遗漏](SEMANTIC_EQUIVALENCE.md)：按“等价表示应获得相同业务结果和评分”检查，Banking 新增 298 组配对回放中，有 33 个不同任务仅因表示变化从 reward=1 变成 0。包括 JSON 原文比较和嵌套数值改变交易 ID；这些问题尚未接入 v2 修复，数量不表示 rollout 错误率或训练收益。

[收益与边界](GAIN_ASSESSMENT.md)：旧运行的 2,192 条 Telecom 轨迹原样重评分，均值 33.12%→54.43%；这是评分纠错量。400 混合组窗口的曲线标准差 5.43→4.42 pp，仍有 21.11 pp 峰谷差。没有 v2 重训或官方提分结论。

## 四领域检查结果

对完整 2,524 个训练任务执行黄金动作回放，并检查等值数值表示。全部黄金动作可执行；随后检查 Retail 多商品修改的实际字段和顺序等价性。数字表示检查中的任务数表示可复现该差异的任务，不等于实际 rollout 的失败次数。

| 领域 | 检查任务数 | 发现的问题 | 修复 |
|---|---:|---|---|
| Airline | 1,148 | 274 个任务的行李数量等字段可保存为整数或等值浮点数，旧 DB 比较区分两种表示 | 按结构化字段值比较，接受 `2 == 2.0`，保留真实数值差异 |
| Telecom | 271 | 155 个黄金目标执行后仍有未同步状态，其中 136 个属于默认 DB 评分任务；102 个任务的充值账单文字存在数值格式差异 | 每个黄金动作后同步；只在评分快照中统一充值描述的数字表示 |
| Retail | 563 | 137 个修改订单任务中，13 个多商品修改错误复用最后一个商品的价格/规格 | 保存每组 old/new 商品的对应 variant，逐项写入，保留重复商品数量；金额按分计算 |
| Banking | 542 | 16 个信用卡申请任务因 `annual_income` 的整数/浮点表示不同生成不同业务 ID | 调用原业务 ID 生成函数前，按声明的浮点类型统一收入 |

Retail 的具体例子是 `retail_14`：原工具把应为 $272.33 的键盘写成 $29.28，并套用了笔记本的 A4/软皮规格。修复后键盘和笔记本分别保留各自正确的价格和选项。

证据：[汇总](audit_summary.json)、[全任务回放](reference_audit.jsonl)、[修复复核](verification.jsonl)、[Retail 商品检查](retail_items_audit.jsonl)。前序 Telecom 的实际轨迹和字段差异见 [诊断报告](../tau2-mix-rl-sft4505-credit-signal-v1-20260927/telecom_diagnosis/README.md)。

## 实现与隔离

[reward_v2.py](../../../examples/tau2-bench/rl/reward_v2.py)提供统一的评分快照和同步参考构造，修复后的 terminal DB 比较与原 progress 计算共用它们。真实工具响应、无关文本、其他列表顺序和真实金额差异不会被评分归一化删除。Banking 的收入规范化发生在原业务 ID 生成函数之前；Retail 的修复发生在实际工具执行层，覆盖 live rollout、评估回放和目标构造。

[独立训练入口](../../../train_tau2_reward_v2.py)通过 Ray 的 `worker_process_setup_hook` 在新作业的 worker 启动时接入修复，包括 RolloutManager 创建的子环境 worker。不是运行中热修改；原 Python 文件和旧作业进程不会被改写。继续使用 `credit_signal_v1` 的组过滤和已有的 turn-advantage/TIS/token-mask 路径。

本版保留各任务的 `reward_basis`：205 个未写字段的 Telecom 任务仍按原默认 DB+COMMUNICATE 评分，显式 ENV_ASSERTION/ACTION/COMMUNICATE 条件继续生效。此次修复实现错误；将这些任务改成 ENV-only 是另一项奖励方案，需要另做对照。

Banking 的业务 ID 和 Retail 的工具返回状态在 v2 中可能与旧实现不同。旧轨迹继续使用旧版本复现；新版本应生成新的轨迹。历史官方评测文件与脚本未改，本版本没有新的官方评测成绩，不能将修复后的训练 reward 提升直接当成策略能力提升。

## 验证

- 全量审查 2,524 个任务，未发现黄金动作执行错误。
- 447 项定向复核全部通过：Airline274、Retail1、Telecom155、Banking17；等价调用得到相同 DB，DB 任务完成后的 progress 距离为 0。
- 额外检查全部 137 个 Retail 修改订单任务：修复后商品字段错误为 0，成对反转输入商品顺序后的 DB 差异为 0。另验证工具 schema 保持一致，`retail_14` 完整回放 reward=1、最终 progress=0。
- [17 项 CPU 回归测试](../../../examples/tau2-bench/rl/test_reward_v2.py)通过，覆盖实际 Telecom 误判轨迹、全部 271 个 Telecom 参考同步、四域 reward/progress 一致性、Banking ID、Retail 多商品/重复商品/失败调用、ACTION/COMMUNICATE 约束和真实 Ray 父子 worker 的接入。
- shell 语法和配置 dry-run 通过。验证环境为 Python3.12.14、CPU Torch2.6.0、Ray2.58.0；未运行 GPU 训练。

这些测试是新实验启动前的独立检查。为保留现有文件，未修改固定 CI 矩阵。

```bash
python3 examples/tau2-bench/rl/test_reward_v2.py
python3 output/experiments/tau2-mix-rl-sft4505-reward-v2-20260927/audit_reference_states.py
python3 output/experiments/tau2-mix-rl-sft4505-reward-v2-20260927/verify_reward_fixes.py
python3 output/experiments/tau2-mix-rl-sft4505-reward-v2-20260927/audit_retail_items.py
```

## 启动配置

[run_user.sh](run_user.sh)准备独立的 4-GPU Qwen3.6-27B User 服务；[run_mix.sh](run_mix.sh)准备 8-GPU mix RL。训练从 SFT4505、新优化器开始，保持 140 updates、K=8、batch128、LR2e-6、KL/entropy0、progress/format1/1、gamma0.98、seeds1234/42、随机任务池及四环境 worker 配置。输出写入本实验目录，默认等待本实验的 `user_endpoint.env`；也支持用 `TAU2_USER_ENDPOINT_FILE` 指定服务。

```bash
bash output/experiments/tau2-mix-rl-sft4505-reward-v2-20260927/run_mix.sh --dry-run
```

实际提交时先启动 4 卡 User，接口就绪后提交 8 卡训练，两者使用 normal priority，并将作业号和相对运行日志链接补到本 README。
