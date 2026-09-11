# Tau2 AReaL 三域异步 Agentic RL

## 目标与范围

在 `feature/async-tau2` 分支实现并验证 Tau2 AReaL 三域
（airline、retail、telecom）的异步 RL。工作目录：
`/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2`。
起点为主分支提交 `1efd1d3`。本文件仅为任务交接；尚未实现或运行实验。

模型角色：训练目标是 **Agent**；trainer 更新 Agent 参数，generator
用 Agent 的行为策略生成动作；**User** 是固定参数的用户模拟器，不参与优化。
这是对原需求中模型名称笔误的解释；若用户要求同时训练 User，需重新确定算法范围。

实现两层异步：

- 对话层：不同对话中的 Agent 和 User 推理请求可以交错执行、并行服务。
  单条对话仍须遵循工具执行、环境状态更新和 Agent/User 消息的因果顺序。
- 训练层：trainer 处理当前批时，generator + User 继续生成后续完整对话。
  处理好行为策略概率、权重同步及由此产生的策略滞后。

本任务不混入 Banking、不做 SFT 或 OPD，不同时更换奖励/优势配方来解释吞吐变化。
先复用已有异步入口；若采用更深的队列或连续权重更新，说明必要性和训练语义。

## 数据与实现入口

- 输入：`/mnt/afs/users/fush/projects/ServiceAgent/datasets/tau2/rl/areal_tasks_slime.jsonl`，
  1,982 个任务（airline 1,148、retail 563、telecom 271）。
- 数据说明：`/mnt/afs/users/fush/projects/ServiceAgent/datasets/tau2/README.md`。
  JSONL 内的数据库绝对路径仍依赖原 `datasets/AReaL-tau2-data/`，须保持可读。
- 检查 `train_async.py`、`slime/ray/placement_group.py`、
  `slime/rollout/sglang_rollout.py`、`examples/tau2-bench/rl/rollout.py`、
  `agent.py`、`filters.py` 和现有 RL 启动脚本。
- 最近的同步 official-native 对照入口：
  `examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_sft_raw_vs_processed_vanilla_grpo.sh`，
  使用 `processed` 臂作候选基线。现有 Agent SFT 初始化为
  `checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903`；
  User 为 `models/Qwen3.6-27B`，相对路径均从 ServiceAgent 根目录算起。
  这些是已有可复用配置，不代表用户指定了新的算法或长期训练预算。

初步阅读发现 `train_async.py` 已有一批预取与训练重叠，并在完整 rollout 结束后换权重；
Tau2 也已有 `asyncio.to_thread` 并发入口。先检查实际瓶颈再修改。
尤其核查 Tau2 的 `rollout_log_probs` 占位值和启动脚本的 colocate 配置；
不能只改入口名称或添加版本字段就视为完成异步训练。

## 实现与验收

1. 梳理同步基线、现有并发行为、两层异步的实际缺口，以及 trainer / generator / User
   各自 GPU 和进程归属。使用当前任务的独立启动配置。
2. 明确 PPO/GRPO 概率比的分母来自哪一版行为策略；保留实际生成 token 的 logprob
   并正确对应训练 token/loss mask，不能用全零概率或无说明地用当前 trainer 概率替代。
   不截掉对话前缀以迁就长度。沿用 16,384 token 上限及已有超长轨迹重采样策略。
3. 使 rollout 与训练重叠；明确何时更新 generator 权重、允许多少策略滞后及恢复语义。
   避免无意在一条对话中混用参数版本；若允许，必须正确支持该语义。
4. 保持三域任务比例、K、reward、advantage 归一化、零方差组处理、LR/KL、User、
   生成长度与终止协议和同步对照一致。确需改变时单独说明，不能作为纯异步加速对比。
5. 用必要测试验证对话隔离、行为概率与 mask 对齐、权重更新边界、尾批收尾和恢复。
   运行最小 GPU smoke，日志展示真实的跨对话并发与 trainer/rollout 时间重叠。
   比较相同任务预算下的吞吐、等待时间、KL、有效训练样本、截断及零方差组率。
   若做效果评测，使用官方 binary reward，并报告 pass@1、pass@4(any)、pass^4。

先完成可验证的异步实现及 smoke；用户本次未指定新的长程 RL 更新数或消融规模。
诊断指标用于解释实验，不把普通行为波动变成额外停止门槛。

## 工作区与交付

遵循本工作树 `AGENTS.md` 和相关技能。只改本分支，不改主工作树；
`datasets/`、`models/` 和已有 checkpoint 作为共享只读输入。
外部 `tau2-bench/` 是共享代码目录，若必须改动，先建立该仓库的独立分支/worktree，
并让安装入口指向它，不直接影响其他任务。

启动脚本须将 `PROJECT_ROOT` 指向本 worktree。通过项目提交器提交作业时，
显式传入下列参数，使 setup 阶段的 editable install 也使用本分支：

```bash
bash scripts/submit.sh --experiment tau2-areal-async-rl --gpus <实际需要的卡数> \
  --project-root /mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2 \
  -e PROJECT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent/slime-async-tau2 \
  <本任务运行脚本>
```

使用独立实验名、checkpoint 根目录和作业输出；集群 priority 为 normal。
提交作业时维护本工作树 `output/doc/INDEX.md` 与实验 README，记录相对 run-log 链接。
GPU 与另一条 SFT 任务共享配额，提交前查 quota/queue 并协调，不能停止其他任务。

交付代码、必要测试、可复现启动命令和实际运行结果/未验证限制。
提交到 `feature/async-tau2`，不自行合并 main 或推送远程。
