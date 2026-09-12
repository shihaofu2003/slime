# Tau2 Credit Assignment 探索

## 目标与边界

工作目录：`/mnt/afs/users/fush/projects/ServiceAgent/slime-credit-assignment`。
分支：`feature/credit-assignment`；共同起点：`1efd1d3`。
本次仅搭建工作区并明确任务，尚未实施代码修改或提交实验作业。

探索 Tau2 agentic RL 的训练 reward、优势估计和逐回合 credit assignment，
使学习信号偏向正确动作、参数和最终 DB/环境状态，并分配到有因果贡献的 Agent 回合。
Agent 是优化对象，User 是固定模拟用户。遵循 AGENTS.md：先优化并通过受控官方评测
选择 reward/advantage 配方，再考虑 OPD；本任务不自动开展 OPD。

默认先在 AReaL airline、retail、telecom 三域及现有同步 RL 路径上探索，
不同时引入异步框架或混合 SFT 改动。若要结合其他分支，先明确依赖和新的对照设计。

## 数据与调查入口

共享只读输入：
`/mnt/afs/users/fush/projects/ServiceAgent/datasets/tau2/rl/areal_tasks_slime.jsonl`。
含 1,982 个任务，数据库绝对路径仍依赖原 AReaL 数据目录。
先阅读 `/mnt/afs/users/fush/projects/ServiceAgent/datasets/tau2/README.md`。

阅读 `examples/tau2-bench/rl/` 的 rollout、reward、filters、Agent 实现及训练入口，
追踪训练后端实际使用的 advantage、loss mask、GRPO group 和概率比计算。
参考 `output/doc/EXP_QA.md` 和既有 reward/turn-credit 实验，核对实现与历史结论。
不要把仅有正确工具名或更多工具调用当作正确行动的代理目标。

## 探索与验收

1. 先说明现有 reward 分量、group mean/std 归一化、零方差组处理及逐 token/turn
   优势的真实计算路径。区分轨迹奖励塑形与组内优势向回合的再分配。
2. 提出少量有明确机制假设的候选，例如针对正确动作/参数/状态的信号，或基于
   回合贡献的优势分配。说明信号如何得到、额外开销及可能的奖励投机，不预设某方案更优。
3. 明确 credit 对应的 Assistant 回合、工具返回及 token/loss mask；历史上下文、
   User 和工具文本不承担 Agent loss。检查多工具调用、失败工具、延迟状态变化和终止回合。
4. 检查同一 task 的 K 条完整轨迹构成 GRPO group，避免跨 task 或把多个 turn 当作
   多条轨迹。明确全对/全错组的 advantage 及保留/替换行为，不能静默改变采样分布。
5. 与现有基线从同一 SFT 初始化，固定 User、任务采样/配额、LR/KL、K、rollout
   预算、长度上限和评测协议。reward 与 credit 变化尽量分开对照；记录额外采样成本。
6. 先完成必要单元测试和最小训练 smoke，再进行受控实验。长期更新数、候选数量、
   具体初始化及 User/评测配置尚未指定，实施 agent 应先列出具体方案再安排资源。
7. 官方 held-out 评测始终使用 binary task success，报告 pass@1、pass@4(any)、pass^4，
   以及 action/DB accuracy、KL、截断和零方差组率。区分训练 reward 改善与真实成功率改善；
   用同任务/seed 的配对比较和不确定性支持选择，不仅凭训练曲线选配方。

不增加复杂评分网络、签名/哈希、审计状态或多层重复验证；
不把一般行为诊断当作硬失败，遵循项目禁止过度工程化的要求。

## 工作区与交付

只修改本 worktree，提交到本分支，不自行合并 main 或推送远程。
共享 datasets、models 和历史 checkpoints 只读；外部 tau2-bench 源码也被其他任务使用，
如确需修改，应为其建立独立 checkout 并配置安装路径。
新实验使用独立目录和 checkpoint 根目录，不覆盖既有结果。

通过项目提交器运行时，必须显式设置 worktree 路径，包括 setup 阶段：

```bash
bash scripts/submit.sh --experiment tau2-credit-assignment --gpus <实际所需卡数> \
  --project-root /mnt/afs/users/fush/projects/ServiceAgent/slime-credit-assignment \
  -e PROJECT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent/slime-credit-assignment \
  <本任务运行脚本>
```

GPU 与其他分支共享配额，提交前查询 quota/queue 并协调；priority 固定 normal。
不要在交互机器执行带 pkill 的训练 launcher，应在独立作业容器中运行。
每次提交维护本工作树 output/doc/INDEX.md 和实验 README 的相对 run-log 链接。
数据和自动生成轨迹不提交 Git。

交付：现有机制调查、候选及对照设计、实现与必要测试、可复现命令、
实际实验结果与限制。未完成官方受控评测时，不宣称已选出更好的训练配方。
