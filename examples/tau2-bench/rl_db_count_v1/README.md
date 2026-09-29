# progress-db-count-v1

目的：比较同一任务、相同总 DB 差异条数处的 Assistant 决策，替换按 turn index 分组。
本目录是 `../rl/` 的独立版本副本；原目录和原启动入口不修改。
Slime、tau2 和 shared 公共模块继续读取现有版本。

## 计数与优势

- 基准为每次 rollout 初始化（包括 task initialization 和环境同步）完成后的 DB。
- 每次 Assistant 决策前取当前状态，包含之前的 Agent 工具、User 工具和同步结果。
- Agent DB 按表内记录计数：字典使用原 ID，Telecom 列表使用各表记录 ID。
  同一记录多个字段变化只计一条；新增/删除各计一条，恢复原值后不再计数。
- User DB 的 device/surroundings 是状态对象，按叶子字段计数，列表为原子字段。
  两侧分开比较后求和；同步造成两侧都变化时两侧分别计数，不跨 DB 去重。
- 组键为同一任务采样组内的总差异条数，不跨任务混组。跨轨迹和 turn 收集全部访问。
  重复访问参与组统计，诊断同时记录桶内 turn 数和独立轨迹数。
- progress reward/RTG 保持原语义（目标 DB 字段距离或环境断言；gamma=0.98）。
  只替换局部归一化分组；outcome/progress/format 权重仍为 1，格式惩罚 −0.1。
  单元素/零方差桶的局部优势为零，保留 outcome 和 format 项。
- `credit.jsonl` 的各 turn 记录总计数、两侧计数、桶大小及独立轨迹数。

## 使用

在项目作业环境中先运行 `bash examples/tau2-bench/rl_db_count_v1/preflight.sh`。
测试依赖 tau2 数据、模型和训练环境，作为手动实验 preflight，不注册通用 CI。

```bash
export PROJECT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent/slime-credit-assignment
export RUN_DIR="$PROJECT_ROOT/output/experiments/tau2-db-count-v1/<new-run>"
bash examples/tau2-bench/rl_db_count_v1/run_qwen3_4b_instruct_2507_tau2_rl_progress_db_count_v1.sh smoke
```

支持 smoke/train40/train100/continue100；使用新的 RUN_DIR 从相同 SFT 开始。
启动入口显式使用本目录的模块路径，Ray workers 同样使用本目录。
