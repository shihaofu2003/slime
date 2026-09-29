# Tau2 DB-count advantage v1

目的：实现并验证按 Agent/User 总 DB 差异条数进行 turn-level advantage 分组。
实验名：`tau2-db-count-v1`。独立代码位于
[rl_db_count_v1](../../../examples/tau2-bench/rl_db_count_v1/README.md)，原 `rl/` 不修改。
CPU preflight 已通过；Job19187 已提交，依次执行独立 smoke 和从 SFT 新初始化的 100 步训练。

## 作业

- Job 19184：CPU preflight 提交失败，8 CPU / 64 GB 不是集群支持的资源组合；未启动。
  [提交日志](jobs/19184-db-count-preflight-0914-164006746/submit_20260914_164006746.log)。
- Job 19185：CPU preflight，0 GPU / 8 CPU / 32 GB，通过。
  [运行日志](jobs/19185-db-count-preflight-0914-164048402/run_0_20260914_164048402.log)。

## 验证

2026-09-14：30 项测试通过，完整 rollout preflight 通过，作业 exit_code=0。
覆盖 Agent/User 计数、初始化后基准、原地修改和恢复、Telecom 列表按 ID 比较、
跨 turn 分组、任务隔离、单元素/零方差桶、重复访问、token mask 和 CP=1/2。

实际执行：`python3 -m pytest examples/tau2-bench/rl_db_count_v1/test_progress.py examples/tau2-bench/rl_db_count_v1/test_db_count.py -q -p no:cacheprovider`，
随后 `python3 examples/tau2-bench/rl_db_count_v1/test_rollout_logic.py`。
Python 语法与三个新 shell 入口的 `bash -n` 检查通过。
测试作为依赖项目数据/模型的手动实验 preflight，不加入通用 CI。

原 `rl/` 源码及训练入口未修改。四域 checkpoint 评测与小时级监控已提交。

## DB-count 混合领域训练

Job **19187**（pt-seyqmy77），2026-09-14 16:53 已提交；8 GPU / 112 CPU / 1584 GB，priority normal。
[配置与启动脚本](20260914_165226-mixed-smoke-train100/README.md)，[运行日志](jobs/19187-db-count-mixed-smoke-train100-0914-165312975/run_0_20260914_165312975.log)。
1 步 smoke 成功后从相同 SFT 新初始化训练 100 步；两阶段使用不同 checkpoint 目录。
对照 Job19000 的时间分组版本，保持 User、任务采样、种子、LR/KL 和奖励权重相同。

截至 2026-09-14 23:55，Job19187 已完成 step60；iter49（第50步）和 iter59（第60步）checkpoint 完整，日志未见 NaN。

## Checkpoint evaluation and monitoring

- Job **19303**：iter49（第50步）torch-dist→HF 转换，并运行 Airline/Retail/Telecom/Banking 四域、BM25、seed300、4 trials 官方评测；[运行日志](jobs/19303-db-count-four-domain-eval-iter0049-0914-235555116/run_0_20260914_235555116.log)。
- Job **19304**：初版监控因容器缺少 `rg` 停止；未影响 checkpoint 扫描。
- Job **19310**：修正版因扫描 HF 目录产生一次可恢复的格式错误后停止。
- Job **19311**：最终修正版只扫描严格的 `iter_XXXXXXX` torch-dist checkpoint，每5分钟扫描、每小时记录 Job19187 状态，并自动提交第50、100…步四域评测；[运行日志](jobs/19311-db-count-monitor-hourly-v3-0914-235922687/run_0_20260914_235922687.log)。
