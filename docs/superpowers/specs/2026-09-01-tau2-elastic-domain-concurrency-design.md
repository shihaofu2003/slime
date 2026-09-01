# tau2 异步评估弹性领域并发设计

## 背景

当前官方异步评估为 Airline、Retail、Telecom 分别启动一个
`run_domain` 进程，并给每个进程固定的 `max_concurrency=3`。11756 中三个
领域分别耗时 10.43、19.18、47.07 分钟；领域结束后，其轨迹并发不会转移，
全局并发从 9 降为 6，最后降为 3，Telecom 决定整体尾部。

## 目标与边界

- 官方 test 集仍为 Airline/Retail/Telecom `20/40/40` 个任务、每任务 4 个
  trial；任务、trial、seed、checkpoint、重试和计分语义不变。
- 支持不同领域的初始并发，首个生产配置为
  `airline:2,retail:2,telecom:5`，全局硬上限为 9。
- 开启弹性借槽后，仅在一个领域全部轨迹成功完成时转移其槽位，不在单条
  轨迹等待 Agent 或 User 响应时抢占。
- 未设置新配置时继续使用现有统一 `MAX_CONCURRENCY` 行为。
- 后续八卡评测拓扑固定为两个 Agent TP1 副本和三个 User TP2 副本：Agent
  使用 GPU `0;1`，User 使用 GPU `2,3;4,5;6,7`。

## 方案选择

采用“tau2 runner 可选槽位门控 + slime 跨领域协调”方案。tau2 继续负责加载
任务、建立 trial seed、执行轨迹、重试、checkpoint 和结果聚合；slime 只负责
初始配额、领域完成事件和配额转移。不在 slime 中复制官方 batch runner，也
不通过停止、重启和 auto-resume 模拟动态扩容。

## 接口与配置

`tau2.runner.batch.run_tasks` 和 `run_domain` 增加两个可选关键字参数：

- `executor_max_workers: int | None`：领域线程池可创建的最大 worker 数；未设置
  时继续使用 `config.max_concurrency`。
- `slot_semaphore`：轨迹开始前获取、结束后释放的跨进程信号量；未设置时不
  增加门控。

slime 官方评测增加：

- `DOMAIN_CONCURRENCY` / `--domain-concurrency`，格式
  `airline:2,retail:2,telecom:5`。
- `GLOBAL_CONCURRENCY` / `--global-concurrency`，生产值为 `9`。
- `BORROW_COMPLETED_DOMAIN_SLOTS` / `--borrow-completed-domain-slots`，生产值
  为 `1`。

只有选择弹性模式时才创建跨进程信号量；静态不同领域并发直接传给各领域的
`TextRunConfig.max_concurrency`。

## 调度数据流

父进程按初始配额创建三个 `multiprocessing.Manager().Semaphore`，并以
`as_completed()` 处理领域 Future。每个领域线程池最多准备 9 个 worker，
但 `_run_tracked` 只有取得本领域信号量后才开始轨迹。

当 Airline 以配额 2 完成时，父进程按剩余领域初始权重、使用确定性的整数
余数分配，将两个槽位各转一个给 Retail 和 Telecom，配额变为 `0:3:6`；
Retail 完成后，其 3 个槽位全部转给 Telecom，配额变为 `0:0:9`。父进程始终
按请求中的领域顺序返回结果，不因完成顺序改变 summary 布局。

## 失败与结果语义

只有领域 Future 正常返回后才转移槽位。领域执行异常时保持现有失败行为并
向上抛出，不把失败领域视为已完成，也不启动恢复状态机。信号量在轨迹的
`finally` 中释放，避免一次轨迹异常占住槽位。

改变调度只会改变轨迹开始顺序；trial seed 在任务提交前按现有 runner 逻辑
生成，因此任务与 seed 的对应不变。正式结果仍要求 100 个唯一任务、400 条
唯一轨迹、每任务 trial `0..3`、基础设施错误为 0。

## 日志与验证

summary 增加初始领域配额、全局上限、是否借槽及简短的配额转移事件；现有
领域 wall time、Agent/User 延迟和结果指标继续保留。配额事件是诊断信息，
不作为新的评测失败条件。

测试采用 TDD：覆盖配置解析与兼容默认值、确定性配额转移、全局活动轨迹数
不超过 9、领域完成后阻塞轨迹获得新槽位、Future 异常不触发借槽、结果顺序
不随完成顺序变化，以及 tau2 runner 在未传新参数时保持原行为。随后运行
CPU preflight、真实小规模三领域 smoke，并以同模型、任务和 trial 对比固定
`3:3:3` 与弹性 `2:2:5` 的 wall time、请求延迟、worker 请求分布和结果完整性。

## 实施文件

- `../tau2-bench/src/tau2/runner/batch.py`
- `examples/tau2-bench/eval/official/run_eval.py`
- `examples/tau2-bench/eval/official/run_eval.sh`
- `examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh`
- `tests/test_tau2_official_async_eval.py`
- `examples/tau2-bench/eval/official/README.md`
- `output/experiments/tau2-eval-qwen36-user-async-timed/README.md`
