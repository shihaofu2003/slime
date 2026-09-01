# 第一个改变：tau2 异步评估

## S：背景（Situation）

tau2 的正式评测包含 Airline、Retail 和 Telecom 三个领域，共 100 个任务。
每个任务运行 4 次，一次完整评测需要生成 400 条轨迹。原来的执行方式按领域
串行推进，Agent 输出后再等待 User 模型回复。两次串行基线的实际耗时分别是：

| 任务 | User 配置 | 任务数 / 轨迹数 | 总耗时 | GPU-hours |
|---|---|---:|---:|---:|
| 11723 | TP1 | 100 / 400 | 229.94 分钟 | 7.66 |
| 11728 | TP2 | 100 / 400 | 174.26 分钟 | 11.62 |

一次评测要等三到四个小时，导致模型迭代后的验证周期过长。与此同时，不同领域、
不同任务和不同 trial 之间没有状态依赖，本来可以并行执行。

## T：任务（Task）

目标是在不改变官方任务、对话协议和评分方式的前提下缩短评测时间。新的执行路径
需要满足以下条件：

- 保持同一组 Agent/User 模型、test split、100 个任务、4 trials、seed 300 和
  `max_steps=200`。
- Agent 生成完一轮后，User 请求立即进入共享推理池，不等待其他轨迹结束。
- Airline、Retail 和 Telecom 可以同时运行，同时保留每个任务独立的环境状态。
- 日志能够区分服务启动、Agent 推理、User 推理、环境处理和结果汇总耗时。
- 任务或服务出错时不能只留下一个看似成功的调度状态。

## A：行动（Action）

评测入口改为多进程执行三个领域，每个领域内部并发 3 条轨迹。进程之间不共享
tau2 环境，只共享模型服务：4 个 TP1 Agent worker 使用 GPU 0–3，两个 TP2
Qwen3.6-27B User worker 使用 GPU 4–7，前面各放一个 SGLang Router。这样，一条
轨迹产生 Agent 回复后就能提交 User 请求，其他轨迹可以继续占用空闲 worker。

我们同时补上了三层计时：Shell 记录服务启动和评测主体的墙钟时间；Agent 与 User
适配器记录每次模型请求；汇总程序统计 p50、p95、累计耗时占比和有效并行度。

第一次 smoke 暴露了一个真实问题：SGLang 原生 `/generate` 接口接收
`sampling_seed`，不能直接传 `seed`。修正参数映射后，第二次 smoke 完成 6 个任务、
12 条轨迹且无基础设施错误。评测程序也增加了结果检查，只要 summary 中出现
`infrastructure_error` 就返回非零退出码。随后提交完整任务 11756，并持续检查完成数、
服务 5xx、超时、OOM 和 traceback。

相关实现包括：

- `examples/tau2-bench/eval/official/run_eval.py`：领域并行、汇总和错误退出。
- `examples/tau2-bench/eval/official/run_eval.sh`：worker 池、Router、健康检查和阶段计时。
- `examples/tau2-bench/eval/official/model_request.py`、`timed_user.py`：请求参数和
  Agent/User 计时。
- `tests/test_tau2_official_async_eval.py`：9 个 CPU 单元测试，覆盖并行进程、seed
  映射、计时、汇总和基础设施错误。

## R：结果（Result）

完整异步任务 11756 完成 100 个任务和 400 条轨迹，总耗时 52.84 分钟。其中服务
启动 5.38 分钟，评测和汇总 47.30 分钟。相对 TP1 串行任务加速 4.35 倍，相对 TP2
串行任务加速 3.30 倍。虽然使用了 8 张卡，GPU-hours 仍降到 7.05，比 TP1 串行少
8.1%，比 TP2 串行少 39.4%。

| 任务 | pass@1 / pass^1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |
|---|---:|---:|---:|---:|---:|
| 11723，TP1 串行 | 28.00% | 47.00% | 13.00% | 43.97% | 36.46% |
| 11728，TP2 串行 | 27.00% | 50.00% | 6.00% | 43.34% | 37.60% |
| 11756，异步 | 26.50% | 48.00% | 12.00% | 42.48% | 35.64% |

正确性检查覆盖了结果结构和行为分布。11756 的 400 个 simulation ID 均不重复，
每个任务都有 trial 0、1、2、3；三个领域的基础设施错误为 0，12,527 条 Agent/User
计时记录均为正数。异步结果的 pass@1 比更接近的串行结果低 0.50 个百分点，
pass@4(any) 和 pass^4 位于两次串行结果之间，Action/DB accuracy 与两次串行结果的
最大差距分别是 1.49 和 1.96 个百分点。由于三次运行都没有启用 SGLang 确定性推理，
并发请求顺序也会改变采样轨迹，因此不要求三次分数逐项相同。这些结果没有显示出
任务丢失、trial 重复、基础设施错误或明显的行为分布漂移。

计时结果也指出了下一步。User 推理占累计轨迹耗时的 55.68%，Agent 占 43.99%；
Telecom 单独运行了 47.07 分钟，决定了整体尾部。cache-aware 路由还让两个 Agent
worker 基本空闲。后续可以先做 matched smoke，对比当前拓扑、round-robin 路由以及
“2 个 Agent + 3 个 TP2 User”分配，再决定是否调整正式评测配置。

完整数据见
[异步实验记录](../output/experiments/tau2-eval-qwen36-user-async-timed/README.md)、
[11723 summary](../output/experiments/tau2-eval-qwen36-user-timing/eval/seed300_summary.json)、
[11728 summary](../output/experiments/tau2-eval-qwen36-user-timing-tp2/eval/seed300_summary.json) 和
[11756 summary](../output/experiments/tau2-eval-qwen36-user-async-timed/eval/full/seed300_0901_045448_summary.json)。
