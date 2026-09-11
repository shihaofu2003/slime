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

完整任务 11756 之后又做了两轮调度优化。三个领域先使用固定并发
`Airline:Retail:Telecom=2:2:5`，某个领域完成后将空闲槽位借给剩余领域；User
Router 从 cache-aware 改为 round-robin。最终任务 11878 使用 2 个 TP1 Agent、
3 个 TP2 User 和全局并发 9，实际并发变化为 `2:2:5 -> 0:3:6 -> 0:0:9`。

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
- `tests/test_tau2_official_async_eval.py`：18 个 CPU 单元测试，覆盖并行进程、seed
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
| 11878，2 Agent + 3 User 异步 | 24.75% | 46.00% | 8.00% | 42.94% | 33.51% |

正确性检查覆盖了结果结构和行为分布。11756 的 400 个 simulation ID 均不重复，
每个任务都有 trial 0、1、2、3；三个领域的基础设施错误为 0，12,527 条 Agent/User
计时记录均为正数。异步结果的 pass@1 比更接近的串行结果低 0.50 个百分点，
pass@4(any) 和 pass^4 位于两次串行结果之间，Action/DB accuracy 与两次串行结果的
最大差距分别是 1.49 和 1.96 个百分点。由于三次运行都没有启用 SGLang 确定性推理，
并发请求顺序也会改变采样轨迹，因此不要求三次分数逐项相同。这些结果没有显示出
任务丢失、trial 重复、基础设施错误或明显的行为分布漂移。

调度优化后的完整任务 11878 也完成了 100 个任务和 400 条轨迹，基础设施错误和
single-call 协议违规均为 0。它相对 11728 的三个总体指标变化为
`-2.25/-4.00/+2.00pp`，相对 11723 为 `-3.25/-1.00/-5.00pp`。按领域分层、同任务
配对的 20,000 次 bootstrap 中，11878 相对 11723、11728 和 11756 的九个总体指标
差异，其 95% 区间全部包含 0。因此结果不是逐项复现，但和此前串行及异步结果统计上
相容，没有证据表明 round-robin 或弹性借槽改变了评测语义。

11756 的计时结果指出了下一步：User 推理占累计轨迹耗时的 55.68%，Agent 占
43.99%，Telecom 的 47.07 分钟决定整体尾部；cache-aware 还让四个 Agent worker
中的两个基本空闲。这直接促成了 2 Agent + 3 User 的 matched smoke 和正式复测。

这个 matched 配置已经由 11851 smoke 和 11878 full eval 验证。11878 的服务启动、
评测和作业总耗时分别为 5.37、29.26 和 35.71 分钟，GPU-hours 为 4.76；相对 11756
的评测阶段再加速 1.62 倍，相对串行 TP2 任务 11728 的端到端耗时加速 4.88 倍。
三个 User worker 各完成 2,012 个成功请求，round-robin 达到完全均衡；两个 Agent
在 cache-aware 下仍为 `6662/998`，后续优化重点应放在 Agent 路由和 Telecom 长尾，
而不是继续增加 User 并发。

完整数据见
[异步实验记录](../output/experiments/tau2-eval-qwen36-user-async-timed/README.md)、
[11723 summary](../output/experiments/tau2-eval-qwen36-user-timing/eval/seed300_summary.json)、
[11728 summary](../output/experiments/tau2-eval-qwen36-user-timing-tp2/eval/seed300_summary.json)、
[11756 summary](../output/experiments/tau2-eval-qwen36-user-async-timed/eval/full/seed300_0901_045448_summary.json) 和
[11878 summary](../output/experiments/tau2-eval-qwen36-user-async-timed/eval/full-round-robin/seed300_0901_102800_summary.json)。
