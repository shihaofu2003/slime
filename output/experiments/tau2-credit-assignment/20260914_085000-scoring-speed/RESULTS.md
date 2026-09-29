# 状态评分效率优化

目的：降低 Progress-RTG 的状态评分 CPU 开销，保持评分、奖励和优势语义。
实验：tau2-credit-assignment / 20260914_085000-scoring-speed。

修改：在新鲜 JSON 快照上跳过相等子树，只对变化部分计算原叶字段距离；
展平操作使用单个结果字典，避免逐层复制；参考终态使用进程内 32 项 LRU 缓存，
按领域、数据库绝对路径和完整 task JSON 区分。并发首次未命中可能重复构建；
每次评分仍读取全部最新状态，保留 User 操作、同步、无关字段损坏和列表原子语义。

验证作业 18999（1 GPU / 14 CPU / 198 GB，normal；测试/基准不做模型生成）：
[运行日志](../jobs/18999-credit-scoring-speed-preflight-0914-084738626/run_0_20260914_084738626.log)。
20 项 progress 测试和原 rollout preflight 全部通过，覆盖三域 DB、Telecom 断言、
缓存复用及不同 task 内容隔离、增删/修复、User/Tool 边界、CP 和 policy loss 等价。
本地另外执行 10,000 组随机嵌套状态距离等价检查，无需安装 Torch。

| 领域 | 单线程评分加速 | 8 线程评分加速 | 参考构建冷缓存 → 热缓存 |
|---|---:|---:|---:|
| airline | 5.98× | 6.50× | 133.94 → 0.158 ms |
| telecom | 1.58× | 1.22× | 8.79 → 0.045 ms |
| retail | 5.77× | 5.32× | 137.74 → 0.094 ms |

方法：每域取共享数据第一个 DB 任务（airline_1、retail_1、telecom_1），
逐个执行参考工具动作并检查新旧评分一致，共 20 个状态边界。
在相同初始化状态上测量，包含每次数据库 model_dump；单线程 5 次，
8 线程合计 40 次。旧版函数保留原递归展平及路径并集比较实现。
基准脚本：examples/tau2-bench/analysis/benchmark_progress_scoring.py；
[原始计时](benchmark.json)。Telecom 的毫秒级结果易受线程调度开销影响。

以上为评分模块微基准，未启动新训练，也未测量端到端 40 步加速。
测试依赖共享 tau2 数据和模型，沿用手动实验 preflight，不加入通用 CI。
