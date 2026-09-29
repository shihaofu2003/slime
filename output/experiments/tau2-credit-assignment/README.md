# Tau2 credit assignment

目的：从 Processed SFT 比较 progress-rtg-v1 与历史原始 GRPO，先训练 40 次更新，
seed300 官方 pass@1 > 57.25% 且 pass^4 ≥ 29% 才续训至总计 100 次。
实验名：`tau2-credit-assignment`。KL/entropy=0，progress gamma=0.98、progress
weight=1，format weight=1 且错误惩罚=-0.1。
运行目录：[20260913_055616](20260913_055616/)。代码未暂存、提交或推送。

## 已验证

- 状态评分效率优化：跳过相等子树，缓存最多 32 个任务参考终态，保持全字段距离语义。
  [验证结果](20260914_085000-scoring-speed/RESULTS.md)，集群 preflight 18999 通过：
  [日志](jobs/18999-credit-scoring-speed-preflight-0914-084738626/run_0_20260914_084738626.log)。
  20 项 progress 测试及 rollout preflight 通过；8 线程评分基准 Airline/ Retail/Telecom
  分别加速 6.50/5.32/1.22 倍，端到端训练加速尚未测量。未启动新训练。

- 首轮 11 项测试及 rollout preflight 通过；扩展 16 项测试通过，包括真实三域状态、
  多/失败工具与 User 操作边界、mask、CP=1/2、零辅助权重的 policy loss/梯度等价。
  测试依赖共享 tau2 数据和模型，作为手动实验 preflight，不加入通用 CI。
- Smoke 成功完成 iter0，40 条接受轨迹、5 组，Airline/Retail/Telecom=8/16/16。
  梯度范数 0.45265，loss 0.002126；公式、有限数值和 mask 覆盖检查通过。
  32 条 DB 与 8 条断言型轨迹的终态评分全部符合官方 evaluator。
- Binary 零方差组率 20%，最终无信号组率 0%，辅助项不可用组数 0；
  共 48 条候选轨迹、额外重采样 5 次，同域替换 Telecom 组 1 次。
  [Smoke 统计](20260913_055616/smoke/training_summary.json)。
- 初版 raw 重渲染诊断未剥离原始输出的结束标记；修正后集群重放全部 40 条轨迹，
  mask、response 长度和 Assistant span 全部一致。694 个回合中 1 个存在正文差异：
  既有解析器将 Unicode 转义转换为 `×`，重渲染后少 5 tokens；保持历史解析行为。
  [重放检查](20260913_055616/smoke/tokenization_audit.json)同时确认实际 checkpoint 保存了
  域游标 Airline/Retail/Telecom=1/2/3、group/sample 计数=6/48。
- 已复核历史 iter39 pass@1=57.25%、pass^4=29%；新比较程序对该控制自身做 12 项
  整体/逐域配对比较，delta 和 CI 全部为零。

## 作业

评分优化后的独立 100 步训练：Job **19000**，2026-09-14 08:55 提交；
从 Processed SFT 新初始化，不固定领域配比，每 10 步保存。
[配置与输出](20260914_085453-mixed-fast-train100/README.md)，
[运行日志](jobs/19000-credit-fast-mixed-train100-0914-085503741/run_0_20260914_085503741.log)。

训练与评测使用 8 GPU、112 CPU、1584 GB；转换使用 1 GPU、14 CPU、198 GB。
全部 priority normal，显式设置 worktree project root。

| 阶段 | Job | 状态 | 运行日志 |
|---|---|---|---|
| 首轮 preflight | 18155 / pt-v1drmc4q | 成功 | [log](jobs/18155-credit-preflight-0913-055622841/run_0_20260913_055622841.log) |
| 扩展 preflight → smoke | 18156 / pt-d17vh40c | 成功 | [log](jobs/18156-credit-smoke-0913-060223926/run_0_20260913_060223926.log) |
| 重放检查 → 新 SFT 初始化训练 40 步 | 18157 / pt-8x7gwt4d | 测试夹具失败，未训练 | [log](jobs/18157-credit-train40-0913-062505476/run_0_20260913_062505476.log) |
| 重跑 preflight → 训练 40 步 | 18158 / pt-td3tludx | 成功，完成 iter39（40/40 更新） | [log](jobs/18158-credit-train40-r2-0913-062944606/run_0_20260913_062944606.log) |
| iter19 转换 | 18159 / pt-lf3bvhri | 成功 | [log](jobs/18159-credit-convert19-0913-090557896/run_0_20260913_090557896.log) |
| iter19 seed300 三域评测 | 18160 / pt-y6jd5um7 | 成功，400 trials | [log](jobs/18160-credit-eval19-s300-0913-090837503/run_0_20260913_090837503.log) |
| iter39 转换（普通队列） | 18232 | 已取消，改用闲时任务 | [log](jobs/18232-credit-convert39-0913-125753573/run_0_20260913_125753573.log) |
| iter39 seed300 三域评测（普通队列） | 18235 | 已取消，改用闲时任务 | [log](jobs/18235-credit-eval39-s300-0913-130430848/run_0_20260913_130430848.log) |
| iter39 转换（闲时） | 18236 / pt-ptf1jdho | 成功 | [log](jobs/18236-credit-convert39-spot-0913-131229649/run_0_20260913_131229649.log) |
| iter39 seed300 三域评测（闲时） | 18237 | 已停止，改用普通队列 | [log](jobs/18237-credit-eval39-s300-spot-0913-131237299/run_0_20260913_131237299.log) |
| iter39 seed300 三域评测（普通队列） | 18245 / pt-av2ab6z3 | 成功，400 trials | [log](jobs/18245-credit-eval39-s300-normal-0913-133220649/run_0_20260913_133220649.log) |
| iter39 seed300 三域评测（闲时 spot2） | 18250 | 已停止 | [log](jobs/18250-credit-eval39-s300-spot2-0913-134509541/run_0_20260913_134509541.log) |
| iter39 seed300 三域评测（普通队列，默认 CPU/内存） | 18252 | 已停止 | [log](jobs/18252-credit-eval39-s300-normal-default-0913-134923599/run_0_20260913_134923599.log) |
| gamma=0.98、权重=1 配方 smoke | 18274 / pt-vbr854er | 成功，1 次更新 | [log](jobs/18274-credit-progress-v2-smoke-r2-0913-143910480/run_0_20260913_143910480.log) |
| 新配置正式训练 40 步 | 18310 | 已提交，8 GPU | [log](jobs/fsh-credit-progress-v2-train40-0913-151711/run_*.log) |
| 新配置正式训练 40 步（spot） | 18333 | 已停止，未启动 | [log](jobs/18333-credit-progress-v2-train40-spot-0913-153603074/submit_20260913_153603074.log) |
| 新配置 iter39 转换 | 18583 | 已提交，1 GPU | [log](jobs/fsh-credit-progress-v2-convert39-0913-204631/run_*.log) |
| 新配置 iter39 seed300 全域评测 | 18638 | 成功，400 trials | [log](jobs/18638-credit-progress-v2-eval39-s300-0913-212354028/run_0_20260913_212354028.log) |

18157 的新增 Telecom 损坏/修复测试错误地要求字符串叶节点；该 DB 使用原子列表。
已改用列表重复元素验证距离与修复；不改变训练算法。18158 的 16 项测试及
原 rollout preflight 全部通过（新增字段损坏/修复与结束标记统计检查均覆盖）。

正式训练 iter0：梯度范数 0.25622，loss 0.003579，binary reward 0.30；
40 条接受轨迹配额正确，终态 DB/断言评分全部符合官方 evaluator。Binary 零方差组率
80%，最终无信号组率 0%，辅助项不可用组数 0；681 个回合无正文重渲染差异。
[训练统计（阶段性）](20260913_055616/train40/training_summary.json)。这些是训练诊断，
不代表 held-out 官方评测结果。

截至 iter19，累计完成 20 次更新、800 条接受轨迹，checkpoint 已保存至 iter19；
最新梯度范数 0.55389、loss -0.023751、K2 漂移诊断 0.001182（KL loss 系数仍为 0），
前 20 次更新的已记录训练数值均有限。

[iter19 评测结果](20260913_055616/iter19_results.md)：pass@1 / pass@4(any) / pass^4 = 54.00% / 82.00% / 24.00%；
相对同 seed300 Processed SFT 为 −2.25 / +1.00 / −6.00 个百分点，整体配对 CI 均包含零。

[iter39 评测结果](20260913_055616/iter39_results.md)：旧配方 18158 的 pass@1 / pass@4(any) / pass^4
= 56.50% / 81.00% / 28.00%（226/400 成功）；相对历史 GRPO iter39 为
−0.75 / 0.00 / −1.00 个百分点，相对 Processed SFT 为 +0.25 / 0.00 / −2.00 个百分点，
整体配对 CI 均包含零。Action / DB accuracy 为 76.26% / 48.19%；无基础设施错误，
9/400 条轨迹达到 max_steps，均在 Telecom。未达到旧配方的续训条件。

用户追加的 iter19 三域官方评测采用 seed300、100 tasks × 4 trials，
与 iter39 使用相同评测协议。40 步训练现已成功结束，iter39 checkpoint 已保存。

## 配方更新

当前代码已切换到 `TAU2_PROGRESS_GAMMA=0.98`、`TAU2_PROGRESS_WEIGHT=1`、
`TAU2_FORMAT_WEIGHT=1`，format 错误原始惩罚为 `-0.1`。旧的 18158 训练仍是
gamma=1、progress=0.3 的结果；新配方 smoke 通过后需从同一 SFT 重新开始 40 步，
不能直接恢复旧 checkpoint。

18274 smoke 通过：40 条轨迹按 Airline/Retail/Telecom=8/16/16 接受，奖励为 18 条成功、
22 条失败；724 个回合使用 gamma=0.98，3 个回合触发 -0.1 format 惩罚；辅助状态全部可用。
1 次更新 loss=0.031595、grad_norm=0.564949，数值有限，iter0 checkpoint 已保存。

## 后续

旧配方 iter39 的转换和 seed300 官方评测均已完成；未满足 pass@1 > 57.25%、
pass^4 ≥ 29% 的预定条件，因此不续训至 100 步。新配方按上节从同一 SFT 重新比较。
转换使用 1 GPU、14 CPU、198 GB；训练和评测沿用上表资源配置。
所有运行测试均在集群，不在本地安装 Torch。

新配置 iter39 seed300 评测完成：pass@1/pass@4(any)/pass^4 = 52.50/83.00/23.00%。
未满足续训条件，停止在 iter39，不自动续训至 100 步。
[完整结果与配对置信区间](20260913_151800-v2/iter39_results.md)。

## 不固定领域配比对照（20260913_235650-mixed）

两组使用同一 Processed SFT、同步训练、seed1234/rollout42、全量 1982 条任务打乱，
每次 5 个任务组 × K8，40 次更新；不固定领域配比，零信号组保留，补采样来自全数据池。
普通 GRPO 关闭 shaping/turn credit；Credit 使用 gamma=0.98、progress weight=1、format=-0.1。
学习率、User、长度限制及其他训练配置相同。过滤和完成顺序可能使实际接受任务不同。

| 阶段 | Job | 日志 |
|---|---|---|
| Mixed 普通 GRPO train40 | 18815 | [log](jobs/18815-mixed-vanilla-train40-0913-235700897/run_0_20260913_235700897.log) |
| Mixed Credit GRPO train40 | 18816 | [log](jobs/18816-mixed-credit-train40-0913-235701941/run_0_20260913_235701941.log) |

[后续流程](20260913_235650-mixed/finish_eval.py)等待两组训练成功后，依次提交独立的 1 GPU 转换和
8 GPU seed300 官方全域评测（100 tasks × 4 trials），并生成任务级配对 bootstrap 对比。
[流程日志](20260913_235650-mixed/finish_eval.log)。本轮不自动续训至 100 步。

Mixed vanilla-grpo convert iter39: Job 18990, [log](jobs/18990-mixed-vanilla-grpo-convert39-0914-055551196/run_0_20260914_055551196.log).

Mixed vanilla-grpo eval iter39: Job 18991, [log](jobs/18991-mixed-vanilla-grpo-eval39-0914-055856688/run_0_20260914_055856688.log).

Mixed progress-rtg-v1 convert iter39: Job 18994, [log](jobs/18994-mixed-progress-rtg-v1-convert39-0914-063317133/run_0_20260914_063317133.log).

Mixed progress-rtg-v1 eval iter39: Job 18995, [log](jobs/18995-mixed-progress-rtg-v1-eval39-0914-063557529/run_0_20260914_063557529.log).

Mixed 两组已完成 40 步及 seed300 全域评测：普通 GRPO 为 55.00/77.00/29.00%，
Credit 为 54.75/83.00/23.00%（pass@1/pass@4(any)/pass^4）。
训练作业耗时分别 2.86/5.90 小时。Credit 相对普通 GRPO 的三项整体配对 CI 均包含零，
尚未证明改善；未提交续训。[完整对照](20260913_235650-mixed/results.md)。

优化评分混合训练 Job19000 的 iter49 全域评测：Job **19107**，8 GPU，seed300，三域全部 test 任务 × 4 trials；先转换后评测，提交时等待用户额度。[运行日志](jobs/19107-credit-fast-mixed-eval49-s300-0914-133714143/run_0_20260914_133714143.log)。

优化评分混合训练 Job19000 已完成 100 步；iter99 全域评测 Job **19208** 已提交（seed300，三域全部 test 任务 × 4 trials，8 GPU，先转换后评测）。[运行日志](jobs/19208-credit-fast-mixed-eval99-s300-0914-173517403/run_0_20260914_173517403.log)。

优化评分混合训练 Job19000 的 iter79 全域评测 Job **19231** 已提交（seed300，三域全部 test 任务 × 4 trials，8 GPU，先转换后评测）。[运行日志](jobs/19231-credit-fast-mixed-eval79-s300-0914-184212899/run_0_20260914_184212899.log)。
