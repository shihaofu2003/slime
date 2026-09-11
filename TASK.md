# Tau2 混合 SFT 与领域专家泛化实验

## 目标与范围

在 `feature/sft-domain-generalization` 分支沿用已有同步 SFT 训练流程：
先混合 AReaL 三域和 Banking 数据训练并做全领域评测，
再分别训练四个领域的 SFT 专家并做相同全领域评测，观察目标域收益及跨域泛化。
工作目录：`/mnt/afs/users/fush/projects/ServiceAgent/slime-sft-domain-generalization`。
起点为主分支提交 `1efd1d3`。本文件仅为任务交接；尚未实现或提交实验作业。

本任务不依赖另一分支的异步 RL 修改，不改变 RL 框架、不做 RL 或 OPD。
以下基座、训练和评测参数是根据最近已完成实验整理的默认方案；
用户尚未单独确认这些数值，后续明确指令优先。

## 输入数据与实验矩阵

共享数据根目录：`/mnt/afs/users/fush/projects/ServiceAgent/datasets/tau2`。
先读该目录 `README.md`，使用已复制的数据，避免回头使用旧实验的部分导出。

| 输入文件 | Airline | Retail | Telecom | Banking | 总行数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sft/areal_official_native_target_only.jsonl` | 13,783 | 13,614 | 2,979 | 0 | 30,376 |
| `sft/banking_simplified_target_only.jsonl` | 0 | 0 | 0 | 5,672 | 5,672 |

两份数据均为 native messages/tools、qwen3_full、最后 Assistant target-only 监督。
Banking 文件是完整简化导出，已含 pilot，不再追加 pilot 或其他历史子集。
Banking benchmark 内部域名为 `banking_knowledge`。

| 训练臂 | 输入 | 必须评测的域 |
| --- | --- | --- |
| mixed | 两份文件全部行，默认原始比例，共 36,048 行 | 全四域 |
| airline | AReaL 的 airline 精确子集 | 全四域 |
| retail | AReaL 的 retail 精确子集 | 全四域 |
| telecom | AReaL 的 telecom 精确子集 | 全四域 |
| banking | 完整 5,672 行 Banking 数据 | 全四域 |

五臂都默认从原始 `Qwen3-4B-Instruct-2507` 基座独立初始化，
领域专家不是从 mixed checkpoint 继续训练。
混合采用原始行数比例并固定随机种子打乱；不默认增加均衡重采样或额外消融。
专家数据必须与 mixed 中相应域的数据一致，保留工具 schema 和 target-only mask。

## 训练与评测方案

复用 `examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh`，
参考 official-native-expanded 的 wrapper；给本实验建立独立入口和输出路径。

默认共同配方：8 GPU、batch 16、2 epochs、seed 1234、LR 1e-5 cosine 到 1e-6、
沿用现有 warmup 设置、qwen3_full、16,384 tokens/GPU，不截断训练行。
使用 `models/Qwen3-4B-Instruct-2507` 及对应 `_torch_dist` 初始化。
明确设置 fresh-run 的起始 rollout，避免基座 iteration 导致跳过第一个更新。
记录实际行数、监督 token 数、更新数、样本覆盖、loss 与耗时。
相同 epochs 不等于相同计算量；比较泛化时明确这一限制。

按以下顺序执行：

1. 解析数据，验证域划分、target mask 和模型 tokenizer 下的实际长度；
   若遇到影响结果的格式或超长问题，报告并修复数据流程，不静默删行。
   构建混合数据、四个精确子集及覆盖长样本的最小 smoke 数据。
2. 完成训练 smoke，然后运行 mixed 完整训练、转换 final HF、全域评测。
3. 从同一基座分别训练四个专家，转换 final HF；所有专家执行同样的全域评测。
4. 汇总训练域 × 评测域矩阵，与 mixed 比较；可复用协议完全匹配的已有 raw 基座结果，
   不把旧的 Banking-only 或带时间截断的结果当作匹配的全域基线。

评测默认复用
`examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh`：
official-native Agent、固定 Qwen3.6-27B non-thinking User、Banking BM25 检索、
test 集 airline 20 / retail 40 / telecom 40 / banking 97，共 197 个任务；
每任务 4 trials，seeds 300 和 301。Agent temperature 0.6、top-p 1、
max_tokens 1200、max_steps 200、max_errors 10；沿用相同 retries，
不新增 simulation timeout。先核实实际运行参数和任务数。
评测内部已有并发推理可以照常使用，与本任务不修改异步 RL 框架并不冲突。

不能因为样本数少而只评专家的训练域。各模型均报告每域及 overall 的
pass@1、pass@4(any)、pass^4，明确分别表示单次成功、四次至少一次成功、四次全成功。
同时记录 action/DB accuracy、截断、终止及基础设施错误等诊断。
对同任务/seed 的差异使用配对分析，报告不确定性；不凭单个点估计宣称泛化改善。

## 工作区与交付

遵循本工作树 `AGENTS.md` 和相关技能。只改本分支；共享数据、模型和原 checkpoint
只读，新数据写到本实验目录。不要修改外部共享 `tau2-bench/` 源码。
训练数据、模型和生成日志不提交 Git；维护最小事实 README 和索引。

启动脚本须使用本 worktree 的 `PROJECT_ROOT`，包括作业 setup 阶段：

```bash
bash scripts/submit.sh --experiment tau2-sft-domain-generalization --gpus 8 \
  --project-root /mnt/afs/users/fush/projects/ServiceAgent/slime-sft-domain-generalization \
  -e PROJECT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent/slime-sft-domain-generalization \
  <本任务运行脚本>
```

每个臂、smoke/full 和评测 seed 使用独立输出，不能写入历史实验 checkpoint。
集群 priority 为 normal，GPU 配额与异步 RL 任务共享，提交前协调；不要停止其他任务。
不要在当前交互机器直接执行带 `pkill python/ray` 的训练 launcher，应在独立作业容器运行。
提交时维护本工作树 `output/doc/INDEX.md` 与实验 README 的相对 run-log 链接。

交付可复现的数据准备、训练、HF 转换、评测和汇总入口，必要测试，
五模型 × 四领域的两 seed 结果及局限说明。提交到本分支，不自行合并 main 或推送远程。
