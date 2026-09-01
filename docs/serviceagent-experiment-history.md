# ServiceAgent 实验全记录：tau-bench、tau2 与 VitaBench

> 这份记录把仓库中的实验 README、报告和作业日志串成一条时间线。它记录的是“做了什么、改了什么、得到什么、哪里好、哪里不好”，不是新的评测报告。
>
> **记录边界：** 仓库目前可见的 tau/tau2 实验产物截至 2026-08-16；截至 2026-08-31 没有发现更晚的 tau/tau2 实验 README 或 job 结果。VitaBench 记录也以仓库现有产物为准。

## 1. 范围与口径

**记录范围**

- tau-bench/tau1：从依赖、数据和 checkpoint 转换，到 custom rollout、log-prob 修复和多步 GRPO。
- tau2：依赖、官方评测、Agent/User SFT、数据和协议修复、RL 稳定性、奖励/advantage、谱系续训，以及所有控制实验。
- VitaBench：依赖 smoke、Qwen3.5 远程完整评测尝试、Qwen3.5 本地角色评测、Evaluator thinking A/B，以及 Qwen3-4B 本地角色评测尝试。

**状态标签**

| 标签 | 含义 |
|---|---|
| 成功 | 目标阶段完成，且有可引用的产物或正式汇总。 |
| 部分完成 | 某些阶段或 shard 完成，但没有完整可比结果。 |
| 失败 | 作业在目标阶段前后失败，结果只作为过程证据。 |
| 停止 | 主动停止或因预设 gate 停止；不等于模型或代码损坏。 |
| 取消 | 排队或启动前取消。 |
| 无效谱系 | checkpoint、User 或 max-step 与计划不符，不能用于结论。 |
| 仅诊断 | 用来定位机制或做 A/B，不参与模型选型。 |

**指标和可比性**

- tau2 官方评测中，`pass@1` 是全部 simulation reward 的平均；`pass@4(any)` 是 4 次试验至少成功一次的任务比例；`pass^4` 是 4 次全部成功的任务比例。
- VitaBench 的 `Avg@4 / pass@1`、`pass@4(any)` 和 `pass^4` 采用其自己的任务/窗口协议，不能直接和 tau2 的数值混排；`Avg@4` 是 VitaBench summary 中按四次 trial 聚合的字段，和 `pass@1` 在该实现中并列报告，不应拿来做 tau2 的同名指标替换。
- 结果只有在 Agent、User checkpoint、parser、thinking 状态、temperature、max_steps、任务配额、seed 和试验数都匹配时才做因果比较。早期 Gemini、local User、旧 parser 和 thinking-on 结果保留为背景，不与后期严格协议直接排名。
- 正式结果优先取实验目录中的最终 README、REPORT、`CREDIT_COMPARISON.md` 或 JSON summary；`output/doc/EXP_QA.md` 和 `output/doc/INDEX.md` 用于交叉核对；原始 run log 主要用于确认作业状态和失败原因。缺少汇总的作业明确写“无可比汇总”，不从零散日志推算成绩。
- 交叉核对时发现几处旧文档状态/数字滞后：`EXP_QA.md` 某历史行写过 Qwen3.5 thinking-on `47.20%`，最终 experiment README 和 matched summary 均为 `47.25%`，本文采用后者；`TAU2_SINGLE_CALL_CREDIT_TEST.md` 和 [credit-core README](../output/experiments/tau2-agent-single-call-credit-core/README.md) 仍停留在 `Pending`/“由 held-out 决定”的设计状态，其后更新的 [CREDIT_COMPARISON.md](../output/experiments/tau2-agent-single-call-credit-core/CREDIT_COMPARISON.md) 才是正式选型记录；[TAU2_TURN_CREDIT_V2.md](../output/doc/TAU2_TURN_CREDIT_V2.md) 仍是设计/候选说明，不能覆盖后来的 `v1-matched` 保留决策。

主要索引：[实验总索引](../output/doc/INDEX.md)、[跨实验 QA](../output/doc/EXP_QA.md)、[turn-credit-v2 说明](../output/doc/TAU2_TURN_CREDIT_V2.md)。

## 2. 先看结论

| 主题 | 目前得到的结果 | 做得好的地方 | 仍然不够的地方 |
|---|---|---|---|
| 训练链路 | tau1、tau2 SFT/RL、官方评测和 VitaBench 本地角色评测均已跑通至少一条完整路径。 | 依赖、服务、数据、训练、转换和评测之间的接口问题被逐个定位并留下日志。 | VitaBench 远程 User/Evaluator 的 400-task 完整 run 没有得到最终成绩；Qwen3-4B VitaBench 的正式汇总也未形成。 |
| tau2 显存与数值 | 单条超长轨迹导致的 logits OOM 已通过完整上下文 tokenization、每轨 16,384 cap、重采样和 group filter 解决；K2 KL 使 100-update 稳定性实验不再 NaN。 | 后续 RL 运行能完成 100 updates，健康检查通过。 | K2 只解决数值崩溃，不能自动解决错误参数、重复调用或 max_steps 退化。 |
| 当前全局 Agent 谱系 | Agent-owned Contract + boundary 的 long100 iter99 是全局 selected/default：两 seed `29.50 / 56.50 / 10.50%`（`pass@1 / pass@4(any) / pass^4`）。 | 相对 selected SFT `23.63 / 43.50 / 8.50%`，提升 `+5.88 / +13.00 / +2.00pp`；两 seed 的 pass@1 和 pass@4(any) 都提升。 | pass^4 的区间仍跨 0；这是按预先规则得到的 point-estimate/health/protocol “win”，不是每个差异都显著。 |
| strict-single 与 credit | strict-single-v1 谱系 iter199 为 `31.13 / 61.50 / 10.50%`；credit-core 的 v2 λ=.1 点估计最好，但因 pass@4(any) 与 v1 打平，预注册规则保留 `v1-matched` recipe。 | 通过同一 SFT、User、配额和 KL 的 controlled ablation，避免只看训练 reward 选型。 | DB accuracy 仍低，v2 的优点主要体现在行为规则和局部 credit，尚未证明它严格胜过 v1。 |
| VitaBench | 本地 Qwen3.5-4B + Qwen3.6-27B roles 的 **legacy `artifacts-v2` protocol-v5** 完整评测：400 tasks × 4，`23.0625 / 47.75 / 4.50%`。该数字与 published 22.0 不是严格同协议比较。 | shard 完整性、重复/缺失检查和 SFT export 均通过；本地线避开了远程 API 费用（总 compute cost 未配置）。 | Evaluator non-thinking 虽快 4.36×，链式质量 gate 未通过；远程 full eval 因协议、API、配额和成本多次中止。 |
| 下一步 | 先在固定 SFT/User/配额/LR/KL/评测协议下改 reward 与 advantage，再选 recipe；之后才启动 OPD。 | 顺序和比较条件已经写入实验规范。 | 目前没有任何 OPD 实际作业；Retail teacher iter119 只是第一候选，Telecom 还需补验证，Airline expert 已排除。 |

## 3. 阶段时间线总览

| 时间 | 阶段与实验 | 做了什么、改变了什么 | 结果与评价 |
|---|---|---|---|
| 07-07 | 最早的镜像/框架探针与 checkpoint 转换（`output/jobs/` ad-hoc jobs） | 先验证 Python、Torch、slime、Megatron-Core 和 SGLang，再把 Qwen3-4B HF 权重转成 Megatron `torch_dist`；提交脚本被复制到 job 目录后，先后修正了相对路径问题。 | [image probe](../output/jobs/fush--check_image_env-0707-200007/run_20260707_200007.log) 和 [Megatron/SGLang probe](../output/jobs/fush--check_megatron_sglang-0707-210922/run_20260707_210922.log) 通过；三次转换因找不到 `scripts/models/qwen3-4B.sh` 失败，第四次 [转换](../output/jobs/fush--convert_qwen3_4b_to_torch_dist-0707-220702/run_20260707_220702.log) 成功保存 iteration 1。无 run log 的 submission-only 尝试（[image](../output/jobs/fush--check_image_env-0707-195954/)、[Megatron/SGLang](../output/jobs/fush--check_megatron_sglang-0707-210858/)、[转换](../output/jobs/fush--convert_qwen3_4b_to_torch_dist-0707-212105/)）也保留，但不当作模型结果。 |
| 07-09～07-10 | [tau-bench](../output/experiments/tau-bench/README.md) | 生成 retail mock data，转换 Qwen3-4B checkpoint，接入 custom tau rollout 和 o4-mini User；修复 log-prob 缺失。 | smoke 和 3-step 真实训练成功。最初的 filter 会丢掉约 90% 冷模型 group，User simulator 延迟高。 |
| 07-11～07-21 | [tau2-deps-probe](../output/experiments/tau2-deps-probe/README.md)、[tau2-eval](../output/experiments/tau2-eval/README.md)、[tau2-eval-local-user](../output/experiments/tau2-eval-local-user/README.md) | 找出新镜像缺失包，换成官方 `HalfDuplexAgent + TextRunConfig` 评测，修复断点续跑和增量写入。 | tau2 评测接口固定下来；早期不同 User/temperature/parser 的结果只能作背景。 |
| 07-13～07-21 | [tau2-sft](../output/experiments/tau2-sft/README.md)、[tau2-user-sft](../output/experiments/tau2-user-sft/README.md)、[tau2-eval-user-stop](../output/experiments/tau2-eval-user-stop/README.md)、[tau2-eval-user-stop-parser](../output/experiments/tau2-eval-user-stop-parser/README.md) | 构建 Agent/User SFT；为 User 增加 `###STOP###`；打开 User 的 qwen parser；修复 telecom User 的字面量 `<tool_call>`。 | User 的 max_steps 显著下降，telecom User tool calls 开始执行，后续 RL 统一使用 v1 STOP User/parser-on。 |
| 07-13～07-20 | 数据与协议修复 | 从旧 strict slice 回到保留 multi-tool 的 source；检查 loss mask、tool ownership 和历史/答案格式。 | raw 33,531 行重建为 23,411 条 strict no-thinking 数据；确认主要问题是数据/协议和 telecom 覆盖，不是 loss mask。 |
| 07-20～07-30 | [tau2-rl](../output/experiments/tau2-rl/README.md)、[stability sweep](../output/experiments/tau2-rl-stability-k2-fieldreward-lr-sweep/README.md) | 先处理 actor log-prob、显存和 reward flat/collapse，再比较 K2、field reward、penalty 与学习率。 | 2e-6/3e-6/5e-6 都跑满 100 updates；高 LR 行为退化，2e-6 最稳，但官方 eval 尚不能证明 RL > SFT。 |
| 07-31～08-04 | [traj-pattern](../output/experiments/tau2-traj-pattern/README.md)、[SFT audits](../output/experiments/tau2-sft-quality-audit/README.md)、[multitool audit](../output/experiments/tau2-sft-multitool-quality-audit/README.md)、[local-first SFT](../output/experiments/tau2-sft-local-first-relaxed/README.md) | 用确定性规则和校准 judge 分析失败；比较 single/multi；扩展 local-first target-prefix 数据。 | 80–93% 失败是 action/DB mismatch；local-first 提高 pass@1/@4(any)，但 pass^4 下降 6pp，暂不作为默认。 |
| 07-31～08-03 | [VitaBench smoke](../output/experiments/vitabench-qwen35-smoke/README.md)、[远程 full](../output/experiments/vitabench-qwen35-full-eval/README.md)、[本地 roles](../output/experiments/vitabench-qwen35-local-role-eval/README.md) | 处理 29 个依赖、四套中文任务、shard queue、schema-5 journal 和本地 User/Evaluator。 | Qwen3.5 本地 legacy protocol-v5 完成 400×4；远程 full 因 rubric/API/成本失败或停止；Qwen3-4B 线完成 40 个固定 10-task shards、1,600/1,600 trajectories，但 aggregate validator 因 profile bug 判 invalid。 |
| 08-04～08-06 | [boundary SFT](../output/experiments/tau2-sft-agent-user-boundary-v2/README.md)、[boundary RL](../output/experiments/tau2-rl-agent-user-boundary-v2/README.md)、[long100](../output/experiments/tau2-rl-agent-user-boundary-v2-long100/README.md)、[long200](../output/experiments/tau2-rl-agent-user-boundary-v2-long200/README.md)、[domain experts](../output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts/README.md) | 加入 Agent-owned Contract、boundary anchors、turn-aware-v1；完成 100/200 update 和单域 teacher 探索。 | long100 iter99 成为全局默认；long200 是高 KL、偏 Telecom 的诊断候选；Retail teacher 首选，Airline 排除。 |
| 08-09～08-16 | [Qwen3.5 baseline](../output/experiments/tau2-qwen35-nonthinking-eval/README.md)、[strict-single-v1](../output/experiments/tau2-agent-single-call-v1/README.md)、[credit-core](../output/experiments/tau2-agent-single-call-credit-core/CREDIT_COMPARISON.md) | 把每个 Assistant target 展开成单一 native call；做 User/max_steps/parser 交叉矩阵；比较 turn-credit-v1/v2。 | strict-single iter199 在自身谱系内继续提升；credit-core 按严格主指标规则保留 v1；错误谱系续训被停止或取消。 |

## 4. tau-bench / tau1：先把最小闭环跑起来

| 步骤 | 改动和作业 | 得到的结果 | 好处 | 不足与影响 |
|---|---|---|---|---|
| 依赖与数据 | 初次 mock-data job 因缺 `litellm` 失败；转换 wrapper 因 `submit.sh` 只复制提交脚本而找不到兄弟脚本失败。随后只安装 litellm，移除会降级 protobuf 的 `google-generativeai`，改用 ChatAnywhere 的 OpenAI-compatible `o4-mini`。 | retail train/dev JSONL 和 torch_dist checkpoint 生成成功。相关记录在 [MONITOR.md](../output/experiments/tau-bench/MONITOR.md)。 | 找到镜像依赖与脚本路径的真实边界。 | API User 较慢，不能把这条小规模 run 的 reward 当作学习曲线。 |
| log-prob 修复 | 第一个 8-GPU run 在 `compute_advantages_and_returns` 因所有 log-prob 为 `None` 失败。先用 `kl_coef=0.001` 验证 actor forward，再让 sglang 返回每 token log-prob，按 Qwen template 对齐并将 `kl_coef` 恢复为 0。 | `pt-prux2yjk` smoke 完成 8/8 rollout、actor log-prob forward、backward 和 iter-0 checkpoint；不再出现 `loss.py:703`。 | 采用了真正的 on-policy log-prob 方案，避免用 KL workaround。 | 自定义 rollout 必须显式满足 slime 的 log-prob contract；这是后续 tau2 rollout 设计的前置经验。 |
| 多步真实 run | `pt-8grukdrk` 用 8 GPU、batch 32、3 次 rollout、`kl_coef=0`、关闭冷启动 filter，完成 iter0～2。 | raw reward `0.71875 → 0.875 → 0.78125`，梯度和 checkpoint 正常。 | 证明 rollout→advantage→backward→checkpoint 的循环可重复。 | reward 只是小批次波动，不代表模型已经学会；filter 开启时冷模型约 90% group 被拒绝。 |

关键日志：[log-prob smoke](../output/experiments/tau-bench/jobs/fsh--run-qwen3-4B-serviceagent-tau-smoke-kl0-v2-0710-181141/run_20260710_181141.log)、[3-step run](../output/experiments/tau-bench/jobs/fsh--run-qwen3-4B-serviceagent-tau-real-0710-200409/run_20260710_200409.log)。

最早的 ad-hoc framework run 使用的是 DAPO-math/AIME 数据，不属于 tau-bench 成绩；[07-07 run log](../output/jobs/fush--run-qwen3-4B-serviceagent-0707-222151/run_20260707_222151.log) 和 [07-09 run log](../output/jobs/fush--run-qwen3-4B-serviceagent-0709-183810/run_20260709_183810.log) 仅作为 slime + Ray + SGLang 的早期连通性记录，不纳入后续 Agent 选型。

tau1 的关键实现位于 [generate_with_tau.py](../examples/tau-bench/generate_with_tau.py)、[trainable_agents.py](../examples/tau-bench/trainable_agents.py)、[sglang_tool_parser.py](../examples/tau-bench/sglang_tool_parser.py) 和 [run_qwen3_4B.sh](../examples/tau-bench/run_qwen3_4B.sh)：分别负责环境 rollout、工具调用/每 token log-prob、SGLang tool parsing 和训练入口。

## 5. tau2：数据、协议、SFT、User 和官方评测

**5.1 依赖与评测框架**

[tau2-deps-probe](../output/experiments/tau2-deps-probe/README.md) 确认新镜像主要缺少 `hatchling`、`editables`、`toml`、`deepdiff`、`litellm` 和 `gymnasium`；5-task airline smoke 平均 reward 为 `0.60`。之后官方评测统一到 tau2 的 `HalfDuplexAgent`、`TextRunConfig` 和 `run_domain`，固定为 airline 20、retail 40、telecom 40 个任务，每任务 4 trials。

早期评测还解决了两个工程问题：交互式 resume prompt 会阻塞非交互作业，改为 timestamped save prefix；httpx transient `ReadError` 会丢掉整批结果，改为 retry 和 atomic incremental writes。早期 current protocol 的历史对照为 Qwen3.5 thinking `68.2/91`、raw Qwen3 `26.2/58`、旧 SFT `24.5/43`、旧 GRPO `7.5/17`（这里仅展示 `pass@1/pass@4(any)`，与后期 v1 User/parser-on 结果不可直接比较）。

评测协议迁移期间的关键快照（都只作过程基线，不能跨行做模型排名）如下：

| 协议快照 | 主要变化 | 代表结果 `pass@1/pass@4(any)` | 学到的东西 |
|---|---|---:|---|
| legacy strict parser、thinking off | Qwen3.5 关闭 thinking，严格要求 tool-call；旧 User/temperature | Qwen3.5 `13/36`、raw Qwen3 `5/22`、早期 GRPO `34/62` | parser/解码模式会改变绝对分数，不能把它当模型能力的单独变化。 |
| prose-as-respond 修复 | plain-prose 回复转为 `respond`，Qwen3.5 恢复 thinking | Qwen3.5 `39/73`、GRPO `32/56`、raw Qwen3 `24/48` | 解析策略本身可带来大幅跳变；先固定协议再比较训练。 |
| current official runner（旧 User） | `HalfDuplexAgent`、temperature 0.6/0.0、100 tasks×4 | Qwen3.5 `68.2/91`、raw Qwen3 `26.2/58`、旧 SFT `24.5/43`、旧 GRPO `7.5/17` | 这是 tau2 评测 API 的稳定基线，但 User/parser 与后续 v1 STOP 主线不同。 |

早期 raw-adapter smoke 还经历了“已有结果文件触发 EOF resume prompt”和“sglang 连接瞬断导致整批结果丢失”两次失败；timestamped save prefix、transport retry 和 atomic incremental writes 后才可重复。上述过程证据见 [tau2-eval README](../output/experiments/tau2-eval/README.md) 的 Tasks 日志清单。

[tau2-eval-local-user](../output/experiments/tau2-eval-local-user/README.md) 的旧 local-Qwen User 快照（100 tasks×4；仅作 User 分布背景）为：

| Agent | `pass^1/pass^4` | `pass@1/pass@4(any)` | 备注 |
|---|---:|---:|---|
| Qwen3.5-4B | `38.25/19.00%` | `40.00/54.00%` | local User，非 v1 STOP/parser-on 主线 |
| Qwen3-4B-Instruct-2507 SFT | `24.50/13.00%` | `25.00/43.00%` | 同上 |
| Qwen3-4B-Instruct-2507 raw | `18.00/4.00%` | `16.00/35.00%` | 同上 |
| 旧 tau2 GRPO | `6.25/0.00%` | `8.00/17.00%` | 历史 checkpoint，不用于当前选型 |

该旧 README 的 `pass@1` 是当时定义的 trial-0/task 指标；后期主线采用 sim-weighted `pass@1`，所以这张表只用于说明 User/协议迁移，不能与后面的 v1 STOP 数值直接相减。

[tau2-hf-convert](../output/experiments/tau2-hf-convert/README.md) 负责把 User/Agent 的 torch_dist checkpoint 转成 sglang 可用的 HF 目录；它是评测桥接步骤，不是新的模型训练结果。[tau2-eval-user-sft](../output/experiments/tau2-eval-user-sft/README.md) 的初始 local User 评测暴露出 User 几乎不发 STOP，episode 因此经常撞到 max_steps，这直接促成了下一轮 STOP 数据和 User parser 修复。

**5.2 Agent 数据和协议修复**

数据入口和转换脚本是 [prepare_sft_data.py](../examples/tau2-bench/sft/prepare_sft_data.py)、[build_agent_boundary_v2.py](../examples/tau2-bench/sft/build_agent_boundary_v2.py)、[build_agent_single_call_v1.py](../examples/tau2-bench/sft/build_agent_single_call_v1.py) 和 [prepare_rl_data.py](../examples/tau2-bench/rl/prepare_rl_data.py)；后文的谱系差异都以这些脚本生成的 manifest/README 为准。

| 发现 | 做的改变 | 结果、优点和不足 |
|---|---|---|
| 旧 strict slice 把 33,531 raw rows 压成 7,060 rows（airline 1,939、retail 3,482、telecom 1,639）。Telecom 中 correct=1 仅 574、correct=0 有 1,065，平均 Assistant turns 从 11.84 降到 4.98；工具覆盖集中在 `get_customer_by_phone` 1,142 次和 `get_details_by_id` 48 次。 | 复核发现官方 half-duplex 实际支持一个 Assistant message 的 multi-tool list；原始历史有 33,348 条 multi-tool、答案有 4,147 条。修正转换逻辑，只丢弃 `content+tool` 混合目标，不再按 multi-tool 数量硬删。 | 重建得到 23,411 条 strict no-thinking rows 和 27,779 个 multi-tool Assistant messages。这样保住了真实工具序列，但旧协议仍把 observation 写成 `role=user`/`Tool result`，后续需要 native boundary。 |
| 6-row loss-mask debug 共 644 个 masked tokens、0 warnings，assistant span 对齐正常。 | 保留 qwen3 loss mask，转而检查 ownership、工具覆盖和上下文长度。 | 排除了 mask bug 作为主要根因，避免在错误方向上改训练代码。 |
| 旧 multi-tool SFT 的一个 12.4K-token 样本触发 CUDA OOM；另一个 smoke wrapper 因按 job log 目录解析兄弟脚本而失败。 | 将训练长度过滤到 max8192，并修正 wrapper 的绝对路径解析。 | `pt-1cf3ger2` 完成 SFT，说明当时的 OOM 是数据长度问题；这条经验后来被 RL 的 16,384 per-trajectory cap 取代。 |
| [tau2-sft-quality-audit](../output/experiments/tau2-sft-quality-audit/README.md)：19,318 rows、2,496 dialogs；strict keep 998 rows（5.17%），75.44% dialogs 有 same-turn multi-policy violation。 | 将它保留为 historical single-policy baseline，不直接发布为新训练集。 | 审计暴露了数据噪声，但“multi 多”本身不是质量代理。 |
| [tau2-sft-multitool-quality-audit](../output/experiments/tau2-sft-multitool-quality-audit/README.md)：full-source multi 为 2,295/2,496（91.95%）；matched success multi-single quality diff `+0.005`，95% CI `[-0.034,+0.045]`；341 条 multi-exposed failures 中只有 66 条被归为 multi caused/contributed。 | 采用 dependency-safe-multi，自动 keep 只留 69 dialogs/630 targets 作高精度 seed，不把它当全量训练集。 | 避免错误地 hard-drop multi；代价是数据筛选规则更复杂，且 seed 严重偏域。 |
| [tau2-sft-local-first-relaxed](../output/experiments/tau2-sft-local-first-relaxed/README.md)：2,046 success dialogs → 1,863 dialogs、15,807 canonical targets；广覆盖 budget 为 airline 8,873、retail 10,092、telecom 353。 | 用 calibrated local Qwen3.6-27B 做 target-prefix、dependency-safe-multi、target-only mask 筛选，从 raw init 训练 2 epochs。 | official seed300：raw `20.5/43/6`，旧 SFT `21.25/41/10`，local-first SFT `27/54/4`。coverage 和单次成功变好，但 `pass^4` 下降 6pp，暂不作为默认。 |
| [tau2-sft-agent-user-boundary-v2](../output/experiments/tau2-sft-agent-user-boundary-v2/README.md)：Contract-only 与 Contract+boundary 都固定 19,318-row、2-epoch budget；后者补充 970 ordinary telecom、580 handoff、390 ownership-repair anchors。 | 引入 Agent-owned native `tools=`、`role=tool`、`qwen3_full` mask、User-event ownership boundary；比较 contract-only 与 boundary。 | Contract-only 被 probes/namespace gate 拒绝；Contract+boundary 两 seed selected SFT 为 `23.63/43.50/8.50`，成为后续 boundary RL 的共同起点。 |

**5.3 User-SFT、STOP 和 parser**

- [tau2-user-sft](../output/experiments/tau2-user-sft/README.md) 先做混合 User SFT，再从 AReaL 成功 terminal rows 增加 `###STOP###`。STOP v1 让 User stop 比例升到约 80–98%，max_steps 降到约 0–3%；v2 重新对齐 scenario/guidelines，但在 telecom 上较弱，因此后续主线保留 v1。早期 [User issue-stats ad-hoc 分析](../output/jobs/fsh-tau2-user-sft-issue-stats-0715-015710/run_20260715_015710.log) 记录了未加 STOP 时大量 `max_steps`/`too_many_errors`，是修复动机而非独立模型结果。
- [tau2-eval-user-stop](../output/experiments/tau2-eval-user-stop/README.md) 证明“User 不停止”会把 episode 推到 max_steps；[parser-on 重跑](../output/experiments/tau2-eval-user-stop-parser/README.md) 让 telecom User tool calls 从几乎 0 增至约 1,010–1,715，telecom 指标显著上升。这个 parser 修复是后续 RL/eval 的固定前提。
- [v2 parser 对照](../output/experiments/tau2-eval-user-stop-v2-parser/README.md) 的 common-Agent 平均（顺序为 `pass^1/pass^4/pass@4(any)`）为 Gemini `47.25/22.50/74.50`、original Qwen `28.12/11.50/44.50`、v1 `35.38/12.50/61.50`、v2 `23.88/10.00/41.00`。它说明 v1 是当前 local User 默认，但这是 system-level User/Agent 组合结果，不是 User checkpoint 的单独排名。
- [tau2-rl-usercmp](../output/experiments/tau2-rl-usercmp/README.md) 曾设计 v2 User、v1 STOP User、raw Instruct User 三个只变 User 的 RL 臂；该 README 没有合并后的正式 eval 表，后来的 strict-single User 矩阵才成为可引用的控制证据。

parser-on 的直接前后对照（官方 `pass^1/pass^4`，每格 400 simulations）为：Qwen3.5 `28.7/10.0 → 47.2/19.0`、raw Qwen3 `18.0/4.0 → 23.5/6.0`、multitool SFT `17.8/6.0 → 23.25/8.0`；telecom User-side structured calls 从 0 增至 `1,010–1,715`。因此 parser 是必要的执行修复，但残余的 prose+tool 混合和 User 质量仍限制上限。

**5.4 历史基线和模型诊断**

[tau2-qwen35](../output/experiments/tau2-qwen35/README.md) 的 thinking/full parser smoke 暴露约 40% bad output；关闭 thinking 后 parser error 消失，但一次 smoke 只有 1/3 成功。[最终 Qwen3.5 non-thinking baseline](../output/experiments/tau2-qwen35-nonthinking-eval/README.md) 在 v1 STOP User、parser-on、两 seed、100 tasks×4 下为 `32.75/61/9`，action/DB accuracy `60.20/35.09%`。同 seed 的 thinking-on 为 `47.25/78/19`，所以不能把 non-thinking baseline 当作 Qwen3 RL 的同模型因果对照。

[raw-Agent/raw-User parser-on 矩阵](../output/experiments/tau2-raw-agent-raw-user-parser-on/README.md) 补齐了 User 分布的反事实：Qwen3 raw Agent 在 V1/RAW User 下为 `23.88/45.50/6.50` 与 `17.38/35.50/2.50`，Qwen3.5 thinking 为 `48.50/76/22` 与 `53.25/78/25`，Qwen3.5 non-thinking 为 `32.75/61/9` 与 `25.63/53.50/7`（均为两 seed `pass@1/pass@4(any)/pass^4`）。同一 Agent 会随 User 改变方向，说明 User 必须作为协议变量单独报告。

## 6. tau2 RL：从 OOM/崩溃到 credit 选型

**6.1 初始 RL 和显存根因**

| 阶段 | 发生了什么 | 最终判断 |
|---|---|---|
| 初始真实 8-GPU 尝试 | 先后遇到空 `WANDB_MODE`、global batch 与 DP3 不整除、ref log-prob forward 申请 25.96 GiB、allocator 参数破坏 SGLang TorchMemorySaver、再次 OOM；单纯降低 log-prob chunk size 无效。 | 这些是暴露问题的尝试，不能把每次失败都归为同一原因。 |
| 07-26 verify | p99 trajectory 约 16K，最大约 104K；`first_fit_pack` 允许超过 bin cap 的 singleton，导致单条超长轨迹形成独立 oversized microbatch，最终 `[T,V]` logits OOM。旧 `_limit_training_context` 通过删掉 leading messages 形成 on-policy 失配。 | 根因是“超长样本在训练前没有被截断/替换”，不是整体显存不足，也不是 chunking 能解决的问题。 |
| 修复 | 改为完整 conversation verbatim tokenization；`TAU2_RL_MAX_TRAIN_TOKENS=16384` 与 `max_tokens_per_gpu` 相同；单样本超长最多重采样 2 次，仍超长才 placeholder/remove；加入 `drop_zero_std_or_unsampleable` 和 `test_rollout_logic.py` preflight。 | 后续真实 RL 不再因该机制 OOM；对 on-policy 语义更忠实。代价是超长样本会被重采样或舍弃，必须报告 replacement/zero-variance rate。 |
| reward/recipe 诊断 | 早期 recipe 的 LR `1e-6`、batch 48、temperature .65、clip .2 远弱于 AReaL 对照；约 36% group 因 zero variance 被丢掉。对齐到 LR `1e-5`/temperature 1/clip .4 后，airline 仍在 step49–71 collapse：KL 达到约 .42、reward→0、truncation 最高约 60%、随后 backward NaN。 | 二元 DB/communication reward 太稀疏；1,906 条失败轨迹中 98.6% 仍有 action_match，平均约 .55，主要是“工具名对、参数错”。 |

**6.2 K2、field reward 和学习率稳定性**

[stability sweep](../output/experiments/tau2-rl-stability-k2-fieldreward-lr-sweep/README.md) 固定 airline-only、v1 STOP User、K=8、100 updates、`k2` KL `0.01`、entropy `0`，并加入 tool-name/argument/DB/env/communication field reward、malformed/nonexistent/repetition/max_steps penalties 和 shaped zero-std replacement。

| LR | 训练状态 | 末段 raw/KL/truncation | 行为结论 |
|---:|---|---:|---|
| `2e-6` | 100 updates，finite，无 NaN/Inf | `.379 / .0593 / .42%` | malformed `.115→.031`、nonexistent `.023→.003`；重复调用仍略升 `.185→.211`，但整体最稳。 |
| `3e-6` | 100 updates，finite | `.396 / .1604 / .63%` | 可运行，但 repetition `.107→.247`，KL 漂移更大。 |
| `5e-6` | 100 updates，finite | `.452 / .3443 / .21%` | repetition `.135→2.557`、malformed `.063→.177`，高 LR 退化，拒绝。 |

同协议官方 seed300（[final eval](../output/experiments/tau2-rl-stability-k2-fieldreward-final-eval/README.md)）为：

| Agent | pass@1 | pass@4(any) | pass^4 | 相对 SFT `23.25/46/8` |
|---|---:|---:|---:|---:|
| SFT | 23.25% | 46.00% | 8.00% | 基线 |
| RL `2e-6` | 25.50% | 49.00% | 7.00% | `+2.25/+3/-1pp`，pass@1 CI `[-2.84,+7.34]` |
| RL `3e-6` | 23.50% | 47.00% | 5.00% | 无稳定优势 |
| RL `5e-6` | 18.50% | 39.00% | 3.00% | 高 LR 退化 |

2e-6 是稳定候选，但 CI 包含 0，不能写成“已证明 RL>SFT”。[轨迹分析](../output/experiments/tau2-traj-pattern/README.md) 进一步显示 raw/SFT/RL 的 80–93% 失败属于 action/DB mismatch；SFT→RL 的 multitool batch rate 从 0.3% 升到 24.1%，golden-action accuracy 仅 46.7%→49.9%，说明 field shaping 会鼓励“多调用/叫对工具”，却没有解决正确参数和最终 DB 状态。

**6.3 Contract/boundary、long100、long200 和 domain expert**

| 实验 | 改动 | 结果（pass@1 / pass@4(any) / pass^4） | 好处 | 不足 |
|---|---|---|---|---|
| [boundary Pilot A](../output/experiments/tau2-rl-agent-user-boundary-v2/README.md) | Contract+boundary SFT；turn-credit-v1；Assistant-turn penalty；LR `2e-6`、K=8、K2 `.01`、max_steps 60。 | iter9 两 seed 健康，namespace 改善；但 strict behavior gate fail：framework truncation 30.63%、malformed 18.13%、execution-error 15.63%、max_steps 29.58%。 | 训练状态 finite，非 OOM/NaN/optimizer corruption；边界确实减少 namespace/probe。 | gate 失败是行为诊断，不是 checkpoint 坏了；不能把 iter9 当最终胜者。 |
| [local-first RL](../output/experiments/tau2-rl-local-first-dependency-safe-k2-fieldreward/README.md) | 从 local-first relaxed SFT 做 airline-only K2/field-reward RL，预设 promotion gate 后才续训。 | iter99 `19/38/5`，对应 SFT `27/54/4`；too_many_errors 100、repetition .6224，promotion fail，未续训。 | 把“数据覆盖变好”与“RL 能否利用这些数据”分开验证。 | Telecom 训练覆盖仍只有 353 rows，User/tool boundary 缺失；说明 local-first SFT 不能直接保证 RL 改善。 |
| [boundary long100](../output/experiments/tau2-rl-agent-user-boundary-v2-long100/README.md) | 保留 recipe，换 fresh optimizer/checkpoint root；iter9/19 改用 health-only gate，跑满 100 updates。 | selected SFT `23.63/43.50/8.50`；历史 iter9 `25.25/50/9`；iter99 **`29.50/56.50/10.50`**。每域 pass@1 都提升。 | 两 seed 都比 SFT 和 iter9 提升，训练健康；按预宣规则判 `win`。 | pass^4 CI 仍跨 0；point win 不是显著性声明。 |
| [checkpoint curve](../output/experiments/tau2-rl-agent-user-boundary-v2-checkpoint-curve/README.md) | 同 lineage seed300 检查 iter9/39/69/89/99。 | pass@1 `25.00/26.25/24.00/29.75/27.50`，非单调。 | 证明“继续训练必然单调变好”不成立。 | 单 seed 诊断不能取代两 seed 选型。 |
| [boundary long200](../output/experiments/tau2-rl-agent-user-boundary-v2-long200/README.md) | exact resume optimizer/RNG/sampler；update176 因 K2 10-step `.14287` 且连续两步 `>=.20` 停止，之后 diagnostic waiver 到 iter199。 | iter199 **`32.38/58/11.50`**，相对 iter99 `+2.88/+1.50/+1pp`；Telecom +8.75pp，但 Airline 平均 -5pp，late KL 约 .32374。 | checkpoint 有效且可复现；为 Telecom-oriented teacher 提供候选。 | 高 KL、namespace raw attempts 增加、整体 CI 不稳；不做 200→299，long100 仍是 balanced default。 |
| [domain experts](../output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts/README.md) | 从 long100 iter99 各做 30-update Airline/Retail/Telecom 单域 continuation，与 mixed prefixes 对照。 | Retail iter119 target pass^4 `21.25%`（parent 13.75、mixed 16.25）最适合作为首个 OPD teacher；Telecom iter129 target `1.25%` 低于 parent 2.50/mixed 3.75，需额外验证；Airline 排除。 | 把 teacher 选择和域遗忘显式化。 | 这是 teacher 候选评测，不是 OPD 已启动。 |

long100 的第一次 `b` 阶段在 update 10 前因 Megatron scheduler horizon `480`/`960` 不一致失败；只覆盖 scheduler override 的 retry 成功，保留 optimizer state 并通过 iter19/iter99 health gate。这个工程性 retry 不改变 long100 的 lineage 或最终比较。

**6.4 strict-single-v1、User/max_steps 控制和 credit-core**

**strict-single-v1 主线。** [实验](../output/experiments/tau2-agent-single-call-v1/README.md) 将 19,318-row boundary 数据展开为 25,761 个 source-order target，去掉 53 个超过 16,384 token 的样本后得到 25,708 rows；每个 Assistant message 只序列化一个 native Agent call。固定 v1 User、parser-on、LR `2e-6`、K=8、K2 `.01` 后：

| Checkpoint | pass@1 | pass@4(any) | pass^4 | 变化 |
|---|---:|---:|---:|---|
| SFT | 25.88% | 49.50% | 8.50% | 基线 |
| RL iter99 | 27.50% | 53.50% | 10.00% | 相对 SFT `+1.62/+4/+1.5pp` |
| RL iter199 | **31.13%** | **61.50%** | **10.50%** | 相对 SFT `+5.25/+12/+2pp`，相对 iter99 `+3.63/+8/+0.5pp` |

iter199 的 action/DB accuracy 为 `65.46/27.39%`，termination 为 user_stop 723、too_many_errors 2、max_steps 75、infra 0。它是该 strict-single 谱系的最终 checkpoint，不宣称全面支配 boundary 谱系。

**User 与 max_steps 控制。** User 分布是重要混杂变量：同一 Agent 在 v1 User 与 raw User 下的结果不能互换解释。User-ablation 中，raw User 下 SFT 为 `28/51/9`，v1-trained RL iter199 为 `26.88/52.50/8`；同一 raw User 训练的 RL iter199 为 `31.50/58/8`。raw-User max120 continuation 为 `30.13/58.50/7`；max120 将 training truncation 从 18.38% 降到 1.19%，但没有带来一致性胜利。五个 crossed parser-on cells 都低于 rollout-User-matched controls，说明一致性问题不只是“每轮一个 call”。

为避免把这些控制作业混在一句话里，下面列出同一两 seed/800-simulation 口径下的关键结果（均为 `pass@1 / pass@4(any) / pass^4`）：

| 谱系 | 训练 User | rollout `max_steps` | 评测 User | 结果 |
|---|---|---:|---|---:|
| [single-call SFT](../output/experiments/tau2-agent-single-call-v1-user-ablation/README.md) | — | — | raw | `28.00/51.00/9.00%` |
| [raw-User RL iter99](../output/experiments/tau2-agent-single-call-v1-raw-user-rl100/README.md) | raw | 60 | raw | `29.25/54.00/6.50%` |
| [raw-User RL iter199](../output/experiments/tau2-agent-single-call-v1-raw-user-rl200/README.md) | raw | 60 | raw | `31.50/58.00/8.00%` |
| [raw-User RL iter99 max120](../output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl100/README.md) | raw | 120 | raw | `28.38/56.50/9.50%` |
| [raw-User RL iter199 max120](../output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl200/README.md) | raw | 120 | raw | `30.13/58.50/7.00%` |
| [v1-trained RL iter199](../output/experiments/tau2-agent-single-call-v1-user-ablation/README.md) | v1 | 60 | raw | `26.88/52.50/8.00%` |

raw-User 训练的 iter199 在 raw User 上的 pass@1/pass@4(any) 最高，但 pass^4 仍低于 raw-User SFT；max120 明显减少截断，却没有稳定地提高任务成功或一致性。

[crossed parser-on](../output/experiments/tau2-agent-single-call-v1-rl-crossed-parser-on/README.md) 还把训练 User 与评测 User 交叉：V1-RL99→RAW 为 `25.25/50.50/5.50`，RAW-RL99 max60→V1 为 `29.25/54/6.50`，RAW-RL99 max120→V1 为 `28.38/56.50/9.50`，RAW-RL199 max60→V1 为 `31.50/58/8`，RAW-RL199 max120→V1 为 `30.13/58.50/7`；五个 crossed cell 都低于各自 rollout-User-matched control，支持“分布匹配比单纯 one-call 协议更关键”的判断。

**credit-core controlled comparison。** [CREDIT_COMPARISON.md](../output/experiments/tau2-agent-single-call-credit-core/CREDIT_COMPARISON.md) 固定 strict-single SFT、v1 User、K=8、LR `2e-6`、K2 `.01`、100 updates、600 groups/arm，只改 reward/advantage：

| Arm | pass@1 | pass@4(any) | pass^4 | action / DB |
|---|---:|---:|---:|---:|
| SFT | 24.38% | 47.00% | 9.00% | 64.23 / 28.81% |
| v1-matched | 27.00% | 54.00% | 8.00% | 64.11 / 27.44% |
| v2 λ=0 | 26.25% | 50.50% | 7.00% | 62.85 / 26.50% |
| v2 λ=.1 | 28.75% | 54.00% | 11.00% | 66.19 / 30.36% |

v2 λ=.1 激活 1,071/4,800 trajectories，救回 179/310 个 binary-zero-var groups，并降低 namespace/nonexistent/malformed/repetition；但 pass@4(any) 与 v1 打平，按预先规则没有在两个主指标上严格胜出，因此 **保留 v1-matched**。其 pass@1 相对 SFT 的 bootstrap CI 为 `[+0.75,+8.12]pp`，但 v2-v1 的 pass@4 CI 为 `[-7.50,+7.50]pp`。这次选择来自 held-out official eval，不来自训练 raw reward 排名；旧的 [credit test note](../output/doc/TAU2_SINGLE_CALL_CREDIT_TEST.md) 仍显示 `Pending`，应以更新后的 `CREDIT_COMPARISON.md` 为准。

**无效或未完成的续训。**

- [boundary-v2-domain-continuations-20260816](../output/experiments/tau2-rl-boundary-v2-domain-continuations-20260816/README.md)：使用了 boundary-v2/v1 User/max60，而计划要求 single-call/raw User/max120；因此取消/停止，不能当作结果。
- [single-call-v1-raw-user-maxsteps120-domain-rl130](../output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-domain-rl130/README.md)：错误地从 SFT 重启而不是 iter99，jobs 2736/2737/2738 停止、2740 取消；无有效 iter99 continuation。
- raw-User max120 的第一次 100–199 continuation 因默认镜像缺 `sglang_router` 在 preflight 失败，随后 retry job 1110 才完成；因此“重试成功”与“第一份作业失败”同时保留，不能只看最终 checkpoint。
- [corrected iter99-domain-cont130](../output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130/README.md)：改为从 raw-User max120 iter99 开始，但截至记录边界只有 continuation/convert 计划，没有 consolidated result table，保留为未完成探索。
- [v2 raw-user max120 domain RL](../output/experiments/tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl/README.md)：mixed/Airline/Retail/Telecom 都有 checkpoint/eval，汇总约为 `26.50/50.50/7`、`26.50/49.50/8.5`、`25.88/52/6.5`、`21.38/41/6`；README 没有预注册 selection decision，因此只作 exploratory artifact。

## 7. VitaBench：依赖打通，本地评测完成，远程完整线未闭环

| 实验 | 做了什么 | 结果 | 好处 | 不足 |
|---|---|---|---|---|
| [vitabench-qwen35-smoke](../output/experiments/vitabench-qwen35-smoke/README.md) | 安装并验证 29 个依赖、tau2 shared pins 和 CLI；跑 Qwen3.5-4B delivery 单任务。 | 依赖 v2 29/29 通过；delivery smoke 运行到 max_steps 30，14 次 Agent tool calls 后 reward 0。 | 证明 VitaBench 能在 tau2 环境共存，发现真实服务和终止问题。 | 单任务失败不代表模型正式分数。 |
| [vitabench-qwen35-full-eval](../output/experiments/vitabench-qwen35-full-eval/README.md) | 计划用 `gpt-4.1-ca` User、`gemini-2.5-flash` Evaluator 完成 4×100 tasks×4。 | 多次尝试分别因 readonly endpoint、protocol drift、单 GPU 改 4 GPU、Gemini rubric coverage、spot 未调度、API key rotation 和约 CNY50 spend 停止；最多完成 151/1,600 formal trajectories，未形成正式 score。 | 所有失败 artifact/journal 都保留，问题边界清楚。 | 远程 API 成本和不稳定性阻断了“从 raw Agent 到最终 Vita 分数”的闭环。 |
| [vitabench-qwen35-local-role-eval](../output/experiments/vitabench-qwen35-local-role-eval/README.md) | 用本地 Qwen3.6-27B User/Evaluator，处理 schema-5、shard queue、context overflow 和 AFS lock；protocol-v5 的 `artifacts-v2` 复用 37 shards 并补跑 delivery 7–9。 | 400 tasks、1,600 trajectories、369 successes；总体 `Avg@4/pass@1 23.0625%`、`pass@4(any) 47.75%`、`pass^4 4.50%`。分域：Delivery `28.75/63/6`、Instore `37.75/63/10`、OTA `16/36/2`、Cross-domain `9.75/29/0`。最终 artifacts 经校验无 missing/duplicate pair；56,181 assistant tool calls，其中 977 条 simulation 含 multitool message/call，另有 166 个 tool-response errors。 | 本地 roles 线完整、可复用、避开远程 API 费用；SFT export 84,500/84,500 journals 接受。 | protocol-v5 曾有 context overflow 和多次恢复，但最终 1,600 pairs 完整；分数与 published 22.0 不是严格同协议比较。 |
| [vitabench-evaluator-thinking-ab](../output/experiments/vitabench-evaluator-thinking-ab/README.md) | 固定 Agent/User trajectory，只重放 6,349 evaluator windows，比较 thinking-on/off。 | thinking-off chained replay `21.75/45.75/4.25`，相对 `23.0625/47.75/4.50` 为 `-1.3125/-2/-0.25pp`；Evaluator 4.36× 快、completion tokens 少 69.2%，但 final rubric agreement 96.06%、trajectory agreement 95.56%，质量 gate 未通过。 | 明确了成本和质量的真实折中；fresh local run 的 operational default 改为 non-thinking。 | 这是运行效率选择，不是质量等价结论；预宣 conservative switch gate 仍记录为 fail。 |
| [vitabench-qwen3-4b-instruct-2507-full-eval](../output/experiments/vitabench-qwen3-4b-instruct-2507-full-eval/README.md) | 用 raw Qwen3-4B、local Qwen3.6 roles，尝试四套 100-task suite。 | manifest 和日志显示 40 个固定 10-task shards、1,600/1,600 trajectories 均完成；之后 aggregate validator 报 `official-like Qwen3.5 run requires enable_think=true`，没有有效四套 score（见 [run log](../output/experiments/vitabench-qwen3-4b-instruct-2507-full-eval/jobs/fsh-vitabench-qwen3-4b-instruct-2507-full-0803-170949/run_20260803_170949.log)）。 | 暴露了 runner/validator 对 model-thinking profile 的硬编码假设。 | 这是 profile 校验错误，不是 Qwen3-4B 能力失败；中间 trajectories 也不能当正式 VitaBench score。 |

VitaBench 重要失败日志集中在 [local-role Jobs](../output/experiments/vitabench-qwen35-local-role-eval/README.md)、[remote full Jobs](../output/experiments/vitabench-qwen35-full-eval/README.md) 和 [Evaluator A/B Jobs](../output/experiments/vitabench-evaluator-thinking-ab/README.md)；这些 README 的 Jobs 部分保留了 job ID 和大部分 run log。少数历史相对链接已失效，本文只引用已存在的直接日志，并把没有正式汇总的作业标为部分完成/失败。

## 8. 实现演进与技术债

| 线 | 早期实现 | 后来改变 | 带来的收益 | 仍有的问题 |
|---|---|---|---|---|
| 数据/协议 | legacy text `<tool_call>`，observation 常写成 `role=user`/`Tool result`；解析异常容易退化成普通文本。 | [protocol profiles](../examples/tau2-bench/shared/protocol_profiles.py) 和 [AgentContract](../examples/tau2-bench/shared/agent_contract.py) 统一 native tools、ownership、strict-single；boundary 数据使用 `role=tool`。 | Agent/User tool ownership 可观察，malformed、namespace 和 multi-call 事件可计数。 | legacy、boundary、strict-single 三种 profile 并存，跨谱系比较容易误读。 |
| tokenization/显存 | `_limit_training_context` 取 message suffix，可能丢 leading context；超长 singleton 能绕过 bin cap。 | [tau2 rollout](../examples/tau2-bench/rl/rollout.py) 改为完整会话 `qwen3_full`，16,384 cap、逐样本重采样、placeholder/remove、动态 filter；[mask_utils](../slime/utils/mask_utils.py) 按 native render 对齐 assistant span。 | 保住 on-policy 上下文，解决已确认的 oversized microbatch OOM。 | 超长轨迹仍会被替换/丢弃；zero-variance 和 truncation 必须作为诊断报告。 |
| reward/credit | binary/composite reward，局部错误没有明确 credit。 | [reward_postprocess](../examples/tau2-bench/rl/reward_postprocess.py) 增加 field reward、行为 penalties、turn-credit-v1/v2；官方 held-out 仍保留 binary task success。 | 可区分 tool/name/args/DB/env/communication，能为 causal Assistant turn 分配信号。 | field matcher 对额外允许调用不扣分，可能鼓励“调用更多”；DB/参数正确率仍是瓶颈。 |
| GRPO reducer/元数据 | 默认 reducer 只看到当前 microbatch 的局部 token denominator，自定义 rollout 指标难以跨 DP 对齐。 | [ray rollout plumbing](../slime/ray/rollout.py) 传播 `train_metadata`、按 rollout 汇总 mask token denominator，并支持 CP/DP slicing；官方评测统一输出三种 pass 指标和行为诊断。 | first-fit 拆分时的 token-weighted mean 更正确，评测结果可复核。 | 这部分改动较深，必须和 profile、checkpoint lineage 一起记录。 |
| 提交与实验记录 | 早期脚本直接提交，日志分散。 | [scripts/submit.sh](../scripts/submit.sh) 接入 job-manager，按 experiment 归档 run/submit log。 | 失败重试、资源问题和最终结果能按实验回溯。 | 目前 Contract/lineage gate 还带有较多 fingerprint/signature 产物；它增加流程复杂度，本身不是模型收益。 |

## 9. 当前状态、实验覆盖与下一步

**当前选型**

1. **全局 Agent 默认：** Contract + boundary long100 iter99；它是在同一 SFT/User/协议下完成两 seed、健康检查和正式 held-out comparison 后选出的 balanced default。
2. **strict-single reward recipe：** credit-core 保留 `v1-matched`；v2 λ=.1 继续作为有行为改善但未严格胜出的候选，不替换当前 recipe。
3. **OPD：** 尚未启动。Retail expert iter119 是第一候选；Telecom 先补 seed301 iter109 或和 long200 Telecom-oriented checkpoint 比较；Airline expert 不用。
4. **评测协议：** tau2 继续用 v1 STOP User、parser-on、固定 task quota 和三项 pass 指标；VitaBench fresh local run 默认 Evaluator non-thinking，但保留 thinking-on 对照和质量 gate 失败记录。

**实验目录覆盖台账**

以下目录全部纳入本记录；链接页面的 Jobs 部分保留该实验的完整 job ID 和日志。`仅诊断/未完成/无效谱系` 不参与最终模型排名。

除命名 experiment 外，早期 `output/jobs/` 的环境探针、转换重试、框架 math run 和 User issue-stats 也在第 4、5 节给出直接日志链接；它们没有被误归入 46 个正式 experiment 目录。

| 主线 | 已纳入目录 |
|---|---|
| tau1 | [tau-bench](../output/experiments/tau-bench/README.md) |
| tau2 基础 | [tau2-deps-probe](../output/experiments/tau2-deps-probe/README.md)、[tau2-eval](../output/experiments/tau2-eval/README.md)、[tau2-eval-local-user](../output/experiments/tau2-eval-local-user/README.md)、[tau2-eval-user-sft](../output/experiments/tau2-eval-user-sft/README.md)、[tau2-hf-convert](../output/experiments/tau2-hf-convert/README.md)、[tau2-qwen35](../output/experiments/tau2-qwen35/README.md)、[tau2-qwen35-nonthinking-eval](../output/experiments/tau2-qwen35-nonthinking-eval/README.md) |
| tau2 User/协议 | [tau2-user-sft](../output/experiments/tau2-user-sft/README.md)、[tau2-eval-user-stop](../output/experiments/tau2-eval-user-stop/README.md)、[tau2-eval-user-stop-parser](../output/experiments/tau2-eval-user-stop-parser/README.md)、[tau2-eval-user-stop-v2-parser](../output/experiments/tau2-eval-user-stop-v2-parser/README.md)、[tau2-raw-agent-raw-user-parser-on](../output/experiments/tau2-raw-agent-raw-user-parser-on/README.md) |
| tau2 SFT/审计 | [tau2-sft](../output/experiments/tau2-sft/README.md)、[tau2-sft-quality-audit](../output/experiments/tau2-sft-quality-audit/README.md)、[tau2-sft-multitool-quality-audit](../output/experiments/tau2-sft-multitool-quality-audit/README.md)、[tau2-sft-local-first-relaxed](../output/experiments/tau2-sft-local-first-relaxed/README.md)、[tau2-sft-agent-user-boundary-v2](../output/experiments/tau2-sft-agent-user-boundary-v2/README.md) |
| tau2 初始/稳定性 RL | [tau2-rl](../output/experiments/tau2-rl/README.md)、[tau2-rl-usercmp](../output/experiments/tau2-rl-usercmp/README.md)、[tau2-rl-stability-k2-fieldreward-lr-sweep](../output/experiments/tau2-rl-stability-k2-fieldreward-lr-sweep/README.md)、[tau2-rl-stability-k2-fieldreward-final-eval](../output/experiments/tau2-rl-stability-k2-fieldreward-final-eval/README.md)、[tau2-rl-local-first-dependency-safe-k2-fieldreward](../output/experiments/tau2-rl-local-first-dependency-safe-k2-fieldreward/README.md)、[tau2-traj-pattern](../output/experiments/tau2-traj-pattern/README.md) |
| tau2 boundary RL | [tau2-rl-agent-user-boundary-v2](../output/experiments/tau2-rl-agent-user-boundary-v2/README.md)、[tau2-rl-agent-user-boundary-v2-long100](../output/experiments/tau2-rl-agent-user-boundary-v2-long100/README.md)、[tau2-rl-agent-user-boundary-v2-checkpoint-curve](../output/experiments/tau2-rl-agent-user-boundary-v2-checkpoint-curve/README.md)、[tau2-rl-agent-user-boundary-v2-long200](../output/experiments/tau2-rl-agent-user-boundary-v2-long200/README.md)、[tau2-rl-agent-user-boundary-v2-domain-experts](../output/experiments/tau2-rl-agent-user-boundary-v2-domain-experts/README.md)、[tau2-rl-boundary-v2-domain-continuations-20260816](../output/experiments/tau2-rl-boundary-v2-domain-continuations-20260816/README.md) |
| tau2 strict-single/credit | [tau2-agent-single-call-v1](../output/experiments/tau2-agent-single-call-v1/README.md)、[tau2-agent-single-call-credit-core](../output/experiments/tau2-agent-single-call-credit-core/README.md)、[tau2-agent-single-call-v1-user-ablation](../output/experiments/tau2-agent-single-call-v1-user-ablation/README.md)、[tau2-agent-single-call-v1-rl-crossed-parser-on](../output/experiments/tau2-agent-single-call-v1-rl-crossed-parser-on/README.md)、[tau2-agent-single-call-v1-raw-user-rl100](../output/experiments/tau2-agent-single-call-v1-raw-user-rl100/README.md)、[tau2-agent-single-call-v1-raw-user-rl200](../output/experiments/tau2-agent-single-call-v1-raw-user-rl200/README.md) |
| tau2 max_steps/domain 探索 | [tau2-agent-single-call-v1-raw-user-maxsteps120-rl100](../output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl100/README.md)、[tau2-agent-single-call-v1-raw-user-maxsteps120-rl200](../output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl200/README.md)、[tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl](../output/experiments/tau2-agent-single-call-v2-raw-user-maxsteps120-domain-rl/README.md)、[tau2-agent-single-call-v1-raw-user-maxsteps120-domain-rl130](../output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-domain-rl130/README.md)、[tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130](../output/experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-iter99-domain-cont130/README.md) |
| VitaBench | [vitabench-qwen35-smoke](../output/experiments/vitabench-qwen35-smoke/README.md)、[vitabench-qwen35-full-eval](../output/experiments/vitabench-qwen35-full-eval/README.md)、[vitabench-qwen35-local-role-eval](../output/experiments/vitabench-qwen35-local-role-eval/README.md)、[vitabench-evaluator-thinking-ab](../output/experiments/vitabench-evaluator-thinking-ab/README.md)、[vitabench-qwen3-4b-instruct-2507-full-eval](../output/experiments/vitabench-qwen3-4b-instruct-2507-full-eval/README.md) |

**下一步顺序**

1. 固定 selected SFT、v1 User、parser、domain quota、LR/KL、rollout budget 和 held-out protocol，只改变 reward/advantage；同时报告 `pass@1`、`pass@4(any)`、`pass^4`、action/DB accuracy、KL、truncation、zero-variance group 和行为终止统计。
2. 只从 controlled official evaluation 选出 reward/credit recipe；训练 reward 的 field/turn shaping 不能替代官方 binary task success。
3. 选型完成后再做 OPD，并沿用相同 scoring semantics 选择或加权 teacher rollout。先评估 Retail iter119；Telecom 先补证据；不自动使用 Airline expert。
4. 对每次继续训练先验证 lineage、User、max_steps 和 sampler state；错误谱系应立即标记为无效，不把“跑完了”当成“可比较”。
