# tau2-bench SFT/RL QA

本文按时间记录 Qwen3-4B-Instruct-2507 在 tau2-bench 上的 SFT/RL 问题、证据、修复和验证。开头是 2026-07-14 对 AReaL strict no-thinking SFT 的历史排查快照：loss mask 基本排除，主要问题是 telecom 成功长程监督和 Agent 工具覆盖不足，以及训练/eval 的工具边界不一致。截至 2026-08-06，on-policy/OOM、Agent/User ownership 和过早停止语义均已分别处理；Contract + boundary long100 完成 100 updates，取得当前最稳健的三领域综合提升。Long200 iter199 虽有较高 KL，但 checkpoint 与双 seed 评测完整，总体点估计继续提高且增益集中在 Telecom；因 Airline 回退，long100 iter99 仍是更均衡的默认模型，而 iter199 保留为有效的 Telecom-oriented 候选。2026-08-10 完成的 strict-single-v1 独立谱系则在稳定续训到 iter199 后，对本谱系 SFT 和 iter99 的 pass@1、pass@4(any) 取得双 seed 配对正增益，iter199 成为该单工具谱系的最终 RL checkpoint。2026-08-14 已完成训练 User × 评测 User 的双 seed crossed matrix；结果表明 pass^4 退化并非单工具协议必然造成，而与 reward/advantage、训练 User 和继续训练阶段有关。下一阶段顺序固定为：先优化 reward 与 advantage/credit assignment，再开展 on-policy distillation；OPD 不先于该受控消融启动。最终证据和执行顺序见本文末节。

| 训练阶段 | 做了什么 | 遇到什么问题 | 从哪个角度分析这个遇到的问题 | 分析的结果 | 这个角度是否是造成这个问题的原因 |
|---|---|---|---|---|---|
| SFT 数据准备 | 旧版 `examples/tau2-bench/sft/prepare_sft_data.py` 将 AReaL tau2 数据转成 slime SFT `messages` JSONL；默认训练文件是 `areal_tau2_sft_strict_no_thinking.jsonl`。旧版 strict 规则会丢弃 assistant turn 中多工具调用，或文本和工具调用混合的样本。 | 训练后的模型整体没有超过 base，telecom 失败轨迹更长。首先怀疑训练数据本身是否被筛得过窄。 | strict 前后数据分布。 | AReaL raw 有 33,531 条 turn-level 样本，旧版 strict 后只剩 7,060 条。按 domain：airline 12,842 -> 1,939；retail 11,395 -> 3,482；telecom 9,294 -> 1,639。telecom strict 样本中 correct=1 只有 574，correct=0 有 1,065；raw telecom 平均 assistant turns 11.84，strict 后 4.98。 | 是，主要原因之一。旧版 strict 明显压缩了 telecom 长程监督，并保留了大量失败轨迹片段。 |
| SFT 数据准备 | 旧版 strict 只保留 AReaL 中能满足单工具调用协议的 turn-level 样本。 | telecom 子集训练信号可能不是完整成功轨迹，而是短前缀或失败片段。 | 原始对话级别统计。 | raw telecom 有 500 个 source dialog，其中 success-only 180 个、fail-only 320 个；旧版 strict 后仍有 497 个 dialog，但 success-only 179 个、fail-only 318 个，平均每个 dialog 只剩 3.30 条样本。airline/retail 的 success-only 占比显著更高：airline 879/998，retail 992/1000。 | 是，主要原因之一。telecom 的训练来源本身失败比例高，旧版 strict 后没有把 telecom 转成“成功长轨迹”监督。 |
| SFT 数据准备 | 在 strict no-thinking 文件中训练 assistant 文本和 assistant `<tool_call>`。 | eval 中 SFT telecom 经常调用不存在的工具，导致 `too_many_errors` 或长对话。 | 训练集中 assistant tool-call 覆盖。 | strict 训练集中 telecom assistant tool-call 总共 1,190 次，但只有两个 tool name：`get_customer_by_phone` 1,142 次、`get_details_by_id` 48 次。相比之下 retail 有 10 个 tool name，airline 有 6 个 tool name。telecom 的技术支持动作大量以自然语言指导用户操作，而不是 agent tool call。 | 是，主要原因之一。模型没有从 SFT 中学到丰富的 telecom agent tool 边界，容易把用户侧手机操作误包成 agent `<tool_call>`。 |
| SFT 训练 | 使用 Qwen3-4B-Instruct-2507，AReaL strict no-thinking 数据，2 epoch SFT；checkpoint 转 HF 后用官方 tau2 runner 评测。 | 怀疑 loss mask 是否错误地训练到了 user/tool observation，导致模型学用户或工具输出。 | loss mask 实测。 | 集群诊断 job `pt-ukqcbuci` 成功，日志 `output/experiments/tau2-sft/jobs/fsh-tau2-sft-loss-mask-debug-0714-112841/run_20260714_112841.log` 显示 `SUMMARY rows=6 masked_tokens=644 warnings=0`。抽样 airline/retail/telecom 中，`loss_mask == 1` 命中的都是 assistant 文本或 assistant `<tool_call>`；未发现 system/user/tool observation 被 mask。`<|im_end|>` 被包含在 assistant span 内，这是当前 qwen3 mask 的实际行为。 | 否。当前证据不支持 loss mask 是 SFT 效果不佳的原因。 |
| SFT 评测 | 在官方 tau2 runner 上对 SFT、base、Qwen3.5 做 full test split Pass@4，对比 airline/retail/telecom。 | SFT 总体 pass@4(any) 比 base 低，telecom 经常长对话或异常终止。 | eval 行为和终止原因。 | base telecom：pass@1 19.4%，pass@4(any) 47.5%，`too_many_errors` 5，`max_steps` 0，平均 agent turns 19.59。SFT telecom：pass@1 20.0%，pass@4(any) 32.5%，`too_many_errors` 22，`max_steps` 3，平均 agent turns 26.85，失败样本平均 29.28 turns。Qwen3.5 telecom：pass@1 85.6%，pass@4(any) 100.0%，`too_many_errors` 1。 | 是，现象定位。SFT 没有显著降低 telecom pass@1，但让失败样本更拖、更容易触发工具错误，解释了“超过最大轮数/最大错误”的现象。 |
| SFT 评测 | 读取 SFT telecom results 中 tool error。 | telecom 失败样本里出现大量不存在的工具。 | 工具错误类型。 | SFT telecom 总 tool errors 490，其中 assistant tool errors 408；base telecom 分别是 347/286，Qwen3.5 telecom 是 115/90。SFT telecom top assistant errors 包括：`reboot_device` 45、`toggle_airplane_mode` 37、`check_network_status` 35、`reseat_sim_card` 33、`check_sim_status` 31、`reset_apn_settings` 28、`run_speed_test` 19。这些多是用户侧手机动作或不存在的 agent tools。 | 是，直接原因之一。它直接造成 `too_many_errors`，并使部分轨迹进入长循环或 max steps。 |
| SFT 数据与 eval 接口对齐 | SFT 数据中 tool observation 被转换为 user message；official eval adapter 也把 tool result 喂回 user-role message。 | 怀疑训练和 eval 的 prompt/observation 格式不一致。 | 协议与格式 diff。 | 旧版 SFT 数据使用 `Tool result for <name>:\n...`，代码位于 `prepare_sft_data.py`；eval adapter 使用 `Tool result:\n...`，代码位于 `eval/official/sglang_agent.py`。SFT system prompt 保留 AReaL 原始 `<instructions>` 和 “ALWAYS Think Before You Act”等内容，但 no-thinking 数据又去掉 thinking；eval adapter 使用更短的自定义 system prompt。 | 部分原因，待进一步量化。格式差异不会单独解释所有退化，但会削弱 SFT 和 eval 分布一致性，尤其影响工具结果理解和工具边界。 |
| SFT 数据准备 | 旧版 strict 丢弃多工具调用和文本+工具调用混合样本。 | 用户质疑“原始数据怎么可能是多工具调用”。 | raw AReaL message shape。 | raw AReaL 中确实存在单个 assistant message 带多个 `tool_calls` 的情况，例如 airline 一个 history turn 同时包含多次 `get_reservation_details`，retail 一个 answer 同时包含多个 `get_order_details`。事件级统计显示 history multi-tool 33,348 次、answer multi-tool 4,147 次；这说明 raw 数据生成/采集允许并行工具调用。2026-07-14 复核官方 tau2 orchestrator 后确认 half-duplex eval 也支持多工具调用：它会循环执行一条消息里的所有 tool call，并把多个结果作为连续 tool messages 送回。 | 是，旧版 strict 的单工具限制和官方协议不一致，造成训练分布稀疏。 |
| SFT 训练过程 | 训练了 2 epoch，并评测最终 `iter_0000881_hf`。 | 还不能判断是过训、学习率/epoch 问题，还是数据问题主导。 | checkpoint sweep 和训练曲线。 | 当前已确认最终 checkpoint 效果差；尚未系统评估 epoch 1、中间 checkpoint、不同 save interval 的官方 eval，也未把 W&B loss 曲线和 eval 指标对齐。 | 待验证。若早期 checkpoint 明显更好，则过训/噪声放大会成为重要原因；若所有 checkpoint 都接近，则数据和协议问题更可能是主因。 |

## 2026-07-14 轨迹模板核验

官方 tau2 half-duplex 真实模板：一个 assistant message 可以有 `tool_calls: [ToolCall, ...]`，`content` 为 `null` 或空；orchestrator 对 `tool_calls` 逐个调用 environment；结果在落盘 `messages` 中是多个连续 `role: "tool"` 消息，`requestor: "assistant"`，顺序与 `tool_calls` 一致。

slime SFT 模板：不直接输出 `role: "tool"`，而是按 eval adapter 输入面转换为 user-role observation。一个多工具 assistant turn 渲染为同一 assistant message 中连续多个 `<tool_call>{"name": ..., "arguments": ...}</tool_call>`；紧随其后的每个 tool result 渲染为一个 `{"role":"user","content":"Tool result:\n..."}`，保持顺序。

对应修复：`prepare_sft_data.py` 已不再因多工具调用 drop strict 样本，只 drop 文本+工具混合；tool observation 已统一成 `Tool result:\n...`。`eval/official/sglang_agent.py` 已支持历史回放和模型输出中的多个 `<tool_call>` 块，并返回完整 `AssistantMessage.tool_calls` 给官方 orchestrator。

全量重生成结果：`datasets/tau2-bench-sft/stats.json` 显示 AReaL source rows 33,531，multi-tool turns 37,495，strict 仅因 `content_and_tool` drop 10,120，`areal_tau2_sft_strict_no_thinking.jsonl` 为 23,411 行；不存在 `areal_strict_drop_multi_tool`。抽查默认 strict no-thinking 文件：multi-tool assistant messages 27,779，旧格式 `Tool result for ...` 为 0。

下一步优先级：用新生成的 AReaL strict no-thinking 数据重训并跑官方 eval；再做 telecom-only 数据重构对照，包括只保留成功轨迹、保留更长中后期 turn、显式禁止把用户侧手机动作作为 agent tool call。

## tau2-bench RL QA (2026-07-26 → 07-27)

GRPO RL 训练（`examples/tau2-bench/rl/`，Qwen3-4B-Instruct-2507 agent SFT init，AReaL tau2 RL 数据，本地 user simulator）。三个依次发现并修复的问题：

| 训练阶段 | 做了什么 | 遇到什么问题 | 从哪个角度分析 | 分析的结果 | 是否原因 |
|---|---|---|---|---|---|
| RL rollout→train | slime 默认 sglang rollout + 自定义 `rollout.generate`；trajectory-level GRPO，`kl_coef=0`。 | 8 卡 real-8g run 在 ref/actor **log-prob 前向 OOM**（`compute_log_prob` 分配 49.67 GiB）。 | slime bin-packing + log-prob 路径源码（`slime/utils/seqlen_balancing.py`、`backends/megatron_utils/loss.py`）。 | bin-packer 把每个 microbatch 限制在 `max_tokens_per_gpu`，但**单条超过该上限的轨迹会单独成一个"超大 bin"**（不切分、不丢弃）；tau2 多轮对话长尾（实测 p99 ~16k、max ~104k tokens）触发它，单样本 `[T,V]` logits 在前向 OOM。降低 `max_tokens_per_gpu`（6144）和 `log-probs-chunk-size`（1024）都无效：前者挡不住单样本，后者只分块 softmax、不缩 forward 的 logits。 | 是，直接原因。 |
| RL 训练上下文 | `_limit_training_context` 为凑 `TAU2_TRAIN_MAX_TOKENS` 预算，二分删除对话**前段消息**。 | 训练 log-prob 在裁剪后的上下文上算，与 rollout 采样时的上下文不一致。 | on-policy 语义。 | 删前段消息会让训练算的策略概率 ≠ 动作实际采样时的概率，**破坏 GRPO 的 on-policy 假设**（ratio 不再≈1）。 | 是，方法学错误。 |
| RL 超参 | `lr=1e-6`、`rollout_temperature=0.65`、`eps_clip=0.2`；real-8g A/B/C user-model 对比跑了 ~200 步。 | **reward 曲线走平**：v1 四分位 q1=0.469 → q4=0.456（209 步几乎无上涨），与 AReaL 同任务稳步上涨的曲线不一致。 | 对比 AReaL 官方 `examples/tau2/config_1.7b_airline.yaml`；核查 slime advantage 路径。 | advantage 计算本身正确（`rewards_normalization=True`+`grpo_std_normalization=True` 做 per-group `(r-均值)/std`，故 `rollout/advantages≈0` 是正常的）。真因是**超参**：lr 1e-6 比 AReaL 的 **1.7e-5 低 17×**；每步 batch 48 vs 128（再低 ~2.7×）→ 有效学习量仅 AReaL 的 ~1/45；temperature 0.65 vs 1.0 → 探索不足、~36% 组因零方差被 drop；eps_clip 0.2 vs 0.4 → 每步更新更小。 | 是，主因是 lr 17× 过低。 |

### 修复

1. **OOM**（`rollout.py`、`filters.py`、`run_qwen3_4b_instruct_2507_tau2_rl.sh`）：去掉前段截断，改完整对话 verbatim tokenize（on-policy）；`TAU2_RL_MAX_TRAIN_TOKENS=16384` 同时作为 `max_tokens_per_gpu`（bin 上限 == drop 上限，杜绝超大 bin）；超长轨迹**先 per-sample 重采样**（`TAU2_RL_MAX_ROLLOUT_RETRIES=2`，只重采那一条、保留组内其余 7 条）；反复采不到才标记 `permanently_too_long`，由 `filters.drop_zero_std_or_unsampleable` **整组 drop** 兜底。`test_rollout_logic.py` preflight 在容器内 rollout 前几秒校验 on-policy tokenization / loss_mask / filter。verify run 实测 264 条轨迹只重采 3 条、0 兜底 drop、无 OOM。
2. **走平曲线**（`run_qwen3_4b_instruct_2507_tau2_rl_real_8g.sh`）：超参对齐 AReaL——`LR=1e-5`、`ROLLOUT_TEMPERATURE=1.0`、`EPS_CLIP=EPS_CLIP_HIGH=0.4`。base 脚本把 eps_clip 改为可配置（`EPS_CLIP`/`EPS_CLIP_HIGH`，默认 0.2/0.28 不影响 smoke/verify）。


## tau2-bench RL 走平的真实原因 (2026-07-28)

修正超参（LR 1e-5/temp 1.0/clip 0.4）+ airline-only 后，reward 仍走平/略降。
深入诊断 reward 分布与实际对话，结论：**不是 pipeline bug，是稀疏二元 reward
下的固有问题。**

证据：
- `reward_info` 真实结构：airline 任务 reward_basis 多为 `(DB, COMMUNICATE)`，
  reward = db_reward × communicate_reward。失败几乎全是 **`db_check` 不过**
  (`db_match:false`)，COMMUNICATE 一般过。即 agent **调对了工具但参数错** →
  最终 DB 状态不匹配 → reward 0。
- `user_stop` 是**正常终止**（reward=1 的轨迹 100% 也是 user_stop），不是"过早
  放弃"；失败轨迹反而更长（中位 25 turn vs 成功 21 turn）。
- 实例 airline_613：用户"加 2 件托运行李"，agent 正确 `get_reservation_details`
  → `update_reservation_baggages(total_baggages:5, nonfree_baggages:0)` → 用户
  `###STOP###`。对话 sane、工具对、流程对，但**行李数算错**（未按 existing+2 /
  舱位 policy 算 nonfree）→ DB 不匹配 → reward 0。
- 成功率随训练**下降**（airline dump: 早期 31% → 晚期 21%）；kept-batch reward
  (~0.45) 是假象——动态 filter 丢掉全失败组，剩下混合组恒 ~0.45，掩盖了真实退化。

真实原因：
1. agent 工具调用 sane 但**参数精度不足**（~70% 任务因参数错失败）。
2. tau2 reward 是**二元 exact-match**（最终 DB 状态精确比对），对参数精度要求高。
3. GRPO + 稀疏二元 reward + 小 batch (48) **无法高效教参数精度**；29% 成功率的正
   信号太薄，RL 净效果≈0 甚至略负（边际退化 SFT init）。
4. AReaL 曲线上扬靠的是 batch 128 + 72B user + 更多 effective samples（tree
   training），我们 8 卡 + 4B user 难以匹配。

下一步方向（按性价比）：
- **先测 SFT init 的真实能力上限**（官方 tau2 eval，无 RL）——确认 ~29% 是否就是
  4B SFT 的天花板。
- **更密的 reward**：用 action_checks 的逐动作匹配率（已计算）做部分信用，而非纯
  二元 DB-match；这是最可能让曲线涨的杠杆（但偏离官方 reward 定义）。
- 更大 batch / 更强 user（对齐 AReaL）。

## tau2-bench RL 走平根因定位 + 修复 (2026-07-28, 续)

对 airline run (`pt-...-airline-0727`, dump
`real8g_real8g_airline_20260727_112403.jsonl`, 2813 条) 做了三方对照与逐条诊断：

| 角度 | 现象/证据 | 结论 |
|---|---|---|
| train 指标 | `pg_loss≈±1e-8`、`pg_clipfrac=0`（advantage 中心化后正常），但 `entropy_loss` 0.46→0.13、`kl_loss` 0→0.24 单调变化，`raw_reward` 0.21–0.65 震荡走平、末段跌到 0.21。 | 模型在更新但在退化（稀疏 reward 下 RL collapse），不是 pipeline bug。 |
| reward 分布 | dump 里 27.7% 成功；1906 条失败中 **98.6% 有 `action_match>0`**（均值 0.55），即"调对工具、参数错"。二元 reward 把这些近失与全错都判 0。 | 主因：二元 reward 太稀疏，GRPO 拿不到参数精度的学习信号。 |
| 三方对照 | **社区 cookbook**（`examples/tau-bench/tau-bench-example/tau2`，batch 64、非 thinking、KL 0.01/entropy 0.001、user temp 0.7）用 **shaped reward** `task+0.25·partial` 拿到 57% Pass@4；**AReaL** 用纯二元 reward 但 **batch 256–512**+thinking 才涨。我们 batch 48、纯二元、无 shaping——worst-of-both。 | 不是代码 bug，是 recipe 问题；batch 64 的 cookbook 能 work 证明在我们规模下 shaping 才是分水岭。 |
| filter 交互 | 旧 `drop_zero_std_or_unsampleable` 在 rollout 期按**二元** reward 丢全失败组（21 次 `drop_zero_std_0.0`），把 shaping 最该救的硬任务组在 shaping 前就丢了。 | 加 shaping 必须同时关掉 zero-std drop（保留 permanently_too_long 内存安全兜底）。 |

**修复**（已在 `examples/tau2-bench/rl/` 落地，未提交训练）：
1. 新增 `reward_postprocess.tau2_reward_post_process`（`--custom-reward-post-process-path`）：
   `shaped = task_reward + alpha·partial_score`，partial 取自 `reward_info` 的
   action/env/communicate 匹配率（airline 即 action_match 率），再按 slime 默认做
   per-group `(r-μ)/σ`。本地验证：两条同为 reward=0 的失败（4/5 vs 0/5）被正确区分
   成 advantage [+1, −1]。
2. `filters.drop_zero_std_or_unsampleable`：默认**不再**丢 zero-std 组（shaping 让
   它们可训），仅保留 permanently_too_long 内存安全 drop；`TAU2_DROP_ZERO_STD=1`
   可为纯二元 reward 重新开启。
3. `run_*_real_8g.sh`：`KL_LOSS_COEF=0.01`、`ENTROPY_COEF=0.001`（对冲小 batch 下
   entropy 塌缩/漂移）；LR 1e-5/temp 1.0/clip 0.4 维持。
4. `test_rollout_logic.py` preflight：覆盖 shaping 区分性 + 新 filter 行为。

当时下一步（历史状态）是用上述 recipe 重跑 airline-only real-8g；后续 stability
sweep 已用 K2 KL、field credit 和更小 LR 完成该数值稳定性验证，而 Agent/User
boundary 与训练长度问题由下述 2026-08-04 → 08-05 实验继续解决。

## tau2 turn-aware boundary RL：问题闭环与 long100 结果 (2026-08-04 → 08-05)

这一阶段的目标不是再次更换 reward recipe，而是在同一个 Contract + boundary SFT、
LR `2e-6`、K2 KL `0.01`、K=8、turn-credit-v1、`max_steps=60` 和 v1 User 下，隔离
“训练到 iter9 还是 iter99”这一变量。

| 问题链路 | 证据与判断 | 采用的解决思路/方案 | 取得的结果 |
|---|---|---|---|
| 旧 SFT/eval 没有明确 Agent/User 工具所有权，telecom 会把用户侧动作包装成 Agent tool。 | 早期 SFT telecom 出现大量不存在工具和长失败轨迹；Contract-only 仍有两 seed 合计 1,281 次 namespace attempts、103 次归因终止。 | 用签名 `agent-owned-dependency-safe-multi` Contract、原生 Agent tools、natural handoff 与 ownership-repair anchors 做 matched-budget SFT。 | Contract + boundary 成为 turn-aware-rl-v3 唯一入选 SFT；seed 300/301 namespace 为 `54/326/4`、`55/314/9`（affected/attempts/termination），probe 为 `0/60`、`0/240`。 |
| 旧 `lr=1e-5 + low_var_kl` 在 49–71 步 KL→0.42、reward→0、最终 backward NaN；稳定 sweep 的 `2e-6` 虽不崩溃，却未证明 RL > SFT。 | 旧 `2e-6` seed-300 相对 SFT pass@1 仅 `+2.25pp`，95% CI `[-2.84,+7.34]pp`，pass^4 `-1pp`；同时 KL 一直 `<0.06`。 | 保持已验证的 `2e-6 + K2 0.01` 和 reward/penalty 配置，避免把 recipe 变化与训练长度混在一起。 | 数值稳定问题与“可能训练不足”被拆成两个独立假设；long100 只验证后者。 |
| boundary-v2 Pilot A 在 iter9 已改善 namespace 与能力，却被 safety gate 提前停止。 | iter9 checkpoint/metrics/quota/Contract/turn-credit 均健康；原停止项是 framework truncation `30.63%`、malformed `18.13%`、execution error `15.63%`、`max_steps` `29.58%`。这些是行为/终止诊断，不是 OOM、NaN/Inf、checkpoint 损坏或协议错位；framework `truncated` 还混合了正常 horizon 和真实基础设施/上下文失败。 | 保留历史 artifact 的原标签，但不再将其解释为 iter9 checkpoint 无效；新实验从 SFT 和全新 optimizer/root 启动，把行为错误留到最终评测。 | iter9、iter19、iter99 训练均完整；没有因正常 telecom horizon 再误停。 |
| 分段恢复需要同时换 quota、延长 scheduler horizon，并防止错 root/错 gate。 | 第一次 `long100-b` 在 update 10 前被 Megatron 拒绝：scheduler horizon `480` 与目标 `960` 不一致；iter9 本身未被改写。 | `long100-a/b/final` 使用独立 root 和前置 health artifact；retry 只 override scheduler horizon，精确恢复新 lineage 的 optimizer/RNG；阶段 quota 为 `3/2/1`、`3/1/2`、`2/2/2`。 | retry 从 iter9 正确继续，保存 iter19 及每 10 updates checkpoint，最终到 iter99；历史 iter9 optimizer 从未复用。 |
| 最终能力采用 selected SFT 和历史 iter9 的双 seed、逐域对比。 | 100 tasks × 4 trials，seed 300/301，temperature `0.6`，`max_steps=200`；paired bootstrap 单位为 task。 | iter99 转 HF；两个 2-GPU eval 并行，seed 300 加 namespace probes，同时报告连续指标和不确定性。 | 两个 seed 的 pass@1 与 pass@4(any) 均超过两个控制，long100 获得明确的综合能力提升。 |

### 严格同口径结果

| 模型 | Seed 300：pass@1 / pass@4(any) / pass^4 | Seed 301：pass@1 / pass@4(any) / pass^4 | 两 seed 均值 |
|---|---:|---:|---:|
| selected Contract + boundary SFT | 23.75 / 42.00 / 8.00% | 23.50 / 45.00 / 9.00% | 23.63 / 43.50 / 8.50% |
| 历史 boundary RL iter9 | 26.25 / 52.00 / 10.00% | 24.25 / 48.00 / 8.00% | 25.25 / 50.00 / 9.00% |
| **boundary RL long100 iter99** | **27.50 / 53.00 / 9.00%** | **31.50 / 60.00 / 12.00%** | **29.50 / 56.50 / 10.50%** |
| iter99 - selected SFT | +3.75 / +11.00 / +1.00pp | +8.00 / +15.00 / +3.00pp | **+5.88 / +13.00 / +2.00pp** |
| iter99 - iter9 | +1.25 / +1.00 / -1.00pp | +7.25 / +12.00 / +4.00pp | **+4.25 / +6.50 / +1.50pp** |

两 seed 平均 domain pass@1 也一致提高：selected SFT 为
`31.25/25.63/17.81%`，历史 iter9 为 `32.50/26.56/20.31%`，long100 iter99
为 **`38.13/31.25/23.44%`**（airline/retail/telecom）。Telecom namespace
在 seed 300/301 降至 `8/8/0`、`10/13/0`。

历史 parser-on/stability 评测可用于看模型谱系，但协议和 SFT init 不同，不能替代
上面的因果对照：

| 历史 seed-300 模型 | pass@1 | pass@4(any) | pass^4 |
|---|---:|---:|---:|
| Raw Qwen3-4B-Instruct-2507 | 23.50% | 45.00% | 6.00% |
| 旧 multitool SFT | 23.25% | 46.00% | 8.00% |
| 旧 RL `2e-6` / `3e-6` / `5e-6` iter99 | 25.50 / 23.50 / 18.50% | 49.00 / 47.00 / 39.00% | 7.00 / 5.00 / 3.00% |
| Long100 iter99（新 signed profile） | **27.50%** | **53.00%** | **9.00%** |
| Raw Qwen3.5-4B（不同基座） | 47.20% | 78.00% | 19.00% |

统计解释保持保守：相对 selected SFT，seed 300 的 pass@1 CI
`[-1.00,+8.50]pp` 跨 0，但 pass@4(any) CI `[+2.00,+20.00]pp` 不跨 0；seed
301 两项分别为 `[+2.50,+13.50]pp`、`[+4.00,+26.00]pp`。相对 iter9，seed
301 两项显著，seed 300 不显著；所有 overall pass^4 CI 均跨 0。因此 `win` 是按
预先定义的 point-estimate、health、protocol、namespace 和 capability non-inferiority
规则通过，不表示每个提升都独立达到统计显著。

证据入口：[long100 README](../experiments/tau2-rl-agent-user-boundary-v2-long100/README.md)、
[最终 decision](../experiments/tau2-rl-agent-user-boundary-v2-long100/ITER99_LONG100_DECISION.json)、
[seed 300 paired comparison](../experiments/tau2-rl-agent-user-boundary-v2-long100/eval/comparisons/iter99_seed300.md)、
[seed 301 paired comparison](../experiments/tau2-rl-agent-user-boundary-v2-long100/eval/comparisons/iter99_seed301.md)、
[历史 iter9 gate](../experiments/tau2-rl-agent-user-boundary-v2/ITER9_SAFETY_GATE.json)。

## boundary-v2 checkpoint 曲线与 iter99→199 续训 (2026-08-05)

Long100 已证明 iter99 胜过 selected SFT 和历史 iter9，但仅有同-lineage iter99
终点评测，无法判断 9→99 的增益发生在哪一段，也不能回答继续到 iter199 是上升、平台
还是回落。为避免改变历史证据，新实验只读 long100 root，并把两个问题拆开并行执行：

| 问题 | 方案 | 检查重点 | 当前状态 |
|---|---|---|---|
| 缺少同-lineage 中间曲线。 | 分别转换 long100 iter9/39/69/89 到独立 HF root；各跑 seed 300、100 tasks × 4 trials，并与已有 iter99/SFT 逐 task 配对。 | checkpoint 完整且评测协议一致；历史 iter9 和 raw Instruct 只作背景。 | 四个 2-GPU normal-priority job 均成功并完成 400/400；同-lineage pass@1 为 `25.00/26.25/24.00/29.75/27.50%`（iter9/39/69/89/99），曲线非单调。 |
| iter99→199 需要保持 optimizer/RNG/逐域 cursor，而不是重新初始化。 | 基础 runner 分离 `LOAD_DIR`/`SAVE_DIR`；首次从 long100 iter99 跨 root 读取，之后从 long200 root 的 iter109/119/... 恢复。 | 检查正确的 source/destination、恢复状态和固定训练配方。 | 跨 root 恢复成功，iter109–169 每 10 步 checkpoint 完整；updates 100–176 指标有限、quota/Contract/turn-credit 对齐，随后因原定 KL 条件停止。 |
| iter199 是否应替换 iter99 需要 held-out 能力证据。 | 完成 iter199 后转换模型，跑 seed 300/301 和 seed-300 namespace probes。 | 共同查看总体、逐域能力和训练诊断，不由单个指标决定模型价值。 | 原运行在 update176 先停止；后续诊断续训完成 iter199 和双 seed 评测，结果见下文。 |

提交前验证包括 97 个 CPU tests、`py_compile`、全部相关 shell `bash -n`、
rollout preflight、转换和 8/2-GPU submission dry-run；共享盘可用 6200GB，超过
750GB 门槛。一个 8-GPU 续训任务与四个 2-GPU seed-300 曲线任务重叠运行，
主阶段实际峰值为 16 GPU。任务和实时证据见 [curve README](../experiments/tau2-rl-agent-user-boundary-v2-checkpoint-curve/README.md)
与 [long200 README](../experiments/tau2-rl-agent-user-boundary-v2-long200/README.md)。

曲线结果进一步定位了“上升后回落”而不是单调饱和：9→39、39→69、69→89、
89→99 的 overall pass@1 分别为 `+1.25/-2.25/+5.75/-2.25pp`；其中 69→89 的
paired 95% CI 为 `[+0.25,+11.25]pp`，89→99 的 overall CI 跨 0，但 telecom
下降 `-6.88pp`、CI `[-13.12,-0.62]pp`。iter89 只是 seed-300 定位峰值，按预设
规则不替代已有双 seed `win` 的 iter99，也不影响 iter99→199 续训。

续训暴露出后期 KL 上升，而恢复本身正常。运行记录和 checkpoint 表明 long100 iter99 的
optimizer/RNG/逐域 sampler 被精确恢复，iter109/119/129/139/149/159/169 的
checkpoint 文件完整，updates 100–176 无 OOM、NaN/Inf、quota 或协议失败；训练原始
二元成功率全程均值 `28.81%`、最后 10 步 `31.67%`，未出现 reward collapse。
但 updates 167–176 的 K2 KL 为
`0.070/0.041/0.060/0.096/0.151/0.134/0.175/0.152/0.289/0.260`，10-step 均值
`0.14287` 超过 `<0.10`，且 175–176 连续 `>=0.20`。因此 job 在 update176 后停止，
latest 保持 iter169；`EARLY_STOP_HEALTH_GATE.json` 只失败这两项 KL 检查。

`ITER199_LONG200_DECISION.json` 据此以 `early_health_stop` 记录原始 **`reject`**
标签；当时 iter199 尚不存在，因此该标签描述的是运行流程停止，而不是 iter199 的
能力结论。它说明 2e-6/K2 配方在 100 步内的低 KL 不能直接外推到 200 步。完整
证据见 [long200 README](../experiments/tau2-rl-agent-user-boundary-v2-long200/README.md)。

随后将 KL 作为诊断继续观察最终能力。新增的
`long200-final-kl-waiver` 保留 OOM、NaN/Inf、错误恢复、quota 和协议错位等真实
有效性检查；从完整 iter169 恢复时会
重做未落盘的 updates 170–176，再继续到199。严格 Gate/decision 保持不变，waiver
使用独立 Gate、HF root、eval 目录和结果文件。续训 job `pt-76a8tys1` 成功恢复
model/optimizer/RNG/逐域 sampler，updates 0–169 没有重训；仅重做未落盘的
170–176，再生成 177–199。iter179/189/199 checkpoint 完整，100–199 的训练原始
二元成功率为 `29.98%`，无 OOM、NaN/Inf、quota 或协议错误；最后 10 步 KL 为
`0.32374`，作为训练漂移诊断保留。

转换 job `pt-750lmyik` 和双 seed eval jobs `pt-ravwja7o`/`pt-xjhzn0v1` 均成功。
严格同口径 held-out 结果为：

| 模型 | Seed 300 pass@1 / pass@4(any) / pass^4 | Seed 301 | 双 seed 均值 |
|---|---:|---:|---:|
| Selected boundary SFT | 23.75 / 42.00 / 8.00% | 23.50 / 45.00 / 9.00% | 23.63 / 43.50 / 8.50% |
| Long100 iter99 | 27.50 / 53.00 / 9.00% | 31.50 / 60.00 / 12.00% | 29.50 / 56.50 / 10.50% |
| **Long200 iter199（诊断续训）** | **31.00 / 55.00 / 11.00%** | **33.75 / 61.00 / 12.00%** | **32.38 / 58.00 / 11.50%** |
| iter199 − iter99 | +3.50 / +2.00 / +2.00pp | +2.25 / +1.00 / 0.00pp | **+2.88 / +1.50 / +1.00pp** |

iter199 相对 iter99 的双 seed mean pass@1 在 airline/retail/telecom 分别为
`−5.00/+0.94/+8.75pp`，说明总体增益主要来自 telecom。seed301 airline 单独下降
`−7.50pp`，超过预设 `−5pp` 逐域非劣界；两个 seed 的 overall pass@1 增量 CI
分别为 `[−1.25,+8.50]pp`、`[−2.50,+7.00]pp`，均跨 0。telecom namespace
attributed termination 保持 0，300-case probe 两种温度均为 0 attempt，但正式 eval
的 raw namespace attempts 从 iter99 的 `8/13` 增至 iter199 的 `12/19`（seed
300/301）。因此 [`ITER199_KL_WAIVER_RESULT.json`](../experiments/tau2-rl-agent-user-boundary-v2-long200/ITER199_KL_WAIVER_RESULT.json)
证明 iter199 是完整、可评测且总体 point estimate 更高的 checkpoint。当前证据不
支持它无条件替换更均衡的 long100 iter99，但它是有效的 Telecom-oriented 候选；
本实验不继续 200→299。

## 三领域专家与 OPD teacher 验证 (2026-08-06)

Long200 的总体增益主要来自 telecom，而 airline 出现遗忘；单一 mixed policy 的
后续更新因此不能直接回答“按 domain 路由的冻结 teacher 是否更合适”。新实验
`tau2-rl-agent-user-boundary-v2-domain-experts` 从正式选中的 long100 iter99 同时
分叉 Airline/Retail/Telecom 三个 30-update 专家，只改变每步 quota 为 `6/0/0`、
`0/6/0`、`0/0/6`。三路继续使用完整三域数据、`DomainQuotaDataSource`、LR `2e-6`、
K2 `0.01`、K=8、field reward/penalties、v1 User、signed Contract 和
turn-credit-v1；optimizer、RNG、dataset fingerprint 与所有 domain cursor 从
iter99 精确恢复，scheduler horizon 仅扩展到130。

三个 expert root 相互独立，并分别检查 source、domain/quota、恢复 cursor 和
checkpoint 完整性。首次在 iter109 前中断会从 iter99 重来；有 iter109/119 后从
本 root 最新完整 checkpoint 恢复。已有 mixed long200 iter109/119/129 直接复用，
不重新训练。KL 作为诊断；OOM、NaN/Inf、错误恢复、quota 或协议错位会使结果无效。

三路 8-GPU normal-priority 训练均完成 updates 100–129，9 个 expert checkpoint
和 3 个 mixed prefix 的 checkpoint-scoped Gate 全部 `pass`；KL 保持
diagnostic-only，其余硬检查无失败。入选点的训练 raw-success mean/last-10 mean
分别为 Airline iter129 `38.33/42.29%`、Retail iter119 `34.58/37.71%`、Telecom
iter129 `11.60/12.08%`。12 个 HF 转换和 seed300 曲线评测全部完成，正式选择为
Airline iter129、Retail iter119、Telecom iter129。

Seed300 同 iteration expert↔mixed 的入选点 pass@1/pass@4(any)/pass^4 为：
Airline `37.50/55.00/25.00%` vs `40.00/65.00/15.00%`；Retail
`31.87/47.50/17.50%` vs `29.38/47.50/15.00%`；Telecom
`28.12/67.50/0.00%` vs `28.12/65.00/2.50%`。完整 9 点曲线、同本域样本量
expert109↔mixed129 和 paired task-bootstrap CI 见
[`DOMAIN_EXPERT_CURVE.md`](../experiments/tau2-rl-agent-user-boundary-v2-domain-experts/DOMAIN_EXPERT_CURVE.md)；这些 CI 普遍跨 0，因此结果支持方向性比较，但不声称各项差异
都独立达到统计显著。

Seed301 与遗忘诊断完成后，原 rubric 的逐 seed comparator 标签为：Airline seed300 的
pass@4(any) `−10pp`、seed301 pass@1 `−8.75pp`，因此标为 `reject`；Retail
两 seed 的 pass@1/pass@4(any) delta 为 `+1.875/0pp` 与 `0/−2.5pp`，均值
pass@1 严格为正，故 `advantage`；Telecom 为 `0/+2.5pp` 与
`+0.625/0pp`，两项均值严格为正，且两温度 namespace probe 通过，故
`advantage`。这些是原 rubric 的历史标签；该 rubric 主要按 pass@1 和
pass@4(any) 选择，pass^4 当时只是诊断项，因此 `opd_eligible=true` 不等价于
“一致性不退化”。按目标域双 seed mean 重新核对 pass^4：

| 目标域 | Selected boundary SFT | long100 iter99 父模型 | 同 iteration mixed | Selected expert | Expert 相对父模型 / mixed |
|---|---:|---:|---:|---:|---:|
| Airline iter129 | 20.00% | 20.00% | 10.00% | 20.00% | 0.00 / +10.00pp |
| Retail iter119 | 11.25% | 13.75% | 16.25% | **21.25%** | **+7.50 / +5.00pp** |
| Telecom iter129 | 0.00% | 2.50% | 3.75% | 1.25% | **−1.25 / −2.50pp** |

因此 Retail iter119 是当前最干净的 OPD teacher 候选；Airline 虽未损失
pass^4，但 pass@1/pass@4(any) 弱于 mixed RL，仍不入选。Telecom iter129 在
pass@1/pass@4(any) 上可用，却存在相对父模型和 mixed control 的一致性下降，不能
因旧标签直接进入 OPD。

遗忘诊断（pass@1/pass@4(any)/pass^4）为：Airline expert 在 Retail/Telecom
`30.00/55.00/12.50%`、`28.75/75.00/0.00%`；Retail expert 在
Airline/Telecom `41.25/60.00/20.00%`、`21.88/50.00/2.50%`；Telecom expert 在
Airline/Retail `40.00/60.00/20.00%`、`28.75/45.00/17.50%`。历史决策文件仍如实
记录 Retail 和 Telecom 为 `opd_eligible=true`、Airline 为 false；实验没有启动
任何 distillation。新的执行顺序是在 reward/advantage 优化完成后先使用 Retail，
Telecom 则先补 expert iter109 seed301，或与 boundary long200 iter199 的
Telecom-oriented checkpoint 比较。实现、job/log、checkpoint、CI 与决策证据见
[实验 README](../experiments/tau2-rl-agent-user-boundary-v2-domain-experts/README.md)和
[`DOMAIN_EXPERT_SUMMARY.json`](../experiments/tau2-rl-agent-user-boundary-v2-domain-experts/DOMAIN_EXPERT_SUMMARY.json)。
最后一个 Retail forgetting job 在 summary/results 完整写出且验证通过后卡在
sglang cleanup 的无界 `wait`；等待30分钟后精确 stop 为 `SUSPENDED`，不影响任何
评测产物。官方 eval runner 已改为每个服务 TERM 最多等待30秒后 SIGKILL，避免
后续作业占卡不退出。

本轮以现有 checkpoint 收口，不追加单域训练步数。Airline 虽在 seed300 曲线上升，
但两 seed 结果没有支持其替代 mixed RL；Retail pass@1 在 iter119 达峰后回落；
Telecom 的训练截断率从 `52.71%` 升至 `58.96%`、K2 KL 从 `0.0223` 升至
`0.0507`，同时 seed300 pass^4 从 iter109 的 `5.00%` 降至 iter129 的
`0.00%`。因此当前证据不能把结果简单归因于训练不足。Retail iter119 保留为首选
teacher；Telecom iter109 只完成 seed300，需补 seed301 后再决定；Airline iter129
作为未优于 mixed RL 的有效负面结果保留。更长专家训练可由后续独立实验重新检验。

Qwen3.5 non-thinking 最终 baseline（2026-08-09）：

Raw Qwen3.5-4B 在 v1 User、`current-single`、thinking disabled、100 tasks × 4
trials、temperature `0.6`、seed 300/301 下均完成 400/400，基础设施错误为 0。
最终结果如下；每个单元格为 pass@1 / pass@4(any) / pass^4。

| Scope | Seed 300 | Seed 301 | 双 seed 均值 |
|---|---:|---:|---:|
| Overall | 33.75 / 63.00 / 9.00% | 31.75 / 59.00 / 9.00% | **32.75 / 61.00 / 9.00%** |
| Airline | 43.75 / 70.00 / 15.00% | 35.00 / 60.00 / 15.00% | 39.38 / 65.00 / 15.00% |
| Retail | 26.88 / 50.00 / 5.00% | 26.25 / 50.00 / 7.50% | 26.56 / 50.00 / 6.25% |
| Telecom | 35.62 / 72.50 / 10.00% | 35.62 / 67.50 / 7.50% | 35.62 / 70.00 / 8.75% |

Matched seed-300 thinking-on raw Qwen3.5 为 `47.25/78.00/19.00%`；关闭
thinking 后下降 `−13.50/−15.00/−10.00pp`，paired task-bootstrap 95% 区间为
`[−18.75,−8.50]`、`[−24.00,−6.00]`、`[−18.00,−3.00]pp`。差异主要来自
Retail 和 Telecom；Airline pass@1 为 `+1.25pp`。完整 non-thinking 轨迹共有
20,377 个 assistant turns，thinking marker 和 raw/parsed multi-call turn 均为 0；
聚合 action/DB accuracy 为 `60.20/35.09%`，终止为 `user_stop=684`、
`too_many_errors=84`、`max_steps=32`。该双 seed 均值固定为 raw Qwen3.5
non-thinking 外部 baseline；因其使用 `current-single` 而新单调用训练使用
Agent-only `strict-single-v1`，跨 profile 只作模型定位。任务、日志和完整证据见
[实验 README](../experiments/tau2-qwen35-nonthinking-eval/README.md)。

单工具调用 SFT→RL 最终闭环（2026-08-09 → 08-10）：

新的 `strict-single-v1` 数据将每个 Assistant 多调用 target 按 source order 拆成
单调用消息，并把前序调用与真实 tool result 放回后续 target 的上下文；25,708 条
样本在 raw Qwen3-4B-Instruct-2507 上完成两轮 SFT。该 SFT 的双 seed 正式结果为
`25.88/49.50/8.50%`。同一 SFT 初始化的 GRPO iter99 为
`27.50/53.50/10.00%`，点估计提高但 per-seed CI 多数跨 0，因此保持 recipe 不变，
精确恢复 optimizer/RNG/domain sampler 并继续 updates 100–199。

续训 100 步全部有限且无 OOM；训练 raw reward 均值 `29.81%`，最后 10 步
`32.50%`。K2 均值/最后 10 步/最大值为 `0.02643/0.04252/0.07400`，明显低于
历史 boundary long200 的 `0.07389/0.32374/0.64395`，没有出现旧谱系的后期 KL
与梯度漂移。iter199 转 HF 后，seed300/301 均完成 400/400：

| 模型 | Seed 300 pass@1 / pass@4(any) / pass^4 | Seed 301 | 双 seed 均值 |
|---|---:|---:|---:|
| Single-call SFT | 27.25 / 50.00 / 10.00% | 24.50 / 49.00 / 7.00% | 25.88 / 49.50 / 8.50% |
| Single-call RL iter99 | 28.25 / 50.00 / 11.00% | 26.75 / 57.00 / 9.00% | 27.50 / 53.50 / 10.00% |
| **Single-call RL iter199** | **30.75 / 58.00 / 11.00%** | **31.50 / 65.00 / 10.00%** | **31.13 / 61.50 / 10.50%** |
| iter199 − iter99 | +2.50 / +8.00 / 0.00pp | +4.75 / +8.00 / +1.00pp | **+3.63 / +8.00 / +0.50pp** |
| iter199 − SFT | +3.50 / +8.00 / +1.00pp | +7.00 / +16.00 / +3.00pp | **+5.25 / +12.00 / +2.00pp** |

双 seed task-cluster paired bootstrap 95% 区间中，iter199 − iter99 为
pass@1 `[+0.13,+7.25]pp`、pass@4(any) `[+0.50,+15.50]pp`、pass^4
`[-3.50,+4.50]pp`；iter199 − SFT 为 `[+1.25,+9.25]`、
`[+4.50,+19.50]`、`[-2.50,+6.50]pp`。因此 pass@1 与 pass@4(any) 的
同谱系增益得到支持，pass^4 仍不确定。

iter199 相对 iter99 的双 seed mean delta 在 Airline 为
`−1.25/+5.00/−10.00pp`，Retail 为 `+3.75/+5.00/+5.00pp`，Telecom 为
`+5.94/+12.50/+1.25pp`；提升主要来自 Retail 和 Telecom，Airline 一致性仍是
明确代价。800 条轨迹的 action/DB accuracy 为 `65.46/27.39%`，终止为
`user_stop=723`、`too_many_errors=2`、`max_steps=75`、infra error `0`；raw、
parsed multi-call turn 和 strict protocol error 均为 0。

历史 boundary RL iter99/iter199 分别为 `29.50/56.50/10.50%` 和
`32.38/58.00/11.50%`；新的 iter199 相对二者为
`+1.63/+5.00/0.00pp`、`−1.25/+3.50/−1.00pp`。Raw Qwen3.5 non-thinking
外部 baseline 为 `32.75/61.00/9.00%`，新 iter199 相对为
`−1.63/+0.50/+1.50pp`。这些都是跨 profile 定位，不是因果对照。最终选择是：
strict-single-v1 内由 iter199 替代 iter99；它不被表述为对历史 boundary iter199
或 Qwen3.5 baseline 的全面支配。完整任务、训练诊断和评测证据见
[单工具实验 README](../experiments/tau2-agent-single-call-v1/README.md)。

**User 模型消融（2026-08-10 → 08-11）：** 固定上述 strict-single-v1
iter199 Agent、任务、四次 trial、seed300/301 和所有生成参数，只把 v1 User SFT
替换为同底座未微调的 `Qwen3-4B-Instruct-2507`。两个任务均完成 400/400，infra
error 和 Agent 单工具协议错误均为 0。这里的 iter199 Agent 在 RL 训练 rollout
阶段使用的是 v1 User SFT；raw User 只替换了评测 User，并未重新训练 Agent：

| 范围 | v1 User SFT 双 seed 均值 | Raw User 双 seed 均值 | Raw − v1 |
|---|---:|---:|---:|
| Overall | 31.13 / 61.50 / 10.50% | 26.88 / 52.50 / 8.00% | **−4.25 / −9.00 / −2.50pp** |
| Airline | 30.00 / 50.00 / 12.50% | 41.88 / 62.50 / 25.00% | +11.88 / +12.50 / +12.50pp |
| Retail | 31.25 / 58.75 / 15.00% | 29.69 / 58.75 / 7.50% | −1.56 / 0.00 / −7.50pp |
| Telecom | 31.56 / 70.00 / 5.00% | 16.56 / 41.25 / 0.00% | **−15.00 / −28.75 / −5.00pp** |

整体三项 paired task-bootstrap 95% 区间均跨 0，因为 Airline 与 Telecom 方向
相反；Telecom 三项区间分别为 `[−21.25,−8.75]`、`[−43.75,−13.75]`、
`[−10.00,−1.25]pp`。v1 User 在 77/320 条 Telecom 轨迹中产生 User 侧多调用
turn，raw User 为 0。结论是 User SFT 会显著改变分域交互和评测难度，但 raw
User 并未提高该单工具 Agent 的整体结果；这不能证明 User SFT 导致单工具相对
多工具的差距。严格验证该假设仍需给 multi-call iter199 Agent 补同一 raw User
双 seed 对照。完整记录见
[User 消融实验 README](../experiments/tau2-agent-single-call-v1-user-ablation/README.md)。

**单工具 User 模型统一对比（截至 2026-08-14）：** strict-single-v1 主体和
附加的 raw-Agent 参考均已完成 seed300/301、100 个 test tasks × 4 trials 的正式
评测。结果均为双 seed 均值，Agent temperature 为 `0.6`、评测
`max_steps=200`，两种 User server 都显式开启 Qwen tool-call parser。raw-Agent
行使用不同 Agent protocol，只作外部定位。

- `V1`：`Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf`，即由历史
  multi-tool 轨迹训练的 User simulator。
- `RAW`：未微调的 `models/Qwen3-4B-Instruct-2507`。
- “训练 User”指 RL on-policy rollout User；Agent SFT 是离线监督训练，不存在
  在线 rollout User。

下表每个结果单元均按 pass@1 / pass@4(any) / pass^4 排列；`★` 标在三项指标
都更高的评测 User 一侧。

| Agent checkpoint | 训练 rollout User | 训练 max_steps | 评测 User = V1 | 评测 User = RAW |
|---|---|---:|---:|---:|
| Raw Qwen3-4B-Instruct-2507 | 不适用（raw Agent） | 不适用 | 23.88 / 45.50 / 6.50% ★ | 17.38 / 35.50 / 2.50% |
| Single-call SFT final | 不适用（离线 SFT） | 不适用 | 25.88 / 49.50 / 8.50% | 28.00 / 51.00 / 9.00% ★ |
| V1-RL iter99 | V1 | 60 | 27.50 / 53.50 / 10.00% ★ | 25.25 / 50.50 / 5.50% |
| RAW-RL iter99 | RAW | 60 | 24.25 / 49.50 / 6.00% | 29.25 / 54.00 / 6.50% ★ |
| RAW-RL iter99 max120 | RAW | 120 | 25.25 / 51.00 / 6.00% | 28.38 / 56.50 / 9.50% ★ |
| V1-RL iter199 | V1 | 60 | 31.13 / 61.50 / 10.50% ★ | 26.88 / 52.50 / 8.00% |
| RAW-RL iter199 | RAW | 60 | 25.00 / 53.00 / 6.00% | 31.50 / 58.00 / 8.00% ★ |
| RAW-RL iter199 max120 | RAW | 120 | 25.25 / 49.00 / 5.50% | 30.13 / 58.50 / 7.00% ★ |
| Raw Qwen3.5-4B thinking-on | 不适用（raw Agent） | 不适用 | 48.50 / 76.00 / 22.00% | 53.25 / 78.00 / 25.00% ★ |
| Raw Qwen3.5-4B non-thinking | 不适用（raw Agent） | 不适用 | 32.75 / 61.00 / 9.00% ★ | 25.63 / 53.50 / 7.00% |

所有 RL 行都从同一个 single-call SFT 初始化。这个二维表明确区分了 RL
训练 rollout User 与评测 User。五个 RL crossed cells 在两个 seed 上都已完成，
每份 summary 为 400/400 simulations、infra error 0。五组 RL Agent 都在评测
User 与 rollout User 同族时三项更高，说明存在明显的 User-distribution
specialization，不能把跨 User 的分数直接解释为 User-independent Agent 能力。

pass_hat_4 对应这里的 pass^4，而不是 pass@4(any)。V1-RL 的 matched 结果并未
相对 SFT 下降：`8.50% → 10.00% → 10.50%`。下降集中在 RAW-User 分支：max60
为 `9.00% → 6.50% → 8.00%`，max120 为 `9.00% → 9.50% → 7.00%`。旧
multitool stability 也曾从 SFT `8.00%` 降至 RL `7.00/5.00/3.00%`，而
boundary long100/long200 从 `8.50%` 提高至 `10.50/11.50%`。因此当前证据指向
reward/advantage、rollout User 和继续训练阶段，而不是单工具协议本身。

下表每个 delta 均按 pass@1 / pass@4(any) / pass^4 排列，并固定评测 User：

| 固定的评测 User | 对比 | 唯一主要变化 | Delta |
|---|---|---|---:|
| V1 | V1-RL iter199 − V1-RL iter99 | 增加 100 个 RL updates | +3.63 / +8.00 / +0.50pp |
| RAW | RAW-RL iter199 − RAW-RL iter99 | 增加 100 个 RL updates | +2.25 / +4.00 / +1.50pp |
| RAW | RAW-RL iter199 max120 − RAW-RL iter99 max120 | 增加 100 个 RL updates | +1.75 / +2.00 / −2.50pp |
| RAW | RAW-RL iter199 max120 − RAW-RL iter199 max60 | 训练 rollout max_steps | −1.38 / +0.50 / −1.00pp |
| RAW | RAW-RL iter199 − V1-RL iter199 | 训练 rollout User | +4.62 / +5.50 / 0.00pp |
| V1 | RAW-RL iter199 − V1-RL iter199 | 训练 rollout User | −6.13 / −8.50 / −4.50pp |
| RAW | RAW-RL iter99 max120 − max60 | 训练 rollout max_steps | −0.88 / +2.50 / +3.00pp |
| 相同 SFT Agent | RAW 评测 − V1 评测 | 评测 User | +2.13 / +1.50 / +0.50pp |
| 相同 V1-RL iter199 Agent | RAW 评测 − V1 评测 | 评测 User | −4.25 / −9.00 / −2.50pp |

评测 User 本身会显著改变结果，因此不能把 V1 评测行与 RAW 评测行直接解释为
Agent 能力差。固定 RAW 评测后，RAW-RL iter199 相对 V1-RL iter199 的 pass@1
提高 `4.62pp`，paired 95% 区间为 `[+0.75,+8.62]pp`，支持训练 User 在 RAW
分布内产生影响。反向 crossed cell 也已完成：固定 V1 评测时，RAW-RL iter199
相对 V1-RL iter199 为 `−6.13/−8.50/−4.50pp`。两个方向共同支持分布匹配，
不支持某个训练 User 在两个评测 User 上都更优。`max_steps=120` 在 iter99 时将
训练截断率从 `18.38%` 降到 `1.19%`，但 pass@1 为 `−0.88pp`；继续到 iter199 后达到
`30.13/58.50/7.00%`，相对 max60 iter199 为 `−1.38/+0.50/−1.00pp`，仍未
建立替换 max60 默认模型的理由。其评测 max-step 终止为 `140/800`，高于 max60
iter199 的 `73/800`。

完整训练诊断、分域结果与区间见
[单工具主实验 README](../experiments/tau2-agent-single-call-v1/README.md)、
[User 消融 README](../experiments/tau2-agent-single-call-v1-user-ablation/README.md)、
[RAW-RL100 README](../experiments/tau2-agent-single-call-v1-raw-user-rl100/README.md)、
[RAW-RL200 README](../experiments/tau2-agent-single-call-v1-raw-user-rl200/README.md) 和
[max_steps120 README](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl100/README.md)、
[max_steps120 continuation README](../experiments/tau2-agent-single-call-v1-raw-user-maxsteps120-rl200/README.md)。

**下一阶段固定顺序（2026-08-14）：**

1. **先优化 reward 与 advantage/credit assignment。** 官方 held-out eval 继续只用
   原始 binary task success；训练 reward 则重点区分正确 action、参数和最终
   DB/environment state，避免 field credit 把“多调用/工具名正确”误当成任务完成。
   advantage 需要把信用分配到产生该结果的 Assistant turn，并核对 GRPO group
   normalization、mask 和 zero-variance group 的实际行为。第一轮受控消融固定
   Agent SFT init、rollout User、domain quota、LR/KL、max_steps 和评测协议，只改变
   reward/advantage；同时报告 pass@1、pass@4(any)、pass^4、action/DB accuracy、
   KL、截断率和 zero-variance-group rate。先用官方双 seed 结果选定配方。
2. **配方选定后再做 on-policy distillation。** OPD teacher rollout 的筛选或加权
   使用上一步选定的同一 reward/advantage 语义。第一位 teacher 使用 Retail expert
   iter119；Telecom expert iter129 因 pass^4 低于父模型和 mixed control 暂不直接
   使用，先补 Telecom expert iter109 seed301，或与 boundary long200 iter199
   比较后再选。Airline expert 不进入 OPD。当前没有启动 OPD 任务。
