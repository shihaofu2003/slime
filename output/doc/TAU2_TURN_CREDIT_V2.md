# tau2 GRPO 奖励与 turn-level credit 审计

日期：2026-08-14。范围：当前 `slime` checkout、本地 `../tau2-bench`、AReaL tau2 RL 训练集与官方 `test` split。新算法名为 `turn-credit-v2`，它是待受控实验选择的 candidate，不是已选 recipe。

## 1. 结论

- 训练和官方评测的任务成功都是二值 outcome：只有正常 `agent_stop` / `user_stop` 才可能成功，然后将该任务 `reward_basis` 中的各分量相乘。`Sample.Status.COMPLETED` 只代表正常结束，不代表任务成功。
- 当前 `turn-credit-v1` 先将二值 outcome 与 field partial score 相加并做 GRPO 组内标准化，再在 token 优势上直接减去原始 turn penalty。这确实混合了“组内标准差单位”与“手工 reward 单位”，而且同一个 turn 的 penalty 会被复制到该 turn 的每个 token。
- `turn-credit-v2` 保留官方二值 outcome 作为唯一 trajectory reward；将“完整参数匹配的参考动作”和“明确工具规范错误”变成 `+1/-1/0` directional critique，再在轨迹内重分配固定的零和优势预算。参考参数不一致只作诊断，不是规范处罚。这保留 outcome 目标，也不再把原始 penalty 直接减到 z-score 上。

## 2. 成功和失败如何判定

`examples/tau2-bench/rl/reward.py:evaluate_simulation_with_constructor` 是 RL 本地评估入口；`../tau2-bench/src/tau2/evaluator/evaluator.py:evaluate_simulation` 是官方 eval 入口。两者的核心公式都是：

\[
R(\tau)=\mathbf 1[\text{normal stop}]\prod_{b\in\texttt{reward\_basis}}r_b(\tau),\qquad r_b\in\{0,1\}.
\]

| 分量 | 本地实现的判定 |
|---|---|
| `DB` | 在新环境中重放参考 actions 得到目标终态；重放实际对话得到预测终态；Agent DB 和 User DB 都必须与目标一致。 |
| `ENV_ASSERTION` | 所有 env assertion 都必须成立。 |
| `ACTION` | 每个 reference action 都必须在完整轨迹的 User/Assistant tool calls 中找到匹配。当 `compare_args=None` 时，当前 `Action.compare_with_tool_call` 实际按预测 call 的 keys 比较，因此缺参 call 可能被宽松匹配；同一 predicted call 也可被多个 reference action 复用。 |
| `COMMUNICATE` | 每个必需字符串都要在 Assistant 文本中出现，大小写不敏感并忽略逗号。 |
| `NL_ASSERTION` | 由 LLM judge 对每个自然语言 assertion 判定；官方 eval 支持，当前 RL 本地 evaluator 明确不支持。 |

无 evaluation criteria 的任务返回 1；`max_steps` / `timeout` / `context_window_exceeded` 等提前终止直接返回 0。`rollout._sample_status` 将正常 stop 标记为 `COMPLETED`，而 `sample.reward` 才是成功与否。

本地数据的实际 basis 不完全等于 upstream 文档当前描述，因此实验应以本地 task 文件为准：

- AReaL RL train 共 1,982 任务：1,916 条未显式写 basis，由 `EvaluationCriteria` 默认为 `DB+COMMUNICATE`；50 条为 `ENV_ASSERTION`；16 条为 `ENV_ASSERTION+ACTION`。前 1,916 条的 communication 多数为空，实际上主要由 DB 终态决定。
- 官方 `test` split 共 100 任务：airline 20 条均为 `DB+COMMUNICATE`；retail 为 39 条 `DB+NL_ASSERTION` 加 1 条 `DB`；telecom 为 28 条 `ENV_ASSERTION` 加 12 条 `ENV_ASSERTION+ACTION`。

`examples/tau2-bench/eval/official/run_eval.py:_pass_metrics` 中，`pass@1/pass^1` 是所有 simulation 的二值 reward 平均；`pass@4(any)` 是每题 4 次中至少一次成功的任务比例；`pass^4` 是 4 次全部成功的任务比例。参考 actions 通常只是到达目标终态的一条路径，不是唯一正确路径，这一点也由 [tau2 评测文档](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md) 明确说明。

## 3. 当前 `turn-credit-v1` 的奖励和优势

`examples/tau2-bench/rl/reward_postprocess.py` 的 v1 轨迹分数为：

\[
G_i=R_i+\alpha_{d(i)}P_i,
\qquad
B_i=\frac{G_i-\mu_G}{\sigma_G+10^{-6}},
\qquad
A_{i,t,k}=B_i-C_{i,t}.
\]

`R_i` 是官方二值 task reward。`P_i` 是只对当前任务中存在的分量做加权平均的 partial score；默认权重为 tool-name 0.25、argument 0.35、communication 0.05、env assertion 0.10、DB 0.25，telecom 为 0.20/0.25/0.20/0.15/0.20。`alpha` 默认 retail 0.20、airline 0.25、telecom 0.40。`B_i` 被广播给轨迹内所有可训 Assistant token。

`C_{i,t}` 是直接贴到错误 turn 的手工 penalty：

| 错误 | 每次 | 每 turn cap |
|---|---:|---:|
| nonexistent tool | 0.15 | 0.30 |
| malformed JSON | 0.15 | 0.30 |
| wrong argument field | 0.10 | 0.40 |
| tool execution error | 0.25 | 0.50 |
| repeated call | 0.05 | 0.20 |
| `max_steps` | 0.20 | 0.20 |
| wrong namespace | 首次 1.0，后续 +0.1 | 2.0 |

同一 turn 的各类 cap 可叠加，最大可到 3.9。v1 这条 turn-aware 路径使用 `_TURN_PENALTY_SPECS` 中的固定值；脚本导出的 `TAU2_PENALTY_*` 只影响非 turn-aware legacy 路径，不会改变 v1 的 token penalty。K2 KL `0.01` 在 policy loss 中单独计算，不是这个 advantage 公式的一部分。

## 4. 尺度问题与实证

尺度担心成立。因为

\[
B_i-C_{i,t}=\frac{G_i-\mu_G-\sigma_G C_{i,t}}{\sigma_G},
\]

所以固定的 `C=0.2` 对应的原始 reward 差是随 group 改变的 `0.2 sigma_G`；在 normalized advantage 空间里它又始终是绝对 0.2。对于全组同分的 group，`B_i=0`，v1 可能只训练绝对负 penalty。此外，`C_{i,t}` 被复制给 turn 内每个 token，同一错误的总梯度质量会随输出长度增大。

训练日志中的 `rollout/raw_reward` 只统计 accepted 轨迹的官方二值 `task_reward`；local critique 和 lambda 只进入 token advantage，不回写 raw reward。因此两个 arm 的 raw reward 相同，只表示该批成功条数相同，不能证明其行为、梯度或 credit 信号相同，也不能单独用来调 lambda。

对 `tau2-rl-agent-user-boundary-v2-long100` iter20–99 的 4,000 条轨迹重建 v1 信号：500 个 K=8 group 中 480 个进入训练；234/480 个 accepted group 的二值 outcome 零方差，44/480 个的 `G_i` 也零方差、因而只靠 local penalty 训练。`std(G)` 的 p10/median/p90 为 0.00426/0.374/0.575。在 4,049 个受罚 turn 中，penalty p50/p90/p99 为 0.20/0.35/1.0，最大 1.8；137/1,450（9.45%）原本 `B_i>0` 的受罚 turn 被翻成负优势。这不证明每次翻转都错，但证明 penalty 已不是小的辅助项。

## 5. 近期工作对设计的约束

| 工作 | 与本项目直接相关的结果 | 本次采用方式 |
|---|---|---|
| [AReW / T3](https://arxiv.org/abs/2603.12109) 及[官方代码](https://github.com/unimpor/T3) | 使用 `-1/0/+1` step critique，在轨迹内将优势从负向 step 重分配给正向 step，不修改 outcome reward；论文直接报告 Tau2Bench-Telecom solo 和 standard dual-control。负标签包含 malformed/invalid/repeated/execution failures，正标签包含 expected-action progress 等。 | v2 采用 directional critique + fixed `step_abs_sum` 思路，但只使用当前 pipeline 可靠取得的信号。 |
| [IRC + MT-GRPO/GTPO](https://arxiv.org/abs/2604.02869) | 在 Tau-Bench 训练、Tau2-Bench 评估；朴素 dense turn reward 最多使结果下降 14pp。其 hybrid 使用 discounted local return 的 group normalization，再加缩小的 outcome advantage；并用 reward tier 与成功的经验相关性校准方向。 | 不直接搬用未校准的数值 dense reward；要求后续实验检查每类 critique 与官方成功的相关性和 advantage sign。 |
| [CM2](https://arxiv.org/abs/2602.12268) | 更细的 turn/step checklist advantage 早期学习更快，但在 judge 噪声下更早、更严重地 collapse；最稳定设置仍是 trajectory-level。 | 正向信号只用可验证的完整参数匹配，不引入 LLM judge 型的高频 token/step reward。 |
| [GDPO](https://arxiv.org/abs/2601.05242) | 多 reward 先求和再组内标准化会丢失分量差异；各 reward channel 分开标准化后再组合更稳定。 | v2 将 outcome channel 和 rule/directional channel 分开，不把 field rewards 先 scalarize 进 outcome z-score。 |
| [SENTINEL](https://arxiv.org/abs/2606.12908) | 在 Tau2 Retail、Qwen3-4B-Thinking 和 slime GRPO 上保留 `DBCheck AND CommCheck` outcome，并加入认证、格式、重复调用和长度 shaping；主要增益来自 failure-driven task distribution。 | 证明确定性 rule shaping 在近似技术栈中可用，但其 raw additive penalties 没有解决 advantage 尺度问题，因此不照搬数值。 |

IRC 的训练集、User、模型和评测协议与当前项目不同；AReW 使用 PPO/Qwen2.5 且它的 standard positive proxy 包含更多在线信息增益信号。因此这些结果是设计依据，不是对本地 Qwen3-4B GRPO 的效果保证。

## 6. 已实现的 `turn-credit-v2`

对同一 prompt 的 K 条 rollout，先仅对官方 outcome 做当前 slime 的 GRPO 归一化：

\[
B_i=\operatorname{GN}_{j=1\ldots K}(R_j^{\text{official}}).
\]

对每个可训 Assistant turn 标记：

\[
z_{i,t}=\begin{cases}
-1,&\text{turn 含明确 rule error};\\
+1,&\text{无 rule error 且新匹配一个 tool-name + 全部 reference arguments};\\
0,&\text{其他。}
\end{cases}
\]

负向 rule error 仅为 `wrong_namespace_tool`、`nonexistent_tool`、`malformed_json`、`tool_execution_error`、`repetition`。同 turn 同时完成 reference action 和出现 rule error 时，负标签优先。`wrong_argument_field` 表示偏离一条 reference path，可能仍是官方允许的有效替代路径，因此只保留为诊断；`max_steps` 是轨迹终止诊断，不能唯一归因到最后一个 Assistant turn，二者都不作 v2 local negative。

令 `P_i={t:z=+1}`、`N_i={t:z=-1}`。若二者都有值，直接重分配；若只有负标签，则用规则干净、非最终的 neutral turn 作为比较侧 `P_i`。这是 AReW executable recipe 的 complementary-neutral fallback，使明确的格式/工具错误仍有局部信号，同时不把最终回答奖励为正。若只有正标签，则不合成负例，避免处罚偏离 reference 但有效的替代路径：

\[
u_{i,t}=\begin{cases}
+0.5/|P_i|,&t\in P_i;\\
-0.5/|N_i|,&t\in N_i;\\
0,&\text{otherwise}.
\end{cases}
\qquad \sum_tu_{i,t}=0,\quad\sum_t|u_{i,t}|=1\ \text{（激活时）}.
\]

无法形成正负比较侧时所有 `u` 均为 0。

若整条轨迹有 `M_i` 个可训 token、该 turn 有 `L_{i,t}` 个可训 token，最终 token advantage 为：

\[
A_{i,t,k}=B_i+\lambda M_i\frac{u_{i,t}}{L_{i,t}},\qquad
\lambda=\texttt{TAU2\_TURN\_CREDIT\_REALLOCATION\_WEIGHT}.
\]

`M_i` 不能省略：slime 的实际 policy-loss reducer 会再除以该 rollout 的 trainable-token 总数。按真实 reducer 计算，local channel 的 signed mean 为 `lambda sum_t u_{i,t}=0`；激活时 absolute mean 为 `lambda sum_t |u_{i,t}|=lambda`。因此优势权重预算不随 turn token 数、轨迹长度、错误数量或 group reward 方差改变；它并不声称梯度范数恒定。默认取保守的 `lambda=0.1`，因为修正 reducer 尺度后 `0.5` 已接近主 outcome advantage 的量级；`lambda=0` 是同一代码路径下的 outcome-only GRPO control。

参考 action 仍只是 directional proxy，不是任务 utility。因此不匹配 reference 的 turn 默认是 neutral，不是 negative；最终正确性仍由 DB/env/action/communication/NL 组成的官方 outcome 决定。

## 7. 代码入口和配置

- `reward_postprocess.py:compute_rollout_score`：v2 只返回 official task reward，field partial score 仅作 diagnostics。
- `reward_postprocess.py:build_turn_credit_v2`：将 exact full-argument reference action 和 rule error 对齐到 Assistant response span。
- `reward_postprocess.py:compute_turn_credit_v2_group_advantages`：对 outcome 做 group normalization，计算 fixed-budget turn reallocation，并消除 turn token-length bias。
- `reward_postprocess.py:tau2_reward_post_process`：在 DP 拆分前生成每条 sample 的 response-aligned `token_advantages`。
- `reward_postprocess.py:turn_aware_grpo_advantage`：v1 仍执行 `normalized reward - token_penalty`；v2 直接消费已计算的 token advantage，并按 context-parallel layout 切片。
- `filters.py:drop_zero_std_or_unsampleable`：over-cap / invalid group 始终 replacement；v2 记录 group 是否有实际优势信号。默认 `TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0`，零信号组保留为零 policy advantage，以免不同 recipe 改变 prompt 分布；独立的 K2 KL loss 仍然存在。只有显式设为 `1` 才 replacement。
- `rollout.py:_fill_sample_from_simulation`：将 critique 严格对齐到完整对话 tokenization 的可训 Assistant spans。

历史 boundary 脚本仍默认 `turn-credit-v1`，只有显式设置才启用 candidate。一个隔离的 smoke 提交示例：

```bash
bash scripts/submit.sh \
  --experiment tau2-rl-turn-credit-v2-smoke \
  --name tau2-turn-credit-v2-smoke \
  --gpus 8 \
  --env TAU2_RL_EXPERIMENT_NAME=tau2-rl-turn-credit-v2-smoke \
  --env TAU2_TURN_CREDIT_VERSION=turn-credit-v2 \
  --env TAU2_TURN_CREDIT_REALLOCATION_WEIGHT=0.1 \
  --env TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0 \
  --env SMOKE_CKPT_ROOT=/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_turn_credit_v2_smoke_20260814 \
  examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_boundary_v2.sh \
  -- smoke
```

上面的 boundary-v2 命令是实现阶段的 smoke 示例。正式受控实验使用 strict-single-v1 SFT 与对应 runner；实验记录见 `output/experiments/tau2-agent-single-call-credit-core/README.md`。

## 8. 验证状态

`examples/tau2-bench/rl/test_rollout_logic.py` 新增了以下 preflight：完整参数匹配为 positive、部分参数匹配为 neutral、rule error 为 negative；reference 参数不一致和 `max_steps` 不作 local negative；negative-only complementary-neutral 与 positive-only 不合成负例；全失败 group 的 official outcome 不被改写；不同轨迹/turn 长度经真实 sample-mean reducer 后都保持 signed budget 0、L1 budget `lambda`；zero-signal replacement 的开关；以及 custom advantage 对 response token vector 的消费。原有 v1 checks 显式锁定 v1，因此 v2 配置下也会检查历史行为未改变。

已通过五个相关 Python 文件的 `py_compile`、两个 Bash runner 的 `bash -n`，以及用轻量 `Sample` stub 执行的 v2 pure/sample-mean checks。四 token fixture 在测试专用 `lambda=0.5` 下的期望 advantage 为 `[0.5, 0.5, 0.0, -1.0]`，sample mean 的 signed local budget 为 0、L1 budget 为 0.5。当前 host 的 Python 3.8 缺少 job image 内的完整依赖，而源码使用 Python 3.10+ 功能，所以 host 未执行带真实 tokenizer、tau2 环境以及 slime DP/CP reducer 的全量 preflight；正式 Stage A、Stage B 和 Stage Final 均已在 job image 内通过该 preflight。

用 long100 的 frozen v1 dump 重新执行当前 v2 的完整 golden-argument matcher 与修正版 rule 集合，排除 5 个确定 over-cap group 后，986/3,960（24.90%）条 trajectory、362/495（73.13%）个 group 会激活 modifier；trajectory 激活率为 retail 33.77%、airline 23.02%、telecom 17.34%。这是旧 policy 上的 signal-coverage 检查，不是 v2 的效果估计；online replacement 的反事实轨迹也不在该 dump 中。

## 9. 受控实验与已知限制

不应仅根据 training reward 选 recipe。实际实验从同一个 strict-single-v1 SFT `Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809/final_hf` 初始化，只比较预注册的三个核心 arm：

1. `v1-matched`：采用 `turn-credit-v1` credit 语义的 matched control；与 v2 arms 一样显式关闭 zero-signal replacement。
2. `v2-l000`：`turn-credit-v2, lambda=0`，即 official-outcome-only 消融，同时控制 v2 代码路径差异。
3. `v2-l010`：`turn-credit-v2, lambda=0.1`，加入固定零和 local budget。

训练固定 User v1、K=8、每更新 6 个 group、global batch 48、100 updates、LR `2e-6`、K2 KL `0.01`、temperature `1.0`、`max_steps=60`、每 trajectory 16,384-token cap、train seed 1234、rollout seed 42，以及分段但三臂一致的三域 quota；所有 arm 均设置 `TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0`。训练诊断报告 accepted official outcome、exact reference-action rate、DB success、KL、truncation、rule-error/repetition、原始 outcome 零方差 group 比例、v2 modifier activation 和 token-level advantage 尺度。

三臂均完成 100 个有限更新，每臂实际接受 600 个 K8 group、4,800 条 trajectory。`v1-matched/v2-l000/v2-l010` 的全程 raw reward 分别为 `26.29/26.77/26.00%`，后 20 步为 `25.10/26.88/28.13%`；因此 raw reward 没有选出新算法。`v2-l010` 在 1,071/4,800 条 accepted trajectory 上激活 local credit，并挽救 179/310 个 binary-zero-variance group，使真正无 outcome/local 信号的 group 降至 131/600。每条活跃 trajectory 的 modifier 和严格为 0、local reducer L1 严格为 `0.1`；local/global L1 总量为 `107.1/1806.95`（`5.93%`）。per-token absolute local advantage 的 p99/max 为 `.6644/19.27`，但只有 `.115%` token 超过 `2.5`。`v2-l010` 明显降低 namespace、nonexistent-tool、malformed-JSON、repetition 和 too-many-errors；`v2-l000` 虽有最高的全程 raw/action/DB，却出现明显 rule 退化及后期 K2 峰值 `.19109`。这证明 `lambda=0.1` 有效参与 credit assignment，但不能证明其 held-out 效果或应继续调大。

held-out official eval 固定同一 v1 User、三域全部 100 个 test tasks、4 trials、seed 300/301、temperature `0.6`、`max_steps=200`，并保持历史随机采样语义：Agent/User 均不启用 deterministic inference，tau2 trial seed 不转发为 Agent 的 SGLang `sampling_seed`。只有某个 arm 的两 seed mean `pass@1` 和 `pass@4(any)` 都严格高于另两个 arm 时才选它；完全相同的 primary score 才依次比较 DB、action accuracy，否则保留 `v1-matched`。`pass^4`、paired interval 和训练 raw reward 只作报告与解释，不参与选臂。

已知限制：完整 reference action 仍可能是非必要的 read-only 路径，所以它只能作 weak direction；negative-only 只有在存在非最终 neutral counterpart 时才有 local signal，positive-only 始终不添加 local signal。当前没有逐 turn 的 DB/env potential，因为仅知“状态变化”不能判定是向目标前进，IRC 也观察到未校准的 state-change reward 与成功负相关。当前 trajectory JSONL 在 group post-process 前写出，因此能审计 raw critique/modifier，但最终 outcome advantage 和 token vector 需从 preflight 或训练侧 metadata 检查。下一候选应在可重放环境上定义“到官方目标终态的距离改善”，并先用 frozen rollout 验证其与 outcome 的方向一致性。在奖励/优势 recipe 通过受控官方评测前不进入 OPD。
