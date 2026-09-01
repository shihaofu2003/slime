# Qwen3.5-4B non-thinking single-call baseline

实验名：`tau2-qwen35-nonthinking-eval`。目的：在正式 tau2 三领域协议下评测
Qwen3.5-4B 的非 thinking 模式，并记录它与现有 SFT/RL 模型在工具调用粒度上的差异；
同时检查已选 SFT 数据能否改造成单轮单工具训练数据。seed 300/301 的结果作为
Qwen3.5-4B raw、non-thinking、single-call 的最终外部 baseline。

## Evaluation setup

本轮使用 Qwen3.5-4B、v1 STOP User、test 全量 100 tasks × 4 trials、seed
300/301、temperature 0.6、`max_steps=200`。Agent 使用 `current-single`：每个
assistant turn 只能输出文本或恰好一个工具调用。chat template 显式传入
`enable_thinking=false`，最大生成长度为 8,192。两个 seed 各使用 2 GPU，均已完成。

完整 800 条轨迹共包含 20,377 个 assistant turns；thinking marker、原始多调用输出
和解析后的多调用 turn 均为 0，单轮最大解析调用数为 1，记录的协议均为
`current-single`。因此实际运行配置与目标一致。

## Final baseline results

两个任务均完成 400/400 simulations，基础设施错误为 0。每个单元格为
pass@1 / pass@4(any) / pass^4。

| Scope | Seed 300 | Seed 301 | Two-seed mean |
|---|---:|---:|---:|
| Overall | 33.75 / 63.00 / 9.00% | 31.75 / 59.00 / 9.00% | **32.75 / 61.00 / 9.00%** |
| Airline | 43.75 / 70.00 / 15.00% | 35.00 / 60.00 / 15.00% | **39.38 / 65.00 / 15.00%** |
| Retail | 26.88 / 50.00 / 5.00% | 26.25 / 50.00 / 7.50% | **26.56 / 50.00 / 6.25%** |
| Telecom | 35.62 / 72.50 / 10.00% | 35.62 / 67.50 / 7.50% | **35.62 / 70.00 / 8.75%** |

聚合 action/DB accuracy 为 `60.20%/35.09%`；seed 300 分别为
`60.14%/36.18%`，seed 301 为 `60.26%/33.93%`。800 条轨迹的终止分布为
`user_stop=684`、`too_many_errors=84`、`max_steps=32`，无 infrastructure error。

历史 thinking-on Qwen3.5 使用相同 raw 模型、v1 User、seed 300、100 tasks × 4
trials、temperature 0.6 和 `current-single` 主协议。下面给出 matched seed-300
point estimate；每个单元格仍为三个 pass 指标。

| Scope | Thinking on | Thinking off | Off - on |
|---|---:|---:|---:|
| Overall | 47.25 / 78.00 / 19.00% | 33.75 / 63.00 / 9.00% | **-13.50 / -15.00 / -10.00pp** |
| Airline | 42.50 / 70.00 / 20.00% | 43.75 / 70.00 / 15.00% | +1.25 / 0.00 / -5.00pp |
| Retail | 43.13 / 70.00 / 22.50% | 26.88 / 50.00 / 5.00% | -16.25 / -20.00 / -17.50pp |
| Telecom | 53.75 / 90.00 / 15.00% | 35.62 / 72.50 / 10.00% | -18.13 / -17.50 / -5.00pp |

Overall paired task bootstrap 95% intervals for off-minus-on are
`[-18.75,-8.50]pp`、`[-24.00,-6.00]pp`、`[-18.00,-3.00]pp`。因此关闭
thinking 的主要损失在 retail 和 telecom；airline pass@1 没有同方向回落。

最终 baseline 定义为两 seed 均值 `32.75/61.00/9.00%`。它是 raw Qwen3.5
在 `current-single` 下的外部参照；新的 single-call SFT/RL 使用
`strict-single-v1` 和 Agent-only ownership view，因此跨 profile 差值只作模型谱系
定位，不解释为训练因果。

## Tool-call behavior

对历史完整轨迹直接计数。Qwen3.5 历史强基线几乎总是逐次调用工具；当前 long100
RL 则频繁在一个 assistant turn 内批量调用。两者的 prompt 协议不同，因此该表说明
现象和数据分布，不单独证明差异来自模型能力。本轮 matched single-call 评测用于隔离
thinking 模式的影响。

| Run | Protocol | Multi-call trajectories | Multi-call tool turns | Max calls/turn |
|---|---|---:|---:|---:|
| Qwen3.5-4B final baseline, thinking off | current-single | 0/800 | 0/7,859 | 1 |
| Qwen3.5-4B historical, thinking on | current-single | 2/400 (0.50%) | 2/2,881 (0.07%) | 3 |
| boundary RL long100 iter99, seed 300 | agent-owned dependency-safe multi | 307/400 (76.75%) | 418/1,831 (22.83%) | 18 |

| Domain | Qwen3.5 multi-call trajectories | long100 RL multi-call trajectories |
|---|---:|---:|
| Airline | 0/80 | 69/80 |
| Retail | 0/160 | 141/160 |
| Telecom | 2/160 | 97/160 |

The historical Qwen3.5 raw outputs contain thinking markers in 8,734 of 8,739
generated assistant turns, even though its executed tool behavior is almost entirely
single-call. The final non-thinking baseline has zero markers and zero batched calls,
so thinking mode and tool-call batching remain separate variables.

## SFT conversion check

检查对象是 selected Contract + boundary SFT 的 19,318-row 数据。每行仅监督最后
一个 assistant target；其 target 分布如下。

| Domain | Rows | Text targets | One-tool targets | Multi-tool targets | Rows after serialization |
|---|---:|---:|---:|---:|---:|
| Airline | 8,783 | 4,992 | 1,991 | 1,800 | 13,351 |
| Retail | 8,595 | 4,493 | 3,009 | 1,093 | 10,460 |
| Telecom | 1,940 | 1,635 | 298 | 7 | 1,950 |
| Total | 19,318 | 11,120 | 5,298 | 2,900 | 25,761 |

结论是可以转换，而且不应靠删除 multi-tool targets 实现。2,900 个 multi-tool
targets 占 15.01%，最多一轮 10 calls；所有 2,900 个批次都能在原始
`AReaL-tau2-data/tau2_sft_train.jsonl` 的同对话后续前缀中找到按原顺序对应的工具
结果。仅使用 selected JSONL 的兄弟前缀也已覆盖 2,557/2,900（88.17%）。

合适的转换方式是把每个调用及其结果序列化为独立的 assistant/tool 回合，并把原来的
multi-call target 展开成多个 target-only 样本；后一个调用会看到前一个调用的真实结果。
这样保留全部调用、参数、结果和最终文本回复。展开会增加 6,443 个监督目标，总数变为
25,761。正式产数时仍需重新分词，并对越过 16,384-token 上限的样本正常过滤或调整；
本轮只完成可行性检查，尚未生成新训练集。

## Tasks

- `pt-9iha2qo2`, `fsh-qwen35-nonthinking-s300-0809-123953`：误用 multi profile
  的 seed 300 提交；在调度前取消，不产生评测结果。
- `pt-ohwmgytt`, `fsh-qwen35-nonthinking-s301-0809-123953`：误用 multi profile
  的 seed 301 提交；在调度前取消，不产生评测结果。
- `pt-usqlo3r5`, `fsh-qwen35-nonthinking-single-s300-0809-124159`：正式
  current-single、non-thinking seed 300，成功完成 400/400，耗时 2h28m；
  [run log](jobs/fsh-qwen35-nonthinking-single-s300-0809-124159/run_20260809_124159.log)。
- `pt-sbh5euvg`, `fsh-qwen35-nonthinking-single-s301-0809-124159`：正式
  current-single、non-thinking seed 301，成功完成 400/400，耗时 2h45m；
  [run log](jobs/fsh-qwen35-nonthinking-single-s301-0809-124159/run_20260809_124159.log)。
