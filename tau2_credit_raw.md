# Tau2-Benchmark 中的 Turn-Level Credit Assignment 设计

## 1. 问题背景

在 Tau2-Benchmark 这类多轮 Agent 任务中，强化学习训练通常只能获得轨迹级终局奖励。例如，一个完整任务最终成功，则整条轨迹得到成功奖励；任务失败，则整条轨迹得到失败奖励。

标准 Interactive GRPO 可以写为：

$$
A_i^{\text{traj}}
=
\frac{
R_i-\mu_R
}{
\sigma_R+\epsilon
}
$$

其中：

- $$R_i$$ 是第 $$i$$ 条完整交互轨迹的终局奖励；
- $$\mu_R$$ 和 $$\sigma_R$$ 是同一任务下多个 rollout 的奖励均值和标准差；
- $$A_i^{\text{traj}}$$ 会被广播给整条轨迹中的所有可训练 Agent token。

这种做法的问题是：

> 一条失败轨迹中可能包含大量正确的信息查询、环境探索、用户确认和恢复动作，但由于最终任务失败，这些动作会一起获得负 advantage。

因此，对于 Tau2 这样的 long-horizon interactive agent，我们希望在保留 trajectory-level outcome signal 的同时，引入一个简单、稳定、无需额外 judge model 的 turn-level credit assignment 机制。

本文希望构造：

$$
A_{i,t}
=
A_i^{\text{traj}}
+
\lambda A_{i,t}^{\text{turn}}
+
\beta r_{i,t}^{\text{format}}
$$

其中真正需要解决的问题是：

> 如何在不使用外部模型判断动作正确性的前提下，构造可靠的 $$A_{i,t}^{\text{turn}}$$？

---

## 2. Tau2 相关 RL 方法中的 Credit Assignment

**2.1 Interactive GRPO**

Interactive GRPO 本质上仍然使用标准 GRPO 的 trajectory-level advantage。

普通 GRPO 的 rollout 是：

$$
x \rightarrow y
$$

而 Interactive GRPO 的 rollout 变为：

$$
x
\rightarrow
a_1
\rightarrow
u_2
\rightarrow
a_2
\rightarrow
u_3
\rightarrow
\cdots
\rightarrow
R
$$

其中：

- $$a_t$$ 是 Agent 的行为；
- $$u_{t+1}$$ 是 user simulator 或环境产生的下一轮 observation；
- $$R$$ 是整条轨迹最终得到的 terminal reward。

其 advantage 仍然是：

$$
A_i^{\text{traj}}
=
\frac{
R_i-\mu_R
}{
\sigma_R+\epsilon
}
$$

因此：

$$
A_{i,1}
=
A_{i,2}
=
\cdots
=
A_{i,T}
=
A_i^{\text{traj}}
$$

Interactive GRPO 改变的是 rollout protocol，而不是 credit assignment estimator。

---

**2.2 FACA**

FACA 在 trajectory outcome advantage 之外增加了 turn-level reaction advantage：

$$
A_{i,t}
=
A_i^{\text{outcome}}
+
\lambda A_{i,t}^{\text{reaction}}
$$

它利用 user simulator 在每一轮交互之后产生的 reaction signal 来判断局部行为质量。

这一结构本身非常有参考价值：

$$
\text{Global Outcome}
+
\text{Local Turn Signal}
$$

但它的问题是 turn-level reaction 依赖 user simulator 提供额外的 private strategy 或 reaction metadata。

如果目标是：

> 不引入额外模型，不依赖 LLM judge，只使用环境本身可以程序化验证的信息，

那么需要寻找更加 environment-grounded 的 turn signal。

---

**2.3 MT-GRPO / GTPO 类方法**

另一类方法尝试构造 turn reward，再使用 reward-to-go：

$$
A_{i,t}
=
A_i^{O}
+
\sum_{l=t}^{T_i}A_{i,l}^{I}
$$

也可以写成带折扣的形式：

$$
G_{i,t}
=
\sum_{l=t}^{T_i}
\gamma^{l-t}r_{i,l}
$$

之后将 turn return 与 trajectory-level advantage 结合。

这类方法提供了一个重要经验：

> 简单地设计 dense turn reward 并不一定有效。

例如，为每一次 read-only action 提供固定正奖励：

$$
r_t^{\text{read}}=+0.3
$$

表面上似乎能够鼓励 Agent 查询必要信息，但这类奖励往往没有足够的 discriminative power。

正确查询、错误查询、重复查询都可能获得相同奖励，因此 dense reward 甚至可能干扰 terminal outcome signal。

---

## 3. Tau2 中最重要的问题：Coverage，而不是 Targeting

Tau2 任务通常具有明显的 prerequisite chain。

例如：

```text
Turn 1: 获取用户信息
Turn 2: 获取订单 / reservation
Turn 3: 查询可用航班
Turn 4: 向用户确认
Turn 5: 修改 reservation
```

真正改变数据库状态的可能只有最后一步：

```text
Turn 5: change_reservation
```

如果 verifier 只观察最终数据库状态，那么只有 Turn 5 会产生显式 progress。

于是直接定义：

$$
r_t^{\text{progress}}
=
\mathbb{1}
[
\text{当前 turn 改变了目标状态}
]
$$

可能得到：

$$
r
=
[0,0,0,0,1]
$$

但 Turn 1 到 Turn 4 显然都是完成 Turn 5 的 prerequisite。

因此，turn-level credit assignment 的目标不应只是：

> 找到真正发生状态变化的 turn。

更重要的是：

> 让 sparse verifier signal 覆盖产生该 progress 所依赖的前缀行为。

这可以通过 reward-to-go 非常简单地实现。

定义：

$$
G_t^{\text{progress}}
=
\sum_{l=t}^{T}
\gamma^{l-t}
r_l^{\text{progress}}
$$

若：

$$
\gamma=1
$$

则：

$$
r=[0,0,0,0,1]
$$

对应：

$$
G=[1,1,1,1,1]
$$

也就是说，最终 progress 会向前传播到 prerequisite turns。

这不意味着我们断言 Turn 1 一定“正确”。

它表达的是：

> Turn 1 位于一个最终能够产生可验证 progress 的行为前缀中。

---

**4. 不建议使用 Golden Action 作为 Turn Reward**

Tau2 中的 reference actions 更适合作为一条参考轨迹，而不是唯一正确行为序列。

因此，不建议简单定义：

$$
r_t
=
\mathbb{1}
[
a_t=a_t^{\text{gold}}
]
$$

因为同一个任务可能存在多条正确路径。

例如：

```text
路径 A:
get_user -> get_booking -> search_flight -> update_booking

路径 B:
get_booking -> get_user -> search_flight -> update_booking
```

如果两条路径最终都达到相同正确数据库状态，那么它们都应该被认为是成功策略。

因此，turn-level reward 最好依赖：

> environment state progress

而不是：

> 与 benchmark 作者提供的 reference trajectory 是否完全一致。

---

## 5. Environment-Grounded Turn Reward

可以构造一个非常简单的 progress reward：

$$
r_{i,t}^{\text{prog}}
=
r_{i,t}^{\text{DB}}
+
\alpha r_{i,t}^{\text{COMM}}
$$

其中包含两类完全可以程序化计算的信号：

1. Database progress；
2. Communication progress。

整个过程不需要 LLM judge。

---

**5.1 Database Progress**

设最终目标数据库状态为：

$$
D^\star
$$

当前 turn 执行之前数据库状态为：

$$
D_{t-1}
$$

执行 tool call 后变成：

$$
D_t
$$

只考虑本轮真正发生变化的字段：

$$
\Delta_t
$$

可以定义：

$$
r_t^{\text{DB}}
=
\frac{1}{Z}
\sum_{j\in\Delta_t}
\left[
\mathbb{1}(D_t[j]=D^\star[j])
-
\mathbb{1}(D_{t-1}[j]=D^\star[j])
\right]
$$

这个 reward 有几个自然性质。

**Read-only action**

如果数据库没有发生变化：

$$
D_t=D_{t-1}
$$

则：

$$
r_t^{\text{DB}}=0
$$

**正确修改**

如果一个字段被修改为 target value：

$$
D_{t-1}[j]\neq D^\star[j]
$$

且：

$$
D_t[j]=D^\star[j]
$$

则产生正 reward。

**错误修改**

如果原本正确的字段被修改错误：

$$
D_{t-1}[j]=D^\star[j]
$$

但：

$$
D_t[j]\neq D^\star[j]
$$

则产生负 reward。

**修复行为**

如果之前错误的字段被重新修改回正确值，则再次获得正 reward。

因此：

| 行为 | Database Progress |
|---|---:|
| read-only | $$0$$ |
| 修改为目标值 | 正 |
| 把正确字段改错 | 负 |
| 修复错误字段 | 正 |

这里最重要的是：

> read-only reward 为 $$0$$ 并不意味着 read action 得不到 credit。

read action 的 credit 会通过后面的 reward-to-go 获得。

---

**6. Communication Progress**

Tau2 的任务往往还要求 Agent 向用户正确传递某些信息，例如：

```text
航班号 UA123
起飞时间 09:30
改签费用 $50
```

这些 required information 通常可以程序化检查。

因此，可以定义：

$$
r_{i,t}^{\text{COMM}}
=
\frac{
\text{本轮首次正确覆盖的 required information 数量}
}{
\text{required information 总数}
}
$$

例如总共需要告诉用户三个信息：

```text
UA123
09:30
$50
```

如果 Turn 4 第一次正确告诉用户：

```text
UA123
```

则：

$$
r_{i,4}^{\text{COMM}}
=
\frac{1}{3}
$$

如果 Turn 5 又第一次给出：

```text
09:30
$50
```

则：

$$
r_{i,5}^{\text{COMM}}
=
\frac{2}{3}
$$

这样：

$$
r_{i,t}^{\text{prog}}
=
r_{i,t}^{\text{DB}}
+
\alpha r_{i,t}^{\text{COMM}}
$$

就可以同时反映：

- 环境状态是否朝目标推进；
- 用户是否逐渐获得任务要求的信息。

---

**7. 不应该直接使用 Local Progress 作为 Advantage**

即使我们已经有：

$$
r_{i,t}^{\text{prog}}
$$

也不建议直接：

$$
A_{i,t}^{\text{turn}}
=
GN(r_{i,t}^{\text{prog}})
$$

因为这样仍然只奖励真正发生 progress 的 turn。

例如：

$$
r^{\text{prog}}
=
[0,0,0,0,1]
$$

只有最后一个 turn 会得到正 credit。

这仍然没有解决 prerequisite credit assignment。

因此，更合理的做法是构造 Progress Reward-to-Go。

---

## 8. Progress Reward-to-Go

定义：

$$
G_{i,t}^{\text{prog}}
=
\sum_{l=t}^{T_i}
\gamma^{l-t}
r_{i,l}^{\text{prog}}
$$

第一版实验可以直接使用：

$$
\gamma=1
$$

因为这里的目标不是：

> 更强调靠近 outcome 的动作。

而是：

> 保证 sparse progress signal 能够覆盖 prerequisite chain。

例如：

$$
r^{\text{prog}}
=
[0,0,0,0,1]
$$

则：

$$
G^{\text{prog}}
=
[1,1,1,1,1]
$$

如果存在两个 progress point：

$$
r^{\text{prog}}
=
[0,0,0.5,0,0.5]
$$

则：

$$
G^{\text{prog}}
=
[1,1,1,0.5,0.5]
$$

这样可以自然形成 turn-level structure。

---

## 9. Turn-Wise Group Normalization

仅使用 reward-to-go 仍然不够。

更加符合 GRPO 思路的做法是：

> 对同一个 task、同一个 turn position 下不同 rollout 的 future progress 进行相对比较。

假设一个 task 采样：

$$
K
$$

条 trajectory。

对于第 $$t$$ 个 turn，收集所有仍然存在该 turn 的 trajectory：

$$
G_{1,t}^{\text{prog}},
G_{2,t}^{\text{prog}},
\ldots
$$

然后定义：

$$
A_{i,t}^{\text{turn}}
=
\frac{
G_{i,t}^{\text{prog}}
-
\mu_{G_t}
}{
\sigma_{G_t}
+
\epsilon
}
$$

也就是：

$$
A_{i,t}^{\text{turn}}
=
GN_t
\left(
G_{i,t}^{\text{prog}}
\right)
$$

这个 quantity 的意义不是：

> 这个 turn 本身是否发生 progress？

而是：

> 从当前 turn 开始，这条 rollout 后续产生的可验证 progress 是否优于同一个 task 下其他 rollout？

例如：

| Rollout | Turn 2 行为 | Future Progress |
|---|---|---:|
| A | 查询正确 reservation | $$1.0$$ |
| B | 查询错误 reservation | $$0$$ |
| C | 查询正确 reservation | $$1.0$$ |
| D | 重复查询 | $$0.2$$ |

虽然这些动作在 Turn 2 都可能是 read-only：

$$
r_{i,2}^{\text{prog}}=0
$$

但：

$$
G_{A,2}^{\text{prog}}
>
G_{B,2}^{\text{prog}}
$$

因此：

$$
A_{A,2}^{\text{turn}}>0
$$

$$
A_{B,2}^{\text{turn}}<0
$$

这样我们就在不使用任何外部 judge 的情况下，对 read action 产生了 turn-level credit。

---

**10. Format Reward**

Format reward 应该非常克制。

不建议对合法 tool call 提供正奖励，例如：

$$
r_t^{\text{format}}=+0.1
$$

因为这可能鼓励 Agent 产生更多没有必要的 tool calls。

例如：

```text
query_user
query_user
query_user
query_user
```

如果每次合法调用都得到正奖励，模型可能学会 reward hacking。

更加安全的方式是：

> Format reward 只作为 constraint penalty。

定义：

$$
r_{i,t}^{\text{format}}
=
\begin{cases}
-1,
&
\text{invalid JSON / parser failure}
\\
-1,
&
\text{unknown tool}
\\
-1,
&
\text{missing or type-invalid arguments}
\\
0,
&
\text{otherwise}
\end{cases}
$$

例如：

```text
get_booking(id="ABCDE")
```

如果调用格式完全合法，但 booking 不存在，则：

$$
r_{i,t}^{\text{format}}=0
$$

因为这是：

> valid action + legitimate environment failure

而不是 format error。

因此 format reward 只负责：

> Agent 是否按照环境协议合法交互。

它不负责判断：

> 这个动作是否聪明。

---

## 11. 最终 Advantage 设计

最终可以采用非常简单的组合：

$$
A_{i,t}^{\text{total}}
=
A_i^{\text{traj}}
+
\lambda A_{i,t}^{\text{turn}}
+
\beta r_{i,t}^{\text{format}}
$$

其中 trajectory-level advantage：

$$
A_i^{\text{traj}}
=
GN(R_i^{\text{terminal}})
$$

turn-level advantage：

$$
A_{i,t}^{\text{turn}}
=
GN_t
\left(
\sum_{l=t}^{T_i}
\gamma^{l-t}
r_{i,l}^{\text{prog}}
\right)
$$

progress reward：

$$
r_{i,t}^{\text{prog}}
=
r_{i,t}^{\text{DB}}
+
\alpha r_{i,t}^{\text{COMM}}
$$

因此最终：

$$
A_{i,t}^{\text{total}}
=
GN(R_i^{\text{terminal}})
+
\lambda
GN_t
\left(
\sum_{l=t}^{T_i}
\gamma^{l-t}
\left(
r_{i,l}^{\text{DB}}
+
\alpha r_{i,l}^{\text{COMM}}
\right)
\right)
+
\beta r_{i,t}^{\text{format}}
$$

整个方法只需要 Tau2 environment 已经存在的信息：

$$
R^{\text{terminal}}
$$

$$
D_t
$$

$$
D^\star
$$

$$
\text{communicate\_info}
$$

以及：

$$
\text{tool parser / schema validation}
$$

不需要：

$$
\text{LLM judge}
$$

不需要：

$$
V(s)
$$

不需要：

$$
Q(s,a)
$$

不需要额外 reference model。

也不需要 additional rollout。

---

**12. 推荐的第一版超参数**

第一版不建议进行过多超参数搜索。

可以直接使用：

$$
\gamma=1.0
$$

$$
\lambda=0.3
$$

$$
\beta=0.1
$$

$$
\alpha=1.0
$$

因此：

$$
A_{i,t}
=
A_i^{\text{traj}}
+
0.3A_{i,t}^{\text{turn}}
+
0.1r_{i,t}^{\text{format}}
$$

其中 trajectory-level outcome 仍然是最主要的优化信号。

turn-level advantage 只作为局部 credit correction。

format reward 只作为较弱的 protocol constraint。

---

**13. 一个重要的性质：失败轨迹中的 Useful Prefix**

考虑下面这条最终失败的 trajectory：

```text
Turn 1: 正确获取用户
Turn 2: 正确获取 reservation
Turn 3: 正确搜索航班
Turn 4: 用户完成确认
Turn 5: change_flight 参数写错
```

最终：

$$
R_i=0
$$

如果同组存在成功 rollout，则通常：

$$
A_i^{\text{traj}}<0
$$

标准 Interactive GRPO 会得到：

$$
A_{i,1}
=
A_{i,2}
=
A_{i,3}
=
A_{i,4}
=
A_{i,5}
=
A_i^{\text{traj}}<0
$$

也就是说，前四个 useful actions 会一起受到惩罚。

而在本文设计中，前四个 turn 可能具有较高的：

$$
A_{i,t}^{\text{turn}}>0
$$

因此：

$$
A_{i,t}^{\text{total}}
=
A_i^{\text{traj}}
+
\lambda A_{i,t}^{\text{turn}}
$$

会减弱对这些 useful prefix actions 的惩罚。

需要强调的是：

> turn-level advantage 并不一定必须将这些动作立即翻转成正 advantage。

更合理的目标是：

> 整条轨迹失败，所以总体上仍然保留负 outcome signal；但能够通向 partial verified progress 的前缀动作应该受到更小惩罚。

这种设计比简单设置：

$$
r^{\text{read}}=+0.3
$$

更加合理。

---

**14. 全部 Rollout 失败时仍然能够学习**

这是这个设计非常重要的优点。

假设同一个任务的所有 rollout 都失败：

$$
R_1=R_2=\cdots=R_K=0
$$

那么 GRPO：

$$
\sigma_R=0
$$

最终：

$$
A_i^{\text{traj}}\approx0
$$

也就是说：

> vanilla GRPO 几乎无法从这一组 rollout 中学习。

但是，即使所有 rollout 都失败，它们的 partial progress 仍然可能不同。

例如：

$$
G_{1,t}^{\text{prog}}
\neq
G_{2,t}^{\text{prog}}
$$

那么：

$$
A_{1,t}^{\text{turn}}
\neq
A_{2,t}^{\text{turn}}
$$

因此模型仍然能够学习：

> 哪些失败轨迹实际上已经更加接近成功。

对于 Tau2 这类长程任务，这可能是 turn-level credit assignment 最重要的实际价值之一。

---

## 15. 推荐的核心 Ablation

第一阶段不需要设计很多复杂变体。

只需要比较下面四组。

**15.1 Interactive GRPO**

$$
A_t
=
A^{\text{traj}}
$$

作为标准 baseline。

---

**15.2 Local Progress**

$$
A_t
=
A^{\text{traj}}
+
\lambda
GN_t
\left(
r_t^{\text{prog}}
\right)
$$

用于验证：

> 只奖励 progress turn 是否有效。

---

**15.3 RTG Progress**

核心方法：

$$
A_t
=
A^{\text{traj}}
+
\lambda
GN_t
\left(
\sum_{l=t}^{T}
r_l^{\text{prog}}
\right)
$$

用于验证：

> 将 progress signal 向 prerequisite prefix 传播是否真正重要。

如果：

$$
\text{RTG Progress}
>
\text{Local Progress}
$$

就说明：

> Tau2 中的关键问题不是精确定位发生 progress 的单个 action，而是让 sparse verifier signal 覆盖 prerequisite chain。

---

**15.4 RTG Progress + Format**

$$
A_t
=
A^{\text{traj}}
+
\lambda A_t^{\text{turn}}
+
\beta r_t^{\text{format}}
$$

验证 protocol constraint 是否能够进一步稳定训练。

---

**16. Gamma 的重要实验**

建议专门比较：

$$
\gamma=1.0
$$

$$
\gamma=0.98
$$

$$
\gamma=0.95
$$

甚至：

$$
\gamma=0.9
$$

如果：

$$
\gamma=1.0
$$

在 Tau2 上更加稳定，则可能意味着：

> long-horizon Agent credit assignment 的关键并不是 aggressively targeting actions close to outcome，而是保持 prerequisite credit coverage。

换句话说：

$$
\text{Temporal Targeting}
$$

可能不如：

$$
\text{Credit Coverage}
$$

重要。

这可以与 Gamma-GRPO 中对 $$\gamma$$ 敏感性的现象形成联系。

---

## 17. 最终方法总结

本文推荐的 turn-level credit assignment 可以概括为：

$$
\boxed{
A_{i,t}
=
A_i^{\text{traj}}
+
\lambda A_{i,t}^{\text{turn}}
+
\beta r_{i,t}^{\text{format}}
}
$$

其中：

$$
\boxed{
A_i^{\text{traj}}
=
GN(R_i^{\text{terminal}})
}
$$

$$
\boxed{
A_{i,t}^{\text{turn}}
=
GN_t
\left(
\sum_{l=t}^{T_i}
\gamma^{l-t}
r_{i,l}^{\text{prog}}
\right)
}
$$

$$
\boxed{
r_{i,t}^{\text{prog}}
=
r_{i,t}^{\text{DB}}
+
\alpha r_{i,t}^{\text{COMM}}
}
$$

Format reward：

$$
\boxed{
r_{i,t}^{\text{format}}
=
\begin{cases}
-1,&\text{protocol / parser error}\\
0,&\text{otherwise}
\end{cases}
}
$$

整个设计可以概括为三个层次：

```text
Trajectory Outcome
        +
Verifier-Grounded Progress-to-Go
        +
Local Protocol Constraint
```

即：

$$
\boxed{
\text{Global Outcome}
+
\text{Turn-Level Progress Coverage}
+
\text{Format Constraint}
}
$$

它的核心思想不是寻找一个外部模型去判断：

> “这个动作是不是正确？”

而是利用环境已经提供的可验证信号，回答：

> “从这个 turn 开始，这条行为前缀最终带来了多少真实、可验证的任务进展？”

这样既能够保留 GRPO 的简单性，也能够在 Tau2 的 long-horizon interactive setting 中形成真正的 turn-level advantage。
