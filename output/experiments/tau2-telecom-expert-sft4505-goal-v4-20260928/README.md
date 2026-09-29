# tau2-telecom-expert-sft4505-goal-v4-20260928

Purpose：使用新版 credit assignment 训练 Telecom 单领域 expert。采用 shuffled task deck 保证 domain coverage，保留当前 credit-aware group filter；固定 Telecom domain-experts 预算 20 updates。

数据为 [telecom_train.jsonl](data/telecom_train.jsonl)，从已验证的 v4 混合数据按 `metadata.domain=telecom` 筛选，共 271 个 task。初始化、reward、batch、seed、pool/pending、User 和长度限制沿用 Retail goal-v4 expert 配置。

## 运行配置

| 项目 | 配置 |
|---|---|
| 初始化 | SFT4505 `Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920/iter_0004505_hf` |
| 训练长度 | 20 updates，每10步保存，最终iter19 |
| Batch | 16 task groups × K8 = 128 trajectories |
| 采样 | Telecom-only；`ShuffledTaskDataSource`；train/rollout seed1234/42；pool32、pending64、max_policy_lag=-1 |
| Credit | `progress-db-count-v1`，progress/format 1/1，gamma0.98；当前 credit-aware dynamic group filter；不启用旧 outcome filter/retry |
| 资源 | 8 GPU，normal priority；复用 Retail expert 的常驻 User 服务 |

## 作业

| 任务 | 状态 | 运行日志 |
|---|---|---|
| Telecom expert（显式环境变量配置） | 24882（已提交） | [运行日志](jobs/fsh-tau2-telecom-goal-v4-shuffled20-expert-0928-v2-0928-181121/run_0_20260928_181122170.log) |
| 三域 iter9 official eval | 24938 (`pt-s4j8wdoc`) | 8 卡，`airline,retail,telecom` 全量 test×4 trials，seed300，400 simulations，SUCCEEDED；overall pass@1/pass@4(any)/pass^4=`50.50/77.00/23.00%`；[评测脚本](eval_three_domain_iter9.sh)，[运行日志](jobs/24938-three-domain-telecom-expert-iter9-eval-0928-214005898/run_*_20260928_214005898.log) |
| 三域 iter19 official eval | 24939 (`pt-bj13j0gl`) | 8 卡，`airline,retail,telecom` 全量 test×4 trials，seed300，400 simulations，SUCCEEDED；overall pass@1/pass@4(any)/pass^4=`47.50/74.00/21.00%`；[评测脚本](eval_three_domain_iter19.sh)，[运行日志](jobs/24939-three-domain-telecom-expert-iter19-eval-0928-214007169/run_*_20260928_214007169.log) |

Job24878 was stopped immediately after submission because custom environment variables are not inherited by job-manager unless passed with `-e`; it did not run the Telecom configuration. Job24882 carries the explicit Telecom/data/budget/User endpoint environment.
