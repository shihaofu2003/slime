# tau2 SFT multi-tool quality audit

实验名：`tau2-sft-multitool-quality-audit`。目的：在不改动 tau2 领域 policy、也不直接
发布训练 JSONL 的前提下，用 dependency-safe-multi 反事实协议重新审计当前 Agent SFT
数据；分别回答 multi/non-multi 的标签失败比例、成功示范质量和失败决定性原因，再据此
起草 turn-level 数据筛选标准。现有 `tau2-sft-quality-audit` 保留为 single-policy baseline。

结论：multi 数量本身不能作为质量或失败代理。matched success 总体质量差为 `+0.005`
（95% CI `[-0.034, +0.045]`）；telecom 的 multi/non-multi consensus-failure 差为
`+0.94pp`（95% CI `[-7.95pp, +9.82pp]`），未检出差异且不具 task-level common
support。341 条 multi-exposed 失败中只有 66 条被 Luna 归为 multi caused/contributed。
最终自动 keep 只有 69 dialogs / 630 targets，领域严重偏斜，不能直接作为训练集；采用
标准见 `FILTER_SPEC.md`。

下游 local-first 扩展、raw-init SFT 和同协议 official eval 已完成：新 SFT 的 pass@1 /
pass@4(any) 为 `27.00% / 54.00%`，相对 raw 和旧 SFT 均显著提升；pass^4 为 `4.00%`，
相对旧 SFT 显著下降 6pp。因此 `dependency-safe-multi` 与 broad tier 获得保留，但当前
训练配方不升级为唯一默认。完整结果见
[FINAL_RECOMMENDATION.md](../tau2-sft-local-first-relaxed/FINAL_RECOMMENDATION.md)。

## Tasks

- `fsh-tau2-sft-multitool-prepare-0802-154326`（job `pt-mhnmh1uh`）：只生成
  full/training/converted 三视图、匹配集合、calibration 集合、长上下文路由和 hash
  manifest，不启动 judge 或 API；集群 Transformers 5.8 的 BatchEncoding 曾被误计为
  2 tokens，产物无效且不进入复审；[run log](jobs/fsh-tau2-sft-multitool-prepare-0802-154326/run_20260802_154326.log)。
- `fsh-tau2-sft-multitool-prepare-r2-0802-155401`（job `pt-3tr00tgj`）：修复原生
  `dict` 输出后重跑，但 Transformers 5.8 的 BatchEncoding 是通用 Mapping，prepare
  fail-fast，产物未使用；[run log](jobs/fsh-tau2-sft-multitool-prepare-r2-0802-155401/run_20260802_155401.log)。
- `fsh-tau2-sft-multitool-prepare-r2-0802-155441`（job `pt-p58nw34x`）：发现与上一任务
  重复后在 STARTING 阶段取消，未创建 run log、也未用于分析；
  [submit log](jobs/fsh-tau2-sft-multitool-prepare-r2-0802-155441/submit_20260802_155441.log)。
- `fsh-tau2-sft-multitool-prepare-r3-0802-155748`（job `pt-pekfrvjh`）：兼容通用
  Mapping/属性式 BatchEncoding 后的正式 prepare；成功生成 663/26 条 quality
  local/long、424/26 条 failure local/long，sealed 验收通过；[run log](jobs/fsh-tau2-sft-multitool-prepare-r3-0802-155748/run_20260802_155748.log)。
- `fsh-tau2-sft-multitool-prepare-r3-0802-155746`（job `pt-y39ej6j3`）：与上一任务
  同 SHA 重复，发现共享输出竞态后取消；[run log](jobs/fsh-tau2-sft-multitool-prepare-r3-0802-155746/run_20260802_155746.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-0802-160338`（job `pt-znjw5hb7`）：先用
  Qwen3.6-27B local 与独立 fallback 双审 72 条 calibration（64 full/full、8
  window/full）；local 72/72 成功，fallback 在 8 路并发下 24 成功、40 失败，任务在
  gate 前停止；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-0802-160338/run_20260802_160338.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-r2-0802-162216`（job `pt-7xpwuobu`）：复用有效
  calibration 缓存并降低 fallback 并发；定位到失败是 Gemini 把 `confidence` 的 0–1
  概率误写为 0–4 维度分后取消，尚未发起 judge 请求；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-r2-0802-162216/run_20260802_162216.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-gpt41-0802-162956`（job `pt-1z3rce4a`）：改用已在
  普通、17 个 multi-turn 及 70k-token 样本上通过严格 schema 探针的独立 GPT-4.1
  fallback；完成 64 条短上下文和 2 条长上下文 review 后，按用户指定模型变更取消，
  结果不进入最终审计；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-gpt41-0802-162956/run_20260802_162956.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-0802-164410`（job `pt-ctj5trwl`）：使用用户
  指定的 `gpt-5.6-luna` 作为独立 fallback；首轮 64 条短上下文中 59 条有效，5 条因
  引用了不存在的 evidence turn 而 fail-fast；未进入 gate；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-0802-164410/run_20260802_164410.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-r2-0802-170301`（job `pt-dowp4dex`）：使用
  prompt v2 显式约束 literal/required turn indices；Qwen 72/72 有效，Luna 短上下文
  60/64 有效，余下 4 条仍为非法 evidence citation，未进入 gate；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-r2-0802-170301/run_20260802_170301.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-r3-0802-172649`（job `pt-kvgf7g8u`）：复用
  prompt v3，并用逐 payload dynamic schema 将引用绑定到 literal turn enum；local Qwen
  72/72 有效，Luna 首个 full-source 分区 63/64 有效，1 条在三次 repair 后仍有非法三态
  枚举和 evidence 数组超限。结构失败使任务退出；未运行 calibration gate，也未启动 full
  quality/failure review；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-r3-0802-172649/run_20260802_172649.log)。
- `fsh-fsh-tau2-sft-multitool-qwen36-27b-luna-r4-0802-174900-0802-174911`（job
  `pt-6rts9zec`）：复用 72 条 Qwen 与 63 条 Luna v3 有效缓存；第三次 repair 对 evidence
  数组上限和 multi-turn 三态枚举给出精确约束；缺失样本仍重新生成了整份 review 并重复
  同类格式错误，任务在 gate 前停止；[run log](jobs/fsh-fsh-tau2-sft-multitool-qwen36-27b-luna-r4-0802-174900-0802-174911/run_20260802_174911.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-r5-0802-180006`（job `pt-6y03vorx`）：在 repair
  请求中附上 Luna 自己的上一份非法 JSON，要求只做最小字段修复并保留实质判断；复用其余
  135 条有效 calibration 缓存；sealed prepare 后评审代码发生实质版本变更，为避免 manifest
  SHA 与运行代码不一致，在 Luna 调用前取消；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-r5-0802-180006/run_20260802_180006.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-r6-0802-181236`（job `pt-rgwltmop`）：固定
  normalization 版本后重新 sealed prepare；三次 judge 自修复仍失败时，仅 allowlist
  citation/assessment 结构修复、confidence 强制降至门槛以下并写入诊断 sidecar；45 项测试
  通过；Qwen 72/72、Luna 普通 64/64、Luna long 6/8 有效，另 2 条在 2048 输出预算下
  连续返回空 content，任务在 gate 前停止；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-r6-0802-181236/run_20260802_181236.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-r7-long4096-0802-183737`（job `pt-bt54kclk`）：
  普通 Luna 输出预算仍为 2048，只把 full long-context 提高到 4096；复用 Qwen 72 条和
  Luna 普通 64 条有效缓存，8 条 long 全部复审成功。72 条均 reliable，batch 61/72、
  multi-turn exact 38/51、机械 hard false-keep 0/28；旧 gate 仅因三分类 verdict 36/72
  失败。事后诊断显示 Luna 的 14 个 keep 全被 Qwen 支持且无 keep/drop 极端冲突，故该
  集合冻结为 diagnostic pilot，不用于验证据此修订的 gate；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-r7-long4096-0802-183737/run_20260802_183737.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-r8-validation-0802-190104`（job
  `pt-dey1etp2`）：用 seed `20260803` 封存与 pilot 不重叠的新 72-dialog quality
  validation cohort，并以 Qwen3.6-27B + `gpt-5.6-luna` 复审；72/72 均生成有效 review，
  70/72 双方 reliable。十项 gate 通过，唯一失败项为对称 keep/drop 4/70（5.71%）；四例
  全是 local keep / Luna drop，最终 keep 交集会安全否决，且与 multi batching 无关。
  该集合冻结为 validation-v1 diagnostic，未启动 failure/full review；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-r8-validation-0802-190104/run_20260802_190104.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-r9-v2-validation-0802-193116`（job
  `pt-yr0azi60`）：用 seed `20260804` 封存与 pilot、validation-v1 都不重叠的第三个
  72-dialog cohort，独立验证与最终 keep 交集一致的非对称 gate；仍使用 Qwen3.6-27B +
  `gpt-5.6-luna`。quality gate 10/10 通过：Luna keep 21/21 均获 local keep 支持，
  reliable 72/72，batch 68/72，multi-turn exact 44/48，机械 hard false-keep 0/33。
  随后的 failure pilot 显示 exact unique-cause consensus 不可靠（mode 35/72、role 12/58），
  因此 hard gate 失败并阻止 full review；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-r9-v2-validation-0802-193116/run_20260802_193116.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-r10-failure-v2-validation-0802-203038`（job
  `pt-zwz2nw0n`）：封存与旧 failure diagnostic pilot 完全不重叠的 72-dialog validation，
  以用户指定的 `gpt-5.6-luna` 作为主归因器，并用 causal mode / decisive turn 的机械一致性
  gate 验证；Qwen 仅作 sensitivity audit。failure gate 全部通过：Luna reliable 70/72、
  mode evidence-consistent 68/70、primary/mode unknown 均 11/70；Qwen/Luna exact cause
  41/72、mode 39/72 只作诊断。任务 `SUCCEEDED`：quality 两侧各 689/689、failure 两侧
  各 450/450，0 请求失败；发布 `COMPARISON_REPORT.md`、`FAILURE_ATTRIBUTION.md`、
  `summary.json` 和 19,318-row provenance-only 决策草案；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-r10-failure-v2-validation-0802-203038/run_20260802_203038.log)。
- `fsh-tau2-sft-multitool-qwen36-27b-luna-r2-0802-170345`（job `pt-qj0v9pb3`）：与上一
  任务 runner SHA 相同，发现共享输出竞态后取消；[run log](jobs/fsh-tau2-sft-multitool-qwen36-27b-luna-r2-0802-170345/run_20260802_170345.log)。

## Scope

- 全量风险按 reward、correct、consensus 三种失败定义分开报告；airline/retail 缺少
  non-multi common support 时不解释为因果效果。
- success comparison 包含全部 92 个 full-source single consensus-success anchors，以及
  1,954 个 multi success 中 147 个全局去重的 matched controls；它只估计 matched 示范
  质量差，不估计 multi-success 的总体安全率或可保留比例。
- failure attribution 覆盖 consensus failure 和两类 label-discordant 轨迹，并区分
  multi exposure 与 decisive multi-turn cause；旧 72-dialog exact-consensus pilot 与新
  72-dialog Luna evidence validation 完全不重叠。稀有的 retail failure、correct-only 和
  airline consensus-failure 已被 pilot 穷尽，不能伪造 held-out 覆盖，最终只作诊断性分层。
- 超过本地上下文的完整轨迹不截断：local 使用带省略清单的 structured window，
  long-context fallback 使用 full source；两类 agreement 分开记录。

## Artifacts

- `AUDIT_DESIGN.md` / `COMPARISON_REPORT.md` / `FAILURE_ATTRIBUTION.md`：问题定义、
  标签风险、matched quality 与失败原因。
- `CALIBRATION_PILOT_REPORT.*`：旧 seed 的 72-dialog 诊断结果；它只用于发现两个 judge
  的 severity-scale 偏移，不用于新 gate 验收。
- `CALIBRATION_VALIDATION_V1_REPORT.*`：seed `20260803` 的方向性诊断；它冻结了四个
  local keep / Luna drop 的保守 veto，不用于 v2 gate 验收。
- `CALIBRATION_REPORT.*` / `FAILURE_CALIBRATION_REPORT.*`：两个独立的 72-dialog hard
  gate；前者约束 quality keep 的交集支持和机械错误 false-keep，后者以 Luna 为主归因器，
  约束 reliability、unknown 覆盖及 causal mode 与 sealed turn type 的证据一致性；Qwen
  exact agreement 只作 sensitivity/contradiction 诊断。
- `FAILURE_CALIBRATION_PILOT_REPORT.*`：旧 exact-consensus failure gate 的诊断结果；
  它记录 Luna evidence consistency 与双 judge unique-cause 分歧，不用于新 gate 验收。
- `full_source_features.jsonl` / `training_visible_features.jsonl`：完整轨迹行为与训练可见
  raw prefix；`features.jsonl` 另带实际 converted row/system hash 和逐行转换检查。
- `comparison_selection.jsonl` / `comparison_pairs.jsonl` / `calibration_selection.jsonl` /
  `failure_calibration_pilot_selection.jsonl` / `failure_calibration_selection.jsonl`：预声明
  样本、匹配关系和互斥的 sealed calibration 集合。
- `payload_manifest.jsonl` / `prepare_manifest.json`：prompt、数据、tokenizer、full/window
  payload 和模型配置的可复现 hash；summarize 拒绝 stale 或缺失 review。
- `FILTER_SPEC_DRAFT.md`：judge 前封存的候选标准；`FILTER_SPEC.md`：结合全量结果定稿的
  三层筛选规范、当前产物验收和进入训练前的约束；本实验不生成最终训练数据。
- `filter_decisions_draft.jsonl`：每个 converted target 的 provenance-only 草案决策；只含
  ID、索引、hash、标签、judge/manifest 来源和决定，不含 prompt、conversation、target
  文本或训练 messages，因此不是训练 JSONL。
- [tau2-sft-local-first-relaxed](../tau2-sft-local-first-relaxed/README.md)：按本规范扩展的
  全量本地筛选、19,318-row 训练数据、raw-init SFT、三模型 official eval 与最终决策。

## Decision rule

quality 与 failure-attribution 两个 sealed hard gate 都必须通过，`summarize` 才能发布完整
报告和草案决策。failure 主统计采用用户指定的 `gpt-5.6-luna`；低置信度、unknown 或 causal
mode 与引用 turn type 不一致的样本留在固定分母中并计为 unknown，Qwen 只报告敏感性差异，
模型归因不称为 ground truth。三次 judge repair 后的 allowlisted conservative structural
repair 会把 confidence 压到门槛以下并标记 error，因此不能进入主归因或 keep。
`keep-candidate` 仍要求 consensus-success、双 judge quality keep、strict batch-safe 且无机械
hard issue；失败原因不作为 success keep 的循环 hard rule。下游 matched-budget SFT 与
held-out official eval 已完成，结果支持放宽为 local-first broad tier，但 pass^4 回退要求
另设 consistency anchor。当前 69-dialog / 630-target strict 子集仍只作为高精度 seed，不能
直接作为完整训练集。
