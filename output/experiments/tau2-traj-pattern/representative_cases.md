# 配对代表案例

案例均来自相同 task/trial 的 Raw→SFT→RL 配对轨迹。Reward 变化是官方评测事实；模式与证据轮次来自 Gemini 判断，并经过确定性规则裁决。

## 好模式案例

- **意图持续跟踪**（`intent_tracking`）— `raw | airline | 13 | trial 0`，证据轮次 `[1, 18]`：正确保持用户目标、话题切换和多个子任务。
- **行动前澄清**（`clarify_before_act`）— `raw | airline | 13 | trial 0`，证据轮次 `[2]`：行动前收齐身份、实体、选项、原因和支付信息。
- **先读取、再规划、后行动**（`read_plan_act`）— `raw | airline | 13 | trial 0`，证据轮次 `[4, 8]`：先读取真实状态，再依据状态与 policy 选择流程。
- **原子化提交**（`atomic_commit`）— `raw | airline | 16 | trial 0`，证据轮次 `[20]`：列明变更、费用和影响，获得最新确认后只写入一次。
- **渐进式诊断**（`progressive_diagnosis`）— `rl2e6 | telecom | [mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard] | trial 1`，证据轮次 `[6, 92]`：Telecom 按 service→data→MMS 依赖顺序诊断并复验。
- **基于工具结果恢复**（`grounded_recovery`）— `raw | airline | 13 | trial 0`，证据轮次 `[6, 16]`：理解工具错误后改变方案，不重复碰撞。
- **基于事实收尾**（`grounded_closeout`）— `raw | airline | 13 | trial 0`，证据轮次 `[18]`：仅在工具结果支持时宣称成功，并完成全部子任务。
- **领域 policy 合规**（`domain_policy_compliance`）— `raw | airline | 13 | trial 0`，证据轮次 `[]`：遵守领域 policy 和所有必需流程。

## 主要失败模式

- **确认违规**（`confirmation_violation`）— `raw | retail | 86 | trial 0`，证据轮次 `[32]`。
- **失败恢复不当**（`failed_recovery`）— `sft | retail | 111 | trial 2`，证据轮次 `[5, 6]`。
- **虚假成功声明**（`false_success_claim`）— `raw | airline | 22 | trial 2`，证据轮次 `[28, 32]`。
- **任务未完成**（`incomplete_task`）— `raw | airline | 13 | trial 1`，证据轮次 `[22]`。
- **缺少前置条件**（`missing_precondition`）— `raw | airline | 26 | trial 0`，证据轮次 `[10]`。
- **其他**（`other`）— `raw | airline | 6 | trial 1`，证据轮次 `[30, 200]`。
- **Policy 违规**（`policy_violation`）— `sft | airline | 13 | trial 0`，证据轮次 `[12]`。
- **工具命名空间混淆**（`tool_namespace_confusion`）— `raw | telecom | [mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard] | trial 0`，证据轮次 `[8, 14]`。
- **参数错误**（`wrong_arguments`）— `sft | airline | 13 | trial 2`，证据轮次 `[20]`。
- **流程错误**（`wrong_workflow`）— `raw | airline | 19 | trial 1`，证据轮次 `[6, 26]`。

## 官方 reward 改善案例

- `airline | 16 | trial 3`：reward `0→0→1`，行为有效 `0→0→0`；主要模式 未识别到主要 Agent 错误（`none`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。
- `airline | 22 | trial 3`：reward `0→0→1`，行为有效 `0→0→0`；主要模式 未识别到主要 Agent 错误（`none`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。
- `airline | 24 | trial 2`：reward `0→0→1`，行为有效 `0→0→0`；主要模式 任务未完成（`incomplete_task`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。

## 官方 reward 退化案例

- `airline | 16 | trial 0`：reward `1→0→0`，行为有效 `1→0→0`；主要模式 未识别到主要 Agent 错误（`none`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。
- `airline | 19 | trial 0`：reward `1→0→0`，行为有效 `1→0→0`；主要模式 未识别到主要 Agent 错误（`none`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。
- `airline | 2 | trial 0`：reward `1→0→0`，行为有效 `1→0→0`；主要模式 未识别到主要 Agent 错误（`none`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。

## 官方 reward 持续成功案例

- `airline | 13 | trial 0`：reward `1→1→1`，行为有效 `1→0→0`；主要模式 未识别到主要 Agent 错误（`none`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。
- `airline | 13 | trial 1`：reward `1→1→1`，行为有效 `0→0→0`；主要模式 任务未完成（`incomplete_task`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。
- `airline | 13 | trial 2`：reward `1→1→1`，行为有效 `1→0→0`；主要模式 未识别到主要 Agent 错误（`none`） → 参数错误（`wrong_arguments`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。

## 官方 reward 持续失败案例

- `airline | 16 | trial 1`：reward `0→0→0`，行为有效 `0→0→0`；主要模式 未识别到主要 Agent 错误（`none`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。
- `airline | 16 | trial 2`：reward `0→0→0`，行为有效 `0→0→0`；主要模式 未识别到主要 Agent 错误（`none`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。
- `airline | 18 | trial 0`：reward `0→0→0`，行为有效 `0→0→0`；主要模式 任务未完成（`incomplete_task`） → Policy 违规（`policy_violation`） → Policy 违规（`policy_violation`）。证据模式：单轮多工具违规。
