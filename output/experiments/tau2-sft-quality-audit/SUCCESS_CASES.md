# 高质量成功轨迹画像

以下案例通过 source 正标签、确定性规则和模型盲审三道门槛。证据轮次对应
`judge_payloads.jsonl` 的 conversation index。
部分训练 prefix 恰好结束于正确的 target tool call；此时 `keep/uncertain` 表示
可见动作值得模仿，不表示缺失 observation 之后的任务成功已被直接观察。

状态说明（2026-08-03）：这些案例来自历史 single-policy 基线，案例中的流程、参数依据、
授权和恢复模式仍有效；“每轮最多一个调用”已被后续协议中立证据取代，不能继续作为
高质量轨迹的共性或自动筛除条件。

## 代表案例

| domain | dialog | score | 成功原因 |
|---|---|---:|---|
| airline | `airline_dialog_693` | 4.00 | firm_policy_enforcement: The agent correctly identified that waiving fees was against policy and refused the user's repeated attempts to get an exception, maintaining integrity while remaining helpful @ [8, 10, 12]; explicit_confirmation: The agent clearly summarized the booking details and obtained explicit user consent before executing the state-changing tool call @ [21, 22] |
| airline | `airline_dialog_721` | 4.00 | correct policy refusal: The agent accurately communicated that checked bags cannot be removed. @ [3, 5] |
| airline | `airline_dialog_763` | 4.00 | Empathetic communication: The agent remained polite and empathetic while firmly upholding the policy. @ [5, 7, 9]; Adherence to negative constraints: The agent correctly recognized that the baggage removal request could not be fulfilled. @ [3, 5, 7] |
| airline | `airline_dialog_767` | 4.00 | task_completion: The agent successfully booked the requested flight (earliest available direct flight since none existed before 5 AM) with the correct cabin, passenger, insurance, and baggage. The  @ [11, 12, 13]; tool_choice: The agent used `search_direct_flight` to find options, `get_user_details` to retrieve profile info, and `book_reservation` to finalize. All tools were appropriate for the task. @ [3, 7, 11]; workflow_and_dependencies: The agent followed the correct workflow: search flights -> present options -> get user ID -> get user details -> confirm booking -> book. It correctly handled the dependency of @ [3, 5, 7] |
| retail | `retail_dialog_201` | 4.00 | Correct item identification from ambiguous descriptions: The user described items relatively ('cheaper navy jacket', 'red t-shirt that's not the small one'). The agent correctly parsed the order details to identify the specific item IDs @ [10, 11]; Strict adherence to authentication policy: The agent initiated authentication before accessing any user data, as required by policy. @ [3, 5]; Explicit confirmation before state change: The agent clearly outlined the consequences (refund amount, method, timeline) and obtained explicit 'Yes' confirmation before calling the return tool. @ [11, 12] |
| retail | `retail_dialog_69` | 4.00 | task_completion: The agent correctly identified that the order was delivered and thus could not be cancelled. It offered the correct alternative (return) and handled the user's decision to keep the @ [11, 12, 13]; tool_choice: The agent used the correct tools for authentication (`find_user_id_by_email`), user lookup (`get_user_details`), and order status verification (`get_order_details`). No invalid or  @ [5, 7, 9]; workflow_and_dependencies: The agent followed the correct dependency chain: authenticate user -> get user details to find order ID -> get order details to check status. This is the standard and correct @ [5, 7, 9] |
| retail | `retail_dialog_127` | 4.00 | task_completion: The agent successfully completed the user's request to change the shipping address. The tool call in turn 15 was executed correctly, and the tool observation in turn 16 confirms a  @ [15, 16, 17]; tool_choice: The agent selected the correct tools for each step: `find_user_id_by_email` for authentication, `get_user_details` and `get_order_details` for information retrieval, and `modify_` @ [5, 7, 9]; workflow_and_dependencies: The agent followed the correct dependency chain: authenticate -> get user details -> get order details -> confirm changes -> execute modification. Each step relied on the output of @ [5, 7, 9] |
| retail | `retail_dialog_106` | 4.00 | task_completion: The agent correctly initiates the required workflow: authenticating the user and retrieving their profile to access order information. The trajectory ends before the specific order @ [5, 7]; tool_choice: The agent correctly selects `find_user_id_by_email` to authenticate the user based on the provided email, and then `get_user_details` to retrieve the user's profile and order list. @ [5, 7]; workflow_and_dependencies: The agent correctly sequences the authentication step before the data retrieval step. It waits for the result of the first tool call before proceeding to the next dependent call. @ [5, 6, 7] |
| telecom | `10` | 4.00 | error_recovery: The agent correctly identified the invalid argument error from the tool, explained the valid options to the user, and successfully retried the action with the correct argument. @ [27, 29]; dependency_management: The agent correctly identified that resetting APN settings requires a reboot to take effect and guided the user through this necessary step. @ [42, 46]; user_empathy: The agent effectively alleviated user anxiety about data loss when performing actions like toggling airplane mode and rebooting the device. @ [13, 45] |
| telecom | `105` | 4.00 | Correctly enforces identity verification before tool use: The agent did not attempt to look up the customer until after explicitly asking for and receiving the necessary PII. @ [3, 5]; Adheres to single-tool-call policy: Only one tool call was made in the final turn. @ [5] |
| telecom | `109` | 4.00 | Excellent rapport and de-escalation of user anxiety: The agent used the radio station analogy to comfort the user about their data and photos, successfully reducing hesitation. @ [21]; Robust recovery from tool error: The agent read the error message provided by the tool, understood the valid arguments, and corrected the call in the very next turn. @ [27, 28] |
| telecom | `116` | 4.00 | task_completion: The agent correctly identified the need to verify the customer before proceeding with technical support, as mandated by the policy. The trajectory ends at the correct next step (re @ [5]; tool_choice: The agent chose `get_customer_by_phone` which is the appropriate tool for looking up a customer using the phone number provided by the user in turn 4. @ [5]; workflow_and_dependencies: The agent correctly established the dependency: Customer Identification -> Technical Support. It requested the necessary information in turn 3 and executed the lookup in turn 5. @ [3, 5] |

## 规则未覆盖的语义反例

这些 dialog 有 source 正标签且通过机械规则，但模型认为可见步骤仍不值得直接模仿。

| domain | dialog | tier/score | 语义问题 |
|---|---|---:|---|
| retail | `retail_dialog_205` | review/2.75 | failed_recovery: communication_and_closeout: The final turn contains only a tool call with no text response. While this is technically compliant with the 'no text with tool' rule, it leaves the user's second request (exchange @ [15]; recovery: No errors occurred to require recovery. However, the agent's failure to handle the multi-intent turn (address change + exchange request) represents a workflow gap that wasn't @ []; task_completion: The agent successfully completed the first sub-task (address modification) but failed to address the second sub-task (exchange) introduced by the user in turn 14. The trajectory is @ [15] |
| retail | `retail_dialog_166` | review/3.38 | incomplete_task: incomplete_task_handling: The agent ignored the user's request to cancel a second order and prematurely asked if there was anything else they could help with, failing the full task requirements. @ [17] |
| telecom | `420` | drop/1.88 | failed_recovery: Memory/State blindness: The agent completely ignored the fact that Wi-Fi calling was already turned off in turn 33 and repeatedly instructed the user to turn it off again. @ [33, 39, 43] |
| telecom | `154` | drop/2.25 | false_success_claim: State Memory Failure: The agent forgets that it already performed the 'turn off Wi-Fi calling' step (turn 47) and asks the user to do it again in the final turn. @ [58] |

## 共性定义

高质量不是只有最终 reward=1；还要求意图完整、读后再写、工具与参数有 observation
依据、写操作经过当轮确认、错误后能改变策略、最终陈述与工具结果一致。允许同轮多调用，
但每个调用都必须在调用前已有独立依据、不依赖同批其他调用的 observation、写入已明确
授权且 order-independent，并且每个调用都有必要；否则应单调用并等待结果。

后续 matched success 结果中，multi − single 总质量差为
`+0.005 [-0.034, +0.045]`，没有整体优劣证据；因此调用数量只记录为行为特征，最终筛选
依据是 dependency safety、grounding、policy、authorization、necessity 与 completion。
完整标准见
[FILTER_SPEC.md](../tau2-sft-multitool-quality-audit/FILTER_SPEC.md)和
[FINAL_RECOMMENDATION.md](../tau2-sft-local-first-relaxed/FINAL_RECOMMENDATION.md)。
