# Qwen3–Qwen3.5 Atomic Capability Gap Analysis

Experiment name: `tau2-qwen3-qwen35-atomic-gap-analysis`. Purpose: explain Qwen3-4B-Instruct-2507 failures across the four-domain official benchmark using deterministic evidence, blinded contrastive review, and bounded prefix replay.

Qwen3.5 non-thinking is contrastive evidence rather than a ground-truth teacher.

## Inputs

- Scenario cells: 788 across 197 tasks.
- Runs: two Qwen3-4B-Instruct-2507 seed-300 evaluations and one matched Qwen3.5-4B non-thinking evaluation; official-native, four trials, BM25 Banking retrieval.
- Canonical cases adjudicated: 197/197.

## Outcome matrix

| Outcome | Scenario cells |
|---|---:|
| stable_gap | 166 |
| shared_hard | 413 |
| base_unstable | 98 |
| reverse_control | 19 |
| all_success | 92 |

## Domain outcome rates

Success counts use the same scenario-cell denominator within each domain.

| Domain | Tasks | Qwen3 base A | Qwen3 base B | Qwen3.5 | Stable gap | Shared hard |
|---|---:|---:|---:|---:|---:|---:|
| airline | 20 | 26/80 | 30/80 | 64/80 | 33 | 11 |
| banking_knowledge | 97 | 13/388 | 14/388 | 15/388 | 8 | 360 |
| retail | 40 | 86/160 | 95/160 | 131/160 | 32 | 15 |
| telecom | 40 | 30/160 | 26/160 | 125/160 | 93 | 27 |

## Atomic gaps

Formal prevalence uses only task-weighted, adjudicated high-confidence contrasts.
A task is high confidence when Qwen3 succeeds at most 2/8 times across both repeats and Qwen3.5 succeeds at least 3/4 times.

| Domain | Primary gap | Tasks | Share |
|---|---|---:|---:|
| airline | workflow_precondition_authorization | 3 | 42.9% |
| airline | evidence_sufficiency | 2 | 28.6% |
| airline | action_selection | 1 | 14.3% |
| airline | entity_state_binding | 1 | 14.3% |
| retail | entity_state_binding | 3 | 33.3% |
| retail | action_selection | 2 | 22.2% |
| retail | argument_composition | 1 | 11.1% |
| retail | evidence_sufficiency | 1 | 11.1% |
| retail | goal_constraint_tracking | 1 | 11.1% |
| retail | observation_grounded_recovery | 1 | 11.1% |
| telecom | action_selection | 8 | 29.6% |
| telecom | entity_state_binding | 8 | 29.6% |
| telecom | evidence_sufficiency | 4 | 14.8% |
| telecom | goal_constraint_tracking | 4 | 14.8% |
| telecom | observation_grounded_recovery | 2 | 7.4% |
| telecom | workflow_precondition_authorization | 1 | 3.7% |

## Domain findings

Banking is reported as shared difficulty because it has no high-confidence task-level contrast and Qwen3.5 succeeds only sparsely there.

| Domain | High-confidence tasks | Rows summarized | Leading adjudicated gaps |
|---|---:|---:|---|
| airline | 7 | 7 | workflow_precondition_authorization (3), evidence_sufficiency (2), action_selection (1) |
| banking_knowledge | 0 | 97 | unclear (72), evidence_sufficiency (8), none (7) |
| retail | 9 | 9 | entity_state_binding (3), action_selection (2), argument_composition (1) |
| telecom | 27 | 27 | action_selection (8), entity_state_binding (8), evidence_sufficiency (4) |

Candidate capability priorities (not training results):

- Retrieve decisive evidence and bind it to the correct entity/state before acting.
- Check workflow preconditions and authorization before irreversible writes.
- Select the valid action and compose grounded arguments without inventing values.
- Preserve multi-subtask coverage and execute the required stop or transfer condition.

## Prefix replay

Counts are acceptable next actions over four samples per model/prefix condition.

| Case | Domain | Qwen3/weak | Qwen3.5/weak | Qwen3/strong | Qwen3.5/strong | Pattern |
|---|---|---:|---:|---:|---:|---|
| airline/2/0 | airline | 0/4 | 2/4 | 0/4 | 2/4 | local_decision |
| airline/30/0 | airline | 0/4 | 4/4 | 4/4 | 3/4 | compound |
| airline/37/0 | airline | 0/4 | 3/4 | 0/4 | 1/4 | local_decision |
| banking_knowledge/task_008/1 | banking_knowledge | 0/4 | 1/4 | 0/4 | 4/4 | local_decision |
| banking_knowledge/task_015/3 | banking_knowledge | 0/4 | 0/4 | 4/4 | 4/4 | upstream_evidence_state |
| banking_knowledge/task_034/3 | banking_knowledge | 0/4 | 0/4 | 1/4 | 1/4 | upstream_evidence_state |
| retail/39/1 | retail | 0/4 | 0/4 | 1/4 | 4/4 | upstream_evidence_state |
| retail/55/0 | retail | 0/4 | 4/4 | 1/4 | 3/4 | compound |
| retail/56/0 | retail | 0/4 | 2/4 | 0/4 | 1/4 | local_decision |
| telecom/[mms_issue]bad_network_preference|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:None]/0 | telecom | 0/4 | 0/4 | 4/4 | 4/4 | upstream_evidence_state |
| telecom/[mobile_data_issue]bad_network_preference|bad_vpn|user_abroad_roaming_disabled_off[PERSONA:Hard]/0 | telecom | 4/4 | 0/4 | 0/4 | 0/4 | uncertain |
| telecom/[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin|unseat_sim_card[PERSONA:None]/1 | telecom | 0/4 | 1/4 | 0/4 | 4/4 | local_decision |

Replay aggregate: Qwen3 weak 4/48, Qwen3.5 weak 17/48, Qwen3 strong 15/48, and Qwen3.5 strong 31/48.
Directional case patterns: compound 2, local_decision 5, uncertain 1, upstream_evidence_state 4.

## Representative cases

| Domain/task | Primary gap | First divergence (Qwen3/Qwen3.5) | Minimal corrective mechanism |
|---|---|---|---|
| airline/24 | workflow_precondition_authorization | 8/8 | Qwen3.5 checked the cancellation preconditions against the retrieved reservation and refused the unauthorized write. |
| airline/16 | evidence_sufficiency | 11/11 | Qwen3.5 correctly parsed the `search_onestop_flight` results, calculated the total Economy prices for all valid combinations, identified HAT110 + HAT172 as the cheapest option ($207), and proceeded to update the reservation with these specific flights. |
| banking_knowledge/task_001 | evidence_sufficiency | 4/4 | Qwen3.5 issued targeted follow-up searches and found the no-fee Gold Rewards option compatible with the user's subscription. |
| banking_knowledge/task_015 | entity_state_binding | 10/14 | Qwen3.5 retrieved user and card state before composing the discoverable-tool arguments. |
| retail/101 | entity_state_binding | 10/10 | Qwen3.5 correctly queried all three orders (#W3445693, #W4219264, #W6729841) to identify the specific items in each. It correctly bound the 'two watches' description to order #W4219264 and the 'air purifier and speaker' description to order #W6729841. This allowed it to correctly identify the New York address from order #W3445693 and apply it to the pending order #W4219264, and modify the items in the correct pending orders (#W4219264 and #W6729841) as requested. |
| retail/56 | action_selection | 14/16 | Qwen3.5 correctly interpreted the user's intent to modify the specific item rather than cancel the whole order, and selected the 'modify_pending_order_items' action with the correct arguments (cheapest variant ID and gift card payment method) while the order was still in a modifiable state. |
| telecom/[mms_issue]break_apn_mms_setting|user_abroad_roaming_enabled_off[PERSONA:Hard] | action_selection | 2/12 | Qwen3.5 correctly selects the `can_send_mms` tool call at turn 12 to verify the issue, followed by `check_status_bar` and `check_apn_settings`. This allows the agent to identify the specific configuration errors (MMSC Not Set, Data Roaming Off) and guide the user to execute the precise corrective tool calls (`reset_apn_settings`, `reboot_device`, `toggle_roaming`) that resolve the issue. |
| telecom/[mms_issue]airplane_mode_on|break_app_both_permissions|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None] | entity_state_binding | 14/10 | Qwen3.5 correctly identifies the customer first (Turn 10), then iterates through lines to find the one matching the user's phone number (L1002, Turn 22). This allows it to correctly observe the data usage exceedance (15.1 GB vs 15.0 GB limit) and the disabled roaming status on the correct line, enabling the necessary refuel and roaming enablement actions. |

## Limitations

- Same task/trial trajectories stop being counterfactual after the first model action; offline labels are observational.
- Causal wording is reserved for the directional one-step replay results.
- Banking findings describe retrieval and shared-hard behavior, not a demonstrated Qwen3.5 capability.

## Jobs

- [jobs/12441-qwen36-atomic-gap-review-0902-182923565/run_0_20260902_182923565.log](jobs/12441-qwen36-atomic-gap-review-0902-182923565/run_0_20260902_182923565.log)
- [jobs/12444-qwen36-atomic-gap-review-r2-0902-184105909/run_0_20260902_184105909.log](jobs/12444-qwen36-atomic-gap-review-r2-0902-184105909/run_0_20260902_184105909.log)
- [jobs/12450-qwen3-atomic-gap-replay-0902-191205205/run_0_20260902_191205205.log](jobs/12450-qwen3-atomic-gap-replay-0902-191205205/run_0_20260902_191205205.log)
- [jobs/12451-qwen35-atomic-gap-replay-0902-191205994/run_0_20260902_191205994.log](jobs/12451-qwen35-atomic-gap-replay-0902-191205994/run_0_20260902_191205994.log)
- [jobs/12463-qwen35-atomic-gap-replay-r2-0902-192354467/run_0_20260902_192354467.log](jobs/12463-qwen35-atomic-gap-replay-r2-0902-192354467/run_0_20260902_192354467.log)
