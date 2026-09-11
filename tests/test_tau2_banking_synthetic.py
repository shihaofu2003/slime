import inspect
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


NUM_GPUS = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "examples/tau2-bench/analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from banking_synthetic.constants import (  # noqa: E402
    ACTION_SUPPORT_DOCUMENTS,
    BUSINESS_CATEGORIES,
    CASE_KINDS,
    CHALLENGE_TASK_COUNT,
    DEV_TASK_COUNT,
    EMPTY_DB_TABLES,
    FOLLOWUP_TRAIN_COUNT,
    FORM_ACTION_BOUNDS,
    FORM_DOCUMENT_BOUNDS,
    LOW_FORM_CATEGORIES,
    MIN_GRPO_FRONTIER,
    MIN_TRAIN_SCENARIOS,
    PILOT_FORM_QUOTAS,
    TRAIN_CATEGORY_QUOTAS,
    TRAIN_FORM_QUOTAS,
    TRAIN_TASK_COUNT,
)
from banking_synthetic.assemble import assemble  # noqa: E402
from banking_synthetic.combine_pools import combine_final_pool  # noqa: E402
from banking_synthetic.documents import (  # noqa: E402
    DocumentCatalog,
    communication_anchors,
    decisive_fact,
    empty_db,
    read_json,
    stage_runtime,
)
from banking_synthetic.finalize import (  # noqa: E402
    EXPECTED_SPLIT_COUNTS,
    acceptance_errors,
    build_parser as build_finalize_parser,
)
from banking_synthetic.generate import (  # noqa: E402
    assignments,
    case_matrix_assignments,
    generate,
    summary,
)
from banking_synthetic.isolation import audit_artifacts, build_allowlist  # noqa: E402
from banking_synthetic.merge import merge_artifacts  # noqa: E402
from banking_synthetic.models import scenario_to_task, user_action_protocol  # noqa: E402
from banking_synthetic.naturalize import naturalize_one  # noqa: E402
from banking_synthetic.pools import (  # noqa: E402
    build_pools,
    main as pools_main,
    teacher_trajectories_for_contracts,
)
from banking_synthetic.prepare_eval import select as select_eval_contracts  # noqa: E402
from banking_synthetic.protocol import protocol_failure_reason  # noqa: E402
from banking_synthetic.select_recovery import recovery_selection  # noqa: E402
from banking_synthetic.select_followup import (  # noqa: E402
    build_parser as build_followup_parser,
    select_followup,
)
import banking_synthetic.select_tasks as select_tasks_module  # noqa: E402
from banking_synthetic.select_tasks import balanced_selection  # noqa: E402
from banking_synthetic.select_teacher import select_teacher_tasks  # noqa: E402
from banking_synthetic.summarize import (  # noqa: E402
    _compact_row,
    _group_metrics,
    summarize,
)
from banking_synthetic.teacher import (  # noqa: E402
    _completion,
    _opening_messages,
    shard_task_ids,
)
from banking_synthetic.templates import build_scenario  # noqa: E402
from banking_synthetic.validate import static_errors  # noqa: E402


def _catalog() -> DocumentCatalog:
    documents = {
        document_id: {
            "id": document_id,
            "title": f"Synthetic support policy {index:03d}",
            "content": "Policy support is available for this synthetic operation.",
        }
        for index, document_id in enumerate(
            sorted(
                {
                    document_id
                    for document_ids in ACTION_SUPPORT_DOCUMENTS.values()
                    for document_id in document_ids
                }
            )
        )
    }
    prefixes = (
        "doc_credit_cards_",
        "doc_business_credit_cards_",
        "doc_bank_accounts_",
        "doc_checking_accounts_",
        "doc_business_checking_accounts_",
        "doc_savings_accounts_",
        "doc_business_savings_accounts_",
        "doc_everyone_pay_",
        "doc_buy_now_pay_later_",
        "doc_personal_subscriptions_",
    )
    for index in range(480 - len(documents)):
        document_id = f"{prefixes[index % len(prefixes)]}synthetic_{index:03d}"
        documents[document_id] = {
            "id": document_id,
            "title": f"Synthetic policy document {index:03d}",
            "content": (
                f"Synthetic indexed policy anchor number {index:03d} explains eligibility, "
                "verification, referral, support, rewards, disputes, replacement cards, "
                "annual fees, credit limits, account opening, deposits, ATM fees, declined "
                "transactions, security, savings interest, APY, and bonuses."
            ),
        }
    return DocumentCatalog(documents=documents, allowed_ids=tuple(documents))


class AllowlistTest(unittest.TestCase):
    def test_allowlist_is_exact_complement_of_benchmark_required_documents(self):
        documents = {
            f"doc_{index:03d}": {"id": f"doc_{index:03d}", "title": str(index), "content": "text"}
            for index in range(698)
        }
        required = [f"doc_{index:03d}" for index in range(218)]
        benchmark = [
            {
                "id": f"task_{index:03d}",
                "required_documents": required if index == 0 else [],
                "user_scenario": {"instructions": f"sealed request {index}"},
            }
            for index in range(97)
        ]

        manifest = build_allowlist(benchmark, documents)

        self.assertEqual(manifest["allowed_document_count"], 480)
        self.assertEqual(len(manifest["allowed_document_ids"]), 480)
        self.assertTrue(set(manifest["allowed_document_ids"]).isdisjoint(required))

    def test_empty_db_matches_every_banking_table(self):
        self.assertEqual(set(empty_db()), set(EMPTY_DB_TABLES))
        self.assertTrue(all(value == {"data": {}, "notes": ""} for value in empty_db().values()))
        asset = read_json(
            ANALYSIS_DIR / "banking_synthetic/assets/empty_db.json"
        )
        self.assertEqual(asset, empty_db())

    def test_document_selection_prioritizes_category_topic_within_family(self):
        documents = {
            "doc_credit_cards_unrelated": {
                "id": "doc_credit_cards_unrelated",
                "title": "Travel concierge details",
                "content": "Unrelated card material.",
            },
            "doc_credit_cards_limit": {
                "id": "doc_credit_cards_limit",
                "title": "Credit Limit Increase Guide",
                "content": "Relevant card material.",
            },
        }
        catalog = DocumentCatalog(
            documents=documents, allowed_ids=tuple(documents)
        )

        selected = catalog.select("信用额度调整", offset=100, count=1)

        self.assertEqual(selected, ("doc_credit_cards_limit",))

    def test_decisive_fact_skips_introductions_incomplete_lines_and_table_headers(self):
        closure = {
            "title": "Close an account",
            "content": (
                "We understand that occasionally customers need help.\n"
                "To close an account, you must meet all of the following:\n"
                "1. Zero balance required: Your balance must be $0.00."
            ),
        }
        table = {
            "title": "Daily limits",
            "content": (
                "| Activity | Limit |\n"
                "| --- | --- |\n"
                "| Mobile check deposit | $100,000 |"
            ),
        }

        self.assertEqual(
            decisive_fact(closure),
            "Zero balance required: Your balance must be $0.00.",
        )
        self.assertEqual(
            decisive_fact(table), "Mobile check deposit | $100,000"
        )
        self.assertEqual(
            decisive_fact(
                {
                    "title": "Rewards",
                    "content": "$660.\nRewards are credited monthly after posting.",
                }
            ),
            "Rewards are credited monthly after posting.",
        )
        self.assertEqual(
            decisive_fact(
                {
                    "title": "Rewards eligibility",
                    "content": "What is the minimum score to earn rewards? $660.",
                },
                "返现与奖励核算",
            ),
            "What is the minimum score to earn rewards? $660.",
        )
        self.assertEqual(
            communication_anchors(
                "Your initial line is at least $10,000 and at most $50,000."
            ),
            ("10,000", "50,000"),
        )

    def test_decisive_fact_prefers_a_category_matching_fact(self):
        document = {
            "title": "Evergreen Account FAQ",
            "content": (
                "- You receive impact reports monthly.\n"
                "- Refer friends to open an Evergreen Account.\n"
                "- The referred person must deposit $750 within 60 days."
            ),
        }

        self.assertEqual(
            decisive_fact(document, "账户推荐奖励"),
            "Refer friends to open an Evergreen Account.",
        )


class QuotaTest(unittest.TestCase):
    def test_pilot_assignments_have_exact_category_and_form_quotas(self):
        rows = assignments("pilot", 7)

        self.assertEqual(len(rows), 1500)
        self.assertEqual(Counter(category for category, _ in rows), Counter({name: 100 for name in BUSINESS_CATEGORIES}))
        self.assertEqual(Counter(form for _, form in rows), Counter(PILOT_FORM_QUOTAS))

    def test_train_assignments_have_exact_declared_marginals(self):
        rows = assignments("train", 11)

        self.assertEqual(len(rows), 542)
        self.assertEqual(
            TRAIN_CATEGORY_QUOTAS,
            {
                "信用卡选择与申请资格": 38,
                "身份验证与资料变更": 27,
                "信用卡推荐奖励": 30,
                "知识缺失与转人工": 27,
                "返现与奖励核算": 43,
                "交易争议、购买保护与补卡": 49,
                "信用卡保留与销户": 33,
                "信用额度调整": 30,
                "账户推荐、开户、销户与入金": 54,
                "ATM 费用": 30,
                "卡片遗失或被盗": 38,
                "借记卡交易争议": 38,
                "借记卡拒付与 PIN": 38,
                "储蓄利息": 35,
                "账户推荐奖励": 32,
            },
        )
        self.assertEqual(
            TRAIN_FORM_QUOTAS,
            {
                "l1_evidence": 54,
                "l2_retrieval": 27,
                "l3_decision": 81,
                "l3_single_action": 109,
                "l4_two_skill": 163,
                "l5_medium_workflow": 81,
                "l6_full_workflow": 27,
            },
        )
        self.assertEqual(Counter(category for category, _ in rows), Counter(TRAIN_CATEGORY_QUOTAS))
        self.assertEqual(Counter(form for _, form in rows), Counter(TRAIN_FORM_QUOTAS))
        self.assertTrue(
            all(
                category in LOW_FORM_CATEGORIES
                for category, form in rows
                if form in {"l1_evidence", "l2_retrieval"}
            )
        )

    def test_eval_assignments_cover_each_declared_category_form_cell(self):
        dev = assignments("dev", 11)
        challenge = assignments("challenge", 11)
        dev_forms = {
            "l3_decision",
            "l3_single_action",
            "l4_two_skill",
            "l5_medium_workflow",
            "l6_full_workflow",
        }
        challenge_forms = {"l5_medium_workflow", "l6_full_workflow"}

        self.assertEqual(len(dev), 75)
        self.assertEqual(Counter(category for category, _ in dev), Counter({name: 5 for name in BUSINESS_CATEGORIES}))
        self.assertEqual(Counter(dev), Counter({(category, form): 1 for category in BUSINESS_CATEGORIES for form in dev_forms}))
        self.assertEqual(len(challenge), 30)
        self.assertEqual(Counter(category for category, _ in challenge), Counter({name: 2 for name in BUSINESS_CATEGORIES}))
        self.assertEqual(Counter(challenge), Counter({(category, form): 1 for category in BUSINESS_CATEGORIES for form in challenge_forms}))

    def test_split_serial_namespaces_do_not_overlap(self):
        catalog = _catalog()
        split_scenarios = {
            phase: generate(phase=phase, seed=3, catalog=catalog, limit=2)[0]
            for phase in ("pilot", "train", "dev", "challenge")
        }
        ids = [spec["scenario_id"] for specs in split_scenarios.values() for spec in specs]

        self.assertEqual(len(ids), len(set(ids)))

    def test_template_matrix_covers_every_category_and_case_kind(self):
        rows = case_matrix_assignments()

        self.assertEqual(len(rows), len(BUSINESS_CATEGORIES) * len(CASE_KINDS))
        self.assertEqual(
            {(category, case_kind) for category, _, case_kind in rows},
            {
                (category, case_kind)
                for category in BUSINESS_CATEGORIES
                for case_kind in CASE_KINDS
            },
        )

    def test_assembly_prioritizes_frontier_candidates_within_a_quota_cell(self):
        contracts = [
            {
                "task_id": task_id,
                "scenario_id": f"scenario_{index}",
                "business_category": "category",
                "task_form": "form",
            }
            for index, task_id in enumerate(("task_anchor", "task_frontier", "task_hard"))
        ]
        tasks = [{"id": row["task_id"]} for row in contracts]
        specs = [{"scenario_id": row["scenario_id"]} for row in contracts]

        with patch(
            "banking_synthetic.assemble.assignments",
            return_value=[("category", "form"), ("category", "form")],
        ):
            selected, _, _, _ = assemble(
                tasks=tasks,
                specs=specs,
                contracts=contracts,
                eligible_task_ids={row["task_id"] for row in contracts},
                frontier_task_ids={"task_frontier"},
                seed=1,
            )

        self.assertEqual(
            [row["id"] for row in selected], ["task_frontier", "task_anchor"]
        )

    def test_assembly_prefers_four_trial_candidates_before_incomplete_ones(self):
        contracts = [
            {
                "task_id": "task_incomplete",
                "scenario_id": "scenario_incomplete",
                "business_category": "category",
                "task_form": "form",
            },
            {
                "task_id": "task_complete",
                "scenario_id": "scenario_complete",
                "business_category": "category",
                "task_form": "form",
            },
        ]
        tasks = [{"id": row["task_id"]} for row in contracts]
        specs = [{"scenario_id": row["scenario_id"]} for row in contracts]

        with patch(
            "banking_synthetic.assemble.assignments",
            return_value=[("category", "form")],
        ):
            selected, _, _, _ = assemble(
                tasks=tasks,
                specs=specs,
                contracts=contracts,
                eligible_task_ids={"task_incomplete", "task_complete"},
                frontier_task_ids=set(),
                four_trial_task_ids={"task_complete"},
                seed=1,
            )

        self.assertEqual([row["id"] for row in selected], ["task_complete"])

    def test_assembly_requests_frontier_replacements_from_harder_forms_first(self):
        contracts = [
            {
                "task_id": "task_easy",
                "scenario_id": "scenario_easy",
                "business_category": "category",
                "task_form": "l1_evidence",
            },
            {
                "task_id": "task_hard",
                "scenario_id": "scenario_hard",
                "business_category": "category",
                "task_form": "l5_medium_workflow",
            },
        ]
        tasks = [{"id": row["task_id"]} for row in contracts]
        specs = [{"scenario_id": row["scenario_id"]} for row in contracts]

        with patch(
            "banking_synthetic.assemble.assignments",
            return_value=[
                ("category", "l1_evidence"),
                ("category", "l5_medium_workflow"),
            ],
        ):
            _, _, _, missing = assemble(
                tasks=tasks,
                specs=specs,
                contracts=contracts,
                eligible_task_ids={row["task_id"] for row in contracts},
                frontier_task_ids=set(),
                seed=1,
            )

        frontier_requests = [
            row for row in missing if row.get("reason") == "frontier_shortfall"
        ]
        self.assertEqual(len(frontier_requests), 163)
        self.assertEqual(frontier_requests[0]["task_form"], "l5_medium_workflow")

    def test_reduced_scale_defaults_follow_up_on_all_action_train_tasks(self):
        self.assertEqual(
            EXPECTED_SPLIT_COUNTS,
            {"train": 542, "dev": 75, "challenge": 30},
        )
        self.assertEqual(TRAIN_TASK_COUNT, 542)
        self.assertEqual(DEV_TASK_COUNT, 75)
        self.assertEqual(CHALLENGE_TASK_COUNT, 30)
        self.assertEqual(MIN_TRAIN_SCENARIOS, 542)
        self.assertEqual(MIN_GRPO_FRONTIER, 163)
        self.assertEqual(build_followup_parser().get_default("train_count"), 380)
        self.assertEqual(FOLLOWUP_TRAIN_COUNT, 380)

    def test_quota_selection_reuses_train_candidates_and_keeps_eval_splits(self):
        selector = getattr(select_tasks_module, "quota_selection", None)
        self.assertIsNotNone(selector)
        contracts = [
            {
                "task_id": "train_b",
                "scenario_id": "scenario_b",
                "split": "train",
                "business_category": "category",
                "task_form": "form",
            },
            {
                "task_id": "train_a",
                "scenario_id": "scenario_a",
                "split": "train",
                "business_category": "category",
                "task_form": "form",
            },
            {
                "task_id": "dev",
                "scenario_id": "scenario_dev",
                "split": "dev",
                "business_category": "category",
                "task_form": "form",
            },
            {
                "task_id": "challenge",
                "scenario_id": "scenario_challenge",
                "split": "challenge",
                "business_category": "category",
                "task_form": "form",
            },
        ]
        tasks = [{"id": row["task_id"]} for row in contracts]
        specs = [{"scenario_id": row["scenario_id"]} for row in contracts]

        with patch(
            "banking_synthetic.select_tasks.assignments",
            return_value=[("category", "form")],
        ):
            selected_tasks, selected_specs, selected_contracts = selector(
                tasks, specs, contracts, seed=1
            )

        self.assertEqual(
            [row["id"] for row in selected_tasks],
            ["train_a", "dev", "challenge"],
        )
        self.assertEqual(len(selected_specs), 3)
        self.assertEqual(
            Counter(row["split"] for row in selected_contracts),
            {"train": 1, "dev": 1, "challenge": 1},
        )

    def test_quota_selection_applies_declared_quotas_to_every_split(self):
        contracts = [
            {
                "task_id": f"{split}_{suffix}",
                "scenario_id": f"scenario_{split}_{suffix}",
                "split": split,
                "business_category": "category",
                "task_form": "form",
            }
            for split in ("train", "dev", "challenge")
            for suffix in ("a", "b")
        ]
        tasks = [{"id": row["task_id"]} for row in contracts]
        specs = [{"scenario_id": row["scenario_id"]} for row in contracts]

        with patch(
            "banking_synthetic.select_tasks.assignments",
            return_value=[("category", "form")],
        ):
            selected_tasks, _, selected_contracts = (
                select_tasks_module.quota_selection(
                    tasks, specs, contracts, seed=1
                )
            )

        self.assertEqual(
            [row["id"] for row in selected_tasks],
            ["train_a", "dev_a", "challenge_a"],
        )
        self.assertEqual(
            Counter(row["split"] for row in selected_contracts),
            {"train": 1, "dev": 1, "challenge": 1},
        )


class TemplateTest(unittest.TestCase):
    def test_every_category_and_form_obeys_difficulty_bounds(self):
        catalog = _catalog()
        tasks = []
        specs = []
        contracts = []
        serial = 0
        for category in BUSINESS_CATEGORIES:
            for form in FORM_DOCUMENT_BOUNDS:
                if (
                    form in {"l1_evidence", "l2_retrieval"}
                    and category not in LOW_FORM_CATEGORIES
                ):
                    with self.assertRaises(ValueError):
                        build_scenario(
                            split="train",
                            serial=serial,
                            category=category,
                            form=form,
                            catalog=catalog,
                            case_kind="positive",
                        )
                    serial += 1
                    continue
                spec = build_scenario(
                    split="train",
                    serial=serial,
                    category=category,
                    form=form,
                    catalog=catalog,
                    case_kind="positive",
                )
                task, contract = scenario_to_task(spec)
                tasks.append(task)
                specs.append(spec.to_dict())
                contracts.append(contract.to_dict())
                doc_min, doc_max = FORM_DOCUMENT_BOUNDS[form]
                action_min, action_max = FORM_ACTION_BOUNDS[form]
                self.assertTrue(doc_min <= len(task["required_documents"]) <= doc_max)
                self.assertTrue(
                    action_min
                    <= len(task["evaluation_criteria"]["actions"] or [])
                    <= action_max
                )
                serial += 1

        self.assertEqual(
            static_errors(tasks, specs, contracts, set(catalog.allowed_ids)), []
        )

    def test_each_category_has_refusal_user_dynamic_and_multi_entity_cases(self):
        catalog = _catalog()
        form_for_case = {
            "positive": "l4_two_skill",
            "refusal": "l5_medium_workflow",
            "user_tool": "l5_medium_workflow",
            "dynamic_tool": "l4_two_skill",
            "multi_entity": "l6_full_workflow",
        }
        for category_index, category in enumerate(BUSINESS_CATEGORIES):
            for case_index, case_kind in enumerate(CASE_KINDS):
                spec = build_scenario(
                    split="train",
                    serial=1000 + category_index * 10 + case_index,
                    category=category,
                    form=form_for_case[case_kind],
                    catalog=catalog,
                    case_kind=case_kind,
                )
                names = [node.action["name"] for node in spec.action_nodes]
                if case_kind == "refusal":
                    self.assertIn("request_human_agent_transfer", names)
                    self.assertIn("request_human_agent_transfer", spec.user_tools)
                    self.assertEqual(
                        spec.expected_db_changes,
                        (
                            {
                                "table": "human_transfer_requests",
                                "scope": "synthetic scenario entities",
                            },
                        ),
                    )
                elif case_kind == "user_tool":
                    self.assertTrue(
                        any(
                            node.action["requestor"] == "user"
                            for node in spec.action_nodes
                        )
                    )
                    self.assertTrue(spec.user_tools)
                elif case_kind == "dynamic_tool":
                    self.assertIn("unlock_discoverable_agent_tool", names)
                    self.assertIn("call_discoverable_agent_tool", names)
                elif case_kind == "multi_entity":
                    self.assertGreaterEqual(len(spec.expected_db_changes), 2)

    def test_tasks_have_independent_contract_and_stop_protocol(self):
        spec = build_scenario(
            split="train",
            serial=17,
            category="ATM 费用",
            form="l4_two_skill",
            catalog=_catalog(),
        )
        task, contract = scenario_to_task(spec)
        dumped = contract.to_dict()

        self.assertEqual(dumped["data_origin"], "independent_synthetic")
        self.assertNotIn("source_task_id", dumped)
        self.assertTrue(task["id"].startswith("banking_syn_v1_"))
        self.assertIn("###STOP###", task["user_scenario"]["instructions"])

    def test_user_owned_actions_have_an_exact_private_execution_protocol(self):
        catalog = _catalog()
        pre_given = build_scenario(
            split="train",
            serial=40,
            category="信用卡保留与销户",
            form="l3_single_action",
            catalog=catalog,
            case_kind="user_tool",
        )
        coordinated = build_scenario(
            split="train",
            serial=41,
            category="返现与奖励核算",
            form="l4_two_skill",
            catalog=catalog,
            case_kind="user_tool",
        )
        transfer = build_scenario(
            split="train",
            serial=42,
            category="知识缺失与转人工",
            form="l3_single_action",
            catalog=catalog,
            case_kind="user_tool",
        )
        continued = build_scenario(
            split="train",
            serial=44,
            category="信用卡选择与申请资格",
            form="l6_full_workflow",
            catalog=catalog,
            case_kind="user_tool",
        )
        verified = build_scenario(
            split="train",
            serial=45,
            category="信用卡选择与申请资格",
            form="l5_medium_workflow",
            catalog=catalog,
            case_kind="positive",
        )

        pre_given_protocol = user_action_protocol(pre_given)
        coordinated_protocol = user_action_protocol(coordinated)
        transfer_task, _ = scenario_to_task(transfer)

        self.assertIn("After the Agent's first response", pre_given_protocol)
        self.assertIn(
            "As soon as the Agent gives you `submit_cash_back_dispute_0589`",
            coordinated_protocol,
        )
        self.assertIn(
            "`request_human_agent_transfer`",
            transfer_task["user_scenario"]["instructions"],
        )
        self.assertIn(
            "not to transfer you with its own tool",
            transfer_task["user_scenario"]["instructions"],
        )
        self.assertIn("do not stop early", user_action_protocol(continued))
        self.assertIn(
            "Once the Agent completes the preceding policy or verification steps",
            user_action_protocol(verified),
        )
        self.assertIn(
            "use the already available customer-side card-detail lookup",
            pre_given.canonical_user_request,
        )
        pre_given_task, pre_given_contract = scenario_to_task(pre_given)
        self.assertIn(
            "<<Action clue: customer-side action label `get_card_last_4_digits`",
            pre_given_task["user_scenario"]["instructions"],
        )
        self.assertTrue(
            any("get_card_last_4_digits" in fact for fact in pre_given.user_known_info)
        )
        self.assertEqual(
            static_errors(
                [pre_given_task],
                [pre_given.to_dict()],
                [pre_given_contract.to_dict()],
                set(catalog.allowed_ids),
            ),
            [],
        )

    def test_static_validation_rejects_missing_private_user_tool_protocol(self):
        catalog = _catalog()
        spec = build_scenario(
            split="train",
            serial=43,
            category="账户推荐、开户、销户与入金",
            form="l4_two_skill",
            catalog=catalog,
            case_kind="user_tool",
        )
        task, contract = scenario_to_task(spec)
        task["user_scenario"]["instructions"] = spec.canonical_user_request + " ###STOP###"

        errors = static_errors(
            [task], [spec.to_dict()], [contract.to_dict()], set(catalog.allowed_ids)
        )

        self.assertTrue(any("lacks a private user-tool protocol" in error for error in errors))

    def test_static_validation_rejects_a_hidden_dynamic_action_label(self):
        catalog = _catalog()
        spec = build_scenario(
            split="train",
            serial=46,
            category="储蓄利息",
            form="l4_two_skill",
            catalog=catalog,
            case_kind="positive",
        )
        task, contract = scenario_to_task(spec)
        agent_data = task["initial_state"]["initialization_data"]["agent_data"]
        self.assertTrue((agent_data["verification_history"]["data"]))
        self.assertIn(
            "<<Case precondition: Identity verification was completed and logged",
            task["user_scenario"]["instructions"],
        )
        self.assertIn(
            "`call_discoverable_agent_tool` tool with arguments",
            task["user_scenario"]["instructions"],
        )
        opening = task["initial_state"]["message_history"][0]["content"]
        self.assertIn("`call_discoverable_agent_tool` tool with arguments", opening)
        task["initial_state"]["message_history"][0]["content"] = opening.replace(
            "<<Action clue: Agent must invoke its `call_discoverable_agent_tool` tool",
            "<<Omitted clue: Agent must invoke its `call_discoverable_agent_tool` tool",
        )

        errors = static_errors(
            [task], [spec.to_dict()], [contract.to_dict()], set(catalog.allowed_ids)
        )

        self.assertTrue(
            any("is not visible in the initial User message" in error for error in errors)
        )

    def test_single_action_transfer_allows_equivalent_summary_and_reason_arguments(self):
        spec = build_scenario(
            split="train",
            serial=48,
            category="信用额度调整",
            form="l3_single_action",
            catalog=_catalog(),
            case_kind="refusal",
        )

        transfer = next(
            node.action
            for node in spec.action_nodes
            if node.action["name"] == "transfer_to_human_agents"
        )

        self.assertEqual(transfer["compare_args"], [])

    def test_workflow_refusal_is_scored_by_customer_request_state(self):
        spec = build_scenario(
            split="train",
            serial=49,
            category="信用额度调整",
            form="l5_medium_workflow",
            catalog=_catalog(),
            case_kind="refusal",
        )
        task, _ = scenario_to_task(spec)

        self.assertEqual(task["evaluation_criteria"]["reward_basis"], ["DB"])
        self.assertNotIn(
            "transfer_to_human_agents",
            [node.action["name"] for node in spec.action_nodes],
        )
        self.assertIn(
            "request_human_agent_transfer",
            [node.action["name"] for node in spec.action_nodes],
        )

    def test_evidence_and_decision_forms_require_policy_content(self):
        catalog = _catalog()
        evidence = build_scenario(
            split="train",
            serial=18,
            category="ATM 费用",
            form="l1_evidence",
            catalog=catalog,
        )
        decision = build_scenario(
            split="train",
            serial=19,
            category="ATM 费用",
            form="l3_decision",
            catalog=catalog,
        )

        self.assertEqual(
            evidence.final_response_facts,
            communication_anchors(evidence.decisive_facts[0]),
        )
        self.assertFalse(evidence.decisive_facts[0].startswith("Synthetic support policy"))
        self.assertIn(
            "quote verbatim the first complete policy fact that begins with",
            evidence.canonical_user_request,
        )
        fact_words = evidence.decisive_facts[0].split()
        locator = fact_words[: min(4, max(2, len(fact_words) - 1))]
        self.assertIn(
            f"<<Evidence locator: {' '.join(locator)}>>",
            evidence.canonical_user_request,
        )
        self.assertIn("Quote the complete policy fact verbatim", evidence.canonical_user_request)
        self.assertIn(
            f"<<Policy claim: {decision.decisive_facts[0]}>>",
            decision.canonical_user_request,
        )
        self.assertIn(
            "<<Response requirement: Begin the answer with exactly Yes or No.>>",
            decision.canonical_user_request,
        )
        self.assertEqual(decision.final_response_facts, ("Yes",))

    def test_profile_change_defines_the_requested_new_email(self):
        spec = build_scenario(
            split="train",
            serial=34,
            category="身份验证与资料变更",
            form="l6_full_workflow",
            catalog=_catalog(),
            case_kind="multi_entity",
        )
        task, _ = scenario_to_task(spec)
        expected = "synthetic.updated.train.000034@example.test"

        self.assertIn(f"Requested new email: {expected}", spec.user_known_info)
        self.assertIn(expected, spec.canonical_user_request)
        self.assertIn(expected, task["initial_state"]["message_history"][0]["content"])

    def test_action_scenarios_bind_policy_products_facts_and_state(self):
        catalog = _catalog()

        cashback = build_scenario(
            split="train",
            serial=31,
            category="返现与奖励核算",
            form="l4_two_skill",
            catalog=catalog,
            case_kind="positive",
        )
        cashback_init = cashback.initialization_data
        cashback_card = next(
            iter(cashback_init["credit_card_accounts"]["data"].values())
        )
        cashback_transaction = next(
            iter(
                cashback_init["credit_card_transaction_history"]["data"].values()
            )
        )
        support_documents = ACTION_SUPPORT_DOCUMENTS["返现与奖励核算"]
        self.assertEqual(
            cashback.required_documents[: len(support_documents)],
            support_documents,
        )
        self.assertEqual(cashback_card["card_type"], "Business Gold Rewards Card")
        self.assertEqual(cashback_transaction["category"], "Operations")
        self.assertEqual(cashback_transaction["rewards_earned"], "$0.84 cash back")
        self.assertTrue(
            all("Policy support is available" in fact for fact in cashback.decisive_facts)
        )

        dispute = build_scenario(
            split="train",
            serial=32,
            category="交易争议、购买保护与补卡",
            form="l3_single_action",
            catalog=catalog,
            case_kind="positive",
        )
        dispute_arguments = json.loads(
            dispute.action_nodes[0].action["arguments"]["arguments"]
        )
        self.assertEqual(dispute_arguments["dispute_reason"], "incorrect_amount")
        self.assertEqual(dispute_arguments["resolution_requested"], "partial_refund")
        self.assertEqual(dispute_arguments["partial_refund_amount"], 36.0)

        closure = build_scenario(
            split="train",
            serial=33,
            category="信用卡保留与销户",
            form="l3_single_action",
            catalog=catalog,
            case_kind="positive",
        )
        closure_account = next(
            iter(closure.initialization_data["credit_card_accounts"]["data"].values())
        )
        self.assertEqual(closure_account["current_balance"], "$0.00")
        self.assertEqual(
            closure.action_nodes[0].action["name"],
            "call_discoverable_agent_tool",
        )

        interest = build_scenario(
            split="train",
            serial=34,
            category="储蓄利息",
            form="l4_two_skill",
            catalog=catalog,
            case_kind="positive",
        )
        savings = [
            account
            for account in interest.initialization_data["accounts"]["data"].values()
            if account["account_type"] == "savings"
        ][0]
        self.assertEqual(savings["account_class"], "Platinum Reserve Account")
        self.assertEqual(savings["current_holdings"], "100000.00")
        self.assertEqual(savings["apy"], "5.00%")

        identity = build_scenario(
            split="train",
            serial=35,
            category="身份验证与资料变更",
            form="l4_two_skill",
            catalog=catalog,
            case_kind="positive",
        )
        identity_checking = [
            account
            for account in identity.initialization_data["accounts"]["data"].values()
            if account["account_type"] == "checking"
        ][0]
        self.assertEqual(
            identity.required_documents[0],
            "doc_checking_accounts_gold_years_account_005",
        )
        self.assertEqual(identity_checking["account_class"], "Gold Years Account")

        pin = build_scenario(
            split="train",
            serial=36,
            category="借记卡拒付与 PIN",
            form="l3_single_action",
            catalog=catalog,
            case_kind="positive",
        )
        pin_card = next(
            iter(pin.initialization_data["debit_cards"]["data"].values())
        )
        self.assertEqual(
            pin.required_documents,
            ("doc_checking_accounts_gold_years_account_005",),
        )
        self.assertFalse(pin_card["pin_locked"])
        self.assertEqual(pin_card["pin_attempts_remaining"], 2)

    def test_static_validation_rejects_bad_dates_amounts_and_foreign_keys(self):
        catalog = _catalog()
        spec = build_scenario(
            split="train",
            serial=37,
            category="交易争议、购买保护与补卡",
            form="l4_two_skill",
            catalog=catalog,
            case_kind="positive",
        )
        task, contract = scenario_to_task(spec)
        task = json.loads(json.dumps(task))
        spec_data = spec.to_dict()
        initialization = task["initial_state"]["initialization_data"]["agent_data"]
        user = next(iter(initialization["users"]["data"].values()))
        account = next(iter(initialization["accounts"]["data"].values()))
        transaction = next(
            iter(initialization["credit_card_transaction_history"]["data"].values())
        )
        user["date_of_birth"] = "13/40/2025"
        account["current_holdings"] = "five dollars"
        transaction["credit_card_account_id"] = "missing_card"
        spec_data["initialization_data"] = json.loads(json.dumps(initialization))

        errors = static_errors(
            [task], [spec_data], [contract.to_dict()], set(catalog.allowed_ids)
        )

        self.assertTrue(any("invalid date_of_birth" in error for error in errors))
        self.assertTrue(any("invalid holdings amount" in error for error in errors))
        self.assertTrue(any("credit transaction" in error and "foreign key" in error for error in errors))


class IsolationAuditTest(unittest.TestCase):
    def test_clean_synthetic_task_passes_and_benchmark_entity_is_rejected(self):
        catalog = _catalog()
        spec = build_scenario(
            split="train",
            serial=1,
            category="储蓄利息",
            form="l4_two_skill",
            catalog=catalog,
        )
        task, contract = scenario_to_task(spec)
        benchmark = [
            {
                "id": f"task_{index:03d}",
                "required_documents": ["sealed_doc"] if index == 0 else [],
                "user_scenario": {"instructions": f"sealed benchmark request {index}"},
                "initial_state": {
                    "initialization_data": {
                        "agent_data": {
                            "users": {"data": {"sealed_user": {"user_id": "sealed_user"}}}
                        }
                    }
                },
            }
            for index in range(97)
        ]
        allowlist = {"allowed_document_ids": list(catalog.allowed_ids)}

        errors, _ = audit_artifacts(
            tasks=[task],
            specs=[spec.to_dict()],
            contracts=[contract.to_dict()],
            allowlist=allowlist,
            benchmark_tasks=benchmark,
        )
        self.assertEqual(errors, [])

        contaminated = json.loads(json.dumps(task))
        user_table = contaminated["initial_state"]["initialization_data"]["agent_data"]["users"]["data"]
        user_table["sealed_user"] = {"user_id": "sealed_user"}
        errors, _ = audit_artifacts(
            tasks=[contaminated],
            specs=[spec.to_dict()],
            contracts=[contract.to_dict()],
            allowlist=allowlist,
            benchmark_tasks=benchmark,
        )
        self.assertTrue(any("sealed_user" in error for error in errors))

        contaminated = json.loads(json.dumps(task))
        contaminated["initial_state"]["initialization_data"]["agent_data"][
            "users"
        ]["data"][spec.scenario_id] = {
            "user_id": "sealed_user",
        }
        errors, report = audit_artifacts(
            tasks=[contaminated],
            specs=[spec.to_dict()],
            contracts=[contract.to_dict()],
            allowlist=allowlist,
            benchmark_tasks=benchmark,
        )
        self.assertTrue(any("initialization record" in error for error in errors))
        self.assertEqual(report["benchmark_initial_record_reuse_count"], 1)


class StagingTest(unittest.TestCase):
    def test_runtime_contains_only_allowlisted_documents_and_empty_db(self):
        catalog = _catalog()
        spec = build_scenario(
            split="dev",
            serial=200000,
            category="信用卡选择与申请资格",
            form="l1_evidence",
            catalog=catalog,
        )
        task, contract = scenario_to_task(spec)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            docs = source / "documents"
            prompts = source / "prompts"
            user_sim = root / "user_simulator"
            docs.mkdir(parents=True)
            prompts.mkdir()
            user_sim.mkdir()
            (prompts / "policy.md").write_text("policy", encoding="utf-8")
            (user_sim / "guide.md").write_text("guide", encoding="utf-8")
            for document in catalog.documents.values():
                (docs / f"{document['id']}.json").write_text(json.dumps(document), encoding="utf-8")
            allowlist = root / "allowlist.json"
            allowlist.write_text(
                json.dumps({"allowed_document_ids": list(catalog.allowed_ids)}),
                encoding="utf-8",
            )

            stage_runtime(
                tasks=[task],
                contracts=[contract.to_dict()],
                allowlist_path=allowlist,
                source_documents_dir=docs,
                source_prompts_dir=prompts,
                source_user_simulator_dir=user_sim,
                data_root=root / "runtime",
            )

            staged = root / "runtime/tau2/domains/banking_knowledge"
            self.assertEqual(len(list((staged / "documents").glob("*.json"))), 480)
            self.assertEqual(read_json(staged / "db.json"), empty_db())
            self.assertEqual(len(list((staged / "tasks").glob("task_*.json"))), 1)

    def test_artifact_merge_rejects_duplicate_scenarios(self):
        task = {"id": "task_a"}
        spec = {"scenario_id": "scenario_a"}
        contract = {"task_id": "task_a", "scenario_id": "scenario_a"}

        with self.assertRaises(ValueError):
            merge_artifacts(
                [[task], [{"id": "task_b"}]],
                [[spec], [spec]],
                [[contract], [{"task_id": "task_b", "scenario_id": "scenario_a"}]],
            )


class EvaluationPreparationTest(unittest.TestCase):
    def test_final_acceptance_receives_successful_trajectories(self):
        self.assertIn(
            "successful_trajectories",
            inspect.signature(acceptance_errors).parameters,
        )

    def test_final_acceptance_cli_requires_successful_trajectories(self):
        destinations = {
            action.dest for action in build_finalize_parser()._actions
        }

        self.assertIn("successful_trajectories", destinations)

    def test_final_acceptance_validates_successful_training_trajectories(self):
        errors = acceptance_errors(
            tasks=[],
            specs=[],
            contracts=[
                {
                    "task_id": "train_task",
                    "scenario_id": "train_scenario",
                    "split": "train",
                    "business_category": "信用卡选择与申请资格",
                    "task_form": "l1_evidence",
                }
            ],
            allowed_document_ids=set(),
            pool_manifest={
                "task_count": 1,
                "pools": {"sft_candidate": ["train_task"]},
                "valid_autonomous_task_ids": ["train_task"],
                "four_trial_task_ids": [],
                "successful_training_trajectories": 1,
                "result_model_names_verified": True,
            },
            reference_report={},
            isolation_report={},
            model_manifests=[],
            successful_trajectories=[
                {
                    "task_id": "train_task",
                    "data_origin": "benchmark_derived_diagnostic",
                    "training_eligible": False,
                    "reward_info": {"reward": 0.0},
                }
            ],
        )

        self.assertIn("1 successful trajectories have invalid data origin", errors)
        self.assertIn("1 successful trajectories are not training eligible", errors)
        self.assertIn("1 successful trajectories are not reward-one", errors)
        self.assertIn(
            "1 train tasks lack a verified successful training trajectory", errors
        )

    def test_final_acceptance_rejects_duplicate_successful_trajectory_ids(self):
        trajectory = {
            "id": "duplicate",
            "task_id": "train_task",
            "data_origin": "independent_synthetic",
            "training_eligible": True,
            "reward_info": {"reward": 1.0},
        }
        errors = acceptance_errors(
            tasks=[],
            specs=[],
            contracts=[
                {
                    "task_id": "train_task",
                    "scenario_id": "train_scenario",
                    "split": "train",
                    "business_category": "信用卡选择与申请资格",
                    "task_form": "l1_evidence",
                }
            ],
            allowed_document_ids=set(),
            pool_manifest={
                "task_count": 1,
                "pools": {"sft_candidate": ["train_task"]},
                "valid_autonomous_task_ids": ["train_task"],
                "four_trial_task_ids": [],
                "successful_training_trajectories": 2,
                "result_model_names_verified": True,
            },
            reference_report={},
            isolation_report={},
            model_manifests=[],
            successful_trajectories=[trajectory, dict(trajectory)],
        )

        self.assertIn("successful trajectory ids are not unique", errors)

    def test_final_acceptance_requires_four_trials_for_action_train_tasks(self):
        errors = acceptance_errors(
            tasks=[],
            specs=[],
            contracts=[
                {
                    "task_id": "action_task",
                    "scenario_id": "action_scenario",
                    "split": "train",
                    "business_category": "信用卡选择与申请资格",
                    "task_form": "l3_single_action",
                }
            ],
            allowed_document_ids=set(),
            pool_manifest={
                "task_count": 1,
                "pools": {
                    "sft_candidate": ["action_task"],
                    "grpo_frontier": [],
                },
                "valid_autonomous_task_ids": ["action_task"],
                "four_trial_task_ids": [],
                "result_model_names_verified": True,
            },
            reference_report={},
            isolation_report={},
            model_manifests=[],
        )

        self.assertIn(
            "1 action-bearing train tasks lack four valid trials", errors
        )

    def test_final_acceptance_checks_declared_train_quotas(self):
        errors = acceptance_errors(
            tasks=[],
            specs=[],
            contracts=[],
            allowed_document_ids=set(),
            pool_manifest={"task_count": 0, "pools": {}},
            reference_report={},
            isolation_report={},
            model_manifests=[],
        )

        self.assertIn(
            "train business-category quotas differ from the declared v1 quotas",
            errors,
        )
        self.assertIn(
            "train task-form quotas differ from the declared v1 quotas", errors
        )

    def test_final_acceptance_rejects_eval_tasks_in_training_pools(self):
        errors = acceptance_errors(
            tasks=[],
            specs=[],
            contracts=[
                {
                    "task_id": "dev_task",
                    "scenario_id": "dev_scenario",
                    "split": "dev",
                }
            ],
            allowed_document_ids=set(),
            pool_manifest={
                "task_count": 1,
                "pools": {
                    "sft_candidate": ["dev_task"],
                    "synthetic_challenge": ["dev_task"],
                },
            },
            reference_report={},
            isolation_report={},
            model_manifests=[],
        )

        self.assertIn("training pools contain 1 non-train task ids", errors)
        self.assertIn(
            "synthetic challenge pool contains 1 tasks outside challenge", errors
        )

    def test_naturalizer_retries_when_a_locked_title_is_rewritten(self):
        original = (
            "Find [Policy title: Exact: Policy?] for syn_user_train_000001. "
            "When done, respond ###STOP###."
        )
        task = {
            "id": "banking_syn_v1_train_000001_l2_retrieval",
            "user_scenario": {"instructions": original},
            "initial_state": {
                "message_history": [{"role": "user", "content": original}]
            },
        }
        responses = [
            {
                "choices": [{"message": {"content": "Find Exact Policy."}}],
                "usage": {},
            },
            {
                "choices": [{"message": {"content": original}}],
                "usage": {"completion_tokens": 20},
            },
        ]

        with patch(
            "banking_synthetic.naturalize._post_json", side_effect=responses
        ) as request:
            updated, report = naturalize_one(
                task,
                endpoint="http://local/v1",
                model="test-model",
                timeout=1,
                max_attempts=4,
            )

        self.assertEqual(updated["user_scenario"]["instructions"], original)
        self.assertEqual(
            updated["initial_state"]["message_history"][0]["content"], original
        )
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(request.call_count, 2)

    def test_naturalizer_rejects_a_rewritten_requested_outcome(self):
        original = (
            "Please <<Requested outcome: move $125 between synthetic accounts>>. "
            "When done, respond ###STOP###."
        )
        rewritten = (
            "Please <<Requested outcome: move $215 between synthetic accounts>>. "
            "When done, respond ###STOP###."
        )
        task = {
            "id": "banking_syn_v1_train_000002_l4_two_skill",
            "user_scenario": {"instructions": original},
            "initial_state": {
                "message_history": [{"role": "user", "content": original}]
            },
        }

        with patch(
            "banking_synthetic.naturalize._post_json",
            side_effect=[
                {"choices": [{"message": {"content": rewritten}}], "usage": {}},
                {"choices": [{"message": {"content": original}}], "usage": {}},
            ],
        ) as request:
            updated, report = naturalize_one(
                task,
                endpoint="http://local/v1",
                model="test-model",
                timeout=1,
                max_attempts=4,
            )

        self.assertEqual(updated["user_scenario"]["instructions"], original)
        self.assertEqual(
            updated["initial_state"]["message_history"][0]["content"], original
        )
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(request.call_count, 2)

    def test_naturalizer_preserves_private_protocol_while_rewriting_opening(self):
        original = "Please complete the request."
        instructions = (
            original + " "
            "<<User action protocol: Invoke your `submit_referral` tool exactly once.>> "
            "When done, respond ###STOP###."
        )
        rewritten = "Could you complete this request for me?"
        task = {
            "id": "banking_syn_v1_train_000003_l3_single_action",
            "user_scenario": {"instructions": instructions},
            "initial_state": {
                "message_history": [{"role": "user", "content": original}]
            },
        }

        with patch(
            "banking_synthetic.naturalize._post_json",
            return_value={
                "choices": [{"message": {"content": rewritten}}],
                "usage": {},
            },
        ) as request:
            updated, report = naturalize_one(
                task,
                endpoint="http://local/v1",
                model="test-model",
                timeout=1,
                max_attempts=4,
            )

        self.assertEqual(updated["user_scenario"]["instructions"], instructions)
        self.assertEqual(
            updated["initial_state"]["message_history"][0]["content"], rewritten
        )
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(request.call_count, 1)

    def test_naturalizer_rejects_a_rewritten_action_clue(self):
        original = (
            "Share <<Action clue: Agent-side action label `tool_1234` with case parameters {}.>> "
            "and respond ###STOP### when done."
        )
        rewritten = original.replace("tool_1234", "tool_4321")
        task = {
            "id": "banking_syn_v1_train_000004_l4_two_skill",
            "user_scenario": {"instructions": original},
            "initial_state": {
                "message_history": [{"role": "user", "content": original}]
            },
        }

        with patch(
            "banking_synthetic.naturalize._post_json",
            side_effect=[
                {"choices": [{"message": {"content": rewritten}}], "usage": {}},
                {"choices": [{"message": {"content": original}}], "usage": {}},
            ],
        ) as request:
            updated, report = naturalize_one(
                task,
                endpoint="http://local/v1",
                model="test-model",
                timeout=1,
                max_attempts=4,
            )

        self.assertEqual(updated["user_scenario"]["instructions"], original)
        self.assertEqual(
            updated["initial_state"]["message_history"][0]["content"], original
        )
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(request.call_count, 2)

    def test_naturalizer_masks_and_restores_locked_case_data(self):
        original = (
            "Please use <<Action clue: customer-side action label `tool_1234` "
            "with case parameters {}.>> today."
        )
        task = {
            "id": "banking_syn_v1_train_000005_l4_two_skill",
            "user_scenario": {"instructions": original},
            "initial_state": {
                "message_history": [{"role": "user", "content": original}]
            },
        }

        def respond(_url, payload, _timeout):
            prompt = payload["messages"][1]["content"]
            self.assertIn("<PROTECTED_LITERAL_0>", prompt)
            self.assertNotIn("tool_1234", prompt)
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Could you use <PROTECTED_LITERAL_0> today?"
                        }
                    }
                ],
                "usage": {},
            }

        with patch(
            "banking_synthetic.naturalize._post_json", side_effect=respond
        ):
            updated, report = naturalize_one(
                task,
                endpoint="http://local/v1",
                model="test-model",
                timeout=1,
                max_attempts=4,
            )

        self.assertEqual(
            updated["initial_state"]["message_history"][0]["content"],
            "Could you use "
            "<<Action clue: customer-side action label `tool_1234` with case parameters {}.>> "
            "today?",
        )
        self.assertEqual(report["attempts"], 1)

    def test_naturalizer_falls_back_after_four_invalid_rewrites(self):
        original = "Find [Policy title: Exact Policy]."
        task = {
            "id": "banking_syn_v1_train_000006_l1_evidence",
            "user_scenario": {"instructions": original},
            "initial_state": {
                "message_history": [{"role": "user", "content": original}]
            },
        }

        with patch(
            "banking_synthetic.naturalize._post_json",
            return_value={
                "choices": [{"message": {"content": "Tell me about the policy."}}],
                "usage": {},
            },
        ) as request:
            updated, report = naturalize_one(
                task,
                endpoint="http://local/v1",
                model="test-model",
                timeout=1,
                max_attempts=4,
            )

        self.assertEqual(
            updated["initial_state"]["message_history"][0]["content"], original
        )
        self.assertTrue(report["used_fallback"])
        self.assertEqual(report["attempts"], 4)
        self.assertEqual(request.call_count, 4)

    def test_smoke_selection_is_balanced_and_scenario_unique(self):
        catalog = _catalog()
        specs, tasks, contracts = generate(
            phase="pilot", seed=17, catalog=catalog
        )

        selected_tasks, selected_specs, selected_contracts = balanced_selection(
            tasks, specs, contracts, per_category=2
        )

        self.assertEqual(len(selected_tasks), 30)
        self.assertEqual(
            Counter(row["business_category"] for row in selected_contracts),
            Counter({category: 2 for category in BUSINESS_CATEGORIES}),
        )
        self.assertEqual(
            len({row["scenario_id"] for row in selected_specs}), 30
        )
        self.assertEqual(
            {row["task_form"] for row in selected_contracts},
            set(FORM_DOCUMENT_BOUNDS),
        )
        self.assertEqual(
            {row["case_kind"] for row in selected_contracts}, set(CASE_KINDS)
        )
        self.assertEqual(
            {
                row["task_form"]
                for row in selected_contracts
                if row["case_kind"] == "positive" and row["task_form"] in FORM_ACTION_BOUNDS
                and FORM_ACTION_BOUNDS[row["task_form"]][0] > 0
            },
            {"l2_retrieval", "l3_single_action", "l4_two_skill", "l5_medium_workflow", "l6_full_workflow"},
        )

    def test_protocol_recovery_selects_only_failed_or_missing_tasks(self):
        contracts = [
            {"task_id": "task_ok", "retrieval_variant": "bm25"},
            {"task_id": "task_failed", "retrieval_variant": "bm25"},
            {"task_id": "task_missing", "retrieval_variant": "golden_retrieval"},
        ]
        payload = {
            "tasks": [{"id": row["task_id"]} for row in contracts],
            "simulations": [
                {
                    "task_id": "task_ok",
                    "termination_reason": "user_stop",
                    "messages": [{"role": "assistant", "content": "Done."}],
                },
                {
                    "task_id": "task_failed",
                    "termination_reason": "max_steps",
                    "messages": [{"role": "assistant", "content": "Still working."}],
                },
            ],
        }

        selected, report = recovery_selection([payload], contracts)

        self.assertEqual(selected["bm25"], ["task_failed"])
        self.assertEqual(selected["golden_retrieval"], ["task_missing"])
        self.assertEqual(report["recovery_task_count"], 2)

    def test_protocol_recovery_accepts_results_from_a_larger_candidate_pool(self):
        contracts = [{"task_id": "selected", "retrieval_variant": "bm25"}]
        payload = {
            "tasks": [{"id": "selected"}, {"id": "discarded"}],
            "simulations": [
                {
                    "task_id": "selected",
                    "termination_reason": "user_stop",
                    "messages": [{"role": "assistant", "content": "Done."}],
                },
                {
                    "task_id": "discarded",
                    "termination_reason": "user_stop",
                    "messages": [{"role": "assistant", "content": "Done."}],
                },
            ],
        }

        try:
            selected, report = recovery_selection([payload], contracts)
        except ValueError as error:
            self.fail(f"larger result pool was rejected: {error}")

        self.assertEqual(selected["bm25"], [])
        self.assertEqual(report["first_attempt_task_count"], 1)

    def test_protocol_requires_text_in_the_final_assistant_message(self):
        simulation = {
            "termination_reason": "user_stop",
            "messages": [
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "Please complete the task."},
                {"role": "assistant", "content": "", "tool_calls": [{"name": "some_tool"}]},
                {"role": "user", "content": "###STOP###"},
            ],
        }

        self.assertEqual(protocol_failure_reason(simulation), "no_final_message")

    def test_scale_followup_includes_dev_challenge_and_balances_train_outcomes(self):
        contracts = []
        simulations = []
        for index in range(8):
            split = "train" if index < 6 else ("dev" if index == 6 else "challenge")
            task_id = f"task_{index}"
            contracts.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "business_category": f"category_{index % 2}",
                    "task_form": "l4_two_skill",
                    "retrieval_variant": "bm25" if index % 2 else "golden_retrieval",
                }
            )
            simulations.append(
                {
                    "task_id": task_id,
                    "termination_reason": "user_stop",
                    "messages": [{"role": "assistant", "content": "Done."}],
                    "reward_info": {"reward": float(index % 2)},
                }
            )

        selected, report = select_followup(
            contracts, [{"simulations": simulations}], train_count=4
        )

        selected_ids = set(selected["bm25"] + selected["golden_retrieval"])
        self.assertIn("task_6", selected_ids)
        self.assertIn("task_7", selected_ids)
        self.assertEqual(report["selected_train_count"], 4)
        self.assertGreater(report["selected_train_first_success_count"], 0)
        self.assertGreater(report["selected_train_first_failure_count"], 0)

    def test_scale_followup_accepts_results_from_a_larger_candidate_pool(self):
        contracts = [
            {
                "task_id": "selected",
                "split": "train",
                "business_category": "category",
                "task_form": "l4_two_skill",
                "retrieval_variant": "bm25",
            }
        ]
        simulations = [
            {
                "task_id": task_id,
                "termination_reason": "user_stop",
                "messages": [{"role": "assistant", "content": "Done."}],
                "reward_info": {"reward": 1.0},
            }
            for task_id in ("selected", "discarded")
        ]

        try:
            selected, report = select_followup(
                contracts, [{"simulations": simulations}], train_count=1
            )
        except ValueError as error:
            self.fail(f"larger result pool was rejected: {error}")

        self.assertEqual(selected["bm25"], ["selected"])
        self.assertEqual(report["first_trial_task_count"], 1)

    def test_scale_followup_prioritizes_action_bearing_frontier_candidates(self):
        forms = (
            "l1_evidence",
            "l1_evidence",
            "l3_decision",
            "l3_decision",
            "l4_two_skill",
            "l4_two_skill",
            "l5_medium_workflow",
            "l5_medium_workflow",
        )
        contracts = []
        simulations = []
        for index, form in enumerate(forms):
            task_id = f"task_{index}"
            contracts.append(
                {
                    "task_id": task_id,
                    "split": "train",
                    "business_category": "category",
                    "task_form": form,
                    "retrieval_variant": "bm25",
                }
            )
            simulations.append(
                {
                    "task_id": task_id,
                    "termination_reason": "user_stop",
                    "messages": [{"role": "assistant", "content": "Done."}],
                    "reward_info": {"reward": float(index % 2)},
                }
            )

        selected, report = select_followup(
            contracts, [{"simulations": simulations}], train_count=4
        )

        selected_ids = set(selected["bm25"])
        selected_forms = {
            row["task_form"] for row in contracts if row["task_id"] in selected_ids
        }
        self.assertEqual(selected_forms, {"l4_two_skill", "l5_medium_workflow"})
        self.assertEqual(report["selected_train_count"], 4)
        self.assertGreater(report["selected_train_first_success_count"], 0)
        self.assertGreater(report["selected_train_first_failure_count"], 0)

    def test_teacher_selection_uses_only_valid_zero_success_tasks(self):
        contracts = [
            {
                "task_id": "task_success",
                "split": "train",
                "business_category": "category",
                "task_form": "l4_two_skill",
            },
            {
                "task_id": "task_hard",
                "split": "challenge",
                "business_category": "category",
                "task_form": "l6_full_workflow",
            },
        ]
        payload = {
            "simulations": [
                {
                    "task_id": "task_success",
                    "termination_reason": "user_stop",
                    "messages": [{"role": "assistant", "content": "Done."}],
                    "reward_info": {"reward": 1.0},
                },
                {
                    "task_id": "task_hard",
                    "termination_reason": "user_stop",
                    "messages": [{"role": "assistant", "content": "Not solved."}],
                    "reward_info": {"reward": 0.0},
                },
            ]
        }

        selected, report = select_teacher_tasks(contracts, [payload])

        self.assertEqual(selected, ["task_hard"])
        self.assertEqual(report["by_split"], {"challenge": 1})

        selected, report = select_teacher_tasks(
            contracts, [payload], split="train"
        )
        self.assertEqual(selected, [])
        self.assertEqual(report["by_split"], {})

    def test_teacher_selection_accepts_results_from_a_larger_candidate_pool(self):
        contracts = [
            {
                "task_id": "selected",
                "split": "train",
                "business_category": "category",
                "task_form": "l4_two_skill",
            }
        ]
        simulations = [
            {
                "task_id": task_id,
                "termination_reason": "user_stop",
                "messages": [{"role": "assistant", "content": "Not solved."}],
                "reward_info": {"reward": 0.0},
            }
            for task_id in ("selected", "discarded")
        ]

        try:
            selected, report = select_teacher_tasks(
                contracts, [{"simulations": simulations}], split="train"
            )
        except ValueError as error:
            self.fail(f"larger result pool was rejected: {error}")

        self.assertEqual(selected, ["selected"])
        self.assertEqual(report["autonomous_task_count"], 1)

    def test_teacher_requires_a_real_autonomous_user_opening(self):
        task = {
            "id": "task_hard",
            "user_scenario": {"instructions": "Hidden simulator instructions."},
        }

        with self.assertRaisesRegex(ValueError, "valid autonomous User opening"):
            _opening_messages(task, {})

    def test_teacher_requires_the_single_exposed_node_tool(self):
        tools = [{"type": "function", "function": {"name": "KB_search"}}]
        with patch("banking_synthetic.teacher._post_json", return_value={}) as post:
            _completion(
                endpoint="http://teacher.test/v1",
                model="teacher",
                messages=[{"role": "system", "content": "node"}],
                max_tokens=32,
                timeout=5,
                tools=tools,
            )

        payload = post.call_args.args[1]
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "required")

    def test_teacher_task_shards_are_disjoint_and_complete(self):
        task_ids = {f"task_{index:02d}" for index in range(11)}
        shards = [shard_task_ids(task_ids, index, 3) for index in range(3)]

        self.assertEqual(set().union(*shards), task_ids)
        self.assertFalse(shards[0] & shards[1])
        self.assertFalse(shards[0] & shards[2])
        self.assertFalse(shards[1] & shards[2])

    def test_recovery_emits_one_round_per_missing_valid_trial(self):
        contracts = [{"task_id": "task_a", "retrieval_variant": "bm25"}]
        valid = {
            "task_id": "task_a",
            "termination_reason": "user_stop",
            "messages": [{"role": "assistant", "content": "Done."}],
        }
        failed = {
            "task_id": "task_a",
            "termination_reason": "infrastructure_error",
            "messages": [],
        }
        payloads = [
            {"tasks": [{"id": "task_a"}], "simulations": [valid]},
            {"tasks": [{"id": "task_a"}], "simulations": [valid]},
            {"tasks": [{"id": "task_a"}], "simulations": [failed]},
            {"tasks": [{"id": "task_a"}], "simulations": []},
        ]

        selected, report = recovery_selection(payloads, contracts)

        self.assertEqual(selected["bm25"], ["task_a"])
        self.assertEqual(report["recovery_attempt_count"], 2)
        self.assertEqual(report["recovery_round_count"], 2)
        self.assertEqual(
            [row["bm25"] for row in report["round_task_ids"]],
            [["task_a"], ["task_a"]],
        )

    def test_recovery_attempts_do_not_increase_the_required_trial_count(self):
        contracts = [{"task_id": "task_a", "retrieval_variant": "bm25"}]
        failed = {
            "task_id": "task_a",
            "termination_reason": "infrastructure_error",
            "messages": [],
        }
        succeeded = {
            "task_id": "task_a",
            "termination_reason": "user_stop",
            "messages": [{"role": "assistant", "content": "Done."}],
        }
        first = {"tasks": [{"id": "task_a"}], "simulations": [failed]}
        failed_recovery = {
            "tasks": [{"id": "task_a"}],
            "simulations": [failed],
        }
        successful_recovery = {
            "tasks": [{"id": "task_a"}],
            "simulations": [succeeded],
        }

        selected, report = recovery_selection(
            [first], contracts, [failed_recovery]
        )
        self.assertEqual(selected["bm25"], ["task_a"])
        self.assertEqual(report["recovery_attempt_count"], 1)

        selected, report = recovery_selection(
            [first], contracts, [failed_recovery, successful_recovery]
        )
        self.assertEqual(selected["bm25"], [])
        self.assertEqual(report["recovery_attempt_count"], 0)

    def test_no_final_message_is_a_protocol_failure(self):
        simulation = {
            "termination_reason": "user_stop",
            "messages": [
                {
                    "role": "assistant",
                    "content": "\n",
                    "tool_calls": [{"name": "change_user_email"}],
                }
            ],
        }

        self.assertEqual(
            protocol_failure_reason(simulation), "no_final_message"
        )

    def test_length_exhaustion_is_kept_separate_from_business_failure(self):
        simulation = {
            "termination_reason": "user_stop",
            "messages": [
                {
                    "role": "assistant",
                    "content": "partial",
                    "raw_data": {"choices": [{"finish_reason": "length"}]},
                }
            ],
        }

        self.assertEqual(
            protocol_failure_reason(simulation), "output_length_exhausted"
        )

    def test_recovery_trajectory_resolves_prior_protocol_failure_for_pooling(self):
        contract = {
            "task_id": "task_recovered",
            "split": "train",
        }
        failed = {
            "task_id": "task_recovered",
            "termination_reason": "infrastructure_error",
            "messages": [],
            "reward_info": {"reward": 0.0},
        }
        recovered = {
            "task_id": "task_recovered",
            "termination_reason": "user_stop",
            "messages": [{"role": "assistant", "content": "Completed."}],
            "reward_info": {"reward": 1.0},
        }

        manifest, trajectories, failures = build_pools(
            contracts=[contract],
            result_payloads=[{"simulations": [failed]}, {"simulations": [recovered]}],
            teacher_task_ids=set(),
        )

        self.assertEqual(failures, [])
        self.assertEqual(manifest["pool_counts"]["sft_candidate"], 1)
        self.assertEqual(len(trajectories), 1)

    def test_pooling_deduplicates_repeated_simulation_ids(self):
        simulation = {
            "id": "same-simulation",
            "task_id": "task_repeated",
            "termination_reason": "user_stop",
            "messages": [{"role": "assistant", "content": "Completed."}],
            "reward_info": {"reward": 1.0},
        }

        manifest, trajectories, failures = build_pools(
            contracts=[{"task_id": "task_repeated", "split": "train"}],
            result_payloads=[{"simulations": [simulation]}] * 4,
            teacher_task_ids=set(),
        )

        self.assertEqual(failures, [])
        self.assertNotIn("task_repeated", manifest["four_trial_task_ids"])
        self.assertEqual(len(trajectories), 1)

    def test_pooling_rejects_wrong_runtime_model_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts.jsonl"
            contracts.write_text(
                json.dumps({"task_id": "dev_task", "split": "dev"}) + "\n",
                encoding="utf-8",
            )
            results = root / "results.json"
            results.write_text(
                json.dumps(
                    {
                        "info": {
                            "agent_info": {"llm": "openai/agent"},
                            "user_info": {"llm": "openai/user"},
                        },
                        "simulations": [
                            {
                                "id": "simulation",
                                "task_id": "dev_task",
                                "termination_reason": "user_stop",
                                "messages": [
                                    {"role": "assistant", "content": "Done."}
                                ],
                                "reward_info": {"reward": 0.0},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            model_manifest = root / "runtime_models.json"
            model_manifest.write_text(
                json.dumps(
                    {
                        "agent_model_path": "/wrong/agent",
                        "user_model_path": "/wrong/user",
                        "agent_served_model": "agent",
                        "user_served_model": "user",
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "pools.py",
                "--contracts",
                str(contracts),
                "--results",
                str(results),
                "--model-manifest",
                str(model_manifest),
                "--output",
                str(root / "pool.json"),
                "--successful-trajectories",
                str(root / "trajectories.jsonl"),
                "--failures",
                str(root / "failures.json"),
            ]

            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "runtime model path"):
                    pools_main()

    def test_dev_and_challenge_trajectories_never_enter_training_outputs(self):
        contracts = [
            {"task_id": "train_task", "split": "train"},
            {"task_id": "dev_task", "split": "dev"},
            {"task_id": "challenge_task", "split": "challenge"},
        ]
        simulations = [
            {
                "task_id": contract["task_id"],
                "termination_reason": "user_stop",
                "messages": [{"role": "assistant", "content": "Done."}],
                "reward_info": {"reward": 1.0},
            }
            for contract in contracts
        ]

        manifest, trajectories, failures = build_pools(
            contracts=contracts,
            result_payloads=[{"simulations": simulations}],
            teacher_task_ids=set(),
        )

        self.assertEqual(failures, [])
        self.assertEqual(manifest["pools"]["sft_candidate"], ["train_task"])
        self.assertEqual(
            manifest["pools"]["synthetic_challenge"], ["challenge_task"]
        )
        self.assertEqual([row["task_id"] for row in trajectories], ["train_task"])
        self.assertEqual(trajectories[0]["data_origin"], "independent_synthetic")
        self.assertTrue(trajectories[0]["training_eligible"])

    def test_final_pooling_ignores_teacher_rows_for_unselected_candidates(self):
        rows = [
            {"task_id": "selected"},
            {"task_id": "discarded_candidate"},
        ]

        self.assertEqual(
            teacher_trajectories_for_contracts(rows, {"selected"}),
            [{"task_id": "selected"}],
        )

    def test_final_pool_combines_selected_train_with_eval_coverage(self):
        contracts = [
            {"task_id": "train_task", "split": "train"},
            {"task_id": "dev_task", "split": "dev"},
            {"task_id": "challenge_task", "split": "challenge"},
        ]
        base_manifest = {
            "pools": {
                "sft_candidate": ["train_task", "discarded_train"],
                "grpo_frontier": ["train_task"],
                "behavior_anchor": ["discarded_train"],
                "hard_sft_only": [],
                "synthetic_challenge": [],
            },
            "valid_autonomous_task_ids": ["train_task", "discarded_train"],
            "four_trial_task_ids": ["train_task", "discarded_train"],
            "result_model_names_verified": True,
            "result_file_count": 42,
        }
        eval_manifest = {
            "pools": {
                "sft_candidate": [],
                "grpo_frontier": [],
                "behavior_anchor": [],
                "hard_sft_only": [],
                "synthetic_challenge": ["challenge_task"],
            },
            "valid_autonomous_task_ids": ["dev_task", "challenge_task"],
            "four_trial_task_ids": ["dev_task", "challenge_task"],
            "result_model_names_verified": True,
            "result_file_count": 8,
        }
        trajectories = [
            {"task_id": "train_task", "reward_info": {"reward": 1.0}},
            {"task_id": "discarded_train", "reward_info": {"reward": 1.0}},
        ]

        manifest, selected_trajectories, failures = combine_final_pool(
            contracts=contracts,
            base_manifest=base_manifest,
            base_trajectories=trajectories,
            eval_manifest=eval_manifest,
        )

        self.assertEqual(failures, [])
        self.assertEqual(manifest["task_count"], 3)
        self.assertEqual(manifest["pools"]["sft_candidate"], ["train_task"])
        self.assertEqual(manifest["pools"]["grpo_frontier"], ["train_task"])
        self.assertEqual(manifest["pools"]["behavior_anchor"], [])
        self.assertEqual(
            manifest["pools"]["synthetic_challenge"], ["challenge_task"]
        )
        self.assertEqual(
            manifest["valid_autonomous_task_ids"],
            ["challenge_task", "dev_task", "train_task"],
        )
        self.assertEqual(manifest["four_trial_task_ids"], manifest["valid_autonomous_task_ids"])
        self.assertEqual(manifest["result_file_count"], 50)
        self.assertTrue(manifest["result_model_names_verified"])
        self.assertEqual(
            selected_trajectories,
            [{"task_id": "train_task", "reward_info": {"reward": 1.0}}],
        )

    def test_protocol_failure_is_excluded_from_business_pass_rate(self):
        contract = {"required_documents": []}
        rows = [
            {
                "contract": contract,
                "simulation": {
                    "task_id": "valid",
                    "termination_reason": "user_stop",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "Done.",
                            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                        }
                    ],
                    "reward_info": {"reward": 1.0},
                },
            },
            {
                "contract": contract,
                "simulation": {
                    "task_id": "protocol",
                    "termination_reason": "infrastructure_error",
                    "messages": [],
                    "reward_info": {"reward": 0.0},
                },
            },
        ]

        metrics = _group_metrics(rows)

        self.assertEqual(metrics["simulations"], 2)
        self.assertEqual(metrics["valid_simulations"], 1)
        self.assertEqual(metrics["pass_at_1"], 1.0)
        self.assertEqual(metrics["first_protocol_failure_rate"], 0.5)
        self.assertEqual(metrics["mean_prompt_tokens"], 5)
        self.assertEqual(metrics["mean_completion_tokens"], 1)
        self.assertEqual(metrics["mean_total_tokens"], 6)

    def test_metric_compaction_discards_full_message_payloads(self):
        simulation = {
            "task_id": "task",
            "termination_reason": "user_stop",
            "messages": [
                {
                    "role": "assistant",
                    "content": "large payload " * 1000,
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            ],
            "reward_info": {"reward": 1.0},
        }
        contract = {
            "required_documents": [],
            "business_category": "category",
            "difficulty": "L1",
            "template_id": "template",
        }

        compact = _compact_row(simulation, contract, "first")

        self.assertNotIn("simulation", compact)
        self.assertNotIn("messages", compact)
        self.assertNotIn("large payload", repr(compact))
        self.assertEqual(compact["prompt_tokens"], 7)
        self.assertEqual(compact["completion_tokens"], 3)
        self.assertEqual(_group_metrics([compact])["pass_at_1"], 1.0)

    def test_metric_summary_ignores_unselected_candidate_results(self):
        payload = {
            "info": {
                "agent_info": {"llm": "openai/agent"},
                "user_info": {"llm": "openai/user"},
            },
            "simulations": [
                {
                    "task_id": task_id,
                    "termination_reason": "user_stop",
                    "messages": [{"role": "assistant", "content": "Done."}],
                    "reward_info": {"reward": 1.0},
                }
                for task_id in ("selected", "discarded")
            ],
        }
        contract = {
            "task_id": "selected",
            "required_documents": [],
            "business_category": "category",
            "difficulty": "L1",
            "template_id": "template",
        }
        manifest = {
            "agent_model_path": "/mnt/afs/models/Qwen3.8-27B",
            "user_model_path": "/mnt/afs/models/Qwen3.8-27B",
            "agent_served_model": "agent",
            "user_served_model": "user",
            "attempt_kind": "first",
        }
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "results.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            report = summarize([result_path], [contract], [manifest])

        self.assertEqual(report["overall"]["tasks"], 1)
        self.assertEqual(report["overall"]["simulations"], 1)

    def test_qwen38_runtime_configuration_is_fixed(self):
        script = (
            ANALYSIS_DIR / "banking_synthetic/run_qwen38_eval.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('MODEL_PATH="/mnt/afs/models/Qwen3.8-27B"', script)
        self.assertIn('AGENT_REPLICA_CUDA_GROUPS="0;1;2"', script)
        self.assertIn('USER_REPLICA_CUDA_GROUPS="3"', script)
        self.assertIn('AGENT_TEMPERATURE="1.0"', script)
        self.assertIn('AGENT_TOP_P="0.95"', script)
        self.assertIn('AGENT_MAX_TOKENS="${AGENT_TOKEN_LIMIT}"', script)
        self.assertIn('--context-length 262144', script)
        self.assertIn('USER_TEMPERATURE="0.0"', script)
        self.assertIn('USER_MAX_TOKENS="1024"', script)
        self.assertIn('MAX_RETRIES="0"', script)
        self.assertEqual(script.count('reasoning_effort'), 1)

    def test_reference_and_teacher_environments_use_the_document_allowlist(self):
        validator = (
            ANALYSIS_DIR / "banking_synthetic/validate.py"
        ).read_text(encoding="utf-8")
        teacher_script = (
            ANALYSIS_DIR / "banking_synthetic/run_qwen38_teacher.sh"
        ).read_text(encoding="utf-8")

        self.assertNotIn("KnowledgeBase.load(documents_dir)", validator)
        self.assertIn("allowed_document_ids", validator)
        self.assertIn('--allowlist "${EXPERIMENT_DIR}/allowed_documents.json"', teacher_script)

    def test_only_isolation_module_has_a_benchmark_task_input(self):
        package = ANALYSIS_DIR / "banking_synthetic"
        prohibited = (
            "generate.py",
            "naturalize.py",
            "teacher.py",
            "prepare_eval.py",
        )

        for filename in prohibited:
            text = (package / filename).read_text(encoding="utf-8")
            self.assertNotIn("benchmark-tasks-dir", text)
            self.assertNotIn("source_task_id", text)

    def test_eval_selection_can_filter_split_and_retrieval_variant(self):
        tasks = [
            {"id": "dev_bm25"},
            {"id": "dev_golden"},
            {"id": "challenge_bm25"},
        ]
        contracts = [
            {"task_id": "dev_bm25", "split": "dev", "retrieval_variant": "bm25"},
            {
                "task_id": "dev_golden",
                "split": "dev",
                "retrieval_variant": "golden_retrieval",
            },
            {
                "task_id": "challenge_bm25",
                "split": "challenge",
                "retrieval_variant": "bm25",
            },
        ]

        selected, _ = select_eval_contracts(
            tasks,
            contracts,
            retrieval_variant="bm25",
            split="challenge",
            shard_index=0,
            num_shards=1,
            limit=None,
            task_ids=None,
        )
        self.assertEqual([task["id"] for task in selected], ["challenge_bm25"])


if __name__ == "__main__":
    unittest.main()
