import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


NUM_GPUS = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "examples/tau2-bench/analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from banking_task_curriculum import (  # noqa: E402
    CATEGORIES,
    CURRICULUM_USER_FOLLOWUP_INSTRUCTIONS,
    CURRICULUM_USER_TOOL_INSTRUCTIONS,
    EXPANSION_EXCLUDED_SOURCES,
    EXPECTED_AGGREGATE_DRIFT,
    PILOT_SOURCES,
    PILOT_VARIANTS,
    build_annotation,
    build_systematic_variants,
    build_pilots,
    category_lookup,
    changed_paths,
    concise_intent,
    document_fact_locator,
    graph_errors,
    normalize_snapshot,
    parse_tool_types,
    salient_document_fact,
    validate_expanded_curriculum,
    validate_pilots,
)
from prepare_banking_curriculum_eval import (  # noqa: E402
    build_manifest,
    select_tasks,
)
from summarize_banking_curriculum_eval import summarize_results  # noqa: E402
from build_banking_curriculum_pools import build_pool_manifest  # noqa: E402
from scale_banking_curriculum import (  # noqa: E402
    build_scaled_curriculum,
    retrieval_anchor,
    validate_scaled_curriculum,
)


def _action(action_id, name, *, requestor="assistant", arguments=None):
    return {
        "action_id": action_id,
        "name": name,
        "requestor": requestor,
        "arguments": arguments or {},
    }


def _source(task_id, required_documents, actions):
    return {
        "id": task_id,
        "description": {"purpose": task_id, "notes": f"Reviewed intent for {task_id}."},
        "user_scenario": {"persona": None, "instructions": f"Complete {task_id}."},
        "initial_state": {
            "initialization_data": None,
            "initialization_actions": None,
            "message_history": None,
        },
        "evaluation_criteria": {
            "actions": actions,
            "env_assertions": None,
            "communicate_info": None,
            "nl_assertions": None,
            "reward_basis": ["DB"],
        },
        "required_documents": required_documents,
        "user_tools": None,
    }


def _pilot_sources():
    task_001 = _source(
        "task_001",
        [
            "doc_credit_cards_gold_rewards_card_001",
            "doc_credit_cards_silver_rewards_card_001",
            "doc_credit_cards_bronze_rewards_card_001",
            "doc_credit_cards_platinum_rewards_card_001",
        ],
        [
            _action(
                "001_0",
                "apply_for_credit_card",
                requestor="user",
                arguments={
                    "card_type": "Gold Rewards Card",
                    "customer_name": "Sarah Bosch",
                    "annual_income": 100000,
                    "rho_bank_subscription": True,
                },
            )
        ],
    )
    task_031 = _source(
        "task_031",
        [
            "doc_credit_cards_credit_cards_(general)_013",
            "doc_credit_cards_credit_cards_(general)_014",
            "doc_credit_cards_credit_cards_(general)_015",
        ],
        [
            _action("031_0", "log_verification"),
            _action("031_1", "give_discoverable_user_tool"),
            _action("031_2", "call_discoverable_user_tool", requestor="user"),
            _action("031_3", "unlock_discoverable_agent_tool"),
            _action("031_4", "call_discoverable_agent_tool"),
        ],
    )
    task_036 = _source(
        "task_036",
        ["doc_credit_cards_credit_card_replacements_001"],
        [
            _action("036_0", "log_verification"),
            _action("036_1", "unlock_discoverable_agent_tool"),
            _action("036_2", "call_discoverable_agent_tool"),
        ],
    )
    task_071_docs = [
        "doc_bank_accounts_bank_accounts_(general)_003",
        "doc_bank_accounts_bank_accounts_(general)_004",
        "doc_bank_accounts_bank_accounts_(general)_009",
        "doc_bank_accounts_bank_accounts_(general)_013",
        "doc_bank_accounts_bank_accounts_(general)_014",
        "doc_bank_accounts_bank_accounts_(general)_015",
        "doc_bank_accounts_bank_accounts_(general)_016",
        "doc_business_checking_accounts_sky_blue_001",
        "doc_business_checking_accounts_sky_blue_007",
        "doc_business_checking_accounts_sky_blue_010",
        "doc_business_checking_accounts_lime_green_001",
        "doc_business_checking_accounts_lime_green_007",
        "doc_business_checking_accounts_hunter_green_001",
        "doc_business_checking_accounts_hunter_green_010",
        "doc_business_savings_accounts_gold_saver_account_001",
        "doc_business_savings_accounts_gold_saver_account_006",
        "doc_business_savings_accounts_silver_plus_saver_001",
        "doc_business_savings_accounts_silver_plus_saver_006",
        "doc_business_savings_accounts_gold_plus_saver_001",
        "doc_business_savings_accounts_gold_plus_saver_006",
    ]
    task_071 = _source(
        "task_071",
        task_071_docs,
        [
            _action("071_0", "log_verification"),
            _action("071_1", "unlock_discoverable_agent_tool"),
            _action("071_2", "call_discoverable_agent_tool"),
            _action("071_3", "unlock_discoverable_agent_tool"),
            _action("071_4", "call_discoverable_agent_tool"),
            _action("071_5", "call_discoverable_agent_tool"),
        ],
    )
    return {task["id"]: task for task in (task_001, task_031, task_071, task_036)}


class CategoryContractTest(unittest.TestCase):
    def test_categories_cover_exactly_97_unique_runtime_ids(self):
        lookup = category_lookup()

        self.assertEqual(len(lookup), 97)
        self.assertEqual(Counter(lookup.values()), Counter({name: len(ids) for name, ids in CATEGORIES.items()}))
        self.assertEqual(lookup["task_001"], "信用卡选择与申请资格")
        self.assertEqual(lookup["task_102"], "账户推荐奖励")

    def test_expected_aggregate_drift_is_explicit(self):
        self.assertEqual(len(EXPECTED_AGGREGATE_DRIFT), 13)
        self.assertIn("task_102", EXPECTED_AGGREGATE_DRIFT)

    def test_expansion_exclusions_match_environment_replay_diagnostics(self):
        self.assertEqual(set(EXPANSION_EXCLUDED_SOURCES), {"task_051", "task_077", "task_083"})


class SourceCompatibilityTest(unittest.TestCase):
    def test_snapshot_comparison_ignores_serialization_defaults_but_reports_state_changes(self):
        historical = {"id": "task_058", "ticket": None, "annotations": {"x": 1}}
        runtime = {
            "id": "task_058",
            "initial_state": {"initialization_data": {"agent_data": {"users": {"data": {"u": {}}}}}},
        }

        self.assertEqual(normalize_snapshot({"id": "x", "ticket": None}), {"id": "x"})
        self.assertEqual(
            changed_paths(historical, runtime),
            ["/initial_state"],
        )

    def test_tool_types_are_read_from_both_decorator_kinds(self):
        source = """
@is_tool(ToolType.READ)
def lookup(self):
    pass

@is_discoverable_tool(ToolType.WRITE)
def mutate(self):
    pass
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tools.py"
            path.write_text(source, encoding="utf-8")
            result = parse_tool_types(path)

        self.assertEqual(result, {"lookup": "read", "mutate": "write"})


class AnnotationContractTest(unittest.TestCase):
    def test_intent_and_fact_selection_use_customer_constraints(self):
        task = _source("task_004", ["doc_a"], [_action("a0", "transfer_to_human_agents")])
        task["description"]["notes"] = None
        task["user_scenario"]["instructions"] = (
            "You are playing a customer. Your character wants to change an email address. "
            "The customer cannot complete identity verification and needs a human."
        )
        document = {
            "title": "Transfer reasons",
            "content": (
                "| Reason | Use |\n"
                "|---|---|\n"
                "| unrelated_offer | An advertised offer cannot be found |\n"
                "| account_ownership_dispute | Identity verification failures require a specialist |"
            ),
        }

        intent = concise_intent(task)
        fact = salient_document_fact(document, set(re.findall(r"[a-z_]+", task["user_scenario"]["instructions"].lower())))

        self.assertIn("wants to change", intent)
        self.assertEqual(
            fact,
            "account_ownership_dispute: Identity verification failures require a specialist",
        )
        self.assertEqual(
            document_fact_locator(document, fact),
            "the table row whose first cell is `account_ownership_dispute`",
        )

    def test_annotation_keeps_business_evidence_and_success_separate(self):
        task = _source(
            "task_001",
            ["doc_a"],
            [_action("a0", "apply_for_credit_card", requestor="user", arguments={"card_type": "Gold"})],
        )
        documents = {
            "doc_a": {
                "id": "doc_a",
                "title": "Gold card terms",
                "content": "Annual fee: $0.00\nCash back on all purchases: 2.5%",
            }
        }

        row = build_annotation(task, "信用卡选择与申请资格", documents, {"apply_for_credit_card": "write"})

        self.assertEqual(row["evidence"]["required_facts"][0]["source_document_ids"], ["doc_a"])
        self.assertIn("Annual fee", row["evidence"]["required_facts"][0]["statement"])
        self.assertEqual(
            row["evidence"]["decisive_set_status"],
            "conservative_upper_bound_from_benchmark_gold",
        )
        self.assertEqual(row["success"]["reward_basis"], ["DB"])
        self.assertEqual(row["success"]["final_environment_assertions"], [])
        self.assertEqual(row["success"]["checkpoints"][0]["action_refs"], ["a0"])
        self.assertEqual(row["workflow"]["user_actions"], 1)
        self.assertEqual(graph_errors(row), [])

    def test_graph_validator_rejects_cycles(self):
        row = {
            "task_id": "x",
            "graph": {
                "nodes": [
                    {"id": "a", "kind": "evidence", "depends_on": ["b"]},
                    {"id": "b", "kind": "decision", "depends_on": ["a"]},
                ]
            },
        }

        self.assertTrue(any("cycle" in error for error in graph_errors(row)))


class PilotContractTest(unittest.TestCase):
    def test_four_sources_generate_all_six_variants(self):
        tasks, contracts = build_pilots(_pilot_sources())

        self.assertEqual(len(tasks), 24)
        self.assertEqual(len(contracts), 24)
        self.assertEqual(Counter(item["source_task_id"] for item in contracts), Counter({key: 6 for key in PILOT_SOURCES}))
        for source_id in PILOT_SOURCES:
            self.assertEqual(
                {item["variant"] for item in contracts if item["source_task_id"] == source_id},
                set(PILOT_VARIANTS),
            )

    def test_variant_reward_and_initialization_contracts_are_not_conflated(self):
        tasks, contracts = build_pilots(_pilot_sources())
        by_id = {task["id"]: task for task in tasks}

        retrieval = by_id["banking_curriculum_task_001_retrieval_only"]
        self.assertEqual(retrieval["evaluation_criteria"]["reward_basis"], ["ACTION", "COMMUNICATE"])
        self.assertEqual(retrieval["evaluation_criteria"]["actions"][0]["compare_args"], [])

        single = by_id["banking_curriculum_task_031_single_action"]
        self.assertEqual(len(single["initial_state"]["initialization_actions"]), 4)
        self.assertEqual(len(single["evaluation_criteria"]["actions"]), 1)
        self.assertEqual(single["evaluation_criteria"]["reward_basis"], ["DB", "ACTION"])

        full = by_id["banking_curriculum_task_071_full_composition"]
        self.assertEqual(full["evaluation_criteria"]["reward_basis"], ["DB"])
        self.assertEqual(len(full["evaluation_criteria"]["actions"]), 6)

        decision = by_id["banking_curriculum_task_071_decision_only"]
        composition = by_id["banking_curriculum_task_071_two_skill_composition"]
        self.assertEqual(len(decision["required_documents"]), 2)
        self.assertEqual(len(composition["required_documents"]), 2)

        documents = {
            document_id: {"id": document_id, "title": document_id, "content": "evidence"}
            for task in tasks
            for document_id in task.get("required_documents") or []
        }
        errors, summary = validate_pilots(tasks, contracts, documents)
        self.assertEqual(errors, [])
        self.assertEqual(summary["by_variant"], {variant: 4 for variant in PILOT_VARIANTS})

    def test_atomic_tasks_seed_agent_request_without_exact_gold_label(self):
        tasks, contracts = build_pilots(_pilot_sources())
        task_by_id = {task["id"]: task for task in tasks}

        for item in contracts:
            task = task_by_id[item["task_id"]]
            if item["variant"] == "full_composition":
                self.assertIsNone(task["initial_state"]["message_history"])
                continue

            history = task["initial_state"]["message_history"]
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["role"], "user")
            self.assertEqual(
                task["user_scenario"]["instructions"],
                (
                    CURRICULUM_USER_TOOL_INSTRUCTIONS
                    if task.get("user_tools")
                    else CURRICULUM_USER_FOLLOWUP_INSTRUCTIONS
                ),
            )
            for expected in item["success_contract"]["required_response_facts"]:
                self.assertNotIn(expected, history[0]["content"])

        retrieval = task_by_id["banking_curriculum_task_001_retrieval_only"]
        self.assertIn("Document: <document_id>", retrieval["initial_state"]["message_history"][0]["content"])
        decision = task_by_id["banking_curriculum_task_001_decision_only"]
        self.assertIn("Recommendation: <product_name>", decision["initial_state"]["message_history"][0]["content"])


class ExpandedCurriculumTest(unittest.TestCase):
    def test_clean_source_projects_to_six_bounded_variants(self):
        source = _source(
            "task_004",
            ["doc_profile"],
            [
                _action(
                    "004_0",
                    "change_user_email",
                    arguments={"user_id": "user_1", "new_email": "new@example.com"},
                )
            ],
        )
        source["user_tools"] = ["apply_for_credit_card"]
        documents = {
            "doc_profile": {
                "id": "doc_profile",
                "title": "Changing a customer email",
                "content": "Identity verification is required before changing a customer email address.",
            }
        }
        annotation = build_annotation(
            source,
            "身份验证与资料变更",
            documents,
            {"change_user_email": "write"},
        )

        tasks, contracts = build_systematic_variants(source, annotation, documents)
        errors, summary = validate_expanded_curriculum(
            tasks,
            contracts,
            {source["id"]: source},
            documents,
        )

        self.assertEqual(errors, [])
        self.assertEqual(summary["expanded_tasks"], 6)
        self.assertEqual({item["variant"] for item in contracts}, set(PILOT_VARIANTS))
        task_by_id = {item["id"]: item for item in tasks}
        by_variant = {item["variant"]: task_by_id[item["task_id"]] for item in contracts}
        self.assertEqual(len(by_variant["evidence_given"]["required_documents"]), 1)
        self.assertEqual(len(by_variant["decision_only"]["required_documents"]), 1)
        for variant in ("evidence_given", "decision_only"):
            self.assertIn(
                '"change user email"',
                by_variant[variant]["initial_state"]["message_history"][0]["content"],
            )
        self.assertIn(
            "prose line 1 under section `document start`",
            by_variant["evidence_given"]["initial_state"]["message_history"][0]["content"],
        )
        retrieval_query = by_variant["retrieval_only"]["evaluation_criteria"]["actions"][0]["arguments"]["query"]
        self.assertIn("Identity verification is required", retrieval_query)
        self.assertEqual(len(by_variant["single_action"]["evaluation_criteria"]["actions"]), 1)
        self.assertEqual(by_variant["single_action"]["user_tools"], [])
        self.assertEqual(by_variant["two_skill_composition"]["user_tools"], [])
        self.assertEqual(
            by_variant["full_composition"]["evaluation_criteria"],
            source["evaluation_criteria"],
        )


class CurriculumEvalPreparationTest(unittest.TestCase):
    def test_retrieval_filter_and_sharding_keep_sources_together(self):
        tasks = []
        contracts = []
        for source_index in range(6):
            source_id = f"task_{source_index:03d}"
            for variant, retrieval_variant in (
                ("evidence_given", "golden_retrieval"),
                ("retrieval_only", "bm25"),
                ("full_composition", "bm25"),
            ):
                task_id = f"generated_{source_id}_{variant}"
                tasks.append({"id": task_id})
                contracts.append(
                    {
                        "task_id": task_id,
                        "source_task_id": source_id,
                        "variant": variant,
                        "level": "L1",
                        "business_category": "category",
                        "retrieval_variant": retrieval_variant,
                    }
                )

        shard_zero = select_tasks(
            tasks,
            contracts,
            retrieval_variant="bm25",
            shard_index=0,
            num_shards=2,
        )
        shard_one = select_tasks(
            tasks,
            contracts,
            retrieval_variant="bm25",
            shard_index=1,
            num_shards=2,
        )
        source_sets = [
            {item["source_task_id"] for item in selected_contracts}
            for _, selected_contracts in (shard_zero, shard_one)
        ]

        self.assertFalse(source_sets[0] & source_sets[1])
        self.assertEqual(source_sets[0] | source_sets[1], {f"task_{i:03d}" for i in range(6)})
        for selected_tasks, selected_contracts in (shard_zero, shard_one):
            self.assertEqual(len(selected_tasks), 6)
            self.assertEqual({item["variant"] for item in selected_contracts}, {"retrieval_only", "full_composition"})

        manifest = build_manifest(
            *shard_zero,
            retrieval_variant="bm25",
            shard_index=0,
            num_shards=2,
        )
        self.assertEqual(manifest["task_count"], 6)
        self.assertEqual(manifest["source_count"], 3)

    def test_result_summary_reports_task_and_curriculum_accuracy(self):
        contracts = [
            {
                "task_id": "generated_a",
                "source_task_id": "task_001",
                "variant": "evidence_given",
                "level": "L1",
                "business_category": "cards",
                "retrieval_variant": "golden_retrieval",
            },
            {
                "task_id": "generated_b",
                "source_task_id": "task_002",
                "variant": "retrieval_only",
                "level": "L2",
                "business_category": "cards",
                "retrieval_variant": "bm25",
            },
        ]
        payload = {
            "info": {"agent_info": {"llm": "openai/Qwen3.6-27B"}},
            "simulations": [
                {
                    "task_id": "generated_a",
                    "termination_reason": "user_stop",
                    "reward_info": {
                        "reward": 1,
                        "reward_breakdown": {"COMMUNICATE": 1},
                        "communicate_checks": [{"met": True}],
                    },
                },
                {
                    "task_id": "generated_b",
                    "termination_reason": "max_steps",
                    "reward_info": {
                        "reward": 0,
                        "reward_breakdown": {"ACTION": 0, "COMMUNICATE": 1},
                        "action_checks": [
                            {
                                "action_match": False,
                                "action": {"requestor": "user"},
                            }
                        ],
                        "communicate_checks": [{"met": True}],
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir) / "results.json"
            results_path.write_text(json.dumps(payload), encoding="utf-8")
            run_log = Path(tmp_dir) / "run.log"
            run_log.write_text(
                "Task generated_a failed (attempt 1/4): AssistantMessage must have either content or tool_calls.\n",
                encoding="utf-8",
            )
            summary = summarize_results(
                [results_path],
                contracts,
                expected_task_count=2,
                run_logs=[run_log],
            )
            replacement_path = Path(tmp_dir) / "replacement.json"
            replacement_path.write_text(
                json.dumps(
                    {
                        "info": payload["info"],
                        "simulations": [
                            {
                                "task_id": "generated_b",
                                "termination_reason": "user_stop",
                                "reward_info": {
                                    "reward": 1,
                                    "reward_breakdown": {"ACTION": 1},
                                    "action_checks": [
                                        {
                                            "action_match": True,
                                            "action": {"requestor": "user"},
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            recovered_summary = summarize_results(
                [results_path],
                contracts,
                replacement_results_paths=[replacement_path],
                expected_task_count=2,
            )

        self.assertEqual(summary["tasks"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["by_variant"]["evidence_given"]["accuracy"], 1.0)
        self.assertEqual(summary["by_variant"]["retrieval_only"]["accuracy"], 0.0)
        self.assertEqual(summary["termination_reasons"], {"max_steps": 1, "user_stop": 1})
        self.assertEqual(
            summary["atomic_checks"],
            {
                "ACTION": {"passed": 0, "checks": 1, "pass_rate": 0.0},
                "COMMUNICATE": {"passed": 2, "checks": 2, "pass_rate": 1.0},
            },
        )
        self.assertEqual(
            summary["action_checks_by_requestor"],
            {"user": {"passed": 0, "checks": 1, "pass_rate": 0.0}},
        )
        self.assertEqual(summary["retry_diagnostics"]["events"], 1)
        self.assertEqual(summary["retry_diagnostics"]["tasks"], 1)
        self.assertEqual(summary["retry_diagnostics"]["first_attempt_correct"], 0)
        self.assertEqual(summary["retry_diagnostics"]["first_attempt_accuracy"], 0.0)
        self.assertEqual(recovered_summary["correct"], 2)
        self.assertEqual(recovered_summary["replacement_task_ids"], ["generated_b"])
        self.assertEqual(
            recovered_summary["atomic_checks"],
            {
                "ACTION": {"passed": 1, "checks": 1, "pass_rate": 1.0},
                "COMMUNICATE": {"passed": 1, "checks": 1, "pass_rate": 1.0},
            },
        )

    def test_benchmark_derived_manifest_quarantines_all_variants_from_training(self):
        contracts = [
            {
                "task_id": "retrieval",
                "source_task_id": "task_001",
                "variant": "retrieval_only",
                "level": "L2",
                "business_category": "cards",
                "retrieval_variant": "bm25",
            },
            {
                "task_id": "action",
                "source_task_id": "task_002",
                "variant": "single_action",
                "level": "L3",
                "business_category": "cards",
                "retrieval_variant": "golden_retrieval",
            },
            {
                "task_id": "composition",
                "source_task_id": "task_003",
                "variant": "two_skill_composition",
                "level": "L4",
                "business_category": "cards",
                "retrieval_variant": "golden_retrieval",
            },
            {
                "task_id": "full",
                "source_task_id": "task_004",
                "variant": "full_composition",
                "level": "L6",
                "business_category": "cards",
                "retrieval_variant": "golden_retrieval",
            },
        ]
        payload = {
            "info": {"agent_info": {"llm": "openai/Qwen3.8-27B"}},
            "simulations": [
                {
                    "id": "sim-retrieval",
                    "task_id": "retrieval",
                    "termination_reason": "user_stop",
                    "reward_info": {"reward": 1},
                },
                {
                    "id": "sim-action",
                    "task_id": "action",
                    "termination_reason": "user_stop",
                    "reward_info": {"reward": 0},
                },
                {
                    "id": "sim-composition",
                    "task_id": "composition",
                    "termination_reason": "infrastructure_error",
                    "reward_info": None,
                },
                {
                    "id": "sim-full",
                    "task_id": "full",
                    "termination_reason": "user_stop",
                    "reward_info": {"reward": 1},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = build_pool_manifest(
                [path], contracts, expected_task_count=4
            )
            first_recovery_path = Path(tmp_dir) / "first_recovery.json"
            first_recovery_path.write_text(
                json.dumps(
                    {
                        "info": payload["info"],
                        "simulations": [payload["simulations"][2]],
                    }
                ),
                encoding="utf-8",
            )
            final_recovery_path = Path(tmp_dir) / "final_recovery.json"
            final_recovery_path.write_text(
                json.dumps(
                    {
                        "info": payload["info"],
                        "simulations": [
                            {
                                "id": "sim-composition-recovered",
                                "task_id": "composition",
                                "termination_reason": "user_stop",
                                "reward_info": {"reward": 1},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            recovered_manifest = build_pool_manifest(
                [path],
                contracts,
                replacement_results_paths=[
                    first_recovery_path,
                    final_recovery_path,
                ],
                expected_task_count=4,
            )

        self.assertEqual(manifest["verified_success_trajectories"], 2)
        self.assertEqual(manifest["training_eligible_trajectories"], 0)
        self.assertFalse(manifest["training_eligible"])
        self.assertEqual(
            manifest["task_pool_counts"],
            {
                "benchmark_derived_diagnostic": 3,
                "infrastructure_recovery": 1,
            },
        )
        by_id = {item["task_id"]: item for item in manifest["records"]}
        self.assertFalse(by_id["retrieval"]["sft_opd_candidate"])
        self.assertFalse(by_id["action"]["sft_opd_candidate"])
        self.assertFalse(by_id["full"]["sft_opd_candidate"])
        self.assertTrue(all(not item["training_eligible"] for item in by_id.values()))
        recovered_by_id = {
            item["task_id"]: item for item in recovered_manifest["records"]
        }
        self.assertEqual(recovered_manifest["verified_success_trajectories"], 3)
        self.assertEqual(recovered_manifest["training_eligible_trajectories"], 0)
        self.assertEqual(recovered_manifest["replacement_task_ids"], ["composition"])
        self.assertFalse(recovered_by_id["composition"]["sft_opd_candidate"])
        self.assertEqual(
            recovered_by_id["composition"]["task_pool"],
            "benchmark_derived_diagnostic",
        )
        self.assertEqual(
            recovered_by_id["composition"]["results_file"],
            str(final_recovery_path),
        )


class ScaledCurriculumTest(unittest.TestCase):
    def test_retrieval_anchor_uses_indexed_body_text_not_only_title(self):
        document = {
            "title": "Product title",
            "content": (
                "## Key figures\n\n"
                "| Item | Value |\n"
                "|---|---|\n"
                "| Minimum daily balance to maintain benefits | $112,500 |"
            ),
        }

        self.assertEqual(
            retrieval_anchor(document, "Incoming domestic wire fee: $0.00"),
            "Minimum daily balance to maintain benefits | $112,500",
        )

    def test_scale_expands_each_fact_and_reference_action(self):
        source = _source(
            "task_001",
            ["doc_a"],
            [
                _action("a0", "log_verification", arguments={"name": "A"}),
                _action("a1", "write_account", arguments={"account_id": "x"}),
            ],
        )
        documents = {
            "doc_a": {
                "id": "doc_a",
                "title": "Account policy",
                "content": "Eligible accounts must remain open for 30 days.",
            }
        }
        annotations = {
            "task_001": {
                "business": {"primary_category": "信用卡选择与申请资格"},
                "evidence": {
                    "required_facts": [
                        {
                            "id": "fact_01",
                            "source_document_ids": ["doc_a"],
                            "statement": "Eligible accounts must remain open for 30 days.",
                        }
                    ]
                },
            }
        }

        tasks, contracts = build_scaled_curriculum(
            {"task_001": source}, annotations, documents
        )
        errors, summary = validate_scaled_curriculum(
            tasks, contracts, {"task_001": source}, annotations, documents
        )
        task_by_id = {task["id"]: task for task in tasks}

        self.assertEqual(errors, [])
        self.assertEqual(summary["scaled_tasks"], 4)
        self.assertEqual(
            summary["tasks_by_variant"],
            {"evidence_given": 1, "retrieval_only": 1, "single_action": 2},
        )
        retrieval = task_by_id["banking_scale_v2_task_001_fact_01_retrieval_only"]
        self.assertIn(
            "indexed-body search anchor",
            retrieval["initial_state"]["message_history"][0]["content"],
        )
        second_action = task_by_id["banking_scale_v2_task_001_action_a1"]
        self.assertEqual(
            second_action["initial_state"]["initialization_actions"][0]["func_name"],
            "log_verification",
        )


if __name__ == "__main__":
    unittest.main()
