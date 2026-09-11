import csv
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path


NUM_GPUS = 0

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "examples/tau2-bench/analysis"
RUNNER = ANALYSIS_DIR / "run_atomic_gap_analysis.sh"
sys.path.insert(0, str(ANALYSIS_DIR))

from atomic_gap_analysis import (  # noqa: E402
    build_case_packet,
    build_judge_messages,
    build_official_replay_request,
    build_replay_manifest,
    build_replay_review_queue,
    classify_outcome,
    classify_replay_evidence,
    extract_banking_features,
    extract_trace_features,
    extend_agent_review_queue,
    index_simulations,
    mandatory_agent_review_reasons,
    make_chat_token_counter,
    make_replay_request_fitter,
    merge_adjudications,
    normalize_adjudication,
    pair_run_indexes,
    parse_json_object,
    prepare_analysis,
    request_chat_completion,
    resolve_results_path,
    route_judge_payload,
    run_judge_reviews,
    run_replays,
    select_replay_cases,
    load_official_tool_schemas,
    select_canonical_case,
    summarize_tasks,
    summarize_replay_adjudications,
    validate_atomic_review,
    write_formal_report,
)


def _simulation(task_id, trial, reward):
    return {
        "task_id": str(task_id),
        "trial": trial,
        "reward_info": {"reward": reward},
        "messages": [],
    }


class OutcomePairingTest(unittest.TestCase):
    def test_classify_outcome_preserves_the_five_contrast_groups(self):
        cases = [
            ((0, 0, 1), "stable_gap"),
            ((0, 0, 0), "shared_hard"),
            ((0, 1, 0), "base_unstable"),
            ((1, 0, 1), "base_unstable"),
            ((1, 1, 0), "reverse_control"),
            ((1, 1, 1), "all_success"),
        ]

        for rewards, expected in cases:
            with self.subTest(rewards=rewards):
                self.assertEqual(classify_outcome(*rewards), expected)

    def test_pair_run_indexes_rejects_missing_or_duplicate_scenario_cells(self):
        complete = {
            "simulations": [_simulation("1", 0, 0), _simulation("1", 1, 1)]
        }
        base_a = index_simulations("airline", "base_a", complete, "base-a.json")
        base_b = index_simulations("airline", "base_b", complete, "base-b.json")
        strong = index_simulations("airline", "strong", complete, "strong.json")

        paired = pair_run_indexes(base_a, base_b, strong)

        self.assertEqual([row["case_id"] for row in paired], ["airline/1/0", "airline/1/1"])
        self.assertEqual(
            [row["outcome_class"] for row in paired],
            ["shared_hard", "all_success"],
        )

        missing = index_simulations(
            "airline",
            "strong",
            {"simulations": [_simulation("1", 0, 0)]},
            "strong-missing.json",
        )
        with self.assertRaisesRegex(ValueError, "scenario cells differ"):
            pair_run_indexes(base_a, base_b, missing)

        with self.assertRaisesRegex(ValueError, "duplicate scenario cell airline/1/0"):
            index_simulations(
                "airline",
                "base_a",
                {"simulations": [_simulation("1", 0, 0), _simulation("1", 0, 1)]},
                "base-a-duplicate.json",
            )

    def test_task_summary_uses_eight_baseline_and_four_strong_trials(self):
        base_a = []
        base_b = []
        strong = []
        for trial in range(4):
            base_a.append(_simulation("gap", trial, 1 if trial == 0 else 0))
            base_b.append(_simulation("gap", trial, 1 if trial == 1 else 0))
            strong.append(_simulation("gap", trial, 1 if trial < 3 else 0))

            base_a.append(_simulation("unstable", trial, 1 if trial < 3 else 0))
            base_b.append(_simulation("unstable", trial, 1 if trial < 2 else 0))
            strong.append(_simulation("unstable", trial, 1))

        rows = pair_run_indexes(
            index_simulations("telecom", "base_a", {"simulations": base_a}, "a"),
            index_simulations("telecom", "base_b", {"simulations": base_b}, "b"),
            index_simulations("telecom", "strong", {"simulations": strong}, "s"),
        )

        summaries = {row["task_id"]: row for row in summarize_tasks(rows)}

        self.assertEqual(summaries["gap"]["baseline_successes"], 2)
        self.assertEqual(summaries["gap"]["strong_successes"], 3)
        self.assertTrue(summaries["gap"]["high_confidence_gap"])
        self.assertEqual(summaries["unstable"]["baseline_successes"], 5)
        self.assertFalse(summaries["unstable"]["high_confidence_gap"])

    def test_canonical_case_uses_semantic_precedence_then_lowest_trial(self):
        rows = [
            {"case_id": "retail/7/3", "trial": 3, "outcome_class": "base_unstable"},
            {"case_id": "retail/7/2", "trial": 2, "outcome_class": "stable_gap"},
            {"case_id": "retail/7/1", "trial": 1, "outcome_class": "stable_gap"},
            {"case_id": "retail/7/0", "trial": 0, "outcome_class": "shared_hard"},
        ]

        selected = select_canonical_case(rows)

        self.assertEqual(selected["case_id"], "retail/7/1")


class EvidencePacketTest(unittest.TestCase):
    def test_trace_features_expose_expected_actions_and_observed_tool_failures(self):
        task = {
            "evaluation_criteria": {
                "actions": [
                    {
                        "requestor": "assistant",
                        "name": "change_plan",
                        "arguments": {"line_id": "line-7", "plan": "premium"},
                    }
                ],
                "nl_assertions": ["Explain the new monthly price."],
            }
        }
        simulation = {
            "task_id": "9",
            "trial": 0,
            "termination_reason": "user_stop",
            "reward_info": {
                "reward": 0,
                "db_check": {"db_match": False},
                "action_checks": [
                    {"action_match": False, "action": task["evaluation_criteria"]["actions"][0]}
                ],
            },
            "messages": [
                {"role": "user", "content": "Please move line-7 to premium.", "turn_idx": 1},
                {
                    "role": "assistant",
                    "content": "",
                    "turn_idx": 2,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "change_plan",
                            "arguments": {"line_id": "line-8", "plan": "premium"},
                            "requestor": "assistant",
                        }
                    ],
                },
                {
                    "id": "call-1",
                    "role": "tool",
                    "content": "line not found",
                    "error": True,
                    "turn_idx": 3,
                },
            ],
        }

        features = extract_trace_features("telecom", simulation, task)

        self.assertEqual(features["expected_agent_actions"][0]["name"], "change_plan")
        self.assertEqual(features["observed_tools"][0]["arguments"]["line_id"], "line-8")
        self.assertTrue(features["observed_tools"][0]["error"])
        self.assertEqual(features["golden_matched"], 0)
        self.assertFalse(features["db_match"])

    def test_banking_features_record_queries_document_hits_and_discovery_chain(self):
        task = {
            "required_documents": ["doc_gold_001", "doc_silver_001"],
            "evaluation_criteria": {"actions": []},
        }
        simulation = {
            "task_id": "task_001",
            "trial": 0,
            "reward_info": {"reward": 0},
            "messages": [
                {
                    "role": "assistant",
                    "turn_idx": 2,
                    "tool_calls": [
                        {
                            "id": "search-1",
                            "name": "KB_search",
                            "arguments": {"query": "gold and silver annual fees"},
                            "requestor": "assistant",
                        }
                    ],
                },
                {
                    "id": "search-1",
                    "role": "tool",
                    "turn_idx": 3,
                    "error": False,
                    "content": (
                        "1. Gold card\n   ID: doc_gold_001\n   Score: 8.5\n"
                        "2. Bronze card\n   ID: doc_bronze_001\n   Score: 5.0\n"
                    ),
                },
                {
                    "role": "assistant",
                    "turn_idx": 4,
                    "tool_calls": [
                        {
                            "id": "discover-1",
                            "name": "unlock_discoverable_agent_tool",
                            "arguments": {"tool_name": "apply_card_4821"},
                            "requestor": "assistant",
                        }
                    ],
                },
                {
                    "id": "discover-1",
                    "role": "tool",
                    "turn_idx": 5,
                    "error": False,
                    "content": "Unlocked apply_card_4821",
                },
            ],
        }

        features = extract_banking_features(simulation, task)

        self.assertEqual(features["retrieval_events"][0]["query"], "gold and silver annual fees")
        self.assertEqual(
            features["retrieval_events"][0]["document_ids"],
            ["doc_gold_001", "doc_bronze_001"],
        )
        self.assertEqual(features["required_document_hits"], ["doc_gold_001"])
        self.assertEqual(features["missing_required_documents"], ["doc_silver_001"])
        self.assertEqual(
            features["discovery_events"][0]["name"],
            "unlock_discoverable_agent_tool",
        )

    def test_case_packet_blinds_identity_and_routes_oversized_evidence_without_truncation(self):
        task = {
            "id": "7",
            "description": {"purpose": "Change an address"},
            "user_scenario": {"instructions": "Use the verified address."},
            "evaluation_criteria": {"actions": []},
        }
        result = {"simulations": [_simulation("7", 0, 0)], "tasks": [task]}
        paired = pair_run_indexes(
            index_simulations("retail", "base_a", result, "a.json"),
            index_simulations("retail", "base_b", result, "b.json"),
            index_simulations(
                "retail",
                "strong",
                {"simulations": [_simulation("7", 0, 1)], "tasks": [task]},
                "s.json",
            ),
        )[0]

        packet = build_case_packet(paired, task, ordinal=0)
        judge_text = str(packet["judge_payload"])

        self.assertEqual(packet["trace_labels"], {"A": "base_b", "B": "strong"})
        self.assertNotIn("base_b", judge_text)
        self.assertNotIn("strong", judge_text)
        self.assertNotIn("reward", judge_text)

        captured_messages = []

        def count_local(messages):
            captured_messages.extend(messages)
            return 100

        local_route = route_judge_payload(
            packet["judge_payload"], count_local, context_tokens=128, output_tokens=20
        )
        agent_route = route_judge_payload(
            packet["judge_payload"], lambda _text: 109, context_tokens=128, output_tokens=20
        )
        self.assertEqual(local_route, {"route": "local", "input_tokens": 100})
        self.assertEqual(agent_route, {"route": "agent", "input_tokens": 109})
        self.assertEqual(captured_messages[0]["role"], "system")
        self.assertEqual(captured_messages[1]["role"], "user")
        self.assertEqual(packet["judge_payload"]["task"]["description"], task["description"])

    def test_review_validation_rejects_unknown_gap_and_nonexistent_turn(self):
        valid_review = {
            "case_id": "airline/2/0",
            "preferred_trace": "B",
            "primary_gap": "action_selection",
            "first_divergence": {"A": 4, "B": 6},
            "expected_next_action": "Read the latest reservation.",
            "observed_failure": "Trace A read an unrelated reservation.",
            "corrective_mechanism": "Trace B bound the latest reservation ID.",
            "external_factor": "agent",
            "confidence": "clear",
            "evidence": [{"trace": "A", "turn": 4, "reason": "wrong ID"}],
        }

        self.assertEqual(validate_atomic_review(valid_review, {4, 6}), [])

        invalid = dict(valid_review)
        invalid["primary_gap"] = "more_tools_is_better"
        invalid["first_divergence"] = {"A": 99, "B": 6}
        errors = validate_atomic_review(invalid, {4, 6})
        self.assertIn("invalid primary_gap", errors)
        self.assertIn("first_divergence.A: invalid turn", errors)


class PrepareAnalysisTest(unittest.TestCase):
    def test_mandatory_review_reasons_cover_contrast_controls_without_gating_diagnostics(self):
        reverse = {
            "domain": "retail",
            "outcome_class": "reverse_control",
            "rewards": {"base_b": 1, "strong": 0},
        }
        banking_positive = {
            "domain": "banking_knowledge",
            "outcome_class": "base_unstable",
            "rewards": {"base_b": 0, "strong": 1},
        }

        self.assertEqual(
            mandatory_agent_review_reasons(
                reverse, high_confidence_gap=False, local_route=True
            ),
            ["reverse_control"],
        )
        self.assertEqual(
            mandatory_agent_review_reasons(
                banking_positive, high_confidence_gap=False, local_route=True
            ),
            ["banking_positive_contrast"],
        )
        self.assertEqual(
            mandatory_agent_review_reasons(
                reverse, high_confidence_gap=True, local_route=False
            ),
            ["high_confidence_gap", "over_context", "reverse_control"],
        )

    def test_resolve_results_path_uses_the_summary_path_below_simulations_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            simulations_root = Path(tmp_dir) / "data/simulations"
            expected = simulations_root / "run_airline_test_4trials/results.json"
            expected.parent.mkdir(parents=True)
            expected.write_text("{}", encoding="utf-8")

            resolved = resolve_results_path(
                "data/simulations/run_airline_test_4trials/results.json",
                simulations_root,
            )

            self.assertEqual(resolved, expected)

    def test_prepare_queues_one_canonical_case_per_high_confidence_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            simulations_root = root / "data/simulations"
            summaries = {}
            for run_label, reward in {
                "base_a": 0,
                "base_b": 0,
                "strong": 1,
            }.items():
                run_dir = simulations_root / f"{run_label}_airline_test_4trials"
                run_dir.mkdir(parents=True)
                (run_dir / "results.json").write_text(
                    json.dumps(
                        {
                            "tasks": [
                                {
                                    "id": "gap",
                                    "description": {"purpose": "book a flight"},
                                    "evaluation_criteria": {"actions": []},
                                }
                            ],
                            "simulations": [
                                _simulation("gap", trial, reward)
                                for trial in range(4)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                summary_path = root / f"{run_label}_summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "domains": {
                                "airline": {
                                    "results_file": (
                                        f"data/simulations/{run_dir.name}/results.json"
                                    )
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                summaries[run_label] = summary_path

            output_dir = root / "analysis"
            inventory = prepare_analysis(
                base_a_summary=summaries["base_a"],
                base_b_summary=summaries["base_b"],
                strong_summary=summaries["strong"],
                simulations_root=simulations_root,
                output_dir=output_dir,
                token_counter=lambda _messages: 50,
                context_tokens=128,
                output_tokens=20,
            )

            queued = [
                json.loads(line)
                for line in (output_dir / "agent_review_queue.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(inventory["high_confidence_tasks"], 1)
            self.assertEqual(inventory["mandatory_agent_reviews"], 1)
            self.assertEqual([row["case_id"] for row in queued], ["airline/gap/0"])
            self.assertEqual(queued[0]["review_reasons"], ["high_confidence_gap"])

    def test_prepare_writes_task_weighted_packets_and_separates_long_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            simulations_root = root / "data/simulations"
            summaries = {}
            for run_label, rewards in {
                "base_a": [0, 0],
                "base_b": [0, 1],
                "strong": [1, 1],
            }.items():
                run_dir = simulations_root / f"{run_label}_airline_test_4trials"
                run_dir.mkdir(parents=True)
                tasks = [
                    {
                        "id": "short",
                        "description": {"purpose": "short task"},
                        "evaluation_criteria": {"actions": []},
                    },
                    {
                        "id": "long",
                        "description": {"purpose": "oversized task"},
                        "evaluation_criteria": {"actions": []},
                    },
                ]
                simulations = [
                    _simulation("short", 0, rewards[0]),
                    _simulation("long", 0, rewards[1]),
                ]
                (run_dir / "results.json").write_text(
                    json.dumps({"tasks": tasks, "simulations": simulations}),
                    encoding="utf-8",
                )
                summary_path = root / f"{run_label}_summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "domains": {
                                "airline": {
                                    "results_file": (
                                        f"data/simulations/{run_dir.name}/results.json"
                                    )
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                summaries[run_label] = summary_path

            output_dir = root / "analysis"

            inventory = prepare_analysis(
                base_a_summary=summaries["base_a"],
                base_b_summary=summaries["base_b"],
                strong_summary=summaries["strong"],
                simulations_root=simulations_root,
                output_dir=output_dir,
                token_counter=lambda messages: (
                    200 if "oversized task" in str(messages) else 50
                ),
                context_tokens=128,
                output_tokens=20,
            )

            self.assertEqual(inventory["scenario_cells"], 2)
            self.assertEqual(inventory["tasks"], 2)
            self.assertEqual(inventory["routes"], {"agent": 1, "local": 1})
            self.assertEqual(
                inventory["by_domain"]["airline"],
                {
                    "base_a_successes": 0,
                    "base_b_successes": 1,
                    "cells": 2,
                    "outcomes": {"base_unstable": 1, "stable_gap": 1},
                    "strong_successes": 2,
                    "tasks": 2,
                },
            )

            with (output_dir / "paired_outcomes.csv").open(encoding="utf-8") as handle:
                paired_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["outcome_class"] for row in paired_rows],
                ["base_unstable", "stable_gap"],
            )

            case_packets = [
                json.loads(line)
                for line in (output_dir / "case_packets.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            local_payloads = (output_dir / "judge_payloads.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            agent_queue = (output_dir / "agent_review_queue.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(case_packets), 2)
            self.assertEqual(len(local_payloads), 1)
            self.assertEqual(len(agent_queue), 1)
            self.assertEqual(json.loads(agent_queue[0])["task_id"], "long")
            self.assertEqual(json.loads(agent_queue[0])["review_reasons"], ["over_context"])
            self.assertEqual(
                len(
                    (output_dir / "calibration_payloads.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                1,
            )

            stored_inventory = json.loads(
                (output_dir / "inventory.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored_inventory, inventory)


class LocalJudgeTest(unittest.TestCase):
    def test_parse_json_object_accepts_fenced_structured_output(self):
        parsed = parse_json_object('```json\n{"case_id":"retail/4/0","confidence":"clear"}\n```')

        self.assertEqual(parsed["case_id"], "retail/4/0")
        self.assertEqual(parsed["confidence"], "clear")

    def test_judge_messages_explain_first_divergence_without_outcome_identity(self):
        payload = {
            "case_id": "telecom/3/0",
            "domain": "telecom",
            "task": {"description": {"purpose": "Restore data"}},
            "traces": {"A": {"conversation": []}, "B": {"conversation": []}},
        }

        messages = build_judge_messages(payload)
        rendered = json.dumps(messages, ensure_ascii=False)

        self.assertIn("first causal divergence", rendered)
        self.assertIn("telecom/3/0", rendered)
        self.assertNotIn("Qwen", rendered)
        self.assertNotIn("reward", rendered.lower())

    def test_run_judge_reviews_keeps_valid_rows_and_routes_invalid_output(self):
        payload = {
            "case_id": "airline/2/0",
            "domain": "airline",
            "task": {},
            "traces": {
                "A": {"features": {"valid_turns": [2, 4]}, "conversation": []},
                "B": {"features": {"valid_turns": [2, 6]}, "conversation": []},
            },
        }
        valid = {
            "case_id": "airline/2/0",
            "preferred_trace": "B",
            "primary_gap": "entity_state_binding",
            "first_divergence": {"A": 4, "B": 6},
            "expected_next_action": "Read the latest reservation.",
            "observed_failure": "A used the older reservation.",
            "corrective_mechanism": "B selected the latest reservation ID.",
            "external_factor": "agent",
            "confidence": "clear",
            "evidence": [{"trace": "A", "turn": 4, "reason": "old ID"}],
        }
        responses = iter([json.dumps(valid), "not-json"])

        reviews = run_judge_reviews(
            [payload],
            complete=lambda _messages, _schema: next(responses),
            primary_passes=2,
            concurrency=1,
        )

        self.assertEqual(len(reviews), 2)
        self.assertEqual(reviews[0]["status"], "valid")
        self.assertEqual(reviews[0]["review"]["primary_gap"], "entity_state_binding")
        self.assertEqual(reviews[1]["status"], "agent_review")
        self.assertIn("invalid JSON response", reviews[1]["errors"][0])
        self.assertEqual([row["pass_index"] for row in reviews], [0, 1])

    def test_agent_queue_adds_uncertain_invalid_and_calibration_disagreement_reasons(self):
        packets = [
            {"case_id": "airline/1/0", "task_id": "1"},
            {"case_id": "retail/2/0", "task_id": "2"},
        ]
        qwen_reviews = [
            {
                "case_id": "airline/1/0",
                "pass_index": 0,
                "status": "valid",
                "review": {"confidence": "uncertain"},
                "errors": [],
            },
            {
                "case_id": "retail/2/0",
                "pass_index": 0,
                "status": "agent_review",
                "review": None,
                "errors": ["invalid JSON response"],
            },
        ]
        calibration_reviews = [
            {
                "case_id": "airline/1/0",
                "pass_index": 0,
                "status": "valid",
                "review": {
                    "primary_gap": "action_selection",
                    "preferred_trace": "B",
                },
                "errors": [],
            },
            {
                "case_id": "airline/1/0",
                "pass_index": 1,
                "status": "valid",
                "review": {
                    "primary_gap": "argument_composition",
                    "preferred_trace": "B",
                },
                "errors": [],
            },
        ]

        queue = extend_agent_review_queue(
            base_queue=[],
            packets=packets,
            qwen_reviews=qwen_reviews,
            calibration_reviews=calibration_reviews,
        )

        self.assertEqual(
            {row["case_id"]: row["review_reasons"] for row in queue},
            {
                "airline/1/0": ["calibration_disagreement", "local_uncertain"],
                "retail/2/0": ["local_invalid"],
            },
        )


class ReplayDesignTest(unittest.TestCase):
    def test_normalize_adjudication_maps_blind_trace_labels_back_to_runs(self):
        packet = {
            "case_id": "retail/7/0",
            "domain": "retail",
            "task_id": "7",
            "trial": 0,
            "trace_labels": {"A": "strong", "B": "base_b"},
            "input_tokens": 800,
            "high_confidence_gap": True,
            "run_success": {"base_b": 0, "strong": 1},
            "sources": {
                "base_b": {"source_path": "base.json", "simulation_index": 4},
                "strong": {"source_path": "strong.json", "simulation_index": 9},
            },
        }
        review = {
            "case_id": "retail/7/0",
            "preferred_trace": "A",
            "primary_gap": "argument_composition",
            "first_divergence": {"A": 8, "B": 6},
            "expected_next_action": "Use the verified item id.",
            "observed_failure": "The weaker trace used the parent item id.",
            "corrective_mechanism": "The better trace selected the line-item id.",
            "external_factor": "agent",
            "confidence": "clear",
            "evidence": [{"trace": "B", "turn": 6, "reason": "wrong id"}],
        }

        normalized = normalize_adjudication(packet, review, reviewer="agent")

        self.assertEqual(normalized["preferred_run"], "strong")
        self.assertEqual(normalized["first_divergence"], {"base_b": 6, "strong": 8})
        self.assertEqual(normalized["reviewer"], "agent")
        self.assertEqual(normalized["sources"]["base_b"]["simulation_index"], 4)

    def test_select_replay_cases_uses_top_distinct_gaps_then_shortest_fill(self):
        rows = [
            {
                "case_id": "airline/a/0",
                "domain": "airline",
                "task_id": "a",
                "primary_gap": "action_selection",
                "confidence": "clear",
                "first_divergence": {"base_b": 6, "strong": 8},
                "high_confidence_gap": True,
                "run_success": {"base_b": 0, "strong": 1},
                "input_tokens": 900,
            },
            {
                "case_id": "airline/b/0",
                "domain": "airline",
                "task_id": "b",
                "primary_gap": "action_selection",
                "confidence": "clear",
                "first_divergence": {"base_b": 4, "strong": 4},
                "high_confidence_gap": True,
                "run_success": {"base_b": 0, "strong": 1},
                "input_tokens": 500,
            },
            {
                "case_id": "airline/c/0",
                "domain": "airline",
                "task_id": "c",
                "primary_gap": "entity_state_binding",
                "confidence": "clear",
                "first_divergence": {"base_b": 10, "strong": 12},
                "high_confidence_gap": True,
                "run_success": {"base_b": 0, "strong": 1},
                "input_tokens": 700,
            },
            {
                "case_id": "airline/d/0",
                "domain": "airline",
                "task_id": "d",
                "primary_gap": "workflow_precondition_authorization",
                "confidence": "clear",
                "first_divergence": {"base_b": 2, "strong": 2},
                "high_confidence_gap": True,
                "run_success": {"base_b": 0, "strong": 1},
                "input_tokens": 600,
            },
            {
                "case_id": "banking_knowledge/z/0",
                "domain": "banking_knowledge",
                "task_id": "z",
                "primary_gap": "evidence_sufficiency",
                "confidence": "clear",
                "first_divergence": {"base_b": 4, "strong": 6},
                "high_confidence_gap": False,
                "run_success": {"base_b": 0, "strong": 1},
                "input_tokens": 1000,
            },
        ]

        selected = select_replay_cases(rows, per_domain=3)

        self.assertEqual(
            [row["case_id"] for row in selected if row["domain"] == "airline"],
            ["airline/b/0", "airline/c/0", "airline/d/0"],
        )
        self.assertEqual(
            [row["case_id"] for row in selected if row["domain"] == "banking_knowledge"],
            ["banking_knowledge/z/0"],
        )

    def test_replay_manifest_expands_two_models_two_prefixes_and_four_seeds(self):
        selected = [
            {
                "case_id": "telecom/11/0",
                "domain": "telecom",
                "task_id": "11",
                "primary_gap": "observation_grounded_recovery",
                "first_divergence": {"base_b": 6, "strong": 10},
                "sources": {
                    "base_b": {"source_path": "base.json", "simulation_index": 2},
                    "strong": {"source_path": "strong.json", "simulation_index": 5},
                },
            }
        ]

        manifest = build_replay_manifest(selected, seeds=(300, 301, 302, 303))

        self.assertEqual(len(manifest), 16)
        self.assertEqual(
            {(row["model"], row["prefix"]) for row in manifest},
            {
                ("qwen3", "weak"),
                ("qwen3", "strong"),
                ("qwen35", "weak"),
                ("qwen35", "strong"),
            },
        )
        weak = next(
            row
            for row in manifest
            if row["model"] == "qwen3" and row["prefix"] == "weak" and row["seed"] == 300
        )
        self.assertEqual(weak["prefix_turn"], 6)
        self.assertEqual(weak["source_path"], "base.json")

    def test_official_replay_request_reconstructs_only_agent_visible_prefix(self):
        simulation = {
            "policy": "Always verify the account before changing it.",
            "messages": [
                {"role": "assistant", "content": "Hello", "turn_idx": 0},
                {"role": "user", "content": "Change account 7", "turn_idx": 1},
                {
                    "role": "assistant",
                    "content": "",
                    "turn_idx": 2,
                    "tool_calls": [
                        {
                            "id": "call-7",
                            "name": "get_account",
                            "arguments": {"account_id": "7"},
                            "requestor": "assistant",
                        }
                    ],
                },
                {
                    "id": "call-7",
                    "role": "tool",
                    "content": "account 7",
                    "error": False,
                    "turn_idx": 3,
                },
                {
                    "role": "user",
                    "content": "",
                    "turn_idx": 4,
                    "tool_calls": [
                        {
                            "id": "user-call",
                            "name": "confirm_change",
                            "arguments": {},
                            "requestor": "user",
                        }
                    ],
                },
                {"role": "assistant", "content": "wrong next action", "turn_idx": 6},
            ],
        }
        tools = [
            {
                "type": "function",
                "function": {"name": "get_account", "parameters": {"type": "object"}},
            }
        ]

        request = build_official_replay_request(
            simulation,
            prefix_turn=6,
            tool_schemas=tools,
            model="tau2-replay",
            seed=301,
            max_tokens=1200,
            enable_thinking=None,
        )

        self.assertIn("<policy>\nAlways verify the account", request["messages"][0]["content"])
        self.assertEqual([item["role"] for item in request["messages"]], [
            "system", "assistant", "user", "assistant", "tool"
        ])
        self.assertEqual(
            request["messages"][3]["tool_calls"][0]["function"]["arguments"],
            '{"account_id": "7"}',
        )
        self.assertEqual(request["messages"][4]["tool_call_id"], "call-7")
        self.assertEqual(request["tools"], tools)
        self.assertEqual(request["seed"], 301)
        self.assertNotIn("extra_body", request)

    def test_replay_evidence_distinguishes_local_upstream_and_compound_effects(self):
        self.assertEqual(
            classify_replay_evidence(
                {"qwen3_weak": 1, "qwen35_weak": 3, "qwen3_strong": 1}
            ),
            "local_decision",
        )
        self.assertEqual(
            classify_replay_evidence(
                {"qwen3_weak": 1, "qwen35_weak": 1, "qwen3_strong": 3}
            ),
            "upstream_evidence_state",
        )
        self.assertEqual(
            classify_replay_evidence(
                {"qwen3_weak": 0, "qwen35_weak": 2, "qwen3_strong": 2}
            ),
            "compound",
        )
        self.assertEqual(
            classify_replay_evidence(
                {"qwen3_weak": 2, "qwen35_weak": 2, "qwen3_strong": 1}
            ),
            "uncertain",
        )


class ModelExecutionTest(unittest.TestCase):
    def test_replay_fitter_renders_openai_history_with_mapping_arguments(self):
        class MappingArgumentsTokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
                if not isinstance(arguments, dict):
                    raise TypeError("tool-call arguments must be a mapping")
                return [1, 2, 3]

        messages = [
            {"role": "user", "content": "Look up order 7."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id": "7"}',
                        },
                    }
                ],
            },
        ]
        request = {"messages": messages, "tools": [], "max_tokens": 5}

        fits, input_tokens = make_replay_request_fitter(
            MappingArgumentsTokenizer(), context_tokens=16
        )(request)

        self.assertTrue(fits)
        self.assertEqual(input_tokens, 3)
        self.assertIsInstance(
            request["messages"][1]["tool_calls"][0]["function"]["arguments"],
            str,
        )

    def test_token_helpers_render_chat_templates_and_reserve_generation_budget(self):
        class FakeTokenizer:
            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                token_ids = list(range(90))
                if kwargs.get("return_dict") is False:
                    return token_ids
                return {"input_ids": token_ids, "attention_mask": [1] * 90}

        tokenizer = FakeTokenizer()
        messages = [{"role": "user", "content": "hello"}]

        counter = make_chat_token_counter(tokenizer, enable_thinking=False)
        self.assertEqual(counter(messages), 90)
        self.assertTrue(tokenizer.calls[0][1]["tokenize"])
        self.assertTrue(tokenizer.calls[0][1]["add_generation_prompt"])
        self.assertFalse(tokenizer.calls[0][1]["enable_thinking"])
        self.assertFalse(tokenizer.calls[0][1]["return_dict"])

        fitter = make_replay_request_fitter(tokenizer, context_tokens=100)
        fits, input_tokens = fitter(
            {
                "messages": messages,
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "max_tokens": 10,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        self.assertTrue(fits)
        self.assertEqual(input_tokens, 90)
        self.assertFalse(tokenizer.calls[1][1]["return_dict"])

        too_large, _ = fitter(
            {"messages": messages, "tools": [], "max_tokens": 11}
        )
        self.assertFalse(too_large)

    def test_load_official_tool_schemas_uses_bm25_only_for_banking(self):
        calls = []

        class Tool:
            def __init__(self, name):
                self.openai_schema = {"type": "function", "function": {"name": name}}

        class Environment:
            def __init__(self, domain, kwargs):
                self.domain = domain
                self.kwargs = kwargs

            def get_tools(self):
                return [Tool(f"{self.domain}_lookup")]

        class Registry:
            def get_env_constructor(self, domain):
                def construct(**kwargs):
                    calls.append((domain, kwargs))
                    return Environment(domain, kwargs)

                return construct

        schemas = load_official_tool_schemas(
            ["airline", "banking_knowledge"],
            retrieval_config="bm25",
            registry=Registry(),
        )

        self.assertEqual(calls, [("airline", {}), ("banking_knowledge", {"retrieval_variant": "bm25"})])
        self.assertEqual(
            schemas["banking_knowledge"][0]["function"]["name"],
            "banking_knowledge_lookup",
        )

    def test_chat_completion_uses_openai_endpoint_and_returns_assistant_message(self):
        captured = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                captured["path"] = self.path
                captured["authorization"] = self.headers.get("Authorization")
                length = int(self.headers["Content-Length"])
                captured["body"] = json.loads(self.rfile.read(length))
                response = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "done",
                                "tool_calls": None,
                            }
                        }
                    ]
                }
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        request_body = {
            "model": "local-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        message = request_chat_completion(
            f"http://127.0.0.1:{server.server_port}/v1",
            request_body,
            api_key="local-key",
            timeout=5,
        )

        self.assertEqual(captured["path"], "/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer local-key")
        self.assertEqual(captured["body"], request_body)
        self.assertEqual(message["content"], "done")

    def test_merge_adjudications_requires_agent_for_mandatory_cases_and_maps_clear_local(self):
        packets = [
            {
                "case_id": "airline/1/0",
                "domain": "airline",
                "task_id": "1",
                "trial": 0,
                "trace_labels": {"A": "base_b", "B": "strong"},
                "input_tokens": 100,
                "high_confidence_gap": True,
                "run_success": {"base_b": 0, "strong": 1},
                "sources": {},
                "judge_payload": {
                    "traces": {
                        "A": {"features": {"valid_turns": [2]}},
                        "B": {"features": {"valid_turns": [4]}},
                    }
                },
            },
            {
                "case_id": "retail/2/0",
                "domain": "retail",
                "task_id": "2",
                "trial": 0,
                "trace_labels": {"A": "strong", "B": "base_b"},
                "input_tokens": 120,
                "high_confidence_gap": False,
                "run_success": {"base_b": 0, "strong": 1},
                "sources": {},
                "judge_payload": {
                    "traces": {
                        "A": {"features": {"valid_turns": [6]}},
                        "B": {"features": {"valid_turns": [4]}},
                    }
                },
            },
        ]

        def review(case_id, a_turn, b_turn, gap):
            return {
                "case_id": case_id,
                "preferred_trace": "A",
                "primary_gap": gap,
                "first_divergence": {"A": a_turn, "B": b_turn},
                "expected_next_action": "Do the grounded next step.",
                "observed_failure": "The other trace diverged.",
                "corrective_mechanism": "This trace used observed state.",
                "external_factor": "agent",
                "confidence": "clear",
                "evidence": [{"trace": "A", "turn": a_turn, "reason": "grounded"}],
            }

        qwen_reviews = [
            {
                "case_id": "airline/1/0",
                "pass_index": 0,
                "status": "valid",
                "review": review("airline/1/0", 2, 4, "action_selection"),
                "errors": [],
            },
            {
                "case_id": "retail/2/0",
                "pass_index": 0,
                "status": "valid",
                "review": review("retail/2/0", 6, 4, "entity_state_binding"),
                "errors": [],
            },
        ]

        merged, missing = merge_adjudications(
            packets,
            qwen_reviews,
            agent_reviews=[],
            mandatory_agent_case_ids={"airline/1/0"},
        )

        self.assertEqual([row["case_id"] for row in merged], ["retail/2/0"])
        self.assertEqual(missing, ["airline/1/0"])
        self.assertEqual(merged[0]["reviewer"], "qwen36")

        agent_review = review("airline/1/0", 2, 4, "argument_composition")
        merged, missing = merge_adjudications(
            packets,
            qwen_reviews,
            agent_reviews=[agent_review],
            mandatory_agent_case_ids={"airline/1/0"},
        )
        self.assertEqual(missing, [])
        self.assertEqual(
            {row["case_id"]: row["reviewer"] for row in merged},
            {"airline/1/0": "agent", "retail/2/0": "qwen36"},
        )

    def test_run_replays_records_actions_and_routes_every_generation_for_review(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "results.json"
            simulation = {
                "task_id": "3",
                "trial": 0,
                "policy": "Use account IDs exactly.",
                "messages": [
                    {"role": "assistant", "content": "Hello", "turn_idx": 0},
                    {"role": "user", "content": "Open account 3", "turn_idx": 1},
                    {"role": "assistant", "content": "old", "turn_idx": 2},
                ],
            }
            result_path.write_text(
                json.dumps(
                    {
                        "tasks": [{"id": "3", "evaluation_criteria": {"actions": []}}],
                        "simulations": [simulation],
                    }
                ),
                encoding="utf-8",
            )
            manifest = [
                {
                    "replay_id": "airline/3/0/qwen3/weak/seed300",
                    "case_id": "airline/3/0",
                    "domain": "airline",
                    "task_id": "3",
                    "primary_gap": "action_selection",
                    "model": "qwen3",
                    "prefix": "weak",
                    "source_run": "base_b",
                    "source_path": str(result_path),
                    "simulation_index": 0,
                    "prefix_turn": 2,
                    "seed": 300,
                    "expected_next_action": "Call get_account with account 3.",
                }
            ]

            records = run_replays(
                manifest,
                model_label="qwen3",
                served_model="tau2-replay",
                tool_schemas_by_domain={
                    "airline": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_account",
                                "parameters": {"type": "object"},
                            },
                        }
                    ]
                },
                max_tokens=1200,
                enable_thinking=None,
                request_fits=lambda _request: (True, 77),
                complete=lambda _request: {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "new-call",
                            "type": "function",
                            "function": {
                                "name": "get_account",
                                "arguments": '{"account_id":"3"}',
                            },
                        }
                    ],
                },
                concurrency=1,
            )

            self.assertEqual(records[0]["status"], "generated")
            self.assertEqual(records[0]["input_tokens"], 77)
            self.assertEqual(records[0]["action"]["tool_calls"][0]["name"], "get_account")
            self.assertEqual(
                records[0]["action"]["tool_calls"][0]["arguments"], {"account_id": "3"}
            )
            queue = build_replay_review_queue(records)
            self.assertEqual([row["replay_id"] for row in queue], [manifest[0]["replay_id"]])
            self.assertEqual(queue[0]["expected_next_action"], manifest[0]["expected_next_action"])

    def test_replay_summary_reports_correct_over_four_and_causal_pattern(self):
        records = []
        decisions = []
        acceptable_counts = {
            ("qwen3", "weak"): 1,
            ("qwen35", "weak"): 3,
            ("qwen3", "strong"): 2,
            ("qwen35", "strong"): 4,
        }
        for model, prefix in acceptable_counts:
            for offset, seed in enumerate((300, 301, 302, 303)):
                replay_id = f"telecom/9/0/{model}/{prefix}/seed{seed}"
                records.append(
                    {
                        "replay_id": replay_id,
                        "case_id": "telecom/9/0",
                        "domain": "telecom",
                        "model": model,
                        "prefix": prefix,
                        "status": "generated",
                    }
                )
                decisions.append(
                    {
                        "replay_id": replay_id,
                        "acceptable_next_action": offset < acceptable_counts[(model, prefix)],
                        "reason": "fixture decision",
                    }
                )

        summary = summarize_replay_adjudications(records, decisions)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["conditions"]["qwen3_weak"]["acceptable"], 1)
        self.assertEqual(summary[0]["conditions"]["qwen35_strong"]["evaluated"], 4)
        self.assertEqual(summary[0]["causal_pattern"], "compound")


class RunnerContractTest(unittest.TestCase):
    def test_runner_rejects_unknown_stage_before_starting_any_service(self):
        result = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=ANALYSIS_DIR.parents[2],
            env={**os.environ, "ATOMIC_GAP_STAGE": "unknown"},
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("ATOMIC_GAP_STAGE", result.stderr)

    def test_prepare_stage_runs_cpu_test_and_prepare_subcommand_without_server(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=ANALYSIS_DIR.parents[2],
                env={
                    **os.environ,
                    "ATOMIC_GAP_STAGE": "prepare",
                    "ATOMIC_GAP_PYTHON": "/bin/echo",
                    "ATOMIC_GAP_OUT": tmp_dir,
                },
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("test_tau2_atomic_gap_analysis.py", result.stdout)
        self.assertIn("atomic_gap_analysis.py prepare", result.stdout)
        self.assertNotIn("sglang.launch_server", result.stdout)


class FormalReportTest(unittest.TestCase):
    def test_report_is_task_weighted_domain_complete_and_links_job_logs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            job_log = output_dir / "jobs/123-review/run_0.log"
            job_log.parent.mkdir(parents=True)
            job_log.write_text("complete\n", encoding="utf-8")
            canonical_packets = [
                {
                    "case_id": "airline/1/0",
                    "domain": "airline",
                    "task_id": "1",
                    "high_confidence_gap": True,
                },
                {
                    "case_id": "airline/2/0",
                    "domain": "airline",
                    "task_id": "2",
                    "high_confidence_gap": True,
                },
                {
                    "case_id": "banking_knowledge/task_1/0",
                    "domain": "banking_knowledge",
                    "task_id": "task_1",
                    "high_confidence_gap": False,
                    "trace_labels": {"A": "strong", "B": "base_b"},
                },
                {
                    "case_id": "banking_knowledge/task_2/0",
                    "domain": "banking_knowledge",
                    "task_id": "task_2",
                    "high_confidence_gap": False,
                },
            ]
            adjudicated = [
                {
                    "case_id": "airline/1/0",
                    "domain": "airline",
                    "task_id": "1",
                    "primary_gap": "action_selection",
                    "first_divergence": {"base_b": 4, "strong": 6},
                    "corrective_mechanism": "Selected the required write action.",
                    "confidence": "clear",
                    "reviewer": "agent",
                    "high_confidence_gap": True,
                    "run_success": {"base_b": 0, "strong": 1},
                },
                {
                    "case_id": "airline/2/0",
                    "domain": "airline",
                    "task_id": "2",
                    "primary_gap": "action_selection",
                    "first_divergence": {"base_b": 8, "strong": 10},
                    "corrective_mechanism": "Used the verified entity before writing.",
                    "confidence": "clear",
                    "reviewer": "agent",
                    "high_confidence_gap": True,
                    "run_success": {"base_b": 0, "strong": 1},
                },
                {
                    "case_id": "banking_knowledge/task_1/0",
                    "domain": "banking_knowledge",
                    "task_id": "task_1",
                    "primary_gap": "evidence_sufficiency",
                    "first_divergence": {"base_b": 2, "strong": 4},
                    "corrective_mechanism": "Trace A refined the knowledge-base query.",
                    "confidence": "clear",
                    "reviewer": "agent",
                    "high_confidence_gap": False,
                    "run_success": {"base_b": 0, "strong": 1},
                },
                {
                    "case_id": "banking_knowledge/task_2/0",
                    "domain": "banking_knowledge",
                    "task_id": "task_2",
                    "primary_gap": "action_selection",
                    "first_divergence": {"base_b": 2, "strong": 4},
                    "corrective_mechanism": "Both trajectories still failed.",
                    "confidence": "clear",
                    "reviewer": "qwen36",
                    "high_confidence_gap": False,
                    "run_success": {"base_b": 0, "strong": 0},
                },
            ]
            replay_records = []
            replay_decisions = []
            thresholds = {
                ("qwen3", "weak"): 1,
                ("qwen35", "weak"): 3,
                ("qwen3", "strong"): 2,
                ("qwen35", "strong"): 4,
            }
            for model in ("qwen3", "qwen35"):
                for prefix in ("weak", "strong"):
                    for offset, seed in enumerate((300, 301, 302, 303)):
                        replay_id = f"airline/1/0/{model}/{prefix}/seed{seed}"
                        replay_records.append(
                            {
                                "replay_id": replay_id,
                                "case_id": "airline/1/0",
                                "domain": "airline",
                                "model": model,
                                "prefix": prefix,
                                "status": "generated",
                            }
                        )
                        replay_decisions.append(
                            {
                                "replay_id": replay_id,
                                "acceptable_next_action": offset
                                < thresholds[(model, prefix)],
                                "reason": "fixture",
                            }
                        )
            inventory = {
                "scenario_cells": 788,
                "tasks": 197,
                "by_domain": {
                    "airline": {
                        "cells": 80,
                        "tasks": 20,
                        "base_a_successes": 26,
                        "base_b_successes": 30,
                        "strong_successes": 64,
                        "outcomes": {
                            "stable_gap": 33,
                            "shared_hard": 11,
                        },
                    }
                },
                "outcomes": {
                    "stable_gap": 166,
                    "shared_hard": 413,
                    "base_unstable": 98,
                    "reverse_control": 19,
                    "all_success": 92,
                },
                "high_confidence_tasks": 43,
                "high_confidence_by_domain": {
                    "airline": 7,
                    "retail": 9,
                    "telecom": 27,
                    "banking_knowledge": 0,
                },
            }

            result = write_formal_report(
                adjudicated_cases=adjudicated,
                canonical_packets=canonical_packets,
                replay_records=replay_records,
                replay_decisions=replay_decisions,
                inventory=inventory,
                output_dir=output_dir,
            )

            self.assertEqual(result["canonical_adjudicated"], 4)
            with (output_dir / "atomic_gap_summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            high_airline = next(
                row
                for row in rows
                if row["scope"] == "high_confidence"
                and row["domain"] == "airline"
                and row["primary_gap"] == "action_selection"
            )
            self.assertEqual(high_airline["tasks"], "2")
            self.assertEqual(high_airline["denominator"], "2")
            self.assertEqual(high_airline["share"], "1.000000")

            report = (output_dir / "README.md").read_text(encoding="utf-8")
            headings = [line for line in report.splitlines() if line.startswith("#")]
            self.assertLessEqual(len(headings), 10)
            self.assertIn("Banking is reported as shared difficulty", report)
            self.assertIn(
                "Qwen3 succeeds at most 2/8 times across both repeats and "
                "Qwen3.5 succeeds at least 3/4 times",
                report,
            )
            self.assertIn("| Domain | High-confidence tasks | Rows summarized |", report)
            self.assertIn("| airline | 20 | 26/80 | 30/80 | 64/80 | 33 | 11 |", report)
            self.assertIn("compound", report)
            self.assertIn(
                "Replay aggregate: Qwen3 weak 1/4, Qwen3.5 weak 3/4, "
                "Qwen3 strong 2/4, and Qwen3.5 strong 4/4",
                report,
            )
            self.assertIn("Candidate capability priorities", report)
            self.assertIn("| banking_knowledge/task_1 |", report)
            self.assertNotIn("| banking_knowledge/task_2 |", report)
            self.assertIn("Qwen3.5 refined the knowledge-base query", report)
            self.assertNotIn("Trace A refined the knowledge-base query", report)
            self.assertIn("jobs/123-review/run_0.log", report)
            self.assertNotIn("Qwen3.5 is an oracle", report)


if __name__ == "__main__":
    unittest.main()
