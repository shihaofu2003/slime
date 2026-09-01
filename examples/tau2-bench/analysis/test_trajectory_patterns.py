#!/usr/bin/env python3

import json
import unittest

from trajectory_patterns import (
    PATTERNS,
    PROMPT_VERSION,
    build_judge_payload,
    extract_features,
    validate_review,
)
from judge_trajectories import _schema_example, review_one
from summarize_trajectory_patterns import adjudicate_review


CATALOG = {
    "retail": {
        "agent": [
            {
                "name": "get_order_details",
                "arguments": [{"name": "order_id", "required": True}],
            },
            {
                "name": "modify_pending_order_items",
                "arguments": [{"name": "order_id", "required": True}],
            },
        ],
        "user": [],
    },
    "telecom": {
        "agent": [{"name": "get_data_usage", "arguments": []}],
        "user": [
            {"name": "check_network_status", "arguments": []},
            {"name": "run_speed_test", "arguments": []},
            {"name": "can_send_mms", "arguments": []},
        ],
    },
}


def task(action_name="modify_pending_order_items", requestor="assistant"):
    return {
        "evaluation_criteria": {
            "actions": [
                {
                    "requestor": requestor,
                    "name": action_name,
                    "arguments": {"order_id": "O1"},
                }
            ],
            "reward_basis": ["DB"],
        }
    }


def sim(messages, reward=1.0):
    return {
        "task_id": "fixture",
        "trial": 0,
        "messages": messages,
        "termination_reason": "user_stop",
        "reward_info": {
            "reward": reward,
            "db_check": {"db_match": bool(reward)},
            "action_checks": [{"action_match": bool(reward)}],
        },
    }


def assistant(turn, calls=None, content=None):
    return {
        "role": "assistant",
        "turn_idx": turn,
        "content": content,
        "tool_calls": calls,
    }


def call(call_id, name, requestor="assistant", arguments=None):
    return {
        "id": call_id,
        "name": name,
        "requestor": requestor,
        "arguments": arguments or {},
    }


def tool(turn, call_id, content="ok", error=False, requestor="assistant"):
    return {
        "role": "tool",
        "turn_idx": turn,
        "id": call_id,
        "content": content,
        "error": error,
        "requestor": requestor,
    }


class DeterministicFeatureTest(unittest.TestCase):
    def test_single_tool_normal_write(self):
        messages = [
            {"role": "user", "turn_idx": 0, "content": "Yes, proceed."},
            assistant(1, [call("1", "modify_pending_order_items", arguments={"order_id": "O1"})]),
            tool(2, "1"),
        ]
        record = extract_features(sim(messages), task(), "m", "retail", CATALOG)
        self.assertEqual(record["num_observed_writes"], 1)
        self.assertFalse(record["multitool_policy_violation"])
        self.assertTrue(
            record["write_confirmation_candidates"][0]["candidate_affirmative"]["present"]
        )

    def test_multitool_and_multiwrite_are_policy_violations(self):
        messages = [
            assistant(
                0,
                [
                    call("1", "get_order_details"),
                    call("2", "modify_pending_order_items"),
                ],
            ),
            tool(1, "1"),
            tool(2, "2"),
        ]
        record = extract_features(sim(messages), task(), "m", "retail", CATALOG)
        self.assertTrue(record["multitool_policy_violation"])
        self.assertEqual(record["multiwrite_turns"], 1)
        self.assertEqual(record["multitool_turn_indices"], [0])

    def test_missing_confirmation_candidate(self):
        messages = [
            {"role": "user", "turn_idx": 0, "content": "What are my options?"},
            assistant(1, [call("1", "modify_pending_order_items")]),
            tool(2, "1"),
        ]
        record = extract_features(sim(messages), task(), "m", "retail", CATALOG)
        self.assertFalse(
            record["write_confirmation_candidates"][0]["candidate_affirmative"]["present"]
        )

    def test_nonexistent_tool(self):
        messages = [assistant(0, [call("1", "invented_tool")]), tool(1, "1", "not found", True)]
        record = extract_features(sim(messages, 0), task(), "m", "retail", CATALOG)
        self.assertEqual(record["invalid_tool_names"], ["invented_tool"])

    def test_missing_required_argument(self):
        messages = [
            assistant(0, [call("1", "get_order_details")]),
            tool(1, "1", "Error: missing order_id", True),
        ]
        record = extract_features(sim(messages, 0), task(), "m", "retail", CATALOG)
        self.assertEqual(
            record["tool_argument_schema_errors"][0]["missing_required"], ["order_id"]
        )

    def test_repeated_failure_and_false_success(self):
        messages = [
            assistant(0, [call("1", "get_order_details")]),
            tool(1, "1", "Error: invalid order", True),
            assistant(2, [call("2", "get_order_details")]),
            tool(3, "2", "Error: invalid order", True),
            assistant(4, content="Your order has been updated successfully."),
        ]
        record = extract_features(sim(messages, 0), task(), "m", "retail", CATALOG)
        self.assertEqual(record["adjacent_repeated_calls"], 1)
        self.assertEqual(record["false_success_candidate_turns"], [4])

    def test_correct_telecom_progressive_order(self):
        messages = []
        for index, (name, requestor) in enumerate(
            [
                ("check_network_status", "user"),
                ("get_data_usage", "assistant"),
                ("run_speed_test", "user"),
                ("can_send_mms", "user"),
            ]
        ):
            messages.extend(
                [
                    assistant(index * 2, [call(str(index), name, requestor)]),
                    tool(index * 2 + 1, str(index), requestor=requestor),
                ]
            )
        record = extract_features(
            sim(messages), task("can_send_mms", "user"), "m", "telecom", CATALOG
        )
        self.assertTrue(record["telecom_progressive_order_candidate"])


class ReviewValidationTest(unittest.TestCase):
    def test_judge_payload_blinds_model_stage_and_reward(self):
        cell = {
            "model": "secret-stage",
            "domain": "retail",
            "task": task(),
            "simulation": sim([]),
        }
        payload = build_judge_payload(cell, CATALOG)
        visible = {key: value for key, value in payload.items() if not key.startswith("_")}
        serialized = json.dumps(visible)
        self.assertNotIn("secret-stage", serialized)
        self.assertNotIn('"reward"', serialized)
        self.assertTrue(payload["trajectory_id"].startswith("blind-"))

    def test_review_restores_real_id_after_blind_judging(self):
        cell = {
            "model": "secret-stage",
            "domain": "retail",
            "task": task(),
            "simulation": sim([assistant(0, content="Done.")]),
        }
        payload = build_judge_payload(cell, CATALOG)

        class Client:
            def complete(self, value, repair=None):
                return json.dumps(_schema_example(value["trajectory_id"]))

        review = review_one(Client(), payload, calibration_run=0)
        self.assertEqual(review["trajectory_id"], "secret-stage|retail|fixture|0")

    def test_invalid_turn_is_rejected(self):
        finding = {
            "status": "pass",
            "severity": "none",
            "turn_indices": [],
            "evidence": "",
            "correct_behavior": "",
        }
        review = {
            "prompt_version": PROMPT_VERSION,
            "patterns": {name: dict(finding) for name in PATTERNS},
            "primary_failure_mode": "none",
            "user_simulator_error": {"status": "none", "turn_indices": []},
        }
        review["patterns"]["intent_tracking"]["turn_indices"] = [999]
        self.assertIn(
            "intent_tracking: invalid turn_indices", validate_review(review, {0, 1})
        )

    def test_rule_adjudication_overrides_mechanical_judge_error(self):
        messages = [
            assistant(
                0,
                [
                    call("1", "get_order_details"),
                    call("2", "modify_pending_order_items"),
                ],
            ),
            tool(1, "1"),
            tool(2, "2"),
        ]
        feature = extract_features(
            sim(messages), task(), "raw", "retail", CATALOG
        )
        review = _schema_example(feature["trajectory_id"])
        for finding in review["patterns"].values():
            finding["status"] = "pass"
            finding["turn_indices"] = [0, 1, 2]
        final, overrides = adjudicate_review(feature, review)
        self.assertEqual(
            final["patterns"]["multi_tool_policy_violation"]["status"], "fail"
        )
        self.assertEqual(
            final["patterns"]["domain_policy_compliance"]["severity"], "critical"
        )
        self.assertEqual(
            final["patterns"]["progressive_diagnosis"]["status"],
            "not_applicable",
        )
        self.assertTrue(overrides)
        self.assertTrue(
            all(
                len(finding["turn_indices"]) <= 2
                for finding in final["patterns"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
