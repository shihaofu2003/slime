import copy
import json
import os
import sys
import tempfile
import unittest
from collections import Counter, UserDict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

NUM_GPUS = 0

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "examples/tau2-bench/analysis"
SFT_DIR = Path(__file__).resolve().parents[1] / "examples/tau2-bench/sft"
OFFICIAL_EVAL_DIR = (
    Path(__file__).resolve().parents[1] / "examples/tau2-bench/eval/official"
)
SHARED_DIR = Path(__file__).resolve().parents[1] / "examples/tau2-bench/shared"
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SFT_DIR))
sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
sys.path.insert(0, str(SHARED_DIR))

from sft_quality import (  # noqa: E402
    DIMENSIONS,
    PROMPT_VERSION,
    PROTOCOL_DEPENDENCY_SAFE_MULTI,
    audit_source_dialogs,
    build_samples,
    classify_dialog,
    deterministic_features,
    payload_hash,
    policy_for_profile as audit_policy_for_profile,
    read_jsonl,
    row_messages,
    stable_json_hash,
    validate_review,
    write_jsonl,
)
from judge_sft_quality import (  # noqa: E402
    DEPENDENCY_SAFE_QUALITY_DIMENSIONS,
    MULTITOOL_PROMPT_VERSION,
    REVIEW_NORMALIZATION_VERSION,
    ReviewFailure,
    escalation_reasons,
    extract_json,
    failure_sidecar_path,
    main as judge_main,
    normalize_review,
    reconcile_cached_records,
    reconcile_mechanical_facts,
    request_messages,
    response_schema_for_payload,
    review_once,
    review_payload,
    validate_review_output,
)
from audit_sft_quality import (  # noqa: E402
    adjudicate_review_record,
    aggregate_inventory,
    calibration_metrics,
    filtered_rows,
    judge_diagnostics,
    load_review_map,
    success_cases,
)
from audit_sft_multitool import (  # noqa: E402
    FAILURE_CALIBRATION_LIMIT,
    FAILURE_CALIBRATION_PILOT_SEED,
    FAILURE_CALIBRATION_PILOT_VERSION,
    FAILURE_CALIBRATION_SEED,
    FAILURE_CALIBRATION_VERSION,
    QUALITY_CALIBRATION_LIMIT,
    QUALITY_CALIBRATION_PILOT_SEED,
    QUALITY_CALIBRATION_V1_SEED,
    QUALITY_CALIBRATION_VALIDATION_SEED,
    QUALITY_CALIBRATION_VERSION,
    _counterfactual_payload,
    _failure_mode_evidence_consistent,
    _quality_consensus,
    _selected_judge_identity,
    _token_sequence_length,
    _split_payloads,
    _validated_review_map,
    amend_policy as audit_multitool_amend_policy,
    build_payload_manifest,
    build_calibration_selection,
    build_failure_calibration_pilot_selection,
    build_failure_calibration_selection,
    build_quality_v1_selection,
    build_quality_validation_selection,
    calibration_report,
    file_sha256,
    failure_calibration_report,
    failure_report,
    filter_decision_rows,
    summarize,
    validate_selected_artifacts,
)
from build_local_first_sft import (  # noqa: E402
    AIRLINE_SINGLE_CALL_RULES,
    CANDIDATE_POLICY_RULE,
    FILTER_VERSION,
    amend_training_system,
    apply_target_only_loss_mask,
    matched_budget_rows,
    multitool_prefix_reasons,
    quality_gate_reasons,
)
from shard_local_reviews import merge as merge_review_shards  # noqa: E402
from shard_local_reviews import partition as partition_review_shards  # noqa: E402
from shard_local_reviews import shard_index  # noqa: E402
from compare_official_sft_evals import (  # noqa: E402
    aggregate_behavior,
    aggregate_metrics as aggregate_eval_metrics,
    compare as compare_official_evals,
    load_model_eval,
    paired_bootstrap_ci,
    task_metrics as official_task_metrics,
    trajectory_has_multi_tool_turn,
)
from run_eval import (  # noqa: E402
    _json_finite as eval_json_finite,
    _pass_metrics as eval_pass_metrics,
)
from check_tau2_rl_promotion import (  # noqa: E402
    evaluate_gate,
    parse_training_log,
    select_checkpoint,
    summarize_recent_trajectories,
)
from validate_local_first_sft import validate_row_structure  # noqa: E402
from protocol_profiles import (  # noqa: E402
    DEPENDENCY_SAFE_MULTI_RULE as EVAL_DEPENDENCY_SAFE_MULTI_RULE,
    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    PROTOCOL_CURRENT_SINGLE as EVAL_PROTOCOL_CURRENT_SINGLE,
    PROTOCOL_DEPENDENCY_SAFE_MULTI as EVAL_PROTOCOL_DEPENDENCY_SAFE_MULTI,
    domain_policy_for_profile,
    protocol_block_for_profile,
    protocol_signature,
)
from agent_contract import (  # noqa: E402
    BOUNDARY_ANCHOR_VIEW,
    OFFICIAL_AGENT_VIEW,
    AgentContract,
)
from build_agent_boundary_v2 import (  # noqa: E402
    _tokenize_and_validate_row,
    build_ownership_repair_anchors,
    rebuild_contract_only_rows,
)
from build_agent_single_call_v1 import (  # noqa: E402
    _token_count as single_call_token_count,
    convert_boundary_row,
    index_real_tool_results,
    validate_single_call_row,
)
from namespace_analyzer import (  # noqa: E402
    analyze_namespace_trajectories,
    analyze_single_call_attempts,
    raw_tool_names,
)
from check_agent_boundary_sft_gate import (  # noqa: E402
    GATE_VERSION as BOUNDARY_SFT_GATE_VERSION,
    THRESHOLDS as BOUNDARY_SFT_GATE_THRESHOLDS,
    capability_baseline_values,
    evaluate_candidate as evaluate_boundary_sft_candidate,
    rank_eligible as rank_boundary_sft_eligible,
)
from check_agent_boundary_rl_pilot import (  # noqa: E402
    _quota_observations,
    summarize_trajectories as summarize_boundary_pilot_trajectories,
)
from check_agent_boundary_rl_final import FINAL_QUOTA  # noqa: E402
from check_agent_boundary_rl_health import (  # noqa: E402
    GATE_VERSION as LONG100_HEALTH_GATE_VERSION,
    LONG200_GATE_VERSION,
    LONG200_KL_WAIVER_GATE_VERSION,
    LONG100_STAGES,
    LONG200_STAGES,
    evaluate_health as evaluate_long100_health,
)
from check_agent_boundary_rl_long100 import (  # noqa: E402
    DECISION_VERSION as LONG100_DECISION_VERSION,
    evaluate_long100_decision,
)
from authorize_agent_boundary_checkpoint import authorize_checkpoint  # noqa: E402
from check_domain_expert_health import (  # noqa: E402
    GATE_VERSION as DOMAIN_EXPERT_HEALTH_GATE_VERSION,
    evaluate_prefix_health,
)
from domain_expert_checkpoint import (  # noqa: E402
    expected_lineage as expected_domain_expert_lineage,
    prepare_domain_expert,
    quota_for_domain as expert_quota,
    read_sampler_state,
)
from summarize_domain_experts import (  # noqa: E402
    classify_expert,
    select_checkpoint as select_domain_expert_checkpoint,
)
from check_agent_boundary_rl_long200 import (  # noqa: E402
    DECISION_VERSION as LONG200_DECISION_VERSION,
    evaluate_long200_decision,
    evaluate_long200_early_reject,
    evaluate_long200_kl_waiver_result,
)
from slime.utils.mask_utils import MultiTurnLossMaskGenerator  # noqa: E402


CATALOG = {
    "airline": {
        "agent": [
            {
                "name": "get_user_details",
                "arguments": [{"name": "user_id", "required": True}],
            },
            {
                "name": "book_reservation",
                "arguments": [{"name": "user_id", "required": True}],
            },
        ],
        "user": [{"name": "check_email", "arguments": []}],
    }
}


def call(name, arguments=None, call_id=None):
    value = {"name": name, "arguments": arguments or {}}
    if call_id:
        value["id"] = call_id
    return value


def source_row(turn_index, messages, answer, *, correct=1, reward=1.0):
    return {
        "messages": messages,
        "answer": answer,
        "metadata": {
            "source_dialog_id": "airline_dialog_fixture",
            "turn_index": turn_index,
            "reason_for_call": "Book a flight",
            "scenario_id": "scenario_fixture",
            "correct": correct,
            "reward": reward,
        },
    }


def converted_row(source_index, turn_index, messages=None):
    return {
        "messages": messages
        or [
            {
                "role": "system",
                "content": "In each turn make one or more tool calls.",
            },
            {"role": "assistant", "content": "hello"},
        ],
        "metadata": {
            "source_dialog_id": "airline_dialog_fixture",
            "source_row": source_index,
            "turn_index": turn_index,
        },
    }


def valid_review(**overrides):
    review = {
        "dialog_id": "blind",
        "prompt_version": PROMPT_VERSION,
        "verdict": "keep",
        "confidence": 0.9,
        "apparent_task_outcome": "success",
        "dimensions": {
            name: {"score": 4, "evidence_turns": [1], "rationale": "supported"}
            for name in DIMENSIONS
        },
        "critical_issues": [],
        "strengths": [
            {"pattern": "grounded", "evidence_turns": [1], "explanation": "uses result"}
        ],
        "weaknesses": [],
        "summary": "high quality",
    }
    review.update(overrides)
    return review


def dependency_safe_review(turns=(3, 5), **overrides):
    review = {
        "dialog_id": "blind",
        "prompt_version": MULTITOOL_PROMPT_VERSION,
        "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
        "quality_verdict": "keep",
        "confidence": 0.9,
        "batch_safety": "safe_independent_reads",
        "multi_turn_assessments": [
            {
                "turn_index": turn,
                "batch_safety": "safe_independent_reads",
                "arguments_grounded": "yes",
                "writes_authorized": "not_applicable",
                "calls_necessary": "yes",
            }
            for turn in turns
        ],
        "dimensions": {
            name: {"score": 4, "evidence_turns": [3], "rationale": "supported"}
            for name in DEPENDENCY_SAFE_QUALITY_DIMENSIONS
        },
        "critical_turns": [],
        "summary": "safe independent calls",
    }
    review.update(overrides)
    return review


def judge_record(
    review,
    *,
    confidence_threshold=0.75,
    unresolved=False,
    errors=None,
    label="local:test-model",
    url="http://local.test/v1/chat/completions",
    model="test-model",
):
    return {
        "selected_review": review,
        "selected_provider": label,
        "primary_provider": label,
        "fallback_provider": None,
        "judge_config": {
            "confidence_threshold": confidence_threshold,
            "primary": {"url": url, "model": model},
            "fallback": None,
        },
        "unresolved_escalation": unresolved,
        "errors": list(errors or []),
    }


def failure_judge_record(cause, mode, role, **overrides):
    confidence = overrides.pop("confidence", 0.9)
    decisive_turns = overrides.pop("decisive_turns", [])
    return judge_record(
        {
            "primary_failure_cause": cause,
            "causal_call_mode": mode,
            "multi_causal_role": role,
            "decisive_turns": decisive_turns,
            "confidence": confidence,
        },
        **overrides,
    )


def failure_calibration_fixture():
    """Reproduce the sealed strata and selected IDs without external artifacts."""

    strata = (
        (
            "airline",
            "consensus_failure",
            True,
            "full/full",
            1,
            "airline_dialog_744".split(),
        ),
        (
            "airline",
            "consensus_failure",
            True,
            "window/full",
            1,
            "airline_dialog_378".split(),
        ),
        (
            "airline",
            "correct_only",
            True,
            "full/full",
            3,
            "airline_dialog_104 airline_dialog_636 airline_dialog_938".split(),
        ),
        (
            "airline",
            "correct_only",
            True,
            "window/full",
            2,
            "airline_dialog_310 airline_dialog_444".split(),
        ),
        (
            "airline",
            "reward_only",
            True,
            "full/full",
            94,
            (
                "airline_dialog_124 airline_dialog_169 airline_dialog_268 "
                "airline_dialog_344 airline_dialog_587 airline_dialog_608 "
                "airline_dialog_62 airline_dialog_65 airline_dialog_656 "
                "airline_dialog_660 airline_dialog_737 airline_dialog_79 "
                "airline_dialog_837 airline_dialog_882 airline_dialog_965"
            ).split(),
        ),
        (
            "airline",
            "reward_only",
            True,
            "window/full",
            23,
            (
                "airline_dialog_243 airline_dialog_317 airline_dialog_387 "
                "airline_dialog_437 airline_dialog_454 airline_dialog_524 "
                "airline_dialog_568 airline_dialog_574 airline_dialog_619 "
                "airline_dialog_700 airline_dialog_871 airline_dialog_874 "
                "airline_dialog_9 airline_dialog_919"
            ).split(),
        ),
        (
            "retail",
            "consensus_failure",
            True,
            "full/full",
            8,
            (
                "retail_dialog_122 retail_dialog_394 retail_dialog_506 "
                "retail_dialog_652 retail_dialog_668 retail_dialog_783 "
                "retail_dialog_819 retail_dialog_878"
            ).split(),
        ),
        (
            "telecom",
            "consensus_failure",
            True,
            "full/full",
            209,
            "103 118 134 135 201 214 308 375 394 438 474 494 496 8".split(),
        ),
        (
            "telecom",
            "consensus_failure",
            False,
            "full/full",
            109,
            "111 155 184 250 258 266 321 346 366 419 433 464 497 64".split(),
        ),
    )
    labels = {
        "consensus_failure": (0, 0),
        "correct_only": (1, 0),
        "reward_only": (0, 1),
    }
    features = []
    manifest = []
    for stratum_index, (
        domain,
        group,
        multi,
        route,
        count,
        selected_ids,
    ) in enumerate(strata):
        exposure = "multi_exposed" if multi else "non_multi_exposed"
        key = (domain, group, exposure, route)
        cutoff = min(
            stable_json_hash(
                [
                    FAILURE_CALIBRATION_PILOT_SEED,
                    FAILURE_CALIBRATION_PILOT_VERSION,
                    key,
                    dialog_id,
                ]
            )
            for dialog_id in selected_ids
        )
        dialog_ids = list(selected_ids)
        candidate_index = 0
        while len(dialog_ids) < count:
            candidate = f"fixture-unselected-{stratum_index}-{candidate_index:04d}"
            candidate_index += 1
            candidate_hash = stable_json_hash(
                [
                    FAILURE_CALIBRATION_PILOT_SEED,
                    FAILURE_CALIBRATION_PILOT_VERSION,
                    key,
                    candidate,
                ]
            )
            if candidate_hash < cutoff:
                dialog_ids.append(candidate)
        for dialog_id in dialog_ids:
            correct, reward = labels[group]
            features.append(
                {
                    "dialog_id": dialog_id,
                    "domain": domain,
                    "label_group": group,
                    "source_correct": correct,
                    "source_reward": reward,
                    "full_source_static": {
                        "multitool_turns": [3] if multi else []
                    },
                }
            )
            manifest.append({"dialog_id": dialog_id, "route": route})
    return features, manifest


class SftQualityTest(unittest.TestCase):
    def test_response_schema_binds_literal_payload_turn_indices(self):
        schema = response_schema_for_payload(
            {
                "conversation": [{"turn_index": 2}, {"turn_index": 7}],
                "mechanical_audit_facts": {"multitool_turns": [7]},
            },
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
        )["json_schema"]["schema"]
        properties = schema["properties"]
        dimensions = properties["dimensions"]["properties"]
        for dimension in DEPENDENCY_SAFE_QUALITY_DIMENSIONS:
            self.assertEqual(
                dimensions[dimension]["properties"]["evidence_turns"]["items"][
                    "enum"
                ],
                [2, 7],
            )
        self.assertEqual(properties["critical_turns"]["items"]["enum"], [2, 7])
        assessment = properties["multi_turn_assessments"]["items"]["properties"]
        self.assertEqual(assessment["turn_index"]["enum"], [7])

    def test_response_schema_for_empty_multi_set_forbids_assessments(self):
        schema = response_schema_for_payload(
            {
                "conversation": [{"turn_index": 2}],
                "mechanical_audit_facts": {"multitool_turns": []},
            },
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
        )["json_schema"]["schema"]
        assessments = schema["properties"]["multi_turn_assessments"]
        self.assertEqual(assessments["maxItems"], 0)
        self.assertNotIn("enum", assessments["items"]["properties"]["turn_index"])

    def test_response_schema_rejects_multi_turn_absent_from_conversation(self):
        payload = {
            "_dialog_id": "d-missing",
            "conversation": [{"turn_index": 2}],
            "mechanical_audit_facts": {"multitool_turns": [7]},
        }
        with self.assertRaisesRegex(
            ValueError,
            r"d-missing: mechanical multitool_turns absent from conversation: \[7\]",
        ):
            response_schema_for_payload(
                payload,
                review_mode="quality",
                protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            )

    def test_token_sequence_length_supports_list_and_batch_encoding(self):
        self.assertEqual(_token_sequence_length([1, 2, 3]), 3)
        self.assertEqual(_token_sequence_length({"input_ids": [1, 2, 3, 4]}), 4)
        self.assertEqual(
            _token_sequence_length(UserDict({"input_ids": [1, 2, 3, 4, 5]})),
            5,
        )
        self.assertEqual(_token_sequence_length({"input_ids": [[1, 2, 3]]}), 3)
        with self.assertRaisesRegex(ValueError, "single rendered chat sequence"):
            _token_sequence_length({"input_ids": [[1], [2]]})
        with self.assertRaisesRegex(ValueError, "no input_ids"):
            _token_sequence_length({"attention_mask": [1, 1]})

    def test_reconstructs_longest_training_view_and_counts_prefix_exposure(self):
        system = {"role": "system", "content": "<policy>one call at a time</policy> airline"}
        user = {"role": "user", "content": "Please book"}
        first_answer = {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("get_user_details", {"user_id": "U1"}, "c1")],
        }
        first_tool = {"role": "tool", "id": "c1", "content": '{"id":"U1"}'}
        second_answer = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                call("book_reservation", {"user_id": "U1"}, "c2"),
                call("get_user_details", {"user_id": "U1"}, "c3"),
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            training = Path(directory) / "training.jsonl"
            write_jsonl(
                source,
                [
                    source_row(0, [system, user], first_answer),
                    source_row(1, [system, user, first_answer, first_tool], second_answer),
                ],
            )
            write_jsonl(
                training,
                [
                    converted_row(0, 0, [system, user, first_answer]),
                    converted_row(
                        1,
                        1,
                        [system, user, first_answer, first_tool, second_answer],
                    ),
                ],
            )
            features, payloads, inventory = audit_source_dialogs(source, training, CATALOG)
            aggregate = aggregate_inventory(features, inventory)

        self.assertEqual(inventory["training_rows"], 2)
        self.assertEqual(len(features), 1)
        self.assertEqual(len(payloads), 1)
        feature = features[0]
        self.assertEqual(feature["included_row_count"], 2)
        self.assertEqual(feature["exposures"]["assistant_turn_exposures"], 3)
        self.assertEqual(feature["exposures"]["tool_call_exposures"], 4)
        self.assertEqual(feature["exposures"]["multitool_turn_exposures"], 1)
        self.assertEqual(feature["exposures"]["training_rows_with_multitool_prefix"], 1)
        self.assertEqual(feature["exposures"]["target_multitool_rows"], 1)
        self.assertEqual(feature["static"]["multitool_turns"], [4])
        self.assertIn("multi_tool_policy_violation", feature["hard_issues"])
        self.assertEqual(feature["static"]["missing_prefix_targets"], [])
        self.assertEqual(payloads[0]["domain_policy"], "one call at a time")
        self.assertEqual(aggregate["positive_label_dialogs_by_domain"], {"airline": 1})
        self.assertEqual(aggregate["multitool_dialogs_by_domain"], {"airline": 1})
        self.assertEqual(
            aggregate["deterministic_clean_success_dialogs_by_domain"], {"airline": 0}
        )
        self.assertEqual(
            aggregate["deterministic_clean_success_rows_by_domain"], {"airline": 0}
        )
        self.assertEqual(
            aggregate["deterministic_clean_success_full_dialogs_by_domain"],
            {"airline": 0},
        )
        self.assertEqual(
            aggregate["deterministic_clean_success_state_changing_dialogs_by_domain"],
            {"airline": 0},
        )
        self.assertEqual(
            aggregate["loss_exposures_by_domain"]["airline"]
            ["training_rows_with_multitool_prefix"],
            1,
        )
        self.assertEqual(
            aggregate["loss_exposures_by_label"]["positive"]["target_multitool_rows"],
            1,
        )
        self.assertEqual(
            aggregate["dialog_retention_by_domain"]["airline"]["fully_retained_dialogs"],
            1,
        )
        self.assertEqual(
            aggregate["dialog_retention_by_domain"]["airline"]["median_target_retention"],
            1,
        )

    def test_deterministic_features_find_namespace_schema_and_invalid_tools(self):
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "Legacy narration emitted with the call.",
                "tool_calls": [
                    call("get_user_details"),
                    call("check_email"),
                    call("invented_tool"),
                ],
            },
        ]
        record = deterministic_features(messages, "airline", CATALOG)
        self.assertEqual(record["invalid_tool_names"], ["check_email", "invented_tool"])
        self.assertEqual(record["namespace_confusion_turns"], [1])
        self.assertEqual(
            record["tool_argument_schema_errors"][0]["missing_required"], ["user_id"]
        )
        self.assertEqual(record["multitool_turns"], [1])

    def test_review_validation_and_quality_gates(self):
        review = valid_review()
        self.assertEqual(validate_review(review, {1}), [])
        clean_feature = {"hard_issues": []}
        judge_record = {
            "selected_review": review,
            "selected_provider": "local:model",
            "unresolved_escalation": False,
        }
        self.assertEqual(classify_dialog(clean_feature, judge_record)["tier"], "keep")
        uncertain_prefix = valid_review(apparent_task_outcome="uncertain")
        uncertain_record = {**judge_record, "selected_review": uncertain_prefix}
        self.assertEqual(
            classify_dialog({"hard_issues": [], "static": {"final_has_tool": True}}, uncertain_record)[
                "tier"
            ],
            "keep",
        )
        dirty_feature = {"hard_issues": ["source_correct_not_one"]}
        decision = classify_dialog(dirty_feature, judge_record)
        self.assertEqual(decision["tier"], "drop")
        self.assertEqual(decision["reasons"], ["source_correct_not_one"])

    def test_escalation_detects_disagreement_low_confidence_and_rule_conflict(self):
        first = valid_review(confidence=0.6)
        second = valid_review(verdict="review")
        payload = {
            "mechanical_audit_facts": {"hard_issues": ["multi_tool_policy_violation"]}
        }
        reasons = escalation_reasons(payload, [first, second], 0.75)
        self.assertEqual(
            reasons,
            ["deterministic_rule_conflict", "low_confidence", "verdict_disagreement"],
        )
        uncertain = valid_review(apparent_task_outcome="uncertain")
        boundary_payload = {
            "mechanical_audit_facts": {"hard_issues": [], "final_has_tool": True}
        }
        self.assertEqual(escalation_reasons(boundary_payload, [uncertain], 0.75), [])
        boundary_payload["mechanical_audit_facts"]["final_has_tool"] = False
        self.assertEqual(
            escalation_reasons(boundary_payload, [uncertain], 0.75), ["uncertain_outcome"]
        )

    def test_judge_output_recovery_and_unambiguous_normalization(self):
        raw = 'analysis before {"note":"ignore"}\n' + str(valid_review()).replace("'", '"')
        raw += "\ntrailing text"
        parsed = extract_json(raw)
        parsed["confidence"] = "95%"
        parsed["dimensions"]["tool_choice"]["score"] = "4"
        parsed["dimensions"]["tool_choice"]["evidence_turns"] = ["1"]
        normalized = normalize_review(parsed)
        self.assertEqual(normalized["confidence"], 0.95)
        self.assertEqual(normalized["dimensions"]["tool_choice"]["score"], 4)
        self.assertEqual(normalized["dimensions"]["tool_choice"]["evidence_turns"], [1])
        self.assertEqual(validate_review(normalized, {1}), [])
        parsed["confidence"] = 95
        self.assertEqual(normalize_review(parsed)["confidence"], 0.95)
        parsed["confidence"] = 4
        ambiguous = normalize_review(parsed)
        self.assertEqual(ambiguous["confidence"], 4)
        self.assertIn("invalid confidence", validate_review(ambiguous, {1}))
        normalized["confidence"] = True
        normalized["dimensions"]["tool_choice"]["score"] = True
        normalized["dimensions"]["tool_choice"]["evidence_turns"] = [True]
        errors = validate_review(normalized, {1})
        self.assertIn("invalid confidence", errors)
        self.assertIn("tool_choice: invalid score", errors)
        self.assertIn("tool_choice: invalid evidence_turns", errors)

    def test_review_repair_explains_confidence_scale(self):
        class FakeClient:
            def __init__(self):
                self.repairs = []

            def complete(self, payload, *, temperature, repair=None):
                self.repairs.append(repair)
                review = valid_review(confidence=4 if repair is None else 0.95)
                return json.dumps(review), {
                    "provider": "fake",
                    "usage": {},
                    "structured_output": True,
                }

        client = FakeClient()
        review = review_once(
            client,
            {
                "_dialog_id": "d1",
                "dialog_id": "blind-d1",
                "conversation": [{"turn_index": 1}],
            },
            temperature=0,
        )
        self.assertEqual(review["confidence"], 0.95)
        self.assertIn("never the 0-to-4 dimension scale", client.repairs[1])

    def test_review_repair_lists_valid_turn_indices(self):
        class FakeClient:
            def __init__(self):
                self.repairs = []

            def complete(self, payload, *, temperature, repair=None):
                self.repairs.append(repair)
                review = dependency_safe_review()
                if repair is None:
                    review["dimensions"]["tool_choice"]["evidence_turns"] = [999]
                return json.dumps(review), {
                    "provider": "fake",
                    "usage": {},
                    "structured_output": True,
                }

        client = FakeClient()
        review = review_once(
            client,
            {
                "_dialog_id": "d1",
                "dialog_id": "blind-d1",
                "conversation": [{"turn_index": 3}, {"turn_index": 5}],
                "mechanical_audit_facts": {
                    "multitool_turns": [3, 5],
                    "multiwrite_turns": [],
                },
            },
            temperature=0,
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
        )
        self.assertEqual(review["confidence"], 0.9)
        self.assertIn("valid conversation turn indices are [3, 5]", client.repairs[1])

    def test_second_evidence_repair_requires_empty_citations(self):
        class FakeClient:
            def __init__(self):
                self.repairs = []

            def complete(self, payload, *, temperature, repair=None):
                self.repairs.append(repair)
                review = dependency_safe_review(turns=(3,))
                if len(self.repairs) < 3:
                    review["dimensions"]["tool_choice"]["evidence_turns"] = [999]
                return json.dumps(review), {
                    "provider": "fake",
                    "usage": {},
                    "structured_output": True,
                }

        client = FakeClient()
        review = review_once(
            client,
            {
                "_dialog_id": "d1",
                "dialog_id": "blind-d1",
                "conversation": [{"turn_index": 3}],
                "mechanical_audit_facts": {
                    "multitool_turns": [3],
                    "multiwrite_turns": [],
                },
            },
            temperature=0,
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
        )
        self.assertEqual(review["confidence"], 0.9)
        self.assertEqual(len(client.repairs), 3)
        self.assertIn("set every dimensions.*.evidence_turns", client.repairs[2])

    def test_second_format_repair_constrains_arrays_and_multiturn_enums(self):
        class FakeClient:
            def __init__(self):
                self.repairs = []

            def complete(self, payload, *, temperature, repair=None):
                self.repairs.append(repair)
                review = dependency_safe_review(turns=(3,))
                if len(self.repairs) < 3:
                    review["dimensions"]["task_completion"]["evidence_turns"] = [
                        3,
                        3,
                        3,
                        3,
                    ]
                    assessment = review["multi_turn_assessments"][0]
                    assessment["arguments_grounded"] = "partially"
                    assessment["writes_authorized"] = "unknown"
                    assessment["calls_necessary"] = "mostly"
                return json.dumps(review), {
                    "provider": "fake",
                    "usage": {},
                    "structured_output": True,
                }

        client = FakeClient()
        review = review_once(
            client,
            {
                "_dialog_id": "d1",
                "dialog_id": "blind-d1",
                "conversation": [{"turn_index": 3}],
                "mechanical_audit_facts": {
                    "multitool_turns": [3],
                    "multiwrite_turns": [],
                },
            },
            temperature=0,
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
        )
        self.assertEqual(review["confidence"], 0.9)
        self.assertEqual(len(client.repairs), 3)
        self.assertIn("set every dimensions.*.evidence_turns", client.repairs[2])
        self.assertIn(
            "arguments_grounded and calls_necessary must use exactly one of",
            client.repairs[2],
        )
        self.assertIn(
            "writes_authorized must use exactly one of", client.repairs[2]
        )
        self.assertIn("minimally correct this exact prior JSON", client.repairs[1])
        self.assertIn('"arguments_grounded":"partially"', client.repairs[1])

    def test_final_attempt_conservatively_repairs_only_structural_fields(self):
        class FakeClient:
            def __init__(self):
                self.repairs = []

            def complete(self, payload, *, temperature, repair=None):
                self.repairs.append(repair)
                review = dependency_safe_review(turns=(7,))
                for finding in review["dimensions"].values():
                    finding["evidence_turns"] = [7]
                review["dimensions"]["task_completion"]["evidence_turns"] = [
                    7,
                    7,
                    7,
                    7,
                ]
                review["dimensions"]["call_necessity"]["evidence_turns"] = [999]
                review["critical_turns"] = [999]
                assessment = review["multi_turn_assessments"][0]
                assessment["batch_safety"] = "mostly_safe"
                assessment["arguments_grounded"] = "partially"
                assessment["writes_authorized"] = "unknown"
                assessment["calls_necessary"] = "mostly"
                return json.dumps(review), {
                    "provider": "fake",
                    "response_model": "fake-model",
                    "usage": {},
                    "structured_output": True,
                    "response_schema_sha256": "schema-hash",
                }

        client = FakeClient()
        review = review_once(
            client,
            {
                "_dialog_id": "d1",
                "dialog_id": "blind-d1",
                "conversation": [{"turn_index": 7}, {"turn_index": 9}],
                "mechanical_audit_facts": {
                    "multitool_turns": [7, 9],
                    "multiwrite_turns": [9],
                },
            },
            temperature=0,
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            confidence_threshold=0.75,
        )
        self.assertEqual(len(client.repairs), 3)
        self.assertEqual(
            review["dimensions"]["task_completion"]["evidence_turns"], []
        )
        self.assertEqual(
            review["dimensions"]["call_necessity"]["evidence_turns"], []
        )
        self.assertEqual(review["critical_turns"], [])
        assessments = {
            assessment["turn_index"]: assessment
            for assessment in review["multi_turn_assessments"]
        }
        self.assertEqual(set(assessments), {7, 9})
        self.assertEqual(assessments[7]["batch_safety"], "uncertain")
        self.assertEqual(assessments[7]["arguments_grounded"], "uncertain")
        self.assertEqual(
            assessments[7]["writes_authorized"], "not_applicable"
        )
        self.assertEqual(assessments[7]["calls_necessary"], "uncertain")
        self.assertEqual(assessments[9]["batch_safety"], "uncertain")
        self.assertEqual(assessments[9]["writes_authorized"], "uncertain")
        self.assertLess(review["confidence"], 0.75)
        self.assertEqual(review["quality_verdict"], "keep")
        diagnostic = review["conservative_repair"]
        self.assertTrue(diagnostic["applied"])
        self.assertEqual(diagnostic["version"], REVIEW_NORMALIZATION_VERSION)
        self.assertEqual(diagnostic["defaulted_multi_turns"], [9])
        consensus = _quality_consensus(
            judge_record(review), judge_record(review), {9}
        )
        self.assertFalse(consensus["both_reliable"])
        self.assertFalse(consensus["strict_batch_safe"])
        self.assertFalse(consensus["keep_candidate"])

    def test_final_attempt_keeps_non_allowlisted_semantic_errors_hard(self):
        class FakeClient:
            def complete(self, payload, *, temperature, repair=None):
                review = dependency_safe_review(turns=(3,))
                review["quality_verdict"] = "promote"
                return json.dumps(review), {
                    "provider": "fake",
                    "usage": {},
                    "structured_output": True,
                }

        with self.assertRaises(ReviewFailure) as raised:
            review_once(
                FakeClient(),
                {
                    "_dialog_id": "d1",
                    "dialog_id": "blind-d1",
                    "conversation": [{"turn_index": 3}],
                    "mechanical_audit_facts": {
                        "multitool_turns": [3],
                        "multiwrite_turns": [],
                    },
                },
                temperature=0,
                review_mode="quality",
                protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            )
        attempts = raised.exception.attempt_diagnostics
        self.assertEqual(len(attempts), 3)
        self.assertIn("raw_response", attempts[0])
        self.assertEqual(
            attempts[-1]["conservative_repair"]["non_allowlisted_errors"],
            ["invalid quality_verdict"],
        )

    def test_json_parse_repair_includes_prior_response(self):
        class FakeClient:
            def __init__(self):
                self.repairs = []

            def complete(self, payload, *, temperature, repair=None):
                self.repairs.append(repair)
                if len(self.repairs) == 1:
                    return "not-json", {
                        "provider": "fake",
                        "usage": {},
                        "structured_output": False,
                    }
                return json.dumps(dependency_safe_review(turns=())), {
                    "provider": "fake",
                    "usage": {},
                    "structured_output": True,
                }

        client = FakeClient()
        review = review_once(
            client,
            {
                "_dialog_id": "d1",
                "dialog_id": "blind-d1",
                "conversation": [{"turn_index": 3}],
                "mechanical_audit_facts": {
                    "multitool_turns": [],
                    "multiwrite_turns": [],
                },
            },
            temperature=0,
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
        )
        self.assertEqual(review["confidence"], 0.9)
        self.assertIn("repair this exact prior response", client.repairs[1])
        self.assertIn("not-json", client.repairs[1])

    def test_repair_prompt_lists_literal_valid_turn_indices(self):
        payload = {
            "_dialog_id": "d1",
            "dialog_id": "blind-d1",
            "conversation": [
                {"turn_index": 2, "role": "user", "content": "help"},
                {"turn_index": 7, "role": "assistant", "content": ""},
            ],
            "mechanical_audit_facts": {"multitool_turns": [7]},
        }
        messages = request_messages(
            payload,
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            repair="tool_choice: invalid evidence_turns",
        )
        repair_prompt = messages[-1]["content"]
        self.assertIn("literal trajectory turn_index values: [2, 7]", repair_prompt)
        self.assertIn(
            "multi_turn_assessments array must use exactly these turn values: [7]",
            repair_prompt,
        )

    def test_review_payload_raises_when_all_configured_judges_fail(self):
        class FailingClient:
            def __init__(self, label):
                self.label = label

            def complete(self, payload, *, temperature, repair=None):
                raise RuntimeError(f"{self.label} unavailable")

        payload = {
            "_dialog_id": "d1",
            "dialog_id": "blind-d1",
            "conversation": [{"turn_index": 1, "role": "user", "content": "help"}],
            "mechanical_audit_facts": {"multitool_turns": []},
        }
        with self.assertRaisesRegex(
            ReviewFailure,
            "all configured judge attempts failed.*primary unavailable.*fallback unavailable",
        ):
            review_payload(
                payload,
                FailingClient("primary"),
                FailingClient("fallback"),
                primary_passes=1,
                confidence_threshold=0.75,
                repeat_temperature=0.2,
                force_fallback=False,
            )

    def test_main_persists_redacted_hard_failure_attempts_and_binds_repair_version(self):
        secret = "sk-test-secret-value-123456"
        failure = ReviewFailure(
            f"judge failed with Bearer {secret}",
            [],
            [
                {
                    "validation_attempt": 1,
                    "raw_response": f'{{"leaked":"{secret}"}}',
                    "validation_errors": ["invalid quality_verdict"],
                }
            ],
        )
        primary = SimpleNamespace(
            label="local:test",
            url="http://local.test/v1/chat/completions",
            model="test-model",
            max_tokens=128,
            extra_body={},
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "payloads.jsonl"
            output_path = Path(directory) / "reviews.jsonl"
            write_jsonl(
                input_path,
                [
                    {
                        "_dialog_id": "d1",
                        "dialog_id": "blind-d1",
                        "prompt_version": MULTITOOL_PROMPT_VERSION,
                        "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
                        "review_mode": "quality",
                        "conversation": [{"turn_index": 3}],
                        "mechanical_audit_facts": {
                            "multitool_turns": [3],
                            "multiwrite_turns": [],
                        },
                    }
                ],
            )
            argv = [
                "judge_sft_quality.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--protocol-profile",
                PROTOCOL_DEPENDENCY_SAFE_MULTI,
                "--concurrency",
                "1",
            ]
            with mock.patch.dict(os.environ, {"TEST_API_KEY": secret}):
                with mock.patch(
                    "judge_sft_quality.client_from_args",
                    side_effect=[primary, None],
                ):
                    with mock.patch(
                        "judge_sft_quality.review_payload", side_effect=failure
                    ) as mocked_review:
                        with mock.patch.object(sys, "argv", argv):
                            with self.assertRaises(SystemExit) as raised:
                                judge_main()

            self.assertEqual(raised.exception.code, 1)
            sidecar = failure_sidecar_path(output_path)
            serialized = sidecar.read_text(encoding="utf-8")
            self.assertNotIn(secret, serialized)
            self.assertIn("<redacted>", serialized)
            record = json.loads(serialized)
            self.assertEqual(record["dialog_id"], "d1")
            self.assertEqual(record["attempts"][0]["validation_attempt"], 1)
            self.assertEqual(
                record["review_normalization_version"],
                REVIEW_NORMALIZATION_VERSION,
            )
            judge_config = mocked_review.call_args.kwargs["judge_config"]
            self.assertEqual(
                judge_config["review_normalization_version"],
                REVIEW_NORMALIZATION_VERSION,
            )

    def test_every_calibration_sample_has_a_category(self):
        features = []
        payloads = []
        for index in range(10):
            feature = {
                "dialog_id": f"d{index}",
                "domain": "airline",
                "workflow_type": "read_only",
                "length_bucket": "short",
                "deterministic_clean_success": index == 0,
                "source_correct": 0,
                "source_reward": 0,
                "hard_issues": ["source_correct_not_one"],
            }
            features.append(feature)
            payloads.append({"_dialog_id": feature["dialog_id"]})
        _, calibration = build_samples(features, payloads, calibration_limit=10)
        self.assertEqual(len(calibration), 10)
        self.assertTrue(
            all(item["features"].get("calibration_category") for item in calibration)
        )

    def test_calibration_gate_measures_local_reviews_not_fallback(self):
        sample = [
            {
                "features": {
                    "dialog_id": "d1",
                    "static": {"hard_issues": ["multi_tool_policy_violation"]},
                }
            }
        ]
        local_first = valid_review(confidence=0.6)
        local_second = valid_review(confidence=0.6)
        fallback = valid_review(verdict="drop", confidence=0.99)
        metrics = calibration_metrics(
            sample,
            [
                {
                    "dialog_id": "d1",
                    "primary_reviews": [local_first, local_second],
                    "selected_review": fallback,
                    "unresolved_escalation": False,
                }
            ],
        )
        self.assertEqual(metrics["complete_primary_pairs"], 1)
        self.assertEqual(metrics["hard_issue_keep_rate"], 1)
        self.assertEqual(metrics["median_confidence"], 0.6)
        self.assertFalse(metrics["passed"])

    def test_judge_diagnostics_separates_rule_clean_candidates(self):
        features = [
            {
                "dialog_id": "clean",
                "domain": "airline",
                "source_correct": 1,
                "source_reward": 1,
                "source_row_count": 1,
                "included_row_count": 1,
                "deterministic_clean_success": True,
                "static": {"multitool_turns": [], "multiwrite_turns": []},
            },
            {
                "dialog_id": "dirty",
                "domain": "retail",
                "source_correct": 1,
                "source_reward": 1,
                "source_row_count": 2,
                "included_row_count": 1,
                "deterministic_clean_success": False,
                "static": {"multitool_turns": [2], "multiwrite_turns": [2]},
            },
        ]
        clean_review = valid_review()
        dirty_review = valid_review(
            verdict="drop",
            dimensions={
                name: {"score": 2, "evidence_turns": [1], "rationale": "weak"}
                for name in DIMENSIONS
            },
        )
        diagnostics = judge_diagnostics(
            features,
            {
                "clean": {
                    "selected_review": clean_review,
                    "selected_provider": "local:model",
                    "escalation_reasons": [],
                    "unresolved_escalation": False,
                    "errors": [],
                },
                "dirty": {
                    "selected_review": dirty_review,
                    "selected_provider": "fallback:model",
                    "primary_reviews": [clean_review],
                    "fallback_review": dirty_review,
                    "escalation_reasons": ["low_confidence"],
                    "unresolved_escalation": False,
                    "errors": [],
                },
            },
        )
        self.assertEqual(diagnostics["selected_reviews"], 2)
        self.assertEqual(diagnostics["escalated_dialogs"], 1)
        self.assertEqual(
            diagnostics["dimension_quality"]["all_reviewed"]["tool_choice"]["mean_score"],
            3,
        )
        self.assertEqual(
            diagnostics["dimension_quality"]["positive_rule_clean"]["tool_choice"][
                "mean_score"
            ],
            4,
        )
        self.assertEqual(
            diagnostics["dimension_quality"]["positive_same_turn_multi_with_write"]
            ["tool_choice"]["mean_score"],
            2,
        )
        self.assertEqual(diagnostics["cohort_sizes"]["domain_airline"], 1)
        self.assertEqual(diagnostics["cohort_sizes"]["domain_telecom"], 0)
        self.assertEqual(diagnostics["cohort_sizes"]["positive_fully_retained"], 1)
        self.assertEqual(
            diagnostics["cohort_sizes"]["positive_truncated_training_view"], 1
        )
        self.assertEqual(
            diagnostics["cohort_sizes"]["positive_airline_fully_retained"], 1
        )
        self.assertEqual(
            diagnostics["cohort_sizes"]["positive_airline_truncated_training_view"], 0
        )
        self.assertEqual(diagnostics["local_fallback_pairs"], 1)
        self.assertEqual(diagnostics["local_fallback_verdict_agreement"], 0)
        self.assertEqual(
            diagnostics["local_fallback_dimension_delta"]["tool_choice"],
            {"mean_fallback_minus_local": -2, "mean_absolute_difference": 2},
        )
        self.assertEqual(
            diagnostics["source_label_alignment"]["positive"]["verdicts"],
            {"drop": 1, "keep": 1},
        )

    def test_mechanical_facts_override_false_fallback_claim(self):
        local = valid_review(apparent_task_outcome="uncertain")
        fallback = valid_review(
            verdict="drop",
            critical_issues=["multi_tool_policy_violation"],
        )
        record = {
            "primary_provider": "local:model",
            "primary_reviews": [local],
            "fallback_provider": "fallback:model",
            "fallback_review": fallback,
            "selected_provider": "fallback:model",
            "selected_review": fallback,
            "unresolved_escalation": False,
        }
        adjudicated = adjudicate_review_record(
            {"static": {"multitool_turns": [], "namespace_confusion_turns": []}},
            record,
        )
        self.assertIs(adjudicated["selected_review"], local)
        self.assertEqual(adjudicated["selected_provider"], "local:model")
        self.assertTrue(adjudicated["unresolved_escalation"])
        self.assertEqual(
            adjudicated["adjudication_overrides"][0]["contradictions"],
            ["false_multi_tool_policy_violation"],
        )
        self.assertIs(record["selected_review"], fallback)

        optimistic_fallback = valid_review()
        local_drop = valid_review(
            verdict="drop",
            critical_issues=["multi_tool_policy_violation"],
        )
        optimistic = adjudicate_review_record(
            {
                "static": {
                    "hard_issues": ["multi_tool_policy_violation"],
                    "multitool_turns": [1],
                    "namespace_confusion_turns": [],
                }
            },
            {
                **record,
                "primary_reviews": [local_drop],
                "fallback_review": optimistic_fallback,
                "selected_review": optimistic_fallback,
            },
        )
        self.assertIs(optimistic["selected_review"], local_drop)
        self.assertEqual(
            optimistic["adjudication_overrides"][0]["contradictions"],
            ["keep_despite_multi_tool_policy_violation"],
        )

        no_replacement = adjudicate_review_record(
            {"static": {"multitool_turns": [], "namespace_confusion_turns": []}},
            {
                **record,
                "primary_reviews": [fallback],
                "selected_review": fallback,
            },
        )
        diagnostics = judge_diagnostics(
            [{"dialog_id": "d1", "deterministic_clean_success": True}],
            {"d1": no_replacement},
        )
        self.assertIsNone(no_replacement["selected_review"])
        self.assertEqual(diagnostics["selected_provider_counts"], {"none": 1})

    def test_review_map_ignores_stale_prompt_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.jsonl"
            write_jsonl(
                path,
                [
                    {"dialog_id": "d1", "prompt_version": "stale"},
                    {"dialog_id": "d1", "prompt_version": PROMPT_VERSION},
                ],
            )
            reviews = load_review_map(path)
        self.assertEqual(list(reviews), ["d1"])
        self.assertEqual(reviews["d1"]["prompt_version"], PROMPT_VERSION)

    def test_filtered_rows_preserves_selected_training_data_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            write_jsonl(
                path,
                [
                    {"messages": [], "metadata": {"source_dialog_id": "drop"}},
                    {
                        "messages": [{"role": "assistant", "content": "kept"}],
                        "metadata": {"source_dialog_id": "keep", "source_row": 7},
                    },
                ],
            )
            rows = list(filtered_rows(path, {"keep"}))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["messages"][0]["content"], "kept")
        self.assertEqual(rows[0]["metadata"]["source_row"], 7)
        self.assertEqual(rows[0]["metadata"]["sft_quality_tier"], "keep")

    def test_success_cases_uses_dimension_evidence_when_strengths_are_empty(self):
        review = valid_review(strengths=[])
        report = success_cases(
            [
                {
                    "dialog_id": "d1",
                    "domain": "airline",
                    "tier": "keep",
                    "quality_score": 4.0,
                }
            ],
            {"d1": {"selected_review": review}},
        )
        self.assertIn("task_completion: supported @ [1]", report)

    def test_success_cases_selects_retention_and_workflow_strata(self):
        categories = [
            ("full_write", True, "state_changing"),
            ("full_read", True, "read_only"),
            ("prefix_write", False, "state_changing"),
            ("prefix_read", False, "read_only"),
            ("zz_extra", False, "read_only"),
        ]
        manifest = [
            {
                "dialog_id": dialog_id,
                "domain": "retail",
                "tier": "keep",
                "quality_score": 4.0,
                "fully_retained": retained,
                "workflow_type": workflow,
            }
            for dialog_id, retained, workflow in categories
        ]
        reviews = {
            dialog_id: {"selected_review": valid_review()}
            for dialog_id, _, _ in categories
        }
        report = success_cases(manifest, reviews)
        for dialog_id, _, _ in categories[:4]:
            self.assertIn(f"`{dialog_id}`", report)
        self.assertNotIn("`zz_extra`", report)

    def test_success_cases_includes_rule_clean_semantic_counterexamples(self):
        review = valid_review(
            verdict="review",
            critical_issues=["incomplete_task"],
            weaknesses=[
                {
                    "pattern": "missed second intent",
                    "evidence_turns": [1],
                    "explanation": "The second requested action was omitted.",
                }
            ],
        )
        report = success_cases(
            [
                {
                    "dialog_id": "d1",
                    "domain": "retail",
                    "tier": "review",
                    "quality_score": 2.5,
                    "deterministic_clean_success": True,
                }
            ],
            {"d1": {"selected_review": review}},
        )
        self.assertIn("规则未覆盖的语义反例", report)
        self.assertIn("incomplete_task: missed second intent", report)

    def test_audit_hashes_actual_conversion_but_uses_raw_semantic_training_view(self):
        system = {
            "role": "system",
            "content": "<policy>one call at a time</policy> airline",
        }
        user = {"role": "user", "content": "Please book"}
        answer = {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("get_user_details", {"user_id": "U1"}, "c1")],
        }
        converted_messages = [
            {
                "role": "system",
                "content": system["content"] + "\nIn each turn make one or more tool calls.",
            },
            user,
            {
                "role": "assistant",
                "content": (
                    '<tool_call>{"name":"get_user_details",'
                    '"arguments":{"user_id":"U1"}}</tool_call>'
                ),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            training = Path(directory) / "training.jsonl"
            write_jsonl(source, [source_row(0, [system, user], answer)])
            write_jsonl(training, [converted_row(0, 0, converted_messages)])
            features, _, inventory = audit_source_dialogs(
                source,
                training,
                CATALOG,
                protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            )

        feature = features[0]
        conversion = feature["conversion_checks"][0]
        self.assertTrue(conversion["non_system_messages_match"])
        self.assertTrue(conversion["source_system_preserved"])
        self.assertTrue(conversion["target_fingerprint_match"])
        self.assertEqual(feature["conversion_mismatch_source_rows"], [])
        self.assertEqual(inventory["conversion_mismatch_rows"], 0)
        self.assertEqual(
            feature["actual_training_messages_hash"],
            stable_json_hash(row_messages({"messages": converted_messages})),
        )
        # The converted target is text, but semantic audit facts must come from
        # the raw source prefix, where the structured tool call is still visible.
        self.assertEqual(feature["training_visible_static"]["num_tool_calls"], 1)
        self.assertEqual(
            feature["training_visible_static"]["tool_names"], ["get_user_details"]
        )

    def test_protocol_profile_selects_training_prefix_or_full_source_for_review(self):
        system = {
            "role": "system",
            "content": "<policy>one call at a time</policy> airline",
        }
        user = {"role": "user", "content": "Please book"}
        first_answer = {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("get_user_details", {"user_id": "U1"}, "c1")],
        }
        tool_result = {"role": "tool", "id": "c1", "content": '{"id":"U1"}'}
        second_answer = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                call("book_reservation", {"user_id": "U1"}, "c2"),
                call("get_user_details", {"user_id": "U1"}, "c3"),
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            training = Path(directory) / "training.jsonl"
            write_jsonl(
                source,
                [
                    source_row(0, [system, user], first_answer),
                    source_row(
                        1,
                        [system, user, first_answer, tool_result],
                        second_answer,
                    ),
                ],
            )
            write_jsonl(
                training,
                [converted_row(0, 0, [system, user, first_answer])],
            )
            _, current_payloads, _ = audit_source_dialogs(source, training, CATALOG)
            _, dependency_safe_payloads, _ = audit_source_dialogs(
                source,
                training,
                CATALOG,
                protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            )

        current = current_payloads[0]
        dependency_safe = dependency_safe_payloads[0]
        self.assertEqual(current["conversation_view"], "training_visible")
        self.assertEqual(len(current["conversation"]), 2)
        self.assertEqual(current["mechanical_audit_facts"]["multitool_turns"], [])
        self.assertEqual(dependency_safe["conversation_view"], "full_source")
        self.assertEqual(len(dependency_safe["conversation"]), 4)
        self.assertEqual(
            dependency_safe["mechanical_audit_facts"]["multitool_turns"], [4]
        )
        self.assertEqual(
            dependency_safe["training_visible_view"]["mechanical_audit_facts"]
            ["multitool_turns"],
            [],
        )

    def test_dependency_safe_quality_payload_is_blinded_and_content_addressed(self):
        one_call_rule = (
            "You should only make one tool call at a time, and if you make a tool "
            "call, you should not respond to the user simultaneously. If you respond "
            "to the user, you should not make a tool call at the same time."
        )
        source_payload = {
            "_dialog_id": "real-id",
            "dialog_id": "blind-id",
            "original_domain_policy": one_call_rule + " Keep account data private.",
            "label_group": "consensus_failure",
            "source_correct": 0,
            "source_reward": 0,
            "conversation": [{"turn_index": 1, "role": "user", "content": "help"}],
            "mechanical_audit_facts": {"multitool_turns": []},
        }
        feature = {
            "source_correct": 0,
            "source_reward": 0,
            "label_group": "consensus_failure",
            "full_source_static": {
                "has_multitool_turn": False,
                "multitool_turns": [],
            },
        }
        payload = _counterfactual_payload(feature, source_payload, review_mode="quality")

        for hidden in (
            "original_domain_policy",
            "label_group",
            "source_correct",
            "source_reward",
            "known_outcome",
        ):
            self.assertNotIn(hidden, payload)
        self.assertNotIn("one tool call at a time", payload["domain_policy"].lower())
        self.assertIn("Keep account data private", payload["domain_policy"])
        self.assertEqual(payload["payload_sha256"], payload_hash(payload))
        changed = dict(payload, domain="different")
        self.assertNotEqual(changed["payload_sha256"], payload_hash(changed))

        rendered = json.dumps(
            request_messages(
                payload,
                review_mode="quality",
                protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            ),
            ensure_ascii=False,
        )
        self.assertNotIn('"label_group"', rendered)
        self.assertNotIn('"source_correct"', rendered)
        self.assertNotIn('"source_reward"', rendered)
        self.assertNotIn('"original_domain_policy"', rendered)

    def test_long_payload_routes_through_hashed_projected_local_window(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                characters = sum(len(message["content"]) for message in messages)
                return [0] * ((characters + 7) // 8)

        conversation = [
            {
                "turn_index": index,
                "role": "assistant" if index == 50 else "user",
                "content": "x" * 500,
                **(
                    {
                        "tool_calls": [
                            call("get_user_details", {"user_id": "U1"}, "c1"),
                            call("get_user_details", {"user_id": "U2"}, "c2"),
                        ]
                    }
                    if index == 50
                    else {}
                ),
            }
            for index in range(100)
        ]
        payload = {
            "_dialog_id": "long-id",
            "dialog_id": "blind-long",
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "domain_policy": "dependency-safe multi",
            "conversation": conversation,
            "agent_tools": [],
            "user_tools": [],
            "mechanical_audit_facts": {
                "has_multitool_turn": True,
                "multitool_turns": [50],
                "multitool_details": [
                    {
                        "turn_index": 50,
                        "batch_size": 2,
                        "read_call_count": 2,
                        "write_call_count": 0,
                        "duplicate_call_indices": [],
                        "result_associations": [],
                        "unassociated_result_turns": [],
                        "result_message_hashes": ["must-not-leak-into-projection"],
                        "calls": [
                            {
                                "call_index": 0,
                                "id": "c1",
                                "name": "get_user_details",
                                "arguments": {"user_id": "U1"},
                                "arguments_parse_error": False,
                                "requestor": "assistant",
                                "is_write": False,
                            }
                        ],
                    }
                ],
            },
        }
        payload["payload_sha256"] = payload_hash(payload)

        local, long_payloads, windows = _split_payloads(
            [payload],
            tokenizer=FakeTokenizer(),
            review_mode="quality",
            local_input_limit=2500,
        )
        self.assertEqual(local, [])
        self.assertEqual(len(long_payloads), 1)
        self.assertEqual(len(windows), 1)
        full_payload = long_payloads[0]
        window = windows[0]
        self.assertGreater(full_payload["judge_input_tokens"], 2500)
        self.assertLessEqual(window["judge_input_tokens"], 2500)
        self.assertLess(len(window["conversation"]), len(conversation))
        self.assertEqual(
            window["source_full_payload_sha256"], full_payload["payload_sha256"]
        )
        self.assertEqual(full_payload["payload_sha256"], payload_hash(full_payload))
        self.assertEqual(window["payload_sha256"], payload_hash(window))
        detail = window["mechanical_audit_facts"]["multitool_details"][0]
        self.assertNotIn("result_message_hashes", detail)
        self.assertEqual(window["evidence_scope"]["all_event_turn_indices"], [50])

    def test_dependency_safe_schema_requires_exact_per_turn_coverage_and_reconciles_facts(self):
        review = dependency_safe_review()
        self.assertEqual(
            validate_review_output(
                review,
                {3, 5},
                review_mode="quality",
                protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
                expected_multi_turns={3, 5},
            ),
            [],
        )
        incomplete = dependency_safe_review(turns=(3,))
        self.assertIn(
            "multi_turn_assessments must cover exactly [3, 5]",
            validate_review_output(
                incomplete,
                {3, 5},
                review_mode="quality",
                protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
                expected_multi_turns={3, 5},
            ),
        )

        reconciled = reconcile_mechanical_facts(
            review,
            {
                "mechanical_audit_facts": {
                    "has_multitool_turn": True,
                    "multitool_turns": [3, 5],
                    "namespace_confusion_turns": [3],
                    "tool_argument_schema_errors": [],
                    "invalid_tool_names": ["invented"],
                    "multitool_details": [
                        {
                            "turn_index": 3,
                            "calls": [{"name": "invented"}],
                            "duplicate_call_indices": [],
                        },
                        {
                            "turn_index": 5,
                            "calls": [],
                            "duplicate_call_indices": [0, 1],
                        },
                    ],
                }
            },
            "quality",
            PROTOCOL_DEPENDENCY_SAFE_MULTI,
        )
        assessments = {
            item["turn_index"]: item for item in reconciled["multi_turn_assessments"]
        }
        self.assertEqual(assessments[3]["batch_safety"], "unsafe_tool_or_args")
        self.assertEqual(assessments[5]["batch_safety"], "unsafe_redundant")
        self.assertEqual(assessments[5]["calls_necessary"], "no")
        self.assertEqual(reconciled["batch_safety"], "unsafe_tool_or_args")

    def test_dependency_safe_schema_enforces_write_authorization_applicability(self):
        invalid_write = dependency_safe_review()
        errors = validate_review_output(
            invalid_write,
            {3, 5},
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            expected_multi_turns={3, 5},
            expected_write_turns={3},
        )
        self.assertIn(
            "write multi_turn_assessment cannot use "
            "writes_authorized=not_applicable",
            errors,
        )

        invalid_read = dependency_safe_review()
        invalid_read["multi_turn_assessments"][0]["writes_authorized"] = "yes"
        invalid_read["multi_turn_assessments"][1]["writes_authorized"] = "yes"
        errors = validate_review_output(
            invalid_read,
            {3, 5},
            review_mode="quality",
            protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            expected_multi_turns={3, 5},
            expected_write_turns={3},
        )
        self.assertNotIn(
            "write multi_turn_assessment cannot use "
            "writes_authorized=not_applicable",
            errors,
        )
        self.assertIn(
            "read-only multi_turn_assessment must use "
            "writes_authorized=not_applicable",
            errors,
        )

        valid = dependency_safe_review()
        valid["multi_turn_assessments"][0]["writes_authorized"] = "yes"
        self.assertEqual(
            validate_review_output(
                valid,
                {3, 5},
                review_mode="quality",
                protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
                expected_multi_turns={3, 5},
                expected_write_turns={3},
            ),
            [],
        )

    def test_payload_manifest_routes_and_summarize_rejects_stale_hash(self):
        local_payload = {
            "_dialog_id": "d-local",
            "dialog_id": "blind-local",
            "review_mode": "quality",
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "judge_input_tokens": 100,
            "payload_sha256": "local-hash",
            "conversation": [
                {"turn_index": 1, "role": "user"},
                {
                    "turn_index": 2,
                    "role": "assistant",
                    "tool_calls": [call("first"), call("second")],
                },
            ],
        }
        long_payload = {
            **local_payload,
            "_dialog_id": "d-long",
            "dialog_id": "blind-long",
            "judge_input_tokens": 40000,
            "payload_sha256": "full-hash",
        }
        window = {
            **long_payload,
            "judge_input_tokens": 1000,
            "source_full_payload_sha256": "full-hash",
            "payload_sha256": "window-hash",
            "evidence_scope": {"kind": "deterministic_event_windows"},
        }
        manifest = build_payload_manifest(
            "quality", [local_payload], [long_payload], [window]
        )
        self.assertEqual(
            {row["dialog_id"]: row["route"] for row in manifest},
            {"d-local": "full/full", "d-long": "window/full"},
        )
        local_row = next(row for row in manifest if row["dialog_id"] == "d-local")
        self.assertEqual(
            local_row["full_source_turn_types"],
            {"assistant_multi": [2], "user": [1]},
        )
        long_row = next(row for row in manifest if row["dialog_id"] == "d-long")
        self.assertEqual(long_row["local_input_payload_sha256"], "window-hash")
        self.assertEqual(long_row["fallback_input_payload_sha256"], "full-hash")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.jsonl"
            selection = root / "comparison_selection.jsonl"
            pairs = root / "comparison_pairs.jsonl"
            payload_manifest = root / "payload_manifest.jsonl"
            calibration_pilot = root / "calibration_pilot_selection.jsonl"
            calibration_v1 = root / "calibration_validation_v1_selection.jsonl"
            calibration = root / "calibration_selection.jsonl"
            failure_calibration_pilot = (
                root / "failure_calibration_pilot_selection.jsonl"
            )
            failure_calibration = root / "failure_calibration_selection.jsonl"
            inventory = root / "inventory.json"
            write_jsonl(features, [])
            write_jsonl(selection, [])
            write_jsonl(pairs, [])
            write_jsonl(payload_manifest, manifest)
            write_jsonl(calibration_pilot, [])
            write_jsonl(calibration_v1, [])
            write_jsonl(calibration, [])
            write_jsonl(failure_calibration_pilot, [])
            write_jsonl(failure_calibration, [])
            inventory.write_text(
                json.dumps(
                    {
                        "payload_manifest_sha256": stable_json_hash(manifest + [{}]),
                        "calibration_dialogs": 0,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                features=features,
                inventory=inventory,
                selection=selection,
                pairs=pairs,
                payload_manifest=payload_manifest,
                calibration=calibration,
            )
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                summarize(args)

    def test_review_manifest_rejects_mismatched_judge_config_hash(self):
        expected = {
            "d1": {
                "dialog_id": "d1",
                "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
                "prompt_version": MULTITOOL_PROMPT_VERSION,
                "local_input_payload_sha256": "input-hash",
                "fallback_input_payload_sha256": "input-hash",
                "source_full_payload_sha256": "full-hash",
                "local_evidence_scope": {"kind": "full_source"},
            }
        }
        record = {
            "dialog_id": "d1",
            "review_mode": "quality",
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "input_payload_sha256": "input-hash",
            "source_full_payload_sha256": "full-hash",
            "evidence_scope": {"kind": "full_source"},
            "judge_config": {"model": "actual-model", "temperature": 0},
            "judge_config_sha256": stable_json_hash(
                {"model": "different-model", "temperature": 0}
            ),
            "selected_review": {},
            "selected_provider": "local:actual-model",
        }
        with tempfile.TemporaryDirectory() as directory:
            reviews = Path(directory) / "reviews.jsonl"
            write_jsonl(reviews, [record])
            with self.assertRaisesRegex(ValueError, "judge_config_sha256"):
                _validated_review_map(
                    reviews,
                    expected,
                    review_mode="quality",
                    side="local",
                )

    def test_explicit_cli_artifact_must_match_sealed_prepare_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared-features.jsonl"
            explicit = root / "explicit-features.jsonl"
            prepared.write_text('{"dialog_id":"sealed"}\n', encoding="utf-8")
            explicit.write_text('{"dialog_id":"different"}\n', encoding="utf-8")
            manifest = {
                "artifacts": {
                    "features.jsonl": {"sha256": file_sha256(prepared)},
                }
            }

            with self.assertRaisesRegex(
                ValueError, "does not match sealed prepare artifact features.jsonl"
            ):
                validate_selected_artifacts(
                    manifest,
                    {"features.jsonl": explicit},
                )

    def test_unreliable_agreement_stays_review_or_unknown(self):
        high_confidence_keep = dependency_safe_review(
            turns=(), batch_safety="not_multi", confidence=0.9
        )
        low_confidence_keep = dependency_safe_review(
            turns=(), batch_safety="not_multi", confidence=0.6
        )
        quality = _quality_consensus(
            judge_record(high_confidence_keep),
            judge_record(low_confidence_keep),
        )
        self.assertFalse(quality["both_reliable"])
        self.assertEqual(quality["quality_verdict"], "review")
        self.assertEqual(quality["batch_safety"], "uncertain")
        self.assertFalse(quality["strict_batch_safe"])
        self.assertIn(
            "low_confidence", quality["fallback_reliability"]["reasons"]
        )

        feature = {
            "dialog_id": "failed",
            "domain": "airline",
            "label_group": "consensus_failure",
            "source_correct": 0,
            "source_reward": 0,
            "full_source_static": {"multitool_turns": []},
        }
        agreeing_local = failure_judge_record(
            "wrong_workflow", "single_call_turn", "not_applicable"
        )
        agreeing_but_unresolved = failure_judge_record(
            "wrong_workflow",
            "single_call_turn",
            "not_applicable",
            unresolved=True,
        )
        _, failure = failure_report(
            [feature],
            {"failed": agreeing_local},
            {"failed": agreeing_but_unresolved},
        )
        self.assertEqual(failure["unreliable_judge_pairs"], 1)
        self.assertEqual(failure["field_agreement"], {})
        self.assertEqual(
            failure["field_disagreement"],
            {
                "causal_call_mode": 1,
                "primary_failure_cause": 1,
            },
        )
        self.assertEqual(
            failure["primary_failure_causes_by_label"]["all_non_consensus"],
            {"unknown": 1},
        )

    def test_dimension_disagreement_excludes_dimension_and_mean_quality(self):
        dimension = DEPENDENCY_SAFE_QUALITY_DIMENSIONS[0]
        local = dependency_safe_review(turns=(), batch_safety="not_multi")
        fallback = dependency_safe_review(turns=(), batch_safety="not_multi")
        local["dimensions"][dimension]["score"] = 4
        fallback["dimensions"][dimension]["score"] = 2

        consensus = _quality_consensus(
            judge_record(local),
            judge_record(fallback),
        )

        self.assertIsNone(consensus["dimensions"][dimension])
        self.assertIsNone(consensus["mean_quality"])
        self.assertFalse(consensus["dimension_within_one"])

    def test_calibration_false_keep_uses_raw_reviews_even_when_unresolved(self):
        review = dependency_safe_review(turns=(), batch_safety="not_multi")
        feature = {
            "dialog_id": "hard",
            "domain": "airline",
            "full_source_static": {
                "mechanical_hard_issues": ["invalid_tool_name"],
                "multitool_turns": [],
                "multiwrite_turns": [],
            },
        }
        _, report = calibration_report(
            [{"dialog_id": "hard", "route": "full/full"}],
            [feature],
            {"hard": judge_record(review, unresolved=True)},
            {"hard": judge_record(review, unresolved=True)},
        )

        self.assertEqual(report["metrics"]["mechanical_hard_both_keep"], 1)
        self.assertFalse(
            report["gates"]["mechanical_false_keep_at_most_0.05"]
        )
        self.assertEqual(report["status"], "fail")

    def test_single_dialog_is_excluded_from_multi_per_turn_denominator(self):
        review = dependency_safe_review(
            turns=(), batch_safety="not_multi", confidence=0.9
        )
        feature = {
            "dialog_id": "single",
            "domain": "airline",
            "full_source_static": {
                "multitool_turns": [],
                "multiwrite_turns": [],
                "mechanical_hard_issues": [],
            },
        }
        _, report = calibration_report(
            [{"dialog_id": "single", "route": "full/full"}],
            [feature],
            {"single": judge_record(review)},
            {"single": judge_record(review)},
        )
        metrics = report["metrics"]
        self.assertEqual(metrics["dialogs"], 1)
        self.assertEqual(metrics["verdict_agreement"], 1)
        self.assertEqual(metrics["per_turn_exact_agreement"], 0)
        self.assertEqual(metrics["per_turn_agreement_multi_dialogs"], 0)
        self.assertEqual(metrics["by_route"]["full/full"]["multi_dialogs"], 0)

    def test_quality_validation_is_deterministic_and_disjoint_from_pilot(self):
        features = []
        selection = []
        manifest = []
        for index in range(240):
            dialog_id = f"quality-{index:03d}"
            features.append(
                {
                    "dialog_id": dialog_id,
                    "domain": ("airline", "retail", "telecom")[index % 3],
                    "label_group": "consensus_success",
                    "full_source_static": {
                        "multitool_turns": [3] if index % 2 else [],
                        "multiwrite_turns": [],
                        "mechanical_hard_issues": (
                            ["invalid_tool_name"] if index % 19 == 0 else []
                        ),
                    },
                }
            )
            selection.append({"dialog_id": dialog_id})
            manifest.append(
                {
                    "dialog_id": dialog_id,
                    "route": "window/full" if index % 11 == 0 else "full/full",
                }
            )

        pilot = build_calibration_selection(features, selection, manifest)
        validation_v1 = build_quality_v1_selection(
            features, selection, manifest
        )
        validation = build_quality_validation_selection(
            features, selection, manifest
        )
        reordered = build_quality_validation_selection(
            list(reversed(features)),
            list(reversed(selection)),
            list(reversed(manifest)),
        )

        self.assertEqual(len(pilot), QUALITY_CALIBRATION_LIMIT)
        self.assertEqual(len(validation_v1), QUALITY_CALIBRATION_LIMIT)
        self.assertEqual(len(validation), QUALITY_CALIBRATION_LIMIT)
        self.assertEqual(validation, reordered)
        self.assertFalse(
            {row["dialog_id"] for row in pilot}
            & {row["dialog_id"] for row in validation_v1}
        )
        self.assertFalse(
            {row["dialog_id"] for row in pilot}
            & {row["dialog_id"] for row in validation}
        )
        self.assertFalse(
            {row["dialog_id"] for row in validation_v1}
            & {row["dialog_id"] for row in validation}
        )
        self.assertTrue(
            all(
                row["selection_hash"]
                == stable_json_hash(
                    [QUALITY_CALIBRATION_VALIDATION_SEED, row["dialog_id"]]
                )
                for row in validation
            )
        )
        self.assertNotEqual(
            QUALITY_CALIBRATION_PILOT_SEED,
            QUALITY_CALIBRATION_V1_SEED,
        )
        self.assertNotEqual(
            QUALITY_CALIBRATION_V1_SEED,
            QUALITY_CALIBRATION_VALIDATION_SEED,
        )

    def test_quality_veto_gate_accepts_scale_shift_without_extreme_conflict(self):
        local_verdicts = (
            ["keep"] * 37 + ["review"] * 34 + ["drop"]
        )
        fallback_verdicts = (
            ["keep"] * 14
            + ["review"] * 45
            + ["drop"] * 12
            + ["review"]
        )
        fallback_verdicts[33:37] = ["drop"] * 4
        features = []
        calibration = []
        local = {}
        fallback = {}
        for index, (local_verdict, fallback_verdict) in enumerate(
            zip(local_verdicts, fallback_verdicts)
        ):
            dialog_id = f"validation-{index:02d}"
            features.append(
                {
                    "dialog_id": dialog_id,
                    "domain": "airline",
                    "full_source_static": {
                        "multitool_turns": [3],
                        "multiwrite_turns": [],
                        "mechanical_hard_issues": (
                            ["invalid_tool_name"] if index == 59 else []
                        ),
                    },
                }
            )
            calibration.append(
                {"dialog_id": dialog_id, "route": "full/full"}
            )
            local[dialog_id] = judge_record(
                dependency_safe_review(quality_verdict=local_verdict)
            )
            fallback[dialog_id] = judge_record(
                dependency_safe_review(quality_verdict=fallback_verdict)
            )

        _, report = calibration_report(
            calibration, features, local, fallback
        )

        self.assertEqual(report["metrics"]["verdict_agreement"], 36)
        self.assertEqual(
            report["metrics"]["fallback_keep_supported_by_local"], 14
        )
        self.assertEqual(report["metrics"]["fallback_keep_dialogs"], 14)
        self.assertEqual(report["metrics"]["extreme_verdict_disagreements"], 4)
        self.assertEqual(report["metrics"]["local_keep_fallback_drop"], 4)
        self.assertEqual(report["metrics"]["fallback_keep_local_not_keep"], 0)
        self.assertNotIn(
            "verdict_agreement_at_least_0.80", report["gates"]
        )
        self.assertNotIn(
            "extreme_verdict_disagreement_at_most_0.05",
            report["gates"],
        )
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(report["status"], "pass")

    def test_cache_scope_union_prunes_obsolete_and_retains_other_partition(self):
        payload = {
            "_dialog_id": "current",
            "dialog_id": "blind-current",
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "review_mode": "quality",
            "conversation_view": "full_source",
            "conversation": [],
        }
        fingerprint = payload_hash(payload)
        judge_config = {"confidence_threshold": 0.75, "model": "judge"}
        config_hash = stable_json_hash(judge_config)
        exact = {
            "dialog_id": "current",
            "review_mode": "quality",
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "input_payload_sha256": fingerprint,
            "source_full_payload_sha256": fingerprint,
            "evidence_scope": {"kind": "full_source"},
            "judge_config_sha256": config_hash,
            "judge_config": judge_config,
            "selected_review": {"confidence": 0.9},
        }
        other_partition = {
            "dialog_id": "other-partition",
            "review_mode": "quality",
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
        }
        obsolete = {
            "dialog_id": "obsolete",
            "review_mode": "quality",
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
        }
        retained, exact_hits, stale = reconcile_cached_records(
            [exact, other_partition, obsolete],
            [payload],
            "quality",
            PROTOCOL_DEPENDENCY_SAFE_MULTI,
            config_hash,
            {"current", "other-partition"},
        )
        self.assertEqual(exact_hits, {"current"})
        self.assertEqual(
            {record["dialog_id"] for record in retained},
            {"current", "other-partition"},
        )
        self.assertEqual(stale, 1)

    def test_cache_rejects_missing_review_and_mismatched_judge_config_hash(self):
        payload = {
            "_dialog_id": "current",
            "dialog_id": "blind-current",
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "review_mode": "quality",
            "conversation_view": "full_source",
            "conversation": [],
        }
        fingerprint = payload_hash(payload)
        judge_config = {"model": "judge-model", "temperature": 0}
        config_hash = stable_json_hash(judge_config)
        base = {
            "dialog_id": "current",
            "review_mode": "quality",
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "input_payload_sha256": fingerprint,
            "source_full_payload_sha256": fingerprint,
            "evidence_scope": {"kind": "full_source"},
            "judge_config": judge_config,
            "judge_config_sha256": config_hash,
            "selected_review": {},
        }
        invalid_records = {
            "missing selected review": {**base, "selected_review": None},
            "mismatched config content hash": {
                **base,
                "judge_config": {"model": "different-model", "temperature": 0},
            },
        }
        for case, record in invalid_records.items():
            with self.subTest(case=case):
                retained, exact_hits, stale = reconcile_cached_records(
                    [record],
                    [payload],
                    "quality",
                    PROTOCOL_DEPENDENCY_SAFE_MULTI,
                    config_hash,
                )
                self.assertEqual(exact_hits, set())
                self.assertEqual(retained, [])
                self.assertEqual(stale, 1)

    def test_judge_identity_uses_actual_url_and_model_not_cosmetic_label(self):
        first = judge_record(
            dependency_safe_review(turns=()),
            label="local:shared-label",
            url="HTTP://judge-a.test/v1/chat/completions/",
            model="Model-A",
        )
        same_label_different_judge = judge_record(
            dependency_safe_review(turns=()),
            label="local:shared-label",
            url="http://judge-b.test/v1/chat/completions",
            model="model-b",
        )
        different_label_same_judge = judge_record(
            dependency_safe_review(turns=()),
            label="fallback:cosmetically-different",
            url="http://judge-a.test/v1/chat/completions",
            model="model-a",
        )
        self.assertNotEqual(
            _selected_judge_identity(first),
            _selected_judge_identity(same_label_different_judge),
        )
        self.assertEqual(
            _selected_judge_identity(first),
            _selected_judge_identity(different_label_same_judge),
        )

    def test_failure_report_uses_luna_primary_and_qwen_as_sensitivity(self):
        features = [
            {
                "dialog_id": "success",
                "domain": "airline",
                "label_group": "consensus_success",
                "source_correct": 1,
                "source_reward": 1,
                "full_source_static": {"multitool_turns": []},
            },
            {
                "dialog_id": "failed",
                "domain": "airline",
                "label_group": "consensus_failure",
                "source_correct": 0,
                "source_reward": 0,
                "full_source_static": {"multitool_turns": []},
            },
            {
                "dialog_id": "reward-only",
                "domain": "retail",
                "label_group": "reward_only",
                "source_correct": 0,
                "source_reward": 1,
                "full_source_static": {"multitool_turns": [3]},
            },
            {
                "dialog_id": "correct-only",
                "domain": "telecom",
                "label_group": "correct_only",
                "source_correct": 1,
                "source_reward": 0,
                "full_source_static": {"multitool_turns": []},
            },
        ]

        local = {
            "failed": failure_judge_record(
                "wrong_workflow", "single_call_turn", "not_applicable"
            ),
            "reward-only": failure_judge_record(
                "wrong_arguments", "multi_turn", "caused"
            ),
            "correct-only": failure_judge_record(
                "incomplete_task", "non_tool", "irrelevant"
            ),
        }
        fallback = {
            "failed": failure_judge_record(
                "wrong_workflow", "single_call_turn", "not_applicable"
            ),
            "reward-only": failure_judge_record(
                "wrong_workflow", "multi_turn", "caused"
            ),
            "correct-only": failure_judge_record(
                "incomplete_task", "single_call_turn", "uncertain"
            ),
        }
        markdown, summary = failure_report(features, local, fallback)

        self.assertEqual(summary["attribution_pool"], 3)
        self.assertEqual(
            summary["label_groups"],
            {"consensus_failure": 1, "correct_only": 1, "reward_only": 1},
        )
        self.assertEqual(summary["reward_failures"], 2)
        self.assertEqual(summary["correct_failures"], 2)
        self.assertEqual(
            summary["attribution_semantics"],
            "luna_primary_qwen_sensitivity",
        )
        self.assertEqual(
            summary["field_agreement"],
            {
                "causal_call_mode": 2,
                "multi_causal_role": 1,
                "primary_failure_cause": 2,
            },
        )
        self.assertEqual(
            summary["field_disagreement"],
            {
                "causal_call_mode": 1,
                "primary_failure_cause": 1,
            },
        )
        self.assertEqual(
            summary["primary_failure_causes_by_label"]["all_non_consensus"][
                "wrong_workflow"
            ],
            2,
        )
        self.assertEqual(summary["luna_primary"]["usable_reviews"], 3)
        self.assertIn("reward-only 1", markdown)

    def test_failure_calibration_selection_is_stable_and_matches_sealed_strata(self):
        features, manifest = failure_calibration_fixture()

        selection = build_failure_calibration_pilot_selection(features, manifest)
        reordered = build_failure_calibration_pilot_selection(
            list(reversed(features)), list(reversed(manifest))
        )

        self.assertEqual(selection, reordered)
        self.assertEqual(len(selection), 72)
        self.assertEqual(len({row["dialog_id"] for row in selection}), 72)
        expected_margins = {
            "domain": {"airline": 36, "retail": 8, "telecom": 28},
            "label_group": {
                "consensus_failure": 38,
                "correct_only": 5,
                "reward_only": 29,
            },
            "multi_exposure": {
                "multi_exposed": 58,
                "non_multi_exposed": 14,
            },
            "route": {"full/full": 55, "window/full": 17},
        }
        for field, expected in expected_margins.items():
            self.assertEqual(
                dict(sorted(Counter(row[field] for row in selection).items())),
                expected,
            )
        population_strata = {
            (
                feature["domain"],
                feature["label_group"],
                (
                    "multi_exposed"
                    if feature["full_source_static"]["multitool_turns"]
                    else "non_multi_exposed"
                ),
                next(
                    row["route"]
                    for row in manifest
                    if row["dialog_id"] == feature["dialog_id"]
                ),
            )
            for feature in features
        }
        selected_strata = {
            (
                row["domain"],
                row["label_group"],
                row["multi_exposure"],
                row["route"],
            )
            for row in selection
        }
        self.assertEqual(selected_strata, population_strata)
        self.assertEqual(
            stable_json_hash(selection),
            "ee00c216069b62c8fc764c9db4627388bfd0e35348564a2bd9a55a5c8c9f4c15",
        )

    def test_failure_validation_selection_is_stable_and_disjoint_from_pilot(self):
        features, manifest = failure_calibration_fixture()

        pilot = build_failure_calibration_pilot_selection(features, manifest)
        validation = build_failure_calibration_selection(features, manifest)
        reordered = build_failure_calibration_selection(
            list(reversed(features)), list(reversed(manifest))
        )

        self.assertEqual(validation, reordered)
        self.assertEqual(len(validation), FAILURE_CALIBRATION_LIMIT)
        self.assertEqual(
            len({row["dialog_id"] for row in validation}),
            FAILURE_CALIBRATION_LIMIT,
        )
        self.assertFalse(
            {row["dialog_id"] for row in pilot}
            & {row["dialog_id"] for row in validation}
        )

    def test_failure_calibration_gates_luna_evidence_not_exact_agreement(self):
        features = []
        calibration = []
        manifest = {}
        local = {}
        fallback = {}
        for index in range(FAILURE_CALIBRATION_LIMIT):
            dialog_id = f"failure-validation-{index:02d}"
            multi = index % 2 == 0
            turn_index = 3 if multi else 2
            mode = "multi_turn" if multi else "single_call_turn"
            role = "caused" if multi else "not_applicable"
            features.append(
                {
                    "dialog_id": dialog_id,
                    "domain": "airline",
                    "label_group": "consensus_failure",
                    "full_source_static": {
                        "multitool_turns": [turn_index] if multi else []
                    },
                }
            )
            calibration.append(
                {"dialog_id": dialog_id, "route": "full/full"}
            )
            manifest[dialog_id] = {
                "dialog_id": dialog_id,
                "route": "full/full",
                "full_source_turn_types": {
                    "assistant_multi" if multi else "assistant_single": [
                        turn_index
                    ]
                },
            }
            local[dialog_id] = failure_judge_record(
                "wrong_workflow",
                "non_tool",
                "irrelevant" if multi else "not_applicable",
                decisive_turns=[7],
            )
            fallback[dialog_id] = failure_judge_record(
                "wrong_arguments",
                mode,
                role,
                decisive_turns=[turn_index],
            )

        _, report = failure_calibration_report(
            calibration, features, local, fallback, manifest
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["attribution_semantics"], "luna_primary_qwen_sensitivity")
        self.assertEqual(report["metrics"]["overall"]["fallback_reliable"], 72)
        self.assertEqual(
            report["metrics"]["overall"]["mode_evidence_consistent"], 72
        )
        self.assertEqual(
            report["metrics"]["overall"]["local_fallback_exact_agreement"][
                "primary_failure_cause"
            ],
            0,
        )

    def test_failure_calibration_rejects_mode_without_matching_turn_evidence(self):
        features = []
        calibration = []
        manifest = {}
        reviews = {}
        for index in range(FAILURE_CALIBRATION_LIMIT):
            dialog_id = f"failure-inconsistent-{index:02d}"
            features.append(
                {
                    "dialog_id": dialog_id,
                    "domain": "telecom",
                    "label_group": "consensus_failure",
                    "full_source_static": {"multitool_turns": [3]},
                }
            )
            calibration.append(
                {"dialog_id": dialog_id, "route": "full/full"}
            )
            manifest[dialog_id] = {
                "dialog_id": dialog_id,
                "route": "full/full",
                "full_source_turn_types": {
                    "assistant_single": [2],
                    "assistant_multi": [3],
                },
            }
            reviews[dialog_id] = failure_judge_record(
                "wrong_arguments",
                "multi_turn",
                "caused",
                decisive_turns=[2],
            )

        _, report = failure_calibration_report(
            calibration, features, reviews, reviews, manifest
        )

        self.assertFalse(
            _failure_mode_evidence_consistent(
                reviews["failure-inconsistent-00"]["selected_review"],
                manifest["failure-inconsistent-00"],
            )
        )
        self.assertEqual(report["status"], "fail")
        self.assertFalse(
            report["gates"][
                "causal_mode_evidence_consistency_at_least_0.95"
            ]
        )

    def test_filter_decisions_require_full_keep_intersection_and_are_provenance_only(self):
        cases = {
            "eligible": ("consensus_success", []),
            "label-blocked": ("reward_only", []),
            "mechanical-blocked": (
                "consensus_success",
                ["invalid_tool_name"],
            ),
            "judge-blocked": ("consensus_success", []),
        }
        features = []
        manifest = {}
        local = {}
        fallback = {}

        def reviewed(review, *, fallback_side=False):
            record = judge_record(
                review,
                label=(
                    "fallback:test-model-b"
                    if fallback_side
                    else "local:test-model-a"
                ),
                url=(
                    "http://fallback.test/v1/chat/completions"
                    if fallback_side
                    else "http://local.test/v1/chat/completions"
                ),
                model="test-model-b" if fallback_side else "test-model-a",
            )
            record["judge_config_sha256"] = stable_json_hash(
                record["judge_config"]
            )
            return record

        for index, (dialog_id, (group, mechanical)) in enumerate(cases.items()):
            source_correct, source_reward = {
                "consensus_success": (1, 1),
                "reward_only": (0, 1),
            }[group]
            features.append(
                {
                    "dialog_id": dialog_id,
                    "domain": "airline",
                    "label_group": group,
                    "source_correct": source_correct,
                    "source_reward": source_reward,
                    "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
                    "full_source_static": {
                        "multitool_turns": [],
                        "multiwrite_turns": [],
                        "mechanical_hard_issues": mechanical,
                    },
                    "included_targets": [
                        {
                            "source_row": index,
                            "turn_index": index,
                            "conversation_turn_index": index,
                            "fingerprint": f"target-{index}",
                            "is_multitool": False,
                            "num_tool_calls": 0,
                            "num_write_calls": 0,
                            "tool_names": [],
                        }
                    ],
                    "conversion_checks": [
                        {
                            "source_row": index,
                            "source_non_system_hash": f"source-{index}",
                            "converted_non_system_hash": f"converted-{index}",
                            "target_fingerprint_match": True,
                        }
                    ],
                }
            )
            manifest[dialog_id] = {
                "route": "full/full",
                "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
                "prompt_version": MULTITOOL_PROMPT_VERSION,
                "source_full_payload_sha256": f"full-{index}",
                "local_input_payload_sha256": f"local-{index}",
                "fallback_input_payload_sha256": f"fallback-{index}",
            }
            local[dialog_id] = reviewed(
                dependency_safe_review(turns=(), batch_safety="not_multi")
            )
            fallback_review = dependency_safe_review(
                turns=(), batch_safety="not_multi"
            )
            if dialog_id == "judge-blocked":
                fallback_review["quality_verdict"] = "drop"
            fallback[dialog_id] = reviewed(
                fallback_review, fallback_side=True
            )

        rows = filter_decision_rows(features, local, fallback, manifest)
        by_id = {row["dialog_id"]: row for row in rows}

        self.assertTrue(by_id["eligible"]["keep_candidate"])
        self.assertFalse(by_id["label-blocked"]["keep_candidate"])
        self.assertFalse(by_id["mechanical-blocked"]["keep_candidate"])
        self.assertFalse(by_id["judge-blocked"]["keep_candidate"])
        self.assertTrue(by_id["label-blocked"]["dialog_judge_keep_candidate"])
        self.assertTrue(
            by_id["mechanical-blocked"]["dialog_judge_keep_candidate"]
        )
        self.assertIn(
            "label_group:reward_only",
            by_id["label-blocked"]["decision_reasons"],
        )
        self.assertIn(
            "mechanical_hard_issue:invalid_tool_name",
            by_id["mechanical-blocked"]["decision_reasons"],
        )
        self.assertIn(
            "quality_verdict:disagreement",
            by_id["judge-blocked"]["decision_reasons"],
        )
        forbidden = {
            "messages",
            "conversation",
            "content",
            "domain_policy",
            "candidate_protocol",
            "answer",
        }

        def nested_keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(
                    *(nested_keys(item) for item in value.values()), set()
                )
            if isinstance(value, list):
                return set().union(*(nested_keys(item) for item in value), set())
            return set()

        for row in rows:
            self.assertEqual(
                row["artifact_kind"], "provenance_only_filter_decision"
            )
            self.assertFalse(row["contains_training_messages"])
            self.assertTrue(forbidden.isdisjoint(nested_keys(row)))
            self.assertIsNotNone(row["review_scope"])
            self.assertIsNotNone(row["judge_provenance"])
            decision_sha256 = row["decision_sha256"]
            unhashed = dict(row)
            unhashed.pop("decision_sha256")
            self.assertEqual(decision_sha256, stable_json_hash(unhashed))

    def test_summarize_failure_gate_blocks_decisions_and_final_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quality_features = [
                {
                    "dialog_id": f"quality-{index:03d}",
                    "domain": "airline",
                    "label_group": "consensus_success",
                }
                for index in range(3 * QUALITY_CALIBRATION_LIMIT)
            ]
            features = quality_features + [
                {
                    "dialog_id": f"failure-{index:03d}",
                    "domain": "telecom",
                    "label_group": "consensus_failure",
                }
                for index in range(2 * FAILURE_CALIBRATION_LIMIT)
            ]
            selection = [
                {"dialog_id": feature["dialog_id"]}
                for feature in quality_features
            ]
            calibration_pilot = [
                {"dialog_id": f"quality-{index:03d}"}
                for index in range(QUALITY_CALIBRATION_LIMIT)
            ]
            calibration_v1 = [
                {"dialog_id": f"quality-{index:03d}"}
                for index in range(
                    QUALITY_CALIBRATION_LIMIT,
                    2 * QUALITY_CALIBRATION_LIMIT,
                )
            ]
            calibration = [
                {"dialog_id": f"quality-{index:03d}"}
                for index in range(
                    2 * QUALITY_CALIBRATION_LIMIT,
                    3 * QUALITY_CALIBRATION_LIMIT,
                )
            ]
            failure_calibration_pilot = [
                {"dialog_id": f"failure-{index:03d}", "route": "full/full"}
                for index in range(FAILURE_CALIBRATION_LIMIT)
            ]
            failure_calibration = [
                {"dialog_id": f"failure-{index:03d}", "route": "full/full"}
                for index in range(
                    FAILURE_CALIBRATION_LIMIT,
                    2 * FAILURE_CALIBRATION_LIMIT,
                )
            ]
            manifest_rows = []
            paths = {
                "features": root / "features.jsonl",
                "inventory": root / "inventory.json",
                "selection": root / "comparison_selection.jsonl",
                "pairs": root / "comparison_pairs.jsonl",
                "payload_manifest": root / "payload_manifest.jsonl",
                "calibration_pilot": root
                / "calibration_pilot_selection.jsonl",
                "calibration_v1": root
                / "calibration_validation_v1_selection.jsonl",
                "calibration": root / "calibration_selection.jsonl",
                "failure_calibration": root
                / "failure_calibration_selection.jsonl",
                "failure_calibration_pilot": root
                / "failure_calibration_pilot_selection.jsonl",
            }
            for name, rows in (
                ("features", features),
                ("selection", selection),
                ("pairs", []),
                ("payload_manifest", manifest_rows),
                ("calibration_pilot", calibration_pilot),
                ("calibration_v1", calibration_v1),
                ("calibration", calibration),
                ("failure_calibration_pilot", failure_calibration_pilot),
                ("failure_calibration", failure_calibration),
            ):
                write_jsonl(paths[name], rows)
            paths["inventory"].write_text(
                json.dumps(
                    {
                        "payload_manifest_sha256": stable_json_hash(manifest_rows),
                        "calibration_pilot_dialogs": QUALITY_CALIBRATION_LIMIT,
                        "calibration_validation_v1_dialogs": (
                            QUALITY_CALIBRATION_LIMIT
                        ),
                        "calibration_dialogs": QUALITY_CALIBRATION_LIMIT,
                        "quality_calibration_pilot_seed": (
                            QUALITY_CALIBRATION_PILOT_SEED
                        ),
                        "quality_calibration_validation_seed": (
                            QUALITY_CALIBRATION_VALIDATION_SEED
                        ),
                        "quality_calibration_v1_seed": (
                            QUALITY_CALIBRATION_V1_SEED
                        ),
                        "quality_calibration_version": QUALITY_CALIBRATION_VERSION,
                        "quality_calibration_pilot_selection_sha256": (
                            stable_json_hash(calibration_pilot)
                        ),
                        "quality_calibration_v1_selection_sha256": (
                            stable_json_hash(calibration_v1)
                        ),
                        "quality_calibration_selection_sha256": (
                            stable_json_hash(calibration)
                        ),
                        "failure_calibration_pilot_dialogs": (
                            FAILURE_CALIBRATION_LIMIT
                        ),
                        "failure_calibration_dialogs": FAILURE_CALIBRATION_LIMIT,
                        "failure_calibration_pilot_seed": (
                            FAILURE_CALIBRATION_PILOT_SEED
                        ),
                        "failure_calibration_pilot_version": (
                            FAILURE_CALIBRATION_PILOT_VERSION
                        ),
                        "failure_calibration_pilot_selection_sha256": (
                            stable_json_hash(failure_calibration_pilot)
                        ),
                        "failure_calibration_seed": FAILURE_CALIBRATION_SEED,
                        "failure_calibration_version": FAILURE_CALIBRATION_VERSION,
                        "failure_calibration_selection_sha256": stable_json_hash(
                            failure_calibration
                        ),
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                **paths,
                prepare_manifest=root / "prepare_manifest.json",
                quality_local_reviews=root / "quality-local.jsonl",
                quality_fallback_reviews=root / "quality-fallback.jsonl",
                failure_local_reviews=root / "failure-local.jsonl",
                failure_fallback_reviews=root / "failure-fallback.jsonl",
                out=root,
            )
            quality_manifest = {
                feature["dialog_id"]: {"dialog_id": feature["dialog_id"]}
                for feature in quality_features
            }
            failure_manifest = {
                f"failure-{index:03d}": {
                    "dialog_id": f"failure-{index:03d}"
                }
                for index in range(2 * FAILURE_CALIBRATION_LIMIT)
            }

            def manifest_for_mode(_path, mode):
                return quality_manifest if mode == "quality" else failure_manifest

            def review_map(_path, expected, **_kwargs):
                return {dialog_id: {} for dialog_id in expected}

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool.validate_prepare_manifest",
                        return_value={},
                    )
                )
                stack.enter_context(
                    mock.patch("audit_sft_multitool.validate_selected_artifacts")
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool._manifest_map",
                        side_effect=manifest_for_mode,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool.build_calibration_selection",
                        return_value=calibration_pilot,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool.build_quality_v1_selection",
                        return_value=calibration_v1,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool.build_quality_validation_selection",
                        return_value=calibration,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool.build_failure_calibration_pilot_selection",
                        return_value=failure_calibration_pilot,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool.build_failure_calibration_selection",
                        return_value=failure_calibration,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool._validated_review_map",
                        side_effect=review_map,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool._validate_independent_reviews"
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool.calibration_report",
                        return_value=("quality calibration", {"status": "pass"}),
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "audit_sft_multitool.failure_calibration_report",
                        return_value=("failure calibration", {"status": "fail"}),
                    )
                )
                failure_report_mock = stack.enter_context(
                    mock.patch("audit_sft_multitool.failure_report")
                )
                quality_report_mock = stack.enter_context(
                    mock.patch("audit_sft_multitool.quality_report")
                )
                decisions_mock = stack.enter_context(
                    mock.patch("audit_sft_multitool.filter_decision_rows")
                )
                with self.assertRaisesRegex(
                    SystemExit, "failure calibration gate failed"
                ):
                    summarize(args)

            failure_report_mock.assert_not_called()
            quality_report_mock.assert_not_called()
            decisions_mock.assert_not_called()
            self.assertFalse((root / "FAILURE_ATTRIBUTION.md").exists())
            self.assertFalse((root / "filter_decisions_draft.jsonl").exists())
            self.assertFalse((root / "summary.json").exists())


class LocalFirstSftFilterTest(unittest.TestCase):
    def test_official_comparison_metrics_and_paired_ci(self):
        rewards = {
            ("airline", "1"): [1.0, 0.0, 0.0, 0.0],
            ("retail", "2"): [1.0, 1.0, 1.0, 1.0],
        }
        self.assertEqual(
            official_task_metrics(rewards[("airline", "1")]),
            {
                "pass_at_1": 0.25,
                "pass_at_4_any": 1.0,
                "pass_power_4": 0.0,
            },
        )
        self.assertEqual(
            aggregate_eval_metrics(rewards),
            {
                "pass_at_1": 0.625,
                "pass_at_4_any": 1.0,
                "pass_power_4": 0.5,
            },
        )
        first = paired_bootstrap_ci([0.25, 0.5, 0.75], samples=1000, seed=7)
        second = paired_bootstrap_ci([0.25, 0.5, 0.75], samples=1000, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.5)
        self.assertGreater(first[1], 0.0)

    def test_official_comparison_normalizes_nonfinite_diagnostics(self):
        from compare_official_sft_evals import json_finite

        self.assertEqual(
            json_finite({"cost": float("nan"), "nested": [float("inf"), 1.0]}),
            {"cost": None, "nested": [None, 1.0]},
        )
        self.assertEqual(
            eval_json_finite(
                {"avg_agent_cost": float("nan"), "nested": [-float("inf")]}
            ),
            {"avg_agent_cost": None, "nested": [None]},
        )

    def test_official_comparison_behavior_diagnostics(self):
        self.assertTrue(
            trajectory_has_multi_tool_turn(
                [{"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]}]
            )
        )
        self.assertFalse(
            trajectory_has_multi_tool_turn(
                [{"role": "assistant", "tool_calls": [{"id": "a"}]}]
            )
        )
        diagnostics = aggregate_behavior(
            [
                {
                    "domain": "airline",
                    "reward": 1.0,
                    "call_mode": "multi_exposed",
                    "termination_reason": "user_stop",
                },
                {
                    "domain": "airline",
                    "reward": 0.0,
                    "call_mode": "multi_exposed",
                    "termination_reason": "too_many_errors",
                },
                {
                    "domain": "retail",
                    "reward": 0.0,
                    "call_mode": "non_multi_exposed",
                    "termination_reason": "max_steps",
                },
            ]
        )
        self.assertEqual(
            diagnostics["by_call_mode"]["multi_exposed"],
            {
                "simulations": 2,
                "successes": 1.0,
                "success_rate": 0.5,
                "too_many_errors": 1,
                "max_steps": 0,
            },
        )
        self.assertEqual(
            diagnostics["termination_reasons"],
            {"max_steps": 1, "too_many_errors": 1, "user_stop": 1},
        )

    def test_official_comparison_rejects_protocol_trials_and_infrastructure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "task_split_name": "test",
                        "num_trials": 4,
                        "agent_protocol_profile": "current-single",
                        "domains": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "protocol profile"):
                load_model_eval(
                    label="candidate",
                    summary_path=summary_path,
                    results_root=root,
                    expected_profile="dependency-safe-multi",
                )

            summary_path.write_text(
                json.dumps(
                    {
                        "task_split_name": "test",
                        "num_trials": 3,
                        "agent_protocol_profile": "dependency-safe-multi",
                        "agent_protocol_signature": protocol_signature(
                            "dependency-safe-multi"
                        ),
                        "domains": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly four trials"):
                load_model_eval(
                    label="candidate",
                    summary_path=summary_path,
                    results_root=root,
                    expected_profile="dependency-safe-multi",
                )

            summary_path.write_text(
                json.dumps(
                    {
                        "task_split_name": "test",
                        "num_trials": 4,
                        "agent_protocol_profile": "dependency-safe-multi",
                        "agent_protocol_signature": "sha256:stale",
                        "domains": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "content signature"):
                load_model_eval(
                    label="candidate",
                    summary_path=summary_path,
                    results_root=root,
                    expected_profile="dependency-safe-multi",
                )

            summary_path.write_text(
                json.dumps(
                    {
                        "task_split_name": "test",
                        "num_trials": 4,
                        "agent_protocol_profile": "dependency-safe-multi",
                        "agent_protocol_signature": protocol_signature(
                            "dependency-safe-multi"
                        ),
                        "domains": {
                            "airline": {
                                "metrics": {"infra_error_count": 1},
                                "results_file": "unused.json",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "infra_error_count=1"):
                load_model_eval(
                    label="candidate",
                    summary_path=summary_path,
                    results_root=root,
                    expected_profile="dependency-safe-multi",
                )

    def test_official_comparison_rejects_seed_and_task_set_mismatches(self):
        tasks = {
            **{("airline", str(i)): [0.0] * 4 for i in range(20)},
            **{("retail", str(i)): [0.0] * 4 for i in range(40)},
            **{("telecom", str(i)): [0.0] * 4 for i in range(40)},
        }

        def loaded(label, *, seed=300, task_rewards=None):
            return {
                "label": label,
                "summary_path": f"{label}.json",
                "summary_sha256": label,
                "result_artifacts": {},
                "signature": {"seed": seed},
                "task_rewards": task_rewards or tasks,
                "simulation_rows": [
                    {
                        "domain": "airline",
                        "reward": 0.0,
                        "call_mode": "non_multi_exposed",
                        "termination_reason": "user_stop",
                    }
                ],
            }

        args = SimpleNamespace(
            model=["reference=reference.json", "candidate=candidate.json"],
            candidate="candidate",
            results_root=Path("."),
            protocol_profile="dependency-safe-multi",
            bootstrap_samples=10,
            bootstrap_seed=7,
        )
        with mock.patch(
            "compare_official_sft_evals.load_model_eval",
            side_effect=[loaded("reference", seed=300), loaded("candidate", seed=301)],
        ):
            with self.assertRaisesRegex(ValueError, "protocol signatures differ"):
                compare_official_evals(args)

        missing_task = dict(tasks)
        missing_task.pop(("telecom", "39"))
        with mock.patch(
            "compare_official_sft_evals.load_model_eval",
            side_effect=[loaded("reference"), loaded("candidate", task_rewards=missing_task)],
        ):
            with self.assertRaisesRegex(ValueError, "task sets differ"):
                compare_official_evals(args)

    def test_official_comparison_marks_explicit_legacy_baseline(self):
        tasks = {
            **{("airline", str(i)): [0.0] * 4 for i in range(20)},
            **{("retail", str(i)): [0.0] * 4 for i in range(40)},
            **{("telecom", str(i)): [0.0] * 4 for i in range(40)},
        }

        def loaded(label, *, signed):
            profile = "dependency-safe-multi" if signed else None
            signature = protocol_signature(profile) if signed else None
            return {
                "label": label,
                "summary_path": f"{label}.json",
                "summary_sha256": label,
                "result_artifacts": {},
                "signature": {
                    "task_split_name": "test",
                    "num_trials": 4,
                    "agent_protocol_profile": profile,
                    "seed": 300,
                    "max_steps": 200,
                    "max_errors": 10,
                    "agent": {
                        "implementation": "slime_sglang_agent",
                        "llm": None,
                        "generation": {
                            "temperature": 0.6,
                            "top_p": 1.0,
                            "max_tokens": 8192,
                            "protocol_profile": profile,
                            "protocol_signature": signature,
                        },
                    },
                    "user": {
                        "implementation": "user_simulator",
                        "llm": "openai/user-v1",
                        "generation": {
                            "temperature": 0.0,
                            "top_p": None,
                            "max_tokens": 512,
                            "protocol_profile": None,
                            "protocol_signature": None,
                        },
                    },
                },
                "protocol_provenance": (
                    "signed-current" if signed else "legacy-implicit-current-single"
                ),
                "task_rewards": tasks,
                "simulation_rows": [
                    {
                        "domain": domain,
                        "reward": 0.0,
                        "call_mode": "non_multi_exposed",
                        "termination_reason": "user_stop",
                    }
                    for domain in ("airline", "retail", "telecom")
                ],
            }

        args = SimpleNamespace(
            model=["reference=reference.json", "candidate=candidate.json"],
            candidate="candidate",
            legacy_model=["reference"],
            results_root=Path("."),
            protocol_profile="dependency-safe-multi",
            bootstrap_samples=10,
            bootstrap_seed=7,
        )
        with mock.patch(
            "compare_official_sft_evals.load_model_eval",
            side_effect=[
                loaded("reference", signed=False),
                loaded("candidate", signed=True),
            ],
        ):
            report = compare_official_evals(args)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["comparison_mode"], "legacy-baseline-cross-protocol")
        self.assertEqual(
            report["legacy_models"],
            {"reference": "legacy-implicit-current-single"},
        )
        self.assertIsNotNone(report["comparability_warning"])
        self.assertEqual(
            {row["scope"] for row in report["paired_deltas"]},
            {"overall", "airline", "retail", "telecom"},
        )

    def test_rl_promotion_gate_and_tie_breaker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "run.log"
            log_path.write_text(
                "\n".join(
                    f"rollout {step}: {{'rollout/truncated': 0.0}}\n"
                    f"step {step}: {{'train/kl_loss': 0.05, 'train/grad_norm': 1.0}}"
                    for step in range(90, 100)
                ),
                encoding="utf-8",
            )
            trajectory_path = root / "trajectories.jsonl"
            trajectory_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "metadata": {
                                "tau2_agent_protocol_profile": "dependency-safe-multi",
                                "tau2_agent_protocol_signature": protocol_signature(
                                    "dependency-safe-multi"
                                ),
                                "tau2_termination_reason": "user_stop",
                                "tau2_field_reward_signals": {
                                    "counts": {
                                        "malformed_json": 0,
                                        "nonexistent_tool": 0,
                                        "repetition": 0,
                                        "max_steps": 0,
                                    }
                                },
                            }
                        }
                    )
                    + "\n"
                    for _ in range(384)
                ),
                encoding="utf-8",
            )
            checkpoint_root = root / "checkpoint"
            checkpoint_root.mkdir()
            (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
                "99",
                encoding="utf-8",
            )

            def model(pass1, pass4_any, pass_power4):
                return {
                    "overall": {
                        "pass_at_1": pass1,
                        "pass_at_4_any": pass4_any,
                        "pass_power_4": pass_power4,
                    },
                    "by_domain": {
                        domain: {
                            "pass_at_1": pass1,
                            "pass_at_4_any": pass4_any,
                            "pass_power_4": pass_power4,
                        }
                        for domain in ("airline", "retail", "telecom")
                    },
                    "behavior": {
                        "overall": {
                            "termination_reasons": {
                                "user_stop": 390,
                                "too_many_errors": 8,
                                "max_steps": 2,
                            }
                        }
                    },
                }

            comparison = {
                "status": "pass",
                "protocol_profile": "dependency-safe-multi",
                "agent_protocol_signature": protocol_signature(
                    "dependency-safe-multi"
                ),
                "models": {
                    "new_sft": model(0.28, 0.50, 0.11),
                    "old_rl100": model(0.29, 0.51, 0.10),
                    "new_rl100": model(0.31, 0.53, 0.10),
                },
            }
            gate = evaluate_gate(
                comparison=comparison,
                candidate="new_rl100",
                baseline="new_sft",
                prior="old_rl100",
                gate_mode="promote100",
                training=parse_training_log(log_path, expected_latest=99),
                trajectories=summarize_recent_trajectories(
                    trajectory_path,
                    window=384,
                ),
                checkpoint_root=checkpoint_root,
                expected_latest=99,
            )
            self.assertEqual(gate["status"], "pass")

            tied_comparison = {
                "models": {
                    "new_rl100": model(0.31, 0.53, 0.10),
                    "new_rl200": model(0.31, 0.53, 0.10),
                }
            }
            selection = select_checkpoint(
                tied_comparison,
                {
                    "new_rl100": {**gate, "candidate": "new_rl100"},
                    "new_rl200": {**gate, "candidate": "new_rl200"},
                },
                {"new_rl100": 99, "new_rl200": 199},
            )
            self.assertEqual(selection["selected"]["label"], "new_rl100")

    def test_rl_promotion_training_log_parser_streams_input(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log"
            log_path.write_text(
                "escaped prompt: first line\\nAn ownership record follows\n"
                + "\n".join(
                    f"rollout {step}: {{'rollout/truncated': 0.0}}\n"
                    f"step {step}: {{'train/kl_loss': 0.05}}"
                    for step in range(90, 100)
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("parser must not materialize the log"),
            ):
                parsed = parse_training_log(log_path, expected_latest=99)

            self.assertEqual(parsed["observed_latest_step"], 99)
            self.assertEqual(parsed["missing_steps"], [])
            self.assertAlmostEqual(parsed["last_10_k2_kl_mean"], 0.05)
            self.assertEqual(parsed["fatal_matches"]["nan"], 0)

            log_path.write_text("fatal optimizer value: NaN\n", encoding="utf-8")
            self.assertEqual(
                parse_training_log(log_path, expected_latest=99)["fatal_matches"]["nan"],
                1,
            )

    def test_local_review_shards_partition_seed_and_merge_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_input = root / "full.jsonl"
            second_input = root / "windows.jsonl"
            existing_path = root / "existing.jsonl"
            shard_dir = root / "shards"
            manifest_path = shard_dir / "manifest.json"
            merged_path = root / "merged.jsonl"
            payloads = [{"_dialog_id": f"dialog-{index}"} for index in range(11)]
            write_jsonl(first_input, payloads[:8])
            write_jsonl(second_input, payloads[8:])
            write_jsonl(
                existing_path,
                [
                    {"dialog_id": "dialog-0", "selected_review": {}},
                    {"dialog_id": "out-of-scope", "selected_review": {}},
                ],
            )

            partition_review_shards(
                SimpleNamespace(
                    input=[first_input, second_input],
                    existing=existing_path,
                    shard_dir=shard_dir,
                    manifest=manifest_path,
                    num_shards=4,
                )
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["payload_rows"], 11)
            self.assertEqual(manifest["existing_rows_in_scope"], 1)
            self.assertEqual(
                sum(item["payload_rows"] for item in manifest["shards"].values()),
                11,
            )
            for index in range(4):
                shard_payloads = list(
                    read_jsonl(shard_dir / f"payloads_{index:02d}.jsonl")
                )
                self.assertTrue(
                    all(
                        shard_index(row["_dialog_id"], 4) == index
                        for row in shard_payloads
                    )
                )
                write_jsonl(
                    shard_dir / f"reviews_{index:02d}.jsonl",
                    (
                        {"dialog_id": row["_dialog_id"], "selected_review": {}}
                        for row in shard_payloads
                    ),
                )

            merge_review_shards(
                SimpleNamespace(
                    input=[first_input, second_input],
                    shard_dir=shard_dir,
                    output=merged_path,
                    num_shards=4,
                )
            )
            self.assertEqual(
                {row["dialog_id"] for row in read_jsonl(merged_path)},
                {row["_dialog_id"] for row in payloads},
            )

        with self.assertRaisesRegex(ValueError, "num_shards must be positive"):
            shard_index("dialog", 0)

    def test_eval_protocol_profile_replaces_single_call_clause(self):
        policies = {
            "airline": (
                "Policy start. You should only make one tool call at a time, and if you "
                "make a tool call, you should not respond to the user simultaneously. If "
                "you respond to the user, you should not make a tool call at the same time."
            ),
            "retail": (
                "Policy start. You should at most make one tool call at a time, and if "
                "you take a tool call, you should not respond to the user at the same time. "
                "If you respond to the user, you should not make a tool call at the same time."
            ),
            "telecom": (
                "Policy start. You should only make one tool call at a time, and if you "
                "make a tool call, you should not respond to the user simultaneously. If "
                "you respond to the user, you should not make a tool call at the same time."
            ),
        }
        for domain, policy in policies.items():
            with self.subTest(domain=domain):
                unchanged, changed = domain_policy_for_profile(
                    policy,
                    EVAL_PROTOCOL_CURRENT_SINGLE,
                )
                self.assertEqual(unchanged, policy)
                self.assertFalse(changed)

                amended, changed = domain_policy_for_profile(
                    policy,
                    EVAL_PROTOCOL_DEPENDENCY_SAFE_MULTI,
                )
                self.assertTrue(changed)
                self.assertIn(EVAL_DEPENDENCY_SAFE_MULTI_RULE, amended)
                self.assertNotIn("one tool call at a time", amended)
                self.assertEqual(
                    audit_policy_for_profile(
                        policy,
                        EVAL_PROTOCOL_DEPENDENCY_SAFE_MULTI,
                    ),
                    (amended, changed),
                )
                self.assertEqual(
                    audit_multitool_amend_policy(policy),
                    (amended, changed),
                )

        self.assertEqual(EVAL_DEPENDENCY_SAFE_MULTI_RULE, CANDIDATE_POLICY_RULE)
        self.assertIn(
            EVAL_DEPENDENCY_SAFE_MULTI_RULE,
            protocol_block_for_profile(EVAL_PROTOCOL_DEPENDENCY_SAFE_MULTI),
        )
        with self.assertRaisesRegex(ValueError, "unknown protocol profile"):
            domain_policy_for_profile("policy", "invalid-profile")

    def test_amend_training_system_removes_both_single_call_rules(self):
        system = (
            "## Output Format\nMake one or more tool calls."
            "\n\n---\n\n<instructions>\nAgent instructions.\n\n"
            + AIRLINE_SINGLE_CALL_RULES
            + "\n</instructions>\n<policy>\n"
            "You should only make one tool call at a time, and if you make a tool call, "
            "you should not respond to the user simultaneously. If you respond to the "
            "user, you should not make a tool call at the same time.\n</policy>"
        )

        amended, changes = amend_training_system(system)

        self.assertEqual(amended.count(CANDIDATE_POLICY_RULE), 1)
        self.assertNotIn("exactly ONE tool call", amended)
        self.assertNotIn("CANNOT make multiple tool calls", amended)
        self.assertNotIn("only make one tool call at a time", amended)
        self.assertTrue(changes["protocol_inserted"])
        self.assertTrue(changes["airline_instruction_rules_replaced"])
        self.assertTrue(changes["domain_policy_clauses_replaced"])

    def test_quality_gate_keeps_adequate_review_with_diagnostic_critical_turn(self):
        record = judge_record(
            dependency_safe_review(
                quality_verdict="review",
                critical_turns=[9],
            )
        )

        self.assertEqual(
            quality_gate_reasons(record, side="local"),
            [],
        )

    def test_quality_gate_rejects_drop_and_low_core_score(self):
        review = dependency_safe_review(quality_verdict="drop")
        review["dimensions"]["tool_choice"]["score"] = 2

        reasons = quality_gate_reasons(
            judge_record(review),
            side="local",
        )

        self.assertIn("local:quality_verdict:drop", reasons)
        self.assertIn("local:core_dimension:tool_choice:2", reasons)

    def test_multitool_prefix_requires_every_turn_to_be_strict_safe(self):
        safe_record = judge_record(dependency_safe_review(turns=(3, 5)))
        static = {"multitool_turns": [3, 5], "multiwrite_turns": []}
        self.assertEqual(
            multitool_prefix_reasons(safe_record, static, side="local"),
            [],
        )
        unsafe = dependency_safe_review(turns=(3, 5))
        unsafe["multi_turn_assessments"][1]["calls_necessary"] = "no"
        self.assertEqual(
            multitool_prefix_reasons(
                judge_record(unsafe),
                static,
                side="local",
            ),
            ["local:multi_turn:5:not_strict_safe"],
        )

    def test_target_only_loss_mask_and_budget_schedule_are_deterministic(self):
        row = {
            "messages": [
                {"role": "system", "content": "old"},
                {"role": "assistant", "content": "history"},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": "target"},
            ],
            "metadata": {
                "source_row": 4,
                "source_dialog_id": "airline_dialog_fixture",
                "quality_filter": {
                    "version": FILTER_VERSION,
                    "decision_sha256": "decision-4",
                },
            },
        }
        masked = apply_target_only_loss_mask(row, "amended")
        self.assertEqual(
            [message["step_loss_mask"] for message in masked["messages"]],
            [0, 0, 0, 1],
        )
        self.assertEqual(masked["messages"][0]["content"], "amended")
        self.assertNotIn("step_loss_mask", row["messages"][0])

        first = matched_budget_rows([masked], 3, seed=17)
        second = matched_budget_rows([masked], 3, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(
            [item["metadata"]["training_schedule"]["cycle"] for item in first],
            [0, 1, 2],
        )
        self.assertEqual(
            [item["metadata"]["training_schedule"]["schedule_index"] for item in first],
            [0, 1, 2],
        )

    def test_training_preflight_validates_protocol_mask_and_schedule(self):
        decision = "decision-4"
        row = {
            "messages": [
                {
                    "role": "system",
                    "content": "## Output Format\nMake one or more tool calls.\n\n"
                    "## Dependency-safe multi-tool protocol\n"
                    + CANDIDATE_POLICY_RULE
                    + "\n\n---\n\n<instructions>safe</instructions>",
                    "step_loss_mask": 0,
                },
                {"role": "user", "content": "help", "step_loss_mask": 0},
                {"role": "assistant", "content": "answer", "step_loss_mask": 1},
            ],
            "metadata": {
                "source_row": 4,
                "source_dialog_id": "airline_dialog_fixture",
                "domain": "airline",
                "quality_filter": {
                    "version": FILTER_VERSION,
                    "target_only_loss": True,
                    "decision_sha256": decision,
                },
                "training_schedule": {
                    "version": "matched-row-budget-v1",
                    "schedule_index": 0,
                    "filter_decision_sha256": decision,
                },
            },
        }

        self.assertEqual(
            validate_row_structure(
                row,
                line_number=1,
                expected_schedule_index=0,
            ),
            ("airline", 4, decision),
        )
        row["messages"][1]["step_loss_mask"] = 1
        with self.assertRaisesRegex(ValueError, "only the final target"):
            validate_row_structure(
                row,
                line_number=1,
                expected_schedule_index=0,
            )


class FakeQwen3FullTokenizer:
    """Char tokenizer with native tools and consecutive-tool grouping."""

    chat_template = "fake-qwen3-full-template-v1"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        encoded = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return encoded

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        tools=None,
        add_generation_prompt=False,
        return_dict=False,
        **kwargs,
    ):
        del return_dict, kwargs
        pieces = []
        for index, message in enumerate(messages):
            role = message["role"]
            content = str(message.get("content") or "")
            if role == "system":
                if tools:
                    content += "\n<tools>" + json.dumps(
                        tools,
                        sort_keys=True,
                        separators=(",", ":"),
                    ) + "</tools>"
                pieces.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                pieces.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                rendered = f"<|im_start|>assistant\n{content}"
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    rendered += "<tool_call>" + json.dumps(
                        {
                            "name": function["name"],
                            "arguments": function.get("arguments") or {},
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ) + "</tool_call>"
                pieces.append(rendered + "<|im_end|>\n")
            elif role == "tool":
                rendered = ""
                if index == 0 or messages[index - 1]["role"] != "tool":
                    rendered += "<|im_start|>user"
                rendered += f"\n<tool_response>\n{content}\n</tool_response>"
                if index + 1 == len(messages) or messages[index + 1]["role"] != "tool":
                    rendered += "<|im_end|>\n"
                pieces.append(rendered)
            else:
                raise ValueError(f"unsupported fake role: {role}")
        if add_generation_prompt:
            pieces.append("<|im_start|>assistant\n")
        rendered = "".join(pieces)
        return [ord(character) for character in rendered] if tokenize else rendered


def boundary_tool_schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Fixture schema for {name}",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
    }


def boundary_contract(*, user_tool_count=2, chat_template=None):
    single_call_clause = (
        "You should only make one tool call at a time, and if you make a tool "
        "call, you should not respond to the user simultaneously. If you respond "
        "to the user, you should not make a tool call at the same time."
    )
    return AgentContract.create(
        domain="telecom",
        domain_policy=(
            "<main_policy>Keep records accurate. "
            + single_call_clause
            + "</main_policy>\n<tech_support_policy>Full manual.</tech_support_policy>"
        ),
        agent_tools=[boundary_tool_schema("lookup_account"), boundary_tool_schema("done")],
        user_tools=[
            boundary_tool_schema(f"device_action_{index:02d}")
            for index in range(user_tool_count)
        ],
        chat_template=chat_template or FakeQwen3FullTokenizer.chat_template,
        profile=PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    )


class AgentBoundaryV2Test(unittest.TestCase):
    @staticmethod
    def _materialize_long100_checkpoint(root, iteration):
        root.mkdir(parents=True, exist_ok=True)
        (root / "latest_checkpointed_iteration.txt").write_text(
            str(iteration),
            encoding="utf-8",
        )
        iteration_dir = root / f"iter_{iteration:07d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        (iteration_dir / ".metadata").write_bytes(b"metadata")
        (iteration_dir / "common.pt").write_bytes(b"common")
        (iteration_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
        (iteration_dir / "__0_0.distcp").write_bytes(b"shard")
        rollout_dir = root / "rollout"
        rollout_dir.mkdir(exist_ok=True)
        (rollout_dir / f"global_dataset_state_dict_{iteration}.pt").write_bytes(
            b"rollout"
        )

    @staticmethod
    def _materialize_domain_checkpoint(
        root,
        iteration,
        *,
        fingerprint="a" * 64,
        offsets=None,
        epochs=None,
    ):
        AgentBoundaryV2Test._materialize_long100_checkpoint(root, iteration)
        torch.save(
            {
                "sample_offset": 0,
                "epoch_id": 0,
                "sample_group_index": iteration * 6,
                "sample_index": iteration * 48,
                "metadata": {
                    "tau2_domain_quota_v1": {
                        "dataset_fingerprint": fingerprint,
                        "domain_offsets": offsets
                        or {"airline": 10, "retail": 20, "telecom": 30},
                        "domain_epochs": epochs
                        or {"airline": 0, "retail": 0, "telecom": 0},
                    }
                },
            },
            root / "rollout" / f"global_dataset_state_dict_{iteration}.pt",
        )

    @staticmethod
    def _write_long100_log(
        path,
        *,
        start,
        end,
        load_root,
        load_iteration,
        quota,
        high_kl_steps=(),
        nonfinite_step=None,
        wrong_quota_step=None,
        missing_metric_step=None,
        oom=False,
        save_iteration=None,
        save_root=None,
    ):
        lines = [
            f"successfully loaded checkpoint from {load_root} "
            f"[ t 1/2, p 1/1 ] at iteration {load_iteration}"
        ]
        if oom:
            lines.append("CUDA out of memory")
        for step in range(start, end + 1):
            rollout_metrics = {
                "rollout/truncated": 0.75,
                "rollout/rewards": 0.1,
            }
            for domain, accepted in quota.items():
                prefix = f"rollout/domain_quota/{domain}"
                rollout_metrics[f"{prefix}/accepted"] = float(
                    accepted + int(step == wrong_quota_step and domain == "telecom")
                )
                rollout_metrics[f"{prefix}/rejected"] = 1.0
                rollout_metrics[f"{prefix}/replacement_draws"] = 1.0
            train_metrics = {
                "train/loss": 0.1,
                "train/pg_loss": 0.1,
                "train/kl_loss": 0.2 if step in high_kl_steps else 0.0,
                "train/grad_norm": (
                    float("nan") if step == nonfinite_step else 1.0
                ),
                "train/lr-pg_0": 2e-6,
            }
            if step == missing_metric_step:
                train_metrics.pop("train/pg_loss")
            lines.extend(
                [
                    f"rollout {step}: {rollout_metrics!r}",
                    f"step {step}: {train_metrics!r}",
                ]
            )
        if save_iteration is not None:
            if save_root is None:
                raise ValueError("save_root is required with save_iteration")
            lines.append(
                "successfully saved checkpoint from iteration "
                f"{save_iteration} to {save_root} [ t 1/2, p 1/1 ]"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_long100_trajectories(
        path,
        *,
        rows=480,
        invalid_credit=False,
        invalid_credit_rollout=None,
        rollout_start=None,
        domains=("telecom", "airline", "retail"),
    ):
        with path.open("w", encoding="utf-8") as file:
            for index in range(rows):
                domain = domains[index % len(domains)]
                rollout_id = (
                    rollout_start + index // 48
                    if rollout_start is not None
                    else None
                )
                token_penalty = 0.1 if (
                    invalid_credit and index == 0
                ) or (
                    invalid_credit_rollout is not None
                    and rollout_id == invalid_credit_rollout
                ) else 0.0
                row = {
                    "status": "truncated",
                    "remove_sample": False,
                    "response_length": 1,
                    "loss_mask": [1],
                    "train_metadata": {"token_penalties": [token_penalty]},
                    "metadata": {
                        "tau2_domain": domain,
                        "tau2_agent_protocol_profile": (
                            PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                        ),
                        "tau2_agent_protocol_signature": protocol_signature(
                            PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                        ),
                        "agent_contract_signature": f"sha256:{domain}",
                        "tau2_turn_credit_version": "turn-credit-v1",
                        "tau2_global_score": 0.0,
                        "tau2_turn_credits": [
                            {"response_span": [0, 1], "penalty": 0.0}
                        ],
                        "tau2_termination_reason": "max_steps",
                        "tau2_field_reward_signals": {
                            "counts": {
                                "malformed_json": 1,
                                "nonexistent_tool": 1,
                                "wrong_argument_fields": 1,
                                "tool_execution_error": 1,
                                "repetition": 1,
                                "max_steps": 1,
                            }
                        },
                    },
                    "simulation": {"termination_reason": "max_steps"},
                }
                if rollout_id is not None:
                    row["metadata"]["tau2_domain_quota_rollout_id"] = rollout_id
                file.write(json.dumps(row) + "\n")

    @staticmethod
    def _long100_health_gate(root, stage):
        config = LONG100_STAGES[stage]
        contracts = {
            domain: [f"sha256:{domain}"]
            for domain in ("airline", "retail", "telecom")
        }
        return {
            "status": "pass",
            "gate_version": LONG100_HEALTH_GATE_VERSION,
            "stage": stage,
            "checkpoint_root": str(root.resolve()),
            "expected_checkpoint_root": str(root.resolve()),
            "expected_latest": config["end"],
            "protocol_profile": PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
            "agent_protocol_signature": protocol_signature(
                PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
            ),
            "turn_credit_version": "turn-credit-v1",
            "trajectories": {"agent_contract_signatures": contracts},
            "diagnostics_only": {
                "framework_truncated": {"breakdown": {"max_steps": 1}}
            },
        }

    def test_boundary_eval_wrappers_export_the_requested_seed(self):
        models_dir = OFFICIAL_EVAL_DIR / "models"
        for filename in (
            "run_full_tau2_agent_boundary_v2_baseline_user_stop_parser.sh",
            "run_full_tau2_agent_boundary_v2_user_stop_parser.sh",
            "run_full_tau2_agent_rl_boundary_v2_user_stop_parser.sh",
        ):
            script = (models_dir / filename).read_text(encoding="utf-8")
            self.assertIn("export SEED\n", script, filename)

        for filename in (
            "smoke_qwen3_4b_instruct_2507_sft_agent_boundary_v2.sh",
            "smoke_qwen3_4b_instruct_2507_sft_agent_boundary_v2_cp2.sh",
        ):
            script = (SFT_DIR / filename).read_text(encoding="utf-8")
            self.assertNotIn('$(dirname "$0")', script, filename)
            self.assertIn("${PROJECT_ROOT}/examples/tau2-bench/sft/", script, filename)

    def test_rl_pilot_gate_checks_both_logs_quota_and_trajectory_files(self):
        quota = {"telecom": 3, "airline": 2, "retail": 1}
        metrics = {}
        for domain, accepted in quota.items():
            prefix = f"rollout/domain_quota/{domain}"
            metrics[f"{prefix}/accepted"] = float(accepted)
            metrics[f"{prefix}/rejected"] = 1.0
            metrics[f"{prefix}/replacement_draws"] = 1.0
        passed, observed = _quota_observations(
            {
                "expected_steps": [0],
                "expected_step_metrics": {"0": metrics},
            },
            quota=quota,
        )
        self.assertTrue(passed)
        self.assertEqual(observed["0"]["telecom"]["accepted"], 3.0)

        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"stage-{index}.jsonl" for index in range(2)]
            for index, path in enumerate(paths):
                path.write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "domain": "telecom" if index == 0 else "airline",
                                "agent_contract_signature": "sha256:test",
                                "tau2_agent_protocol_profile": (
                                    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                                ),
                                "tau2_agent_protocol_signature": protocol_signature(
                                    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                                ),
                                "tau2_field_reward_signals": {
                                    "counts": {
                                        "matched_argument_fields": 4,
                                        "wrong_argument_fields": index,
                                        "tool_execution_error": int(index == 0),
                                    }
                                },
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            report = summarize_boundary_pilot_trajectories(paths, window=2)
        self.assertEqual(report["total_rows"], 2)
        self.assertEqual(report["selected_rows"], 2)
        self.assertEqual(report["domains"], {"airline": 1, "telecom": 1})
        self.assertEqual(report["wrong_argument_field_rate"], 1 / 8)
        self.assertEqual(report["tool_execution_error_fraction"], 1 / 2)

    def test_long100_health_accepts_fresh_sft_then_exact_iter9_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft_root = root / "selected-sft"
            sft_root.mkdir()
            checkpoint_root = root / "long100-checkpoint"
            self._materialize_long100_checkpoint(checkpoint_root, 9)
            pilot_a_log = root / "long100-a.log"
            pilot_a_trajectories = root / "long100-a.jsonl"
            self._write_long100_log(
                pilot_a_log,
                start=0,
                end=9,
                load_root=sft_root,
                load_iteration=0,
                quota=LONG100_STAGES["iter9"]["quota"],
            )
            with pilot_a_log.open("a", encoding="utf-8") as file:
                file.write(
                    "(RolloutManager pid=7) infinity generates an error in "
                    "this verbatim telecom response with No OOM value\n"
                )
            self._write_long100_trajectories(pilot_a_trajectories)
            iter9 = evaluate_long100_health(
                stage="iter9",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_root=checkpoint_root,
                sft_checkpoint_root=sft_root,
                training_logs=[pilot_a_log],
                trajectory_paths=[pilot_a_trajectories],
            )
            self.assertEqual(iter9["status"], "pass")
            self.assertEqual(
                iter9["training"]["logs"][0]["recovery_requirement"],
                {
                    "mode": "fresh_sft",
                    "path": str(sft_root.resolve()),
                    "iteration": 0,
                },
            )
            self.assertEqual(
                iter9["diagnostics_only"]["framework_truncated"]["breakdown"][
                    "max_steps"
                ],
                480,
            )

            self._materialize_long100_checkpoint(checkpoint_root, 19)
            pilot_b_log = root / "long100-b.log"
            pilot_b_trajectories = root / "long100-b.jsonl"
            self._write_long100_log(
                pilot_b_log,
                start=10,
                end=19,
                load_root=checkpoint_root,
                load_iteration=9,
                quota=LONG100_STAGES["iter19"]["quota"],
            )
            self._write_long100_trajectories(pilot_b_trajectories)
            iter19 = evaluate_long100_health(
                stage="iter19",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_root=checkpoint_root,
                sft_checkpoint_root=sft_root,
                training_logs=[pilot_b_log],
                trajectory_paths=[pilot_b_trajectories],
                prerequisite_health=iter9,
            )
            self.assertEqual(iter19["status"], "pass")
            self.assertEqual(
                iter19["training"]["logs"][0]["recovery_requirement"],
                {
                    "mode": "resume_rl",
                    "path": str(checkpoint_root.resolve()),
                    "iteration": 9,
                },
            )

            wrong_gate = copy.deepcopy(iter9)
            wrong_gate["checkpoint_root"] = str((root / "old-iter9").resolve())
            rejected = evaluate_long100_health(
                stage="iter19",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_root=root / "wrong-root",
                sft_checkpoint_root=sft_root,
                training_logs=[pilot_b_log],
                trajectory_paths=[pilot_b_trajectories],
                prerequisite_health=wrong_gate,
            )
            failed = {
                check["name"]
                for check in rejected["checks"]
                if not check["passed"]
            }
            self.assertIn("checkpoint_root", failed)
            self.assertIn("prerequisite_health", failed)

            self._materialize_long100_checkpoint(checkpoint_root, 99)
            final_first_log = root / "long100-final-first.log"
            final_resume_log = root / "long100-final-resume.log"
            final_trajectories = root / "long100-final.jsonl"
            self._write_long100_log(
                final_first_log,
                start=20,
                end=59,
                load_root=checkpoint_root,
                load_iteration=19,
                quota=LONG100_STAGES["iter99"]["quota"],
            )
            self._write_long100_log(
                final_resume_log,
                start=60,
                end=99,
                load_root=checkpoint_root,
                load_iteration=59,
                quota=LONG100_STAGES["iter99"]["quota"],
            )
            self._write_long100_trajectories(
                final_trajectories,
                rows=3840,
            )
            iter99 = evaluate_long100_health(
                stage="iter99",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_root=checkpoint_root,
                sft_checkpoint_root=sft_root,
                training_logs=[final_first_log, final_resume_log],
                trajectory_paths=[final_trajectories],
                prerequisite_health=iter19,
            )
            self.assertEqual(iter99["status"], "pass")
            self.assertEqual(
                [
                    item["recovery_requirement"]["iteration"]
                    for item in iter99["training"]["logs"]
                ],
                [19, 59],
            )

    def test_long100_health_hard_fails_kl_finite_quota_and_turn_credit_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft_root = root / "selected-sft"
            sft_root.mkdir()
            checkpoint_root = root / "long100-checkpoint"
            self._materialize_long100_checkpoint(checkpoint_root, 9)
            log_path = root / "unhealthy.log"
            trajectory_path = root / "unaligned.jsonl"
            self._write_long100_log(
                log_path,
                start=0,
                end=9,
                load_root=sft_root,
                load_iteration=0,
                quota=LONG100_STAGES["iter9"]["quota"],
                high_kl_steps=(0, 1),
                nonfinite_step=2,
                wrong_quota_step=3,
                oom=True,
            )
            self._write_long100_trajectories(
                trajectory_path,
                invalid_credit=True,
            )
            report = evaluate_long100_health(
                stage="iter9",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_root=checkpoint_root,
                sft_checkpoint_root=sft_root,
                training_logs=[log_path],
                trajectory_paths=[trajectory_path],
            )
            self.assertEqual(report["status"], "fail")
            checks = {check["name"]: check for check in report["checks"]}
            self.assertFalse(checks["training_finite"]["passed"])
            self.assertFalse(checks["no_consecutive_high_kl"]["passed"])
            self.assertFalse(checks["domain_quota"]["passed"])
            self.assertFalse(checks["trajectory_turn_credit_alignment"]["passed"])
            self.assertLess(report["training"]["last_10_k2_kl_mean"], 0.10)
            self.assertEqual(
                report["diagnostics_only"]["behavior"][
                    "tool_execution_error_fraction"
                ],
                1.0,
            )
            self.assertNotIn("behavior", checks)

    def test_long100_health_rejects_incomplete_per_step_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft_root = root / "selected-sft"
            sft_root.mkdir()
            checkpoint_root = root / "long100-checkpoint"
            self._materialize_long100_checkpoint(checkpoint_root, 9)
            log_path = root / "missing-metric.log"
            trajectory_path = root / "trajectories.jsonl"
            self._write_long100_log(
                log_path,
                start=0,
                end=9,
                load_root=sft_root,
                load_iteration=0,
                quota=LONG100_STAGES["iter9"]["quota"],
                missing_metric_step=4,
            )
            self._write_long100_trajectories(trajectory_path)
            report = evaluate_long100_health(
                stage="iter9",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_root=checkpoint_root,
                sft_checkpoint_root=sft_root,
                training_logs=[log_path],
                trajectory_paths=[trajectory_path],
            )
            checks = {check["name"]: check for check in report["checks"]}
            self.assertEqual(report["status"], "fail")
            self.assertFalse(checks["training_metrics_complete"]["passed"])
            self.assertEqual(
                report["training"]["missing_required_step_metrics"],
                {"4": ["train/pg_loss"]},
            )

    def test_long100_health_reaudits_exact_stage_save_after_root_advances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft_root = root / "selected-sft"
            sft_root.mkdir()
            checkpoint_root = root / "long100-checkpoint"
            self._materialize_long100_checkpoint(checkpoint_root, 9)
            self._materialize_long100_checkpoint(checkpoint_root, 99)
            log_path = root / "iter9.log"
            trajectory_path = root / "trajectories.jsonl"
            self._write_long100_log(
                log_path,
                start=0,
                end=9,
                load_root=sft_root,
                load_iteration=0,
                quota=LONG100_STAGES["iter9"]["quota"],
            )
            with log_path.open("a", encoding="utf-8") as file:
                file.write(
                    "successfully saved checkpoint from iteration 9 to "
                    f"{checkpoint_root} [ t 1/2, p 1/1 ]\n"
                )
            self._write_long100_trajectories(trajectory_path)
            report = evaluate_long100_health(
                stage="iter9",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_root=checkpoint_root,
                sft_checkpoint_root=sft_root,
                training_logs=[log_path],
                trajectory_paths=[trajectory_path],
            )
            self.assertEqual(report["status"], "pass")
            checkpoint_check = next(
                check
                for check in report["checks"]
                if check["name"] == "checkpoint_latest"
            )
            self.assertEqual(
                checkpoint_check["observed"],
                {
                    "current_latest": 99,
                    "stage_iteration": 9,
                    "exact_stage_save_event": True,
                },
            )

            wrong_save = log_path.read_text(encoding="utf-8").replace(
                f"to {checkpoint_root} [",
                f"to {root / 'wrong-root'} [",
            )
            wrong_log = root / "wrong-save.log"
            wrong_log.write_text(wrong_save, encoding="utf-8")
            rejected = evaluate_long100_health(
                stage="iter9",
                checkpoint_root=checkpoint_root,
                expected_checkpoint_root=checkpoint_root,
                sft_checkpoint_root=sft_root,
                training_logs=[wrong_log],
                trajectory_paths=[trajectory_path],
            )
            self.assertEqual(rejected["status"], "fail")
            self.assertFalse(
                next(
                    check
                    for check in rejected["checks"]
                    if check["name"] == "checkpoint_latest"
                )["passed"]
            )

    def test_long200_health_cross_root_resume_deduplicates_ordered_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft_root = root / "selected-sft"
            sft_root.mkdir()
            source_root = root / "long100"
            destination_root = root / "long200"
            self._materialize_long100_checkpoint(source_root, 99)
            self._materialize_long100_checkpoint(destination_root, 199)
            source_gate = self._long100_health_gate(source_root, "iter99")
            source_gate_hash = "a" * 64
            source_gate_path = root / "ITER99_HEALTH_GATE.json"
            lineage = {
                "lineage_version": "boundary-v2-long200-v1",
                "source_checkpoint_root": str(source_root.resolve()),
                "destination_checkpoint_root": str(destination_root.resolve()),
                "source_iteration": 99,
                "target_iteration": 199,
                "source_health_gate": str(source_gate_path.resolve()),
                "source_health_gate_sha256": source_gate_hash,
                "scheduler_horizon_override": True,
            }

            first_log = root / "attempt-1.log"
            resumed_log = root / "attempt-2.log"
            self._write_long100_log(
                first_log,
                start=100,
                end=129,
                load_root=source_root,
                load_iteration=99,
                quota=LONG200_STAGES["iter199"]["quota"],
                wrong_quota_step=120,
            )
            self._write_long100_log(
                resumed_log,
                start=120,
                end=199,
                load_root=destination_root,
                load_iteration=119,
                quota=LONG200_STAGES["iter199"]["quota"],
                save_iteration=199,
                save_root=destination_root,
            )
            first_trajectories = root / "attempt-1.jsonl"
            resumed_trajectories = root / "attempt-2.jsonl"
            self._write_long100_trajectories(
                first_trajectories,
                rows=30 * 48,
                rollout_start=100,
                invalid_credit_rollout=120,
            )
            self._write_long100_trajectories(
                resumed_trajectories,
                rows=80 * 48,
                rollout_start=120,
            )
            report = evaluate_long100_health(
                stage="iter199",
                checkpoint_root=destination_root,
                expected_checkpoint_root=destination_root,
                sft_checkpoint_root=sft_root,
                training_logs=[first_log, resumed_log],
                trajectory_paths=[first_trajectories, resumed_trajectories],
                source_checkpoint_root=source_root,
                source_health=source_gate,
                source_health_sha256=source_gate_hash,
                source_health_artifact=source_gate_path,
                lineage=lineage,
                mode="final",
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["gate_version"], LONG200_GATE_VERSION)
            self.assertEqual(
                [
                    item["recovery_requirement"]
                    for item in report["training"]["logs"]
                ],
                [
                    {
                        "mode": "cross_root_resume",
                        "path": str(source_root.resolve()),
                        "iteration": 99,
                    },
                    {
                        "mode": "resume_rl",
                        "path": str(destination_root.resolve()),
                        "iteration": 119,
                    },
                ],
            )
            self.assertEqual(
                report["trajectories"]["duplicate_rollout_ids"],
                list(range(120, 130)),
            )
            self.assertEqual(report["trajectories"]["ignored_rows"], 10 * 48)
            checks = {check["name"]: check for check in report["checks"]}
            self.assertTrue(checks["domain_quota"]["passed"])
            self.assertTrue(checks["trajectory_turn_credit_alignment"]["passed"])

            first_lines = first_log.read_text(encoding="utf-8").splitlines()
            reference_warning_log = root / "attempt-1-reference-rng-warning.log"
            reference_warning_log.write_text(
                "\n".join(
                    [
                        first_lines[0],
                        "(TP, PP) mismatch after resume: RNG state will be ignored",
                        f"loading distributed checkpoint from {sft_root} at iteration 2413",
                        *first_lines[1:],
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            reference_warning = evaluate_long100_health(
                stage="iter199",
                checkpoint_root=destination_root,
                expected_checkpoint_root=destination_root,
                sft_checkpoint_root=sft_root,
                training_logs=[reference_warning_log, resumed_log],
                trajectory_paths=[first_trajectories, resumed_trajectories],
                source_checkpoint_root=source_root,
                source_health=source_gate,
                source_health_sha256=source_gate_hash,
                source_health_artifact=source_gate_path,
                lineage=lineage,
                mode="final",
            )
            self.assertEqual(reference_warning["status"], "pass")

            main_warning_log = root / "attempt-1-main-rng-warning.log"
            main_warning_log.write_text(
                "\n".join(
                    [
                        "(TP, PP) mismatch after resume: RNG state will be ignored",
                        f"loading distributed checkpoint from {source_root} at iteration 99",
                        *first_lines,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            main_warning = evaluate_long100_health(
                stage="iter199",
                checkpoint_root=destination_root,
                expected_checkpoint_root=destination_root,
                sft_checkpoint_root=sft_root,
                training_logs=[main_warning_log, resumed_log],
                trajectory_paths=[first_trajectories, resumed_trajectories],
                source_checkpoint_root=source_root,
                source_health=source_gate,
                source_health_sha256=source_gate_hash,
                source_health_artifact=source_gate_path,
                lineage=lineage,
                mode="final",
            )
            self.assertEqual(main_warning["status"], "fail")
            self.assertFalse(
                next(
                    check
                    for check in main_warning["checks"]
                    if check["name"] == "recovery_integrity"
                )["passed"]
            )

            wrong_source_gate = copy.deepcopy(source_gate)
            wrong_source_gate["checkpoint_root"] = str((root / "wrong-source").resolve())
            rejected = evaluate_long100_health(
                stage="iter199",
                checkpoint_root=destination_root,
                expected_checkpoint_root=destination_root,
                sft_checkpoint_root=sft_root,
                training_logs=[first_log, resumed_log],
                trajectory_paths=[first_trajectories, resumed_trajectories],
                source_checkpoint_root=source_root,
                source_health=wrong_source_gate,
                source_health_sha256=source_gate_hash,
                source_health_artifact=source_gate_path,
                lineage=lineage,
                mode="final",
            )
            self.assertEqual(rejected["status"], "fail")
            self.assertFalse(
                next(
                    check
                    for check in rejected["checks"]
                    if check["name"] == "source_health"
                )["passed"]
            )

    def test_long200_partial_health_accepts_progress_but_final_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft_root = root / "selected-sft"
            sft_root.mkdir()
            source_root = root / "long100"
            destination_root = root / "long200"
            self._materialize_long100_checkpoint(source_root, 99)
            self._materialize_long100_checkpoint(destination_root, 119)
            source_gate = self._long100_health_gate(source_root, "iter99")
            source_gate_hash = "b" * 64
            source_gate_path = root / "ITER99_HEALTH_GATE.json"
            lineage = {
                "lineage_version": "boundary-v2-long200-v1",
                "source_checkpoint_root": str(source_root.resolve()),
                "destination_checkpoint_root": str(destination_root.resolve()),
                "source_iteration": 99,
                "target_iteration": 199,
                "source_health_gate": str(source_gate_path.resolve()),
                "source_health_gate_sha256": source_gate_hash,
                "scheduler_horizon_override": True,
            }
            log_path = root / "partial.log"
            trajectory_path = root / "partial.jsonl"
            self._write_long100_log(
                log_path,
                start=100,
                end=124,
                load_root=source_root,
                load_iteration=99,
                quota=LONG200_STAGES["iter199"]["quota"],
                save_iteration=119,
                save_root=destination_root,
            )
            self._write_long100_trajectories(
                trajectory_path,
                rows=25 * 48,
                rollout_start=100,
            )
            common = {
                "stage": "iter199",
                "checkpoint_root": destination_root,
                "expected_checkpoint_root": destination_root,
                "sft_checkpoint_root": sft_root,
                "training_logs": [log_path],
                "trajectory_paths": [trajectory_path],
                "source_checkpoint_root": source_root,
                "source_health": source_gate,
                "source_health_sha256": source_gate_hash,
                "source_health_artifact": source_gate_path,
                "lineage": lineage,
            }
            partial = evaluate_long100_health(mode="partial", **common)
            final = evaluate_long100_health(mode="final", **common)
            self.assertEqual(partial["status"], "pass")
            self.assertEqual(partial["completed_steps"], list(range(100, 125)))
            self.assertEqual(final["status"], "fail")
            self.assertFalse(
                next(
                    check
                    for check in final["checks"]
                    if check["name"] == "training_metrics_complete"
                )["passed"]
            )

    def test_long200_kl_waiver_demotes_only_kl_and_authorizes_diagnostic_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft_root = root / "selected-sft"
            sft_root.mkdir()
            source_root = root / "long100"
            destination_root = root / "long200"
            self._materialize_long100_checkpoint(source_root, 99)
            self._materialize_long100_checkpoint(destination_root, 199)
            source_gate = self._long100_health_gate(source_root, "iter99")
            source_gate_hash = "e" * 64
            source_gate_path = root / "ITER99_HEALTH_GATE.json"
            source_gate_path.write_text(json.dumps(source_gate), encoding="utf-8")
            lineage = {
                "lineage_version": "boundary-v2-long200-v1",
                "source_checkpoint_root": str(source_root.resolve()),
                "destination_checkpoint_root": str(destination_root.resolve()),
                "source_iteration": 99,
                "target_iteration": 199,
                "source_health_gate": str(source_gate_path.resolve()),
                "source_health_gate_sha256": source_gate_hash,
                "scheduler_horizon_override": True,
            }
            log_path = root / "kl-waiver.log"
            trajectory_path = root / "kl-waiver.jsonl"
            self._write_long100_log(
                log_path,
                start=100,
                end=199,
                load_root=source_root,
                load_iteration=99,
                quota=LONG200_STAGES["iter199"]["quota"],
                high_kl_steps=(198, 199),
                save_iteration=199,
                save_root=destination_root,
            )
            self._write_long100_trajectories(
                trajectory_path,
                rows=100 * 48,
                rollout_start=100,
            )
            common = {
                "stage": "iter199",
                "checkpoint_root": destination_root,
                "expected_checkpoint_root": destination_root,
                "sft_checkpoint_root": sft_root,
                "training_logs": [log_path],
                "trajectory_paths": [trajectory_path],
                "source_checkpoint_root": source_root,
                "source_health": source_gate,
                "source_health_sha256": source_gate_hash,
                "source_health_artifact": source_gate_path,
                "lineage": lineage,
                "mode": "final",
            }
            strict = evaluate_long100_health(**common)
            waived = evaluate_long100_health(waive_kl=True, **common)
            self.assertEqual(strict["status"], "fail")
            self.assertEqual(waived["status"], "pass")
            self.assertEqual(waived["health_policy"], "kl-waiver-v1")
            self.assertEqual(
                waived["waived_checks"],
                ["last_10_k2_kl", "no_consecutive_high_kl"],
            )
            waived_checks = {check["name"]: check for check in waived["checks"]}
            self.assertFalse(waived_checks["no_consecutive_high_kl"]["passed"])
            self.assertFalse(waived_checks["no_consecutive_high_kl"]["enforced"])
            self.assertTrue(
                all(
                    check["passed"]
                    for check in waived["checks"]
                    if check["name"] not in waived["waived_checks"]
                )
            )

            waiver_gate_path = root / "ITER199_KL_WAIVER_HEALTH_GATE.json"
            waiver_gate_path.write_text(json.dumps(waived), encoding="utf-8")
            authorization = authorize_checkpoint(
                variant="long200-kl-waiver",
                iteration=199,
                checkpoint_root=destination_root,
                health_gate_path=waiver_gate_path,
                reference_root=source_root,
            )
            self.assertEqual(authorization["status"], "pass")
            self.assertEqual(authorization["variant"], "long200-kl-waiver")
            with self.assertRaisesRegex(ValueError, "does not authorize"):
                authorize_checkpoint(
                    variant="long200",
                    iteration=199,
                    checkpoint_root=destination_root,
                    health_gate_path=waiver_gate_path,
                    reference_root=source_root,
                )

            nonfinite_log = root / "kl-waiver-nonfinite.log"
            self._write_long100_log(
                nonfinite_log,
                start=100,
                end=199,
                load_root=source_root,
                load_iteration=99,
                quota=LONG200_STAGES["iter199"]["quota"],
                high_kl_steps=(198, 199),
                nonfinite_step=150,
                save_iteration=199,
                save_root=destination_root,
            )
            nonfinite = evaluate_long100_health(
                waive_kl=True,
                **{**common, "training_logs": [nonfinite_log]},
            )
            self.assertEqual(nonfinite["status"], "fail")
            self.assertFalse(
                next(
                    check
                    for check in nonfinite["checks"]
                    if check["name"] == "training_finite"
                )["passed"]
            )

    def test_checkpoint_curve_authorization_requires_gate_save_and_full_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "long100"
            self._materialize_long100_checkpoint(checkpoint_root, 39)
            self._materialize_long100_checkpoint(checkpoint_root, 99)
            gate = self._long100_health_gate(checkpoint_root, "iter99")
            gate["training"] = {
                "save_events": [
                    {"path": str(checkpoint_root.resolve()), "iteration": 39},
                    {"path": str(checkpoint_root.resolve()), "iteration": 99},
                ]
            }
            gate_path = root / "ITER99_HEALTH_GATE.json"
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            report = authorize_checkpoint(
                variant="curve",
                iteration=39,
                checkpoint_root=checkpoint_root,
                health_gate_path=gate_path,
                reference_root=checkpoint_root,
            )
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["checkpoint_manifest"]["complete"])

            wrong_gate = copy.deepcopy(gate)
            wrong_gate["checkpoint_root"] = str((root / "wrong-root").resolve())
            wrong_gate_path = root / "wrong-gate.json"
            wrong_gate_path.write_text(json.dumps(wrong_gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not authorize"):
                authorize_checkpoint(
                    variant="curve",
                    iteration=39,
                    checkpoint_root=checkpoint_root,
                    health_gate_path=wrong_gate_path,
                    reference_root=checkpoint_root,
                )

            (checkpoint_root / "iter_0000039" / "common.pt").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                authorize_checkpoint(
                    variant="curve",
                    iteration=39,
                    checkpoint_root=checkpoint_root,
                    health_gate_path=gate_path,
                    reference_root=checkpoint_root,
                )

    def test_rl_final_gate_uses_balanced_quota_and_both_prior_models(self):
        self.assertEqual(FINAL_QUOTA, {"telecom": 2, "airline": 2, "retail": 2})
        finalizer = (
            ANALYSIS_DIR / "finalize_agent_boundary_v2_rl.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('--model "selected_sft=${sft_summary}"', finalizer)
        self.assertIn('--model "rl_iter19=${iter19_summary}"', finalizer)
        self.assertIn('--model "rl_iter99=${iter99_summary}"', finalizer)
        self.assertIn("--window 3840", finalizer)

    def test_long100_decision_classifies_win_usable_and_reject(self):
        def summary(affected, attempts, terminations):
            return {
                "agent_protocol_profile": (
                    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                ),
                "domains": {
                    "telecom": {
                        "namespace": {
                            "trajectory_count": 160,
                            "affected_trajectory_count": affected,
                            "raw_attempt_count": attempts,
                            "namespace_attributed_termination_count": terminations,
                        }
                    }
                },
            }

        def model(pass1, pass4, pass_power):
            return {
                "tasks": 100,
                "simulations": 400,
                "overall": {
                    "pass_at_1": pass1,
                    "pass_at_4_any": pass4,
                    "pass_power_4": pass_power,
                },
                "by_domain": {
                    domain: {
                        "pass_at_1": pass1,
                        "pass_at_4_any": pass4,
                        "pass_power_4": pass_power,
                    }
                    for domain in ("airline", "retail", "telecom")
                },
                "behavior": {"overall": {"termination_reasons": {"max_steps": 80}}},
            }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / "long100"
            health = {
                stage: self._long100_health_gate(checkpoint_root, stage)
                for stage in LONG100_STAGES
            }
            seeds = (300, 301)
            sft = {seed: summary(10, 30, 3) for seed in seeds}
            iter9 = {seed: summary(8, 25, 2) for seed in seeds}
            iter99 = {seed: summary(7, 24, 2) for seed in seeds}
            comparisons = {
                seed: {
                    "models": {
                        "selected_sft": model(0.25, 0.45, 0.10),
                        "historical_rl_iter9": model(0.27, 0.47, 0.09),
                        "long100_iter99": model(0.30, 0.50, 0.09),
                    },
                    "paired_deltas": [
                        {
                            "scope": "telecom",
                            "metric": "pass_at_1",
                            "ci95": [-0.01, 0.07],
                        }
                    ],
                }
                for seed in seeds
            }
            common = {
                "checkpoint_root": checkpoint_root,
                "health_gates": health,
                "sft_summaries": sft,
                "iter9_summaries": iter9,
                "iter99_summaries": iter99,
                "probe_complete": True,
                "probe_observation": {"cases": 300},
                "candidate": "long100_iter99",
                "baseline": "selected_sft",
                "prior": "historical_rl_iter9",
            }
            win = evaluate_long100_decision(
                comparisons=comparisons,
                **common,
            )
            self.assertEqual(win["status"], "win")
            self.assertEqual(win["decision_version"], LONG100_DECISION_VERSION)
            self.assertEqual(
                set(win["behavior_diagnostics_only"]["evaluation"]["300"]),
                {"selected_sft", "historical_rl_iter9", "long100_iter99"},
            )
            self.assertEqual(
                win["behavior_diagnostics_only"]["evaluation"]["300"]
                ["historical_rl_iter9"]["overall"]["termination_reasons"],
                {"max_steps": 80},
            )

            usable_comparisons = copy.deepcopy(comparisons)
            for comparison in usable_comparisons.values():
                candidate = comparison["models"]["long100_iter99"]["overall"]
                prior = comparison["models"]["historical_rl_iter9"]["overall"]
                candidate["pass_at_1"] = prior["pass_at_1"]
            usable = evaluate_long100_decision(
                comparisons=usable_comparisons,
                **common,
            )
            self.assertEqual(usable["status"], "usable")
            self.assertTrue(usable["hard_conditions_passed"])
            self.assertFalse(usable["strict_improvements_passed"])

            reject_comparisons = copy.deepcopy(comparisons)
            for comparison in reject_comparisons.values():
                comparison["models"]["long100_iter99"]["by_domain"]["telecom"][
                    "pass_at_1"
                ] = 0.19
            reject = evaluate_long100_decision(
                comparisons=reject_comparisons,
                **common,
            )
            self.assertEqual(reject["status"], "reject")
            self.assertFalse(reject["capability_noninferiority"])

    def test_long200_decision_classifies_win_usable_and_reject(self):
        def summary(affected, attempts, terminations):
            return {
                "agent_protocol_profile": (
                    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                ),
                "domains": {
                    "telecom": {
                        "namespace": {
                            "trajectory_count": 160,
                            "affected_trajectory_count": affected,
                            "raw_attempt_count": attempts,
                            "namespace_attributed_termination_count": terminations,
                        }
                    }
                },
            }

        def model(pass1, pass4, pass_power):
            return {
                "tasks": 100,
                "simulations": 400,
                "overall": {
                    "pass_at_1": pass1,
                    "pass_at_4_any": pass4,
                    "pass_power_4": pass_power,
                },
                "by_domain": {
                    domain: {
                        "pass_at_1": pass1,
                        "pass_at_4_any": pass4,
                        "pass_power_4": pass_power,
                    }
                    for domain in ("airline", "retail", "telecom")
                },
                "behavior": {"overall": {"termination_reasons": {"max_steps": 80}}},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "long200"
            source_root = root / "long100"
            health_gate = {
                "status": "pass",
                "gate_version": LONG200_GATE_VERSION,
                "health_mode": "final",
                "stage": "iter199",
                "checkpoint_root": str(checkpoint_root.resolve()),
                "expected_checkpoint_root": str(checkpoint_root.resolve()),
                "destination_checkpoint_root": str(checkpoint_root.resolve()),
                "source_checkpoint_root": str(source_root.resolve()),
                "source_health_gate_sha256": "c" * 64,
                "expected_latest": 199,
                "protocol_profile": PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
                "agent_protocol_signature": protocol_signature(
                    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                ),
                "turn_credit_version": "turn-credit-v1",
                "checks": [{"name": "all", "passed": True}],
                "diagnostics_only": {"termination_reasons": {"max_steps": 1}},
            }
            seeds = (300, 301)
            sft = {seed: summary(10, 30, 3) for seed in seeds}
            iter99 = {seed: summary(7, 24, 2) for seed in seeds}
            iter199 = {seed: summary(5, 20, 1) for seed in seeds}
            comparisons = {
                seed: {
                    "models": {
                        "selected_sft": model(0.25, 0.45, 0.10),
                        "long100_iter99": model(0.30, 0.55, 0.10),
                        "long200_iter199": model(0.32, 0.57, 0.11),
                    },
                    "paired_deltas": [],
                }
                for seed in seeds
            }
            probe = {
                "reported_status": "pass",
                "temperature_0": {
                    "pass": True,
                    "namespace_attempts": 0,
                },
                "temperature_0_6": {
                    "pass": True,
                    "namespace_attempts": 0,
                },
            }
            common = {
                "checkpoint_root": checkpoint_root,
                "source_checkpoint_root": source_root,
                "health_gate": health_gate,
                "sft_summaries": sft,
                "iter99_summaries": iter99,
                "iter199_summaries": iter199,
                "probe_complete": True,
                "probe_observation": probe,
                "candidate": "long200_iter199",
                "baseline": "selected_sft",
                "prior": "long100_iter99",
            }
            win = evaluate_long200_decision(comparisons=comparisons, **common)
            self.assertEqual(win["status"], "win")
            self.assertEqual(win["decision_version"], LONG200_DECISION_VERSION)

            usable_comparisons = copy.deepcopy(comparisons)
            for comparison in usable_comparisons.values():
                comparison["models"]["long200_iter199"]["overall"][
                    "pass_at_1"
                ] = 0.30
            usable = evaluate_long200_decision(
                comparisons=usable_comparisons,
                **common,
            )
            self.assertEqual(usable["status"], "usable")
            self.assertTrue(usable["hard_conditions_passed"])
            self.assertFalse(usable["strict_improvements_passed"])

            reject_comparisons = copy.deepcopy(comparisons)
            for comparison in reject_comparisons.values():
                comparison["models"]["long200_iter199"]["overall"][
                    "pass_at_1"
                ] = 0.27
            reject = evaluate_long200_decision(
                comparisons=reject_comparisons,
                **common,
            )
            self.assertEqual(reject["status"], "reject")
            self.assertFalse(reject["capability_noninferiority"])

    def test_long200_decision_rejects_valid_early_health_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "long200"
            source_root = root / "long100"
            gate = {
                "status": "fail",
                "gate_version": LONG200_GATE_VERSION,
                "health_mode": "partial",
                "stage": "iter199",
                "checkpoint_root": str(checkpoint_root.resolve()),
                "expected_checkpoint_root": str(checkpoint_root.resolve()),
                "destination_checkpoint_root": str(checkpoint_root.resolve()),
                "source_checkpoint_root": str(source_root.resolve()),
                "source_health_gate_sha256": "d" * 64,
                "expected_latest": 199,
                "protocol_profile": PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
                "agent_protocol_signature": protocol_signature(
                    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                ),
                "turn_credit_version": "turn-credit-v1",
                "completed_steps": list(range(100, 177)),
                "checks": [
                    {"name": "training_finite", "passed": True},
                    {
                        "name": "last_10_k2_kl",
                        "passed": False,
                        "observed": 0.12,
                        "requirement": "< 0.10",
                    },
                ],
                "diagnostics_only": {"max_steps": 12},
            }
            decision = evaluate_long200_early_reject(
                checkpoint_root=checkpoint_root,
                source_checkpoint_root=source_root,
                health_gate=gate,
                candidate="long200_iter199",
                baseline="selected_sft",
                prior="long100_iter99",
            )
            self.assertEqual(decision["status"], "reject")
            self.assertEqual(decision["decision_basis"], "early_health_stop")
            self.assertEqual(decision["completed_latest"], 176)
            self.assertEqual(decision["selected_model"], "long100_iter99")
            self.assertFalse(decision["replacement_authorized"])
            self.assertTrue(decision["evaluation_skipped"]["seed300"])

            wrong_root = copy.deepcopy(gate)
            wrong_root["checkpoint_root"] = str((root / "wrong").resolve())
            with self.assertRaisesRegex(ValueError, "early-stop health gate"):
                evaluate_long200_early_reject(
                    checkpoint_root=checkpoint_root,
                    source_checkpoint_root=source_root,
                    health_gate=wrong_root,
                    candidate="long200_iter199",
                    baseline="selected_sft",
                    prior="long100_iter99",
                )

    def test_long200_kl_waiver_result_never_replaces_the_formal_selection(self):
        def summary(affected, attempts, terminations):
            return {
                "agent_protocol_profile": (
                    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                ),
                "domains": {
                    "telecom": {
                        "namespace": {
                            "trajectory_count": 160,
                            "affected_trajectory_count": affected,
                            "raw_attempt_count": attempts,
                            "namespace_attributed_termination_count": terminations,
                        }
                    }
                },
            }

        def model(pass1, pass4, pass_power):
            return {
                "tasks": 100,
                "simulations": 400,
                "overall": {
                    "pass_at_1": pass1,
                    "pass_at_4_any": pass4,
                    "pass_power_4": pass_power,
                },
                "by_domain": {
                    domain: {
                        "pass_at_1": pass1,
                        "pass_at_4_any": pass4,
                        "pass_power_4": pass_power,
                    }
                    for domain in ("airline", "retail", "telecom")
                },
                "behavior": {},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "long200"
            source_root = root / "long100"
            gate = {
                "status": "pass",
                "gate_version": LONG200_KL_WAIVER_GATE_VERSION,
                "health_mode": "final",
                "health_policy": "kl-waiver-v1",
                "waived_checks": [
                    "last_10_k2_kl",
                    "no_consecutive_high_kl",
                ],
                "stage": "iter199",
                "checkpoint_root": str(checkpoint_root.resolve()),
                "expected_checkpoint_root": str(checkpoint_root.resolve()),
                "destination_checkpoint_root": str(checkpoint_root.resolve()),
                "source_checkpoint_root": str(source_root.resolve()),
                "source_health_gate_sha256": "f" * 64,
                "expected_latest": 199,
                "protocol_profile": PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
                "agent_protocol_signature": protocol_signature(
                    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                ),
                "turn_credit_version": "turn-credit-v1",
                "checks": [
                    {"name": "training_finite", "passed": True},
                    {
                        "name": "last_10_k2_kl",
                        "passed": False,
                        "enforced": False,
                    },
                    {
                        "name": "no_consecutive_high_kl",
                        "passed": False,
                        "enforced": False,
                    },
                ],
                "diagnostics_only": {},
            }
            seeds = (300, 301)
            sft = {seed: summary(10, 30, 3) for seed in seeds}
            iter99 = {seed: summary(7, 24, 2) for seed in seeds}
            iter199 = {seed: summary(5, 20, 1) for seed in seeds}
            candidate = "long200_iter199_kl_waiver"
            comparisons = {
                seed: {
                    "models": {
                        "selected_sft": model(0.25, 0.45, 0.10),
                        "long100_iter99": model(0.30, 0.55, 0.10),
                        candidate: model(0.32, 0.57, 0.11),
                    },
                    "paired_deltas": [],
                }
                for seed in seeds
            }
            probe = {
                "reported_status": "pass",
                "temperature_0": {"pass": True, "namespace_attempts": 0},
                "temperature_0_6": {"pass": True, "namespace_attempts": 0},
            }
            common = {
                "checkpoint_root": checkpoint_root,
                "source_checkpoint_root": source_root,
                "health_gate": gate,
                "sft_summaries": sft,
                "iter99_summaries": iter99,
                "iter199_summaries": iter199,
                "comparisons": comparisons,
                "probe_complete": True,
                "probe_observation": probe,
                "candidate": candidate,
                "baseline": "selected_sft",
                "prior": "long100_iter99",
            }
            result = evaluate_long200_kl_waiver_result(**common)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["strict_capability_point_estimates_passed"])
            self.assertEqual(result["formal_decision_unchanged"], "reject")
            self.assertEqual(result["selected_model"], "long100_iter99")
            self.assertFalse(result["replacement_authorized"])

            nonfinite_gate = copy.deepcopy(gate)
            nonfinite_gate["checks"][0]["passed"] = False
            failed = evaluate_long200_kl_waiver_result(
                **{**common, "health_gate": nonfinite_gate}
            )
            self.assertEqual(failed["status"], "fail")

    def test_long100_runners_use_isolated_root_health_gates_and_two_gpu_eval(self):
        training_runner = (
            Path(__file__).resolve().parents[1]
            / "examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_boundary_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("long100-a|long100-b|long100-final", training_runner)
        self.assertIn(
            "Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804",
            training_runner,
        )
        self.assertEqual(
            training_runner.count("OVERRIDE_OPT_PARAM_SCHEDULER_VALUE=1"),
            4,
        )
        base_training_runner = (
            Path(__file__).resolve().parents[1]
            / "examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--override-opt_param-scheduler", base_training_runner)
        self.assertIn('--load "${LOAD_DIR}"', base_training_runner)
        self.assertIn("long200-final", training_runner)
        self.assertIn("long200-final-kl-waiver", training_runner)
        self.assertIn(
            "Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805",
            training_runner,
        )
        self.assertIn("ITER9_HEALTH_GATE.json", training_runner)
        self.assertIn("ITER19_HEALTH_GATE.json", training_runner)
        self.assertIn("ITER9_SAFETY_GATE.json", training_runner)
        self.assertIn("namespace_strict_reduction", training_runner)

        conversion = (
            Path(__file__).resolve().parents[1]
            / "scripts/convert_tau2_agent_rl_boundary_v2_iter_to_hf.sh"
        ).read_text(encoding="utf-8")
        evaluation = (
            OFFICIAL_EVAL_DIR
            / "models/run_full_tau2_agent_rl_boundary_v2_user_stop_parser.sh"
        ).read_text(encoding="utf-8")
        finalizer = (
            ANALYSIS_DIR / "finalize_agent_boundary_v2_rl_long100.sh"
        ).read_text(encoding="utf-8")
        long200_health_finalizer = (
            ANALYSIS_DIR / "finalize_agent_boundary_v2_rl_long200_health.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("legacy|long100", conversion)
        self.assertIn("curve|long200", conversion)
        self.assertIn("long200-kl-waiver", conversion)
        self.assertIn("ITER${ITERATION}_HEALTH_GATE.json", conversion)
        self.assertIn("legacy|long100", evaluation)
        self.assertIn("curve|long200", evaluation)
        self.assertIn("long200-kl-waiver", evaluation)
        self.assertIn('export AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.6}"', evaluation)
        self.assertIn('export MAX_STEPS="${MAX_STEPS:-200}"', evaluation)
        self.assertIn('--model "historical_rl_iter9=${iter9_summary}"', finalizer)
        self.assertNotIn("--iter19-summary", finalizer)
        self.assertIn("ITER99_LONG100_DECISION.json", finalizer)
        self.assertLess(
            long200_health_finalizer.index(
                "long200-final_iter0100-0199_*.jsonl"
            ),
            long200_health_finalizer.index(
                "long200-final-kl-waiver_iter0100-0199_*.jsonl"
            ),
        )

    def test_domain_expert_lineage_isolated_first_load_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "long100"
            self._materialize_domain_checkpoint(source, 99)
            source_gate_path = root / "ITER99_HEALTH_GATE.json"
            source_gate_path.write_text(
                json.dumps(self._long100_health_gate(source, "iter99")),
                encoding="utf-8",
            )

            destinations = {}
            for domain in ("airline", "retail", "telecom"):
                destination = root / f"{domain}-expert"
                destinations[domain] = destination
                load_root, lineage = prepare_domain_expert(
                    domain=domain,
                    source_root=source,
                    destination_root=destination,
                    source_gate_path=source_gate_path,
                )
                self.assertEqual(load_root, source.resolve())
                self.assertEqual(lineage["expert_domain"], domain)
                self.assertEqual(lineage["domain_quota"], expert_quota(domain))
                self.assertEqual(
                    lineage["dataset_fingerprint"],
                    read_sampler_state(source, 99)["dataset_fingerprint"],
                )
                self.assertEqual(
                    lineage["destination_checkpoint_root"], str(destination.resolve())
                )
            self.assertEqual(len({str(path.resolve()) for path in destinations.values()}), 3)

            airline = destinations["airline"]
            self._materialize_domain_checkpoint(
                airline,
                109,
                offsets={"airline": 70, "retail": 20, "telecom": 30},
            )
            resumed, _ = prepare_domain_expert(
                domain="airline",
                source_root=source,
                destination_root=airline,
                source_gate_path=source_gate_path,
            )
            self.assertEqual(resumed, airline.resolve())

            torch.save(
                {
                    "sample_group_index": 1,
                    "sample_index": 8,
                    "metadata": {
                        "tau2_domain_quota_v1": {
                            "dataset_fingerprint": "b" * 64,
                            "domain_offsets": {"airline": 70, "retail": 20, "telecom": 30},
                            "domain_epochs": {"airline": 0, "retail": 0, "telecom": 0},
                        }
                    },
                },
                airline / "rollout/global_dataset_state_dict_109.pt",
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                prepare_domain_expert(
                    domain="airline",
                    source_root=source,
                    destination_root=airline,
                    source_gate_path=source_gate_path,
                )
            with self.assertRaisesRegex(ValueError, "lineage marker mismatch"):
                prepare_domain_expert(
                    domain="retail",
                    source_root=source,
                    destination_root=airline,
                    source_gate_path=source_gate_path,
                )

            wrong_gate = self._long100_health_gate(source, "iter99")
            wrong_gate["checkpoint_root"] = str((root / "wrong").resolve())
            wrong_gate_path = root / "WRONG_GATE.json"
            wrong_gate_path.write_text(json.dumps(wrong_gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source Gate"):
                prepare_domain_expert(
                    domain="telecom",
                    source_root=source,
                    destination_root=root / "new-telecom",
                    source_gate_path=wrong_gate_path,
                )

    def test_domain_expert_prefix_health_scopes_later_kl_and_authorizes_only_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "long100"
            destination = root / "airline-expert"
            self._materialize_domain_checkpoint(source, 99)
            self._materialize_domain_checkpoint(
                destination,
                109,
                offsets={"airline": 70, "retail": 20, "telecom": 30},
            )
            self._materialize_domain_checkpoint(
                destination,
                119,
                offsets={"airline": 130, "retail": 20, "telecom": 30},
            )
            source_gate = self._long100_health_gate(source, "iter99")
            source_gate_path = root / "ITER99_HEALTH_GATE.json"
            source_gate_path.write_text(json.dumps(source_gate), encoding="utf-8")
            source_state = read_sampler_state(source, 99)
            lineage = expected_domain_expert_lineage(
                domain="airline",
                source_root=source,
                destination_root=destination,
                source_gate_path=source_gate_path,
                source_sampler_state=source_state,
            )

            first_log = root / "first.log"
            later_log = root / "later.log"
            quota = expert_quota("airline")
            self._write_long100_log(
                first_log,
                start=100,
                end=109,
                load_root=source,
                load_iteration=99,
                quota=quota,
                save_iteration=109,
                save_root=destination,
            )
            self._write_long100_log(
                later_log,
                start=110,
                end=119,
                load_root=destination,
                load_iteration=109,
                quota=quota,
                high_kl_steps=(110, 111),
                save_iteration=119,
                save_root=destination,
            )
            first_trajectories = root / "first.jsonl"
            later_trajectories = root / "later.jsonl"
            self._write_long100_trajectories(
                first_trajectories,
                rows=10 * 48,
                rollout_start=100,
                domains=("airline",),
            )
            self._write_long100_trajectories(
                later_trajectories,
                rows=10 * 48,
                rollout_start=110,
                domains=("airline",),
            )

            iter109 = evaluate_prefix_health(
                variant="domain-expert",
                iteration=109,
                domain="airline",
                checkpoint_root=destination,
                source_checkpoint_root=source,
                source_health=source_gate,
                source_health_path=source_gate_path,
                lineage=lineage,
                training_logs=[first_log, later_log],
                trajectory_paths=[first_trajectories, later_trajectories],
            )
            self.assertEqual(iter109["status"], "pass")
            self.assertEqual(iter109["root_current_latest"], 119)
            self.assertEqual(iter109["expected_steps"], [100, 109])
            checks109 = {check["name"]: check for check in iter109["checks"]}
            self.assertFalse(checks109["last_10_k2_kl"]["enforced"])
            self.assertFalse(checks109["no_consecutive_high_kl"]["enforced"])
            self.assertTrue(checks109["no_consecutive_high_kl"]["passed"])

            iter119 = evaluate_prefix_health(
                variant="domain-expert",
                iteration=119,
                domain="airline",
                checkpoint_root=destination,
                source_checkpoint_root=source,
                source_health=source_gate,
                source_health_path=source_gate_path,
                lineage=lineage,
                training_logs=[first_log, later_log],
                trajectory_paths=[first_trajectories, later_trajectories],
            )
            self.assertEqual(iter119["status"], "pass")
            checks119 = {check["name"]: check for check in iter119["checks"]}
            self.assertFalse(checks119["no_consecutive_high_kl"]["passed"])
            self.assertFalse(checks119["no_consecutive_high_kl"]["enforced"])
            self.assertTrue(all(
                check["passed"] for check in iter119["checks"] if check["enforced"]
            ))

            gate_path = root / "AIRLINE_ITER109_HEALTH_GATE.json"
            gate_path.write_text(json.dumps(iter109), encoding="utf-8")
            authorization = authorize_checkpoint(
                variant="domain-expert",
                domain="airline",
                iteration=109,
                checkpoint_root=destination,
                health_gate_path=gate_path,
                reference_root=source,
            )
            self.assertEqual(authorization["expert_domain"], "airline")
            wrong_domain = copy.deepcopy(iter109)
            wrong_domain["expert_domain"] = "retail"
            gate_path.write_text(json.dumps(wrong_domain), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "domain/quota"):
                authorize_checkpoint(
                    variant="domain-expert",
                    domain="airline",
                    iteration=109,
                    checkpoint_root=destination,
                    health_gate_path=gate_path,
                    reference_root=source,
                )

            wrong_quota_log = root / "wrong-quota.log"
            self._write_long100_log(
                wrong_quota_log,
                start=100,
                end=109,
                load_root=source,
                load_iteration=99,
                quota=quota,
                wrong_quota_step=105,
                save_iteration=109,
                save_root=destination,
            )
            rejected = evaluate_prefix_health(
                variant="domain-expert",
                iteration=109,
                domain="airline",
                checkpoint_root=destination,
                source_checkpoint_root=source,
                source_health=source_gate,
                source_health_path=source_gate_path,
                lineage=lineage,
                training_logs=[wrong_quota_log],
                trajectory_paths=[first_trajectories],
            )
            self.assertEqual(rejected["status"], "fail")
            self.assertFalse(next(
                check for check in rejected["checks"] if check["name"] == "domain_quota"
            )["passed"])

    def test_domain_expert_selection_and_five_point_decision_boundaries(self):
        selected = select_domain_expert_checkpoint(
            [
                {
                    "iteration": 129,
                    "health_passed": True,
                    "protocol_passed": True,
                    "metrics": {"pass_at_1": 0.4, "pass_at_4_any": 0.6, "pass_power_4": 0.1},
                },
                {
                    "iteration": 109,
                    "health_passed": True,
                    "protocol_passed": True,
                    "metrics": {"pass_at_1": 0.4, "pass_at_4_any": 0.6, "pass_power_4": 0.1},
                },
                {
                    "iteration": 119,
                    "health_passed": False,
                    "protocol_passed": True,
                    "metrics": {"pass_at_1": 1.0, "pass_at_4_any": 1.0, "pass_power_4": 1.0},
                },
            ]
        )
        self.assertEqual(selected["iteration"], 109)

        baseline = {
            300: {"pass_at_1": 0.30, "pass_at_4_any": 0.50},
            301: {"pass_at_1": 0.30, "pass_at_4_any": 0.50},
        }
        mixed = copy.deepcopy(baseline)
        exact_margin = {
            300: {"pass_at_1": 0.25, "pass_at_4_any": 0.45},
            301: {"pass_at_1": 0.25, "pass_at_4_any": 0.45},
        }
        near = classify_expert(
            expert_by_seed=exact_margin,
            baseline_by_seed=baseline,
            mixed_by_seed=mixed,
            hard_conditions={"health": True, "protocol": True},
        )
        self.assertEqual(near["classification"], "near-parity")
        self.assertTrue(near["opd_eligible"])

        advantage = classify_expert(
            expert_by_seed={
                300: {"pass_at_1": 0.31, "pass_at_4_any": 0.50},
                301: {"pass_at_1": 0.32, "pass_at_4_any": 0.50},
            },
            baseline_by_seed=baseline,
            mixed_by_seed=mixed,
            hard_conditions={"health": True, "protocol": True},
        )
        self.assertEqual(advantage["classification"], "advantage")
        self.assertTrue(advantage["opd_eligible"])

        for rejected in (
            classify_expert(
                expert_by_seed={
                    300: {"pass_at_1": 0.249, "pass_at_4_any": 0.50},
                    301: {"pass_at_1": 0.30, "pass_at_4_any": 0.50},
                },
                baseline_by_seed=baseline,
                mixed_by_seed=mixed,
                hard_conditions={"health": True},
            ),
            classify_expert(
                expert_by_seed=baseline,
                baseline_by_seed=baseline,
                mixed_by_seed=mixed,
                hard_conditions={"telecom_namespace": False},
            ),
        ):
            self.assertEqual(rejected["classification"], "reject")
            self.assertFalse(rejected["opd_eligible"])

    def test_domain_expert_runners_expose_explicit_variants_and_isolated_hf_roots(self):
        training = (
            Path(__file__).resolve().parents[1]
            / "examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl_agent_boundary_v2.sh"
        ).read_text(encoding="utf-8")
        conversion = (
            Path(__file__).resolve().parents[1]
            / "scripts/convert_tau2_agent_rl_boundary_v2_iter_to_hf.sh"
        ).read_text(encoding="utf-8")
        evaluation = (
            OFFICIAL_EVAL_DIR
            / "models/run_full_tau2_agent_rl_boundary_v2_user_stop_parser.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("domain-expert <airline|retail|telecom>", training)
        self.assertIn("NUM_ROLLOUT_VALUE=130", training)
        self.assertIn("airline:6,retail:0,telecom:0", training)
        self.assertIn("domain-expert|mixed-control", conversion)
        self.assertIn("airline_expert_hf_20260806", conversion)
        self.assertIn("retail_expert_hf_20260806", conversion)
        self.assertIn("telecom_expert_hf_20260806", conversion)
        self.assertIn("domain-expert|mixed-control", evaluation)
        self.assertIn("TAU2_DOMAIN_EXPERT_EVAL_SCOPE", evaluation)
        self.assertIn("[target|forgetting|all]", evaluation)
        self.assertIn('export AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.6}"', evaluation)
        self.assertIn('export MAX_STEPS="${MAX_STEPS:-200}"', evaluation)

    def test_capability_gate_excludes_the_other_new_arm_from_baselines(self):
        models = {
            label: {"by_domain": {"telecom": {"pass_at_1": value}}}
            for label, value in {
                "raw_instruct": 0.20,
                "old_sft": 0.25,
                "current_sft": 0.30,
                "contract-boundary": 0.95,
            }.items()
        }
        self.assertEqual(
            capability_baseline_values(
                models,
                domain="telecom",
                labels=("raw_instruct", "old_sft", "current_sft"),
            ),
            [0.20, 0.25, 0.30],
        )

    def test_sft_gate_reports_namespace_totals_and_v3_thresholds(self):
        signature = "sha256:test-contract"
        label = "candidate"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            probe = root / "probe.json"
            probe.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checkpoint": str(checkpoint),
                        "agent_contract_signature": signature,
                        "user_tool_count": 30,
                        "summaries": {
                            "temperature_0": {"cases": 60, "namespace_attempts": 0},
                            "temperature_0_6": {"cases": 240, "namespace_attempts": 0},
                        },
                    }
                ),
                encoding="utf-8",
            )
            eval_paths = []
            comparison_paths = []
            for seed, affected, attempts in ((300, 60, 350), (301, 61, 351)):
                eval_path = root / f"seed{seed}_summary.json"
                eval_path.write_text(
                    json.dumps(
                        {
                            "agent_protocol_profile": (
                                PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                            ),
                            "seed": seed,
                            "domains": {
                                "telecom": {
                                    "namespace": {
                                        "trajectory_count": 160,
                                        "affected_trajectory_count": affected,
                                        "raw_attempt_count": attempts,
                                        "namespace_attributed_termination_count": 0,
                                    },
                                    "agent_contract": {
                                        "agent_contract_signature": signature
                                    },
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                eval_paths.append(eval_path)

                models = {
                    baseline: {
                        "by_domain": {"telecom": {"pass_at_1": value}}
                    }
                    for baseline, value in {
                        "raw_instruct": 0.2,
                        "old_sft": 0.30625,
                        "current_sft": 0.25,
                    }.items()
                }
                models[label] = {
                    "by_domain": {"telecom": {"pass_at_1": 0.25625}},
                    "overall": {"pass_at_1": 0.25625, "pass_power_4": 0.1},
                }
                comparison_path = root / f"seed{seed}_comparison.json"
                comparison_path.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "comparison_mode": "strict-signed",
                            "protocol_profile": (
                                PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI
                            ),
                            "candidate": label,
                            "protocol_signature": {"seed": seed},
                            "agent_contract_signatures": {"telecom": signature},
                            "models": models,
                            "paired_deltas": [
                                {
                                    "candidate": label,
                                    "reference": "old_sft",
                                    "metric": "pass_power_4",
                                    "ci95": [-0.01, 0.01],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                comparison_paths.append(comparison_path)

            decision = evaluate_boundary_sft_candidate(
                label=label,
                checkpoint=checkpoint,
                probe_path=probe,
                eval_paths=eval_paths,
                comparison_paths=comparison_paths,
                baseline="old_sft",
                capability_baselines=("raw_instruct", "old_sft", "current_sft"),
            )
        self.assertEqual(decision["status"], "fail")
        self.assertEqual(decision["namespace_affected_trajectories"], 121)
        self.assertEqual(decision["namespace_attempts"], 701)
        self.assertIn(
            "telecom_affected_above_60:seed301_summary.json:61",
            decision["reasons"],
        )
        self.assertIn(
            "telecom_attempts_above_350:seed301_summary.json:351",
            decision["reasons"],
        )
        self.assertFalse(
            any("pass_at_1_below_best_by_8pp" in reason for reason in decision["reasons"])
        )
        self.assertEqual(BOUNDARY_SFT_GATE_VERSION, "turn-aware-rl-v3")
        self.assertEqual(
            BOUNDARY_SFT_GATE_THRESHOLDS["per_seed_namespace_affected_max"],
            60,
        )

    def test_sft_gate_promotes_contract_boundary_and_rejects_contract_only(self):
        decisions = [
            {
                "label": "contract-only",
                "status": "fail",
                "namespace_affected_trajectories": 0,
                "namespace_attempts": 0,
                "mean_pass_power_4": 1.0,
                "mean_pass_at_1": 1.0,
                "reasons": ["namespace_probe_failed"],
            },
            {
                "label": "contract-boundary",
                "status": "pass",
                "namespace_affected_trajectories": 109,
                "namespace_attempts": 640,
                "mean_pass_power_4": 0.085,
                "mean_pass_at_1": 0.23625,
                "reasons": [],
            },
        ]
        self.assertEqual(
            [decision["label"] for decision in rank_boundary_sft_eligible(decisions)],
            ["contract-boundary"],
        )

    def test_contract_has_one_protocol_agent_only_tools_and_content_signature(self):
        contract = boundary_contract()
        self.assertEqual(
            [schema["function"]["name"] for schema in contract.tools],
            ["lookup_account"],
        )
        self.assertNotIn("<tools>", contract.system_prompt)
        self.assertIn("<tech_support_policy>", contract.system_prompt)
        self.assertNotIn("one tool call at a time", contract.system_prompt)
        self.assertEqual(
            contract.system_prompt.count(EVAL_DEPENDENCY_SAFE_MULTI_RULE),
            1,
        )
        self.assertTrue(contract.agent_contract_signature.startswith("sha256:"))
        self.assertNotEqual(
            contract.agent_contract_signature,
            boundary_contract(chat_template="changed-template").agent_contract_signature,
        )
        self.assertNotEqual(
            protocol_signature(PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI),
            protocol_signature(EVAL_PROTOCOL_DEPENDENCY_SAFE_MULTI),
        )

    def test_source_view_hides_user_events_and_preserves_strict_provenance(self):
        contract = boundary_contract()
        source = [
            {"role": "system", "content": "legacy"},
            {"role": "user", "content": "Please diagnose the line."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "agent-1",
                        "name": "lookup_account",
                        "arguments": {"value": "A1"},
                    }
                ],
            },
            {
                "role": "tool",
                "id": "agent-1",
                "name": "lookup_account",
                "content": "active",
            },
            {
                "role": "assistant",
                "content": "Please run both device checks and tell me what happens.",
            },
            {
                "role": "user",
                "content": "",
                "tool_calls": [
                    {
                        "id": "user-1",
                        "name": "device_action_00",
                        "arguments": {"value": "x"},
                    },
                    {
                        "id": "user-2",
                        "name": "device_action_01",
                        "arguments": {"value": "y"},
                    },
                ],
            },
            {"role": "tool", "content": "first result"},
            {"role": "tool", "content": "second result"},
            {"role": "user", "content": "Both checks completed successfully."},
            {"role": "assistant", "content": "Thanks; the line is healthy."},
        ]
        view = contract.build_source_view(source)
        self.assertEqual(
            [message["role"] for message in view.messages],
            ["system", "user", "assistant", "tool", "assistant", "user", "assistant"],
        )
        self.assertEqual(
            view.messages[2]["tool_calls"][0]["function"]["name"],
            "lookup_account",
        )
        self.assertEqual(view.messages[3]["tool_call_id"], "agent-1")
        self.assertNotIn(
            "Legacy narration emitted with the call.",
            [message.get("content") for message in view.messages],
        )
        self.assertEqual(
            [event["result"] for event in view.user_tool_events],
            ["first result", "second result"],
        )
        self.assertEqual(
            {event["source_report_message_index"] for event in view.user_tool_events},
            {8},
        )
        self.assertTrue(
            all(
                message.get("content")
                for message in view.messages
                if message["role"] == "user"
            )
        )

    def test_source_view_uses_a_mixed_user_turn_as_the_previous_operation_report(self):
        contract = boundary_contract()
        source = [
            {"role": "system", "content": "legacy"},
            {"role": "user", "content": "Please help."},
            {"role": "assistant", "content": "Run the two checks in order."},
            {
                "role": "user",
                "content": "",
                "tool_calls": [
                    {"id": "u1", "name": "device_action_00", "arguments": {}}
                ],
            },
            {"role": "tool", "id": "u1", "content": "first result"},
            {
                "role": "user",
                "content": "The first check passed; I am starting the second.",
                "tool_calls": [
                    {"id": "u2", "name": "device_action_01", "arguments": {}}
                ],
            },
            {"role": "tool", "id": "u2", "content": "second result"},
            {"role": "user", "content": "The second check passed too."},
            {"role": "assistant", "content": "Thanks; both checks are complete."},
        ]
        view = contract.build_source_view(source)
        self.assertEqual(
            [message["role"] for message in view.messages],
            ["system", "user", "assistant", "user", "user", "assistant"],
        )
        self.assertEqual(
            [event["source_report_message_index"] for event in view.user_tool_events],
            [5, 7],
        )
        self.assertNotIn("tool_calls", view.messages[3])

    def test_source_view_rejects_orphans_ambiguous_pairs_and_missing_report(self):
        contract = boundary_contract()
        with self.assertRaisesRegex(ValueError, "orphan tool result"):
            contract.build_source_view(
                [
                    {"role": "system", "content": "legacy"},
                    {"role": "tool", "content": "orphan"},
                ]
            )

        ambiguous = [
            {"role": "system", "content": "legacy"},
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "same", "name": "lookup_account", "arguments": {}},
                    {"id": "same", "name": "lookup_account", "arguments": {}},
                ],
            },
            {"role": "tool", "id": "same", "content": "ambiguous"},
        ]
        with self.assertRaisesRegex(ValueError, "not uniquely paired"):
            contract.build_source_view(ambiguous)

        missing_report = [
            {"role": "system", "content": "legacy"},
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": "Please run the check."},
            {
                "role": "user",
                "content": "",
                "tool_calls": [
                    {"id": "u1", "name": "device_action_00", "arguments": {}}
                ],
            },
            {"role": "tool", "id": "u1", "content": "done"},
        ]
        with self.assertRaisesRegex(ValueError, "no natural report"):
            contract.build_source_view(missing_report)

    def test_qwen3_full_uses_one_native_template_and_target_only_mask(self):
        tokenizer = FakeQwen3FullTokenizer()
        generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
        tools = [boundary_tool_schema("lookup_account")]
        messages = [
            {"role": "system", "content": "SYSTEM", "step_loss_mask": 0},
            {"role": "user", "content": "SEARCH", "step_loss_mask": 0},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a1",
                        "type": "function",
                        "function": {
                            "name": "lookup_account",
                            "arguments": {"value": "A1"},
                        },
                    }
                ],
                "step_loss_mask": 0,
            },
            {"role": "tool", "content": "RESULT_ONE", "tool_call_id": "a1"},
            {"role": "tool", "content": "RESULT_TWO", "tool_call_id": "a2"},
            {"role": "assistant", "content": "FINAL", "step_loss_mask": 1},
        ]
        token_ids, loss_mask = generator.get_loss_mask(messages, tools=tools)
        expected = tokenizer.apply_chat_template(messages, tokenize=True, tools=tools)
        self.assertEqual(token_ids, expected)
        supervised = tokenizer.decode(
            [token for token, mask in zip(token_ids, loss_mask) if mask]
        )
        self.assertIn("FINAL", supervised)
        self.assertNotIn("SEARCH", supervised)
        self.assertNotIn("lookup_account", supervised)
        self.assertNotIn("RESULT_ONE", supervised)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools)
        self.assertEqual(rendered.count("<|im_start|>user\n<tool_response>"), 1)
        self.assertEqual(rendered.count("<tool_response>"), 2)

    def test_boundary_anchors_cover_30_tools_and_never_target_user_calls(self):
        tokenizer = FakeQwen3FullTokenizer()
        generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
        anchors = build_ownership_repair_anchors(
            contract=boundary_contract(user_tool_count=30),
            generator=generator,
        )
        self.assertEqual(len(anchors), 390)
        self.assertEqual(
            Counter(row["metadata"]["ownership_tool_name"] for row in anchors),
            Counter({f"device_action_{index:02d}": 13 for index in range(30)}),
        )
        for row in anchors:
            self.assertEqual(row["metadata"]["view"], BOUNDARY_ANCHOR_VIEW)
            self.assertIn('"requestor":"user"', row["messages"][1]["content"])
            self.assertFalse(row["messages"][-1].get("tool_calls"))

    def test_sft_token_cap_rejects_instead_of_truncating(self):
        tokenizer = FakeQwen3FullTokenizer()
        generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
        row = {
            "messages": [
                {"role": "system", "content": "S", "step_loss_mask": 0},
                {"role": "user", "content": "U", "step_loss_mask": 0},
                {"role": "assistant", "content": "X" * 200, "step_loss_mask": 1},
            ],
            "tools": [boundary_tool_schema("lookup_account")],
        }
        with self.assertRaisesRegex(ValueError, "no-truncation cap"):
            _tokenize_and_validate_row(row, generator=generator, max_total_tokens=100)

    def test_contract_only_replaces_a_rejected_target_with_same_domain_provenance(self):
        tokenizer = FakeQwen3FullTokenizer()
        generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
        contract = boundary_contract()
        rejected = {
            "messages": [
                {"role": "system", "content": "legacy"},
                {"role": "user", "content": "Help me."},
                {"role": "assistant", "content": "Run the device check."},
                {
                    "role": "user",
                    "content": "",
                    "tool_calls": [
                        {"id": "u1", "name": "device_action_00", "arguments": {}}
                    ],
                },
                {"role": "tool", "id": "u1", "content": "done"},
            ],
            "metadata": {"domain": "telecom", "source_dialog_id": "bad"},
        }
        replacement = {
            "messages": [
                {"role": "system", "content": "legacy"},
                {"role": "user", "content": "Help me."},
            ],
            "answer": {"role": "assistant", "content": "I can help."},
            "metadata": {"domain": "telecom", "source_dialog_id": "good"},
        }
        base = [
            {
                "metadata": {
                    "domain": "telecom",
                    "source_row": 0,
                    "training_schedule": {"schedule_index": 7},
                }
            }
        ]
        rows = rebuild_contract_only_rows(
            source_rows=[rejected, replacement],
            base_rows=base,
            contracts={"telecom": contract},
            generator=generator,
            replacement_candidates={
                "telecom": [
                    {
                        "source_row": 1,
                        "filter_version": "fixture-v1",
                        "protocol_profile": "fixture-profile",
                        "review_tier": "local_only",
                        "decision_sha256": "fixture-decision",
                    }
                ]
            },
            expected_counts={"telecom": 1},
        )
        metadata = rows[0]["metadata"]
        self.assertEqual(metadata["source_row"], 1)
        self.assertEqual(metadata["source_dialog_id"], "good")
        self.assertEqual(
            metadata["contract_only_replacement"]["original_source_row"], 0
        )
        self.assertIn(
            "no natural report",
            metadata["contract_only_replacement"]["original_rejection"],
        )
        self.assertEqual(
            metadata["training_schedule"]["filter_decision_sha256"],
            "fixture-decision",
        )

    def test_namespace_analyzer_counts_raw_parsed_executed_and_termination(self):
        trajectories = [
            {
                "task_id": "telecom-1",
                "termination_reason": "too_many_errors",
                "messages": [
                    {
                        "role": "assistant",
                        "raw_data": {
                            "text": (
                                '<tool_call>{"name":"device_action_00",'
                                '"arguments":{}}</tool_call>'
                                '<tool_call>{"name":"device_action_01","arguments":{}}'
                            )
                        },
                        "tool_calls": [
                            {
                                "id": "u1",
                                "function": {
                                    "name": "device_action_00",
                                    "arguments": {},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "id": "u1",
                        "requestor": "assistant",
                        "content": "unknown tool",
                    },
                ],
            },
            {
                "task_id": "telecom-2",
                "termination_reason": "user_stop",
                "messages": [{"role": "assistant", "content": "Done."}],
            },
        ]
        self.assertEqual(
            raw_tool_names(trajectories[0]["messages"][0]["raw_data"]["text"]),
            ["device_action_00", "device_action_01"],
        )
        report = analyze_namespace_trajectories(
            trajectories,
            user_tool_names={"device_action_00", "device_action_01"},
        )
        self.assertEqual(report["affected_trajectory_count"], 1)
        self.assertEqual(report["raw_attempt_count"], 2)
        self.assertEqual(report["parsed_call_count"], 1)
        self.assertEqual(report["executed_call_count"], 1)
        self.assertEqual(report["namespace_attributed_termination_count"], 1)
        self.assertEqual(report["executed_names"], {"device_action_00": 1})

        live_results_report = analyze_namespace_trajectories(
            SimpleNamespace(simulations=trajectories),
            user_tool_names={"device_action_00", "device_action_01"},
        )
        persisted_results_report = analyze_namespace_trajectories(
            {"simulations": trajectories},
            user_tool_names={"device_action_00", "device_action_01"},
        )
        self.assertEqual(live_results_report, report)
        self.assertEqual(persisted_results_report, report)


class AgentSingleCallV1Test(unittest.TestCase):
    @staticmethod
    def boundary_row(*, prefix=None, target_calls=None):
        messages = [
            {
                "role": "system",
                "content": "old preamble\n<policy>\nPOLICY\n</policy>",
                "step_loss_mask": 0,
            },
            {"role": "user", "content": "Help me.", "step_loss_mask": 0},
            *(prefix or []),
        ]
        target = {
            "role": "assistant",
            "content": "" if target_calls else "Done.",
            "step_loss_mask": 1,
        }
        if target_calls:
            target["tool_calls"] = target_calls
        messages.append(target)
        return {
            "messages": messages,
            "tools": [boundary_tool_schema("first"), boundary_tool_schema("second")],
            "metadata": {
                "domain": "retail",
                "source_dataset": "fixture",
                "source_dialog_id": "dialog-1",
                "source_row": 7,
                "turn_index": 3,
                "protocol_hash": "sha256:old",
                "agent_contract_signature": "sha256:old",
                "audit_record": {"hash": "old"},
            },
        }

    def test_target_multi_call_expands_with_real_results_in_source_order(self):
        calls = [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "first", "arguments": {"value": "A"}},
            },
            {
                "id": "b",
                "type": "function",
                "function": {"name": "second", "arguments": {"value": "B"}},
            },
            {
                "id": "c",
                "type": "function",
                "function": {"name": "first", "arguments": {"value": "C"}},
            },
        ]
        source_rows = [
            {
                "metadata": {"source_dialog_id": "dialog-1"},
                "messages": [
                    {"role": "assistant", "content": "", "tool_calls": calls},
                    {"role": "tool", "content": "RESULT_A"},
                    {"role": "tool", "content": "RESULT_B"},
                    {"role": "tool", "content": "RESULT_C"},
                ],
            }
        ]
        result_index = index_real_tool_results(source_rows)
        rows = convert_boundary_row(
            self.boundary_row(target_calls=calls),
            real_tool_results=result_index,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [message["content"] for message in rows[2]["messages"] if message["role"] == "tool"],
            ["RESULT_A", "RESULT_B"],
        )
        self.assertEqual(
            [
                message["tool_calls"][0]["id"]
                for message in rows[2]["messages"]
                if message["role"] == "assistant" and message.get("tool_calls")
            ],
            ["a", "b", "c"],
        )
        for row in rows:
            validate_single_call_row(row)
            self.assertTrue(
                all(
                    len(message.get("tool_calls") or []) <= 1
                    for message in row["messages"]
                    if message["role"] == "assistant"
                )
            )
            metadata_text = json.dumps(row["metadata"]).lower()
            self.assertNotIn("hash", metadata_text)
            self.assertNotIn("signature", metadata_text)

    def test_prefix_multi_call_is_split_into_single_call_result_pairs(self):
        calls = [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "first", "arguments": {}},
            },
            {
                "id": "b",
                "type": "function",
                "function": {"name": "second", "arguments": {}},
            },
        ]
        prefix = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": calls,
                "step_loss_mask": 0,
            },
            {
                "role": "tool",
                "content": "RESULT_A",
                "tool_call_id": "a",
                "step_loss_mask": 0,
            },
            {
                "role": "tool",
                "content": "RESULT_B",
                "tool_call_id": "b",
                "step_loss_mask": 0,
            },
        ]
        [row] = convert_boundary_row(
            self.boundary_row(prefix=prefix),
            real_tool_results={},
        )
        self.assertEqual(
            [message["role"] for message in row["messages"]],
            ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
        )
        self.assertEqual(
            [message.get("tool_call_id") for message in row["messages"] if message["role"] == "tool"],
            ["a", "b"],
        )
        validate_single_call_row(row)

    def test_overlong_expanded_target_does_not_remove_its_siblings(self):
        calls = [
            {
                "id": "short-a",
                "type": "function",
                "function": {"name": "first", "arguments": {}},
            },
            {
                "id": "long",
                "type": "function",
                "function": {"name": "second", "arguments": {}},
            },
            {
                "id": "short-c",
                "type": "function",
                "function": {"name": "first", "arguments": {}},
            },
        ]
        rows = convert_boundary_row(
            self.boundary_row(target_calls=calls),
            real_tool_results={
                ("dialog-1", "short-a"): "A",
                ("dialog-1", "long"): "B",
            },
        )

        class LengthTokenizer:
            def apply_chat_template(
                self,
                messages,
                tools,
                tokenize,
                add_generation_prompt,
            ):
                target_id = messages[-1]["tool_calls"][0]["id"]
                length = 20 if target_id == "long" else 5
                return list(range(length))

        kept = [
            row
            for row in rows
            if single_call_token_count(row, LengthTokenizer()) <= 10
        ]
        self.assertEqual(
            [row["messages"][-1]["tool_calls"][0]["id"] for row in kept],
            ["short-a", "short-c"],
        )

    def test_single_call_attempt_diagnostics_distinguish_raw_and_parsed_batches(self):
        report = analyze_single_call_attempts(
            [
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "raw_data": {
                                "text": (
                                    '<tool_call>{"name":"first","arguments":{}}</tool_call>'
                                    '<tool_call>{"name":"second","arguments":{}}</tool_call>'
                                ),
                                "tau2_single_call_protocol_error": True,
                                "tau2_multi_tool_attempt_count": 2,
                            },
                        }
                    ]
                },
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [{"id": "a"}, {"id": "b"}],
                        }
                    ]
                },
            ]
        )
        self.assertEqual(report["trajectories_with_multi_call_output"], 2)
        self.assertEqual(report["multi_call_output_turns"], 2)
        self.assertEqual(report["parsed_multi_call_turns"], 1)
        self.assertEqual(report["single_call_protocol_error_turns"], 1)

    def test_formal_eval_reports_pass_at_1_any_four_and_power_four(self):
        results = {
            "simulations": [
                {"task_id": "a", "reward_info": {"reward": reward}}
                for reward in (1, 0, 0, 0)
            ]
            + [
                {"task_id": "b", "reward_info": {"reward": reward}}
                for reward in (1, 1, 1, 1)
            ]
        }
        self.assertEqual(
            eval_pass_metrics(results, 4),
            {
                "tasks": 2,
                "simulations": 8,
                "pass_at_1": 0.625,
                "pass_at_4_any": 1.0,
                "pass_power_4": 0.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
