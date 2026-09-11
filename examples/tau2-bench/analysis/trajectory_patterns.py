#!/usr/bin/env python3
"""Shared, dependency-free helpers for tau2 trajectory pattern analysis."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROMPT_VERSION = "tau2-pattern-v1"
PATTERNS = (
    "intent_tracking",
    "clarify_before_act",
    "read_plan_act",
    "atomic_commit",
    "progressive_diagnosis",
    "grounded_recovery",
    "grounded_closeout",
    "wrong_tool_or_workflow",
    "wrong_arguments",
    "missing_precondition",
    "confirmation_violation",
    "unrelated_or_extra_write",
    "multi_tool_policy_violation",
    "tool_namespace_confusion",
    "nonexistent_tool",
    "repeat_or_fanout",
    "failed_recovery",
    "false_success_claim",
    "incomplete_task",
    "premature_transfer_or_close",
    "domain_policy_compliance",
)
KEY_PATTERNS = (
    "intent_tracking",
    "clarify_before_act",
    "read_plan_act",
    "atomic_commit",
    "progressive_diagnosis",
    "grounded_recovery",
    "grounded_closeout",
    "domain_policy_compliance",
)
PRIMARY_FAILURE_MODES = {
    "wrong_workflow",
    "wrong_arguments",
    "missing_precondition",
    "confirmation_violation",
    "tool_namespace_confusion",
    "failed_recovery",
    "false_success_claim",
    "incomplete_task",
    "policy_violation",
    "other",
    "none",
}
WRITE_TOOLS = {
    "airline": {
        "book_reservation",
        "cancel_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "send_certificate",
    },
    "retail": {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    },
    "telecom": {
        "suspend_line",
        "resume_line",
        "send_payment_request",
        "enable_roaming",
        "disable_roaming",
        "refuel_data",
        "suspend_line_for_overdue_bill",
    },
}
AUTH_TOOLS = {
    "airline": {"get_user_details"},
    "retail": {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"},
    "telecom": {"get_customer_by_phone", "get_customer_by_id", "get_customer_by_name"},
}
AFFIRMATIVE_RE = re.compile(
    r"(?i)(?:^|[.!?]\s*)(?:yes|yeah|yep|sure|okay|ok|please do|go ahead|proceed|confirm)"
)
ERROR_RE = re.compile(
    r"(?i)\b(error|invalid|not found|does not exist|missing|required|failed|cannot|unable)\b"
)
SUCCESS_CLAIM_RE = re.compile(
    r"(?i)\b(successfully|has been (?:updated|changed|cancelled|canceled|added|completed)|"
    r"all set|completed successfully|is now active|payment (?:was|has been) processed)\b"
)


def norm_args(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def canonical_args(value: Any) -> str:
    return json.dumps(norm_args(value), sort_keys=True, ensure_ascii=False, default=str)


def trajectory_id(model: str, domain: str, task_id: Any, trial: Any) -> str:
    return f"{model}|{domain}|{task_id}|{trial}"


def load_result_cells(
    sim_root: Path, model_specs: Iterable[tuple[str, str]], domains: Iterable[str]
) -> list[dict[str, Any]]:
    cells = []
    for model, prefix in model_specs:
        for domain in domains:
            path = sim_root / f"tau2_official_{prefix}_{domain}_test_4trials" / "results.json"
            if not path.exists():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            tasks = {str(task["id"]): task for task in data.get("tasks") or []}
            for sim in data.get("simulations") or []:
                task_id = str(sim.get("task_id"))
                cells.append(
                    {
                        "model": model,
                        "domain": domain,
                        "task": tasks[task_id],
                        "simulation": sim,
                        "source": str(path),
                    }
                )
    return cells


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def extract_tool_schemas(source: Path) -> list[dict[str, Any]]:
    """Extract decorated tool signatures without importing tau2 dependencies."""
    source_text = source.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source))
    tools = []
    for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        for fn in (node for node in cls.body if isinstance(node, ast.FunctionDef)):
            decorators = {_decorator_name(item) for item in fn.decorator_list}
            if not ({"is_tool", "is_discoverable_tool"} & decorators):
                continue
            args = []
            positional = list(fn.args.posonlyargs) + list(fn.args.args)
            defaults = [None] * (len(positional) - len(fn.args.defaults)) + list(fn.args.defaults)
            for arg, default in zip(positional, defaults):
                if arg.arg == "self":
                    continue
                args.append(
                    {
                        "name": arg.arg,
                        "type": ast.get_source_segment(source_text, arg.annotation)
                        if arg.annotation
                        else None,
                        "required": default is None,
                        "default": ast.get_source_segment(source_text, default)
                        if default is not None
                        else None,
                    }
                )
            tools.append(
                {
                    "name": fn.name,
                    "arguments": args,
                    "description": (ast.get_docstring(fn) or "").strip(),
                }
            )
    return tools


def load_tool_catalog(tau2_root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    src = tau2_root / "src/tau2/domains"
    catalog: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for domain in ("airline", "retail", "telecom"):
        agent = extract_tool_schemas(src / domain / "tools.py")
        user_path = src / domain / "user_tools.py"
        catalog[domain] = {
            "agent": agent,
            "user": extract_tool_schemas(user_path) if user_path.exists() else [],
        }
    return catalog


def _tool_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results_by_id = {
        message.get("id"): message
        for message in messages
        if message.get("role") == "tool" and message.get("id")
    }
    events = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call_index, call in enumerate(message.get("tool_calls") or []):
            result = results_by_id.get(call.get("id"), {})
            events.append(
                {
                    "message_index": message_index,
                    "turn_index": message.get("turn_idx", message_index),
                    "call_index": call_index,
                    "calls_in_turn": len(message.get("tool_calls") or []),
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "arguments": norm_args(call.get("arguments")),
                    "arguments_parse_error": isinstance(call.get("arguments"), str)
                    and not isinstance(
                        _try_parse_json(call.get("arguments")), dict
                    ),
                    "requestor": call.get("requestor", "assistant"),
                    "result": result.get("content"),
                    "error": bool(result.get("error")),
                    "negative_result": bool(
                        ERROR_RE.search(str(result.get("content") or ""))
                    ),
                }
            )
    return events


def _try_parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _reference_actions(task: dict[str, Any], sim: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = task.get("evaluation_criteria") or {}
    actions = criteria.get("actions")
    if actions is not None:
        return actions
    return [
        check.get("action") or {}
        for check in (sim.get("reward_info") or {}).get("action_checks") or []
    ]


def _preceding_affirmative(
    messages: list[dict[str, Any]], message_index: int
) -> dict[str, Any] | None:
    for index in range(message_index - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        return {
            "present": bool(AFFIRMATIVE_RE.search(content.strip())),
            "turn_index": message.get("turn_idx", index),
            "text": content[:240],
        }
    return None


def extract_features(
    sim: dict[str, Any],
    task: dict[str, Any],
    model: str,
    domain: str,
    tool_catalog: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    messages = sim.get("messages") or []
    reward_info = sim.get("reward_info") or {}
    criteria = task.get("evaluation_criteria") or {}
    events = _tool_events(messages)
    agent_tools = {tool["name"] for tool in tool_catalog[domain]["agent"]}
    user_tools = {tool["name"] for tool in tool_catalog[domain]["user"]}
    valid_tools = agent_tools | user_tools
    reference_actions = _reference_actions(task, sim)
    expected_agent_actions = [
        action
        for action in reference_actions
        if action.get("requestor", "assistant") == "assistant"
    ]
    expected_writes = [
        action for action in expected_agent_actions if action.get("name") in WRITE_TOOLS[domain]
    ]
    observed_agent = [event for event in events if event["requestor"] == "assistant"]
    observed_writes = [event for event in observed_agent if event["name"] in WRITE_TOOLS[domain]]

    signatures = [(event["name"], canonical_args(event["arguments"])) for event in events]
    adjacent_repeats = sum(a == b for a, b in zip(signatures, signatures[1:]))
    prior: dict[tuple[str, str], int] = {}
    nonadjacent_repeats = 0
    for index, signature in enumerate(signatures):
        if signature in prior and index - prior[signature] > 1:
            nonadjacent_repeats += 1
        prior[signature] = index
    name_args: dict[str, set[str]] = defaultdict(set)
    for name, args in signatures:
        name_args[str(name)].add(args)
    fanout_names = sorted(name for name, args in name_args.items() if len(args) >= 3)
    fanout_calls = sum(max(0, len(args) - 1) for args in name_args.values())

    multitool_messages = [
        message
        for message in messages
        if message.get("role") == "assistant" and len(message.get("tool_calls") or []) > 1
    ]
    multiwrite_turns = 0
    for message in multitool_messages:
        calls = message.get("tool_calls") or []
        if any(
            call.get("requestor", "assistant") == "assistant"
            and call.get("name") in WRITE_TOOLS[domain]
            for call in calls
        ):
            multiwrite_turns += 1

    invalid = sorted({str(event["name"]) for event in events if event["name"] not in valid_tools})
    namespace_confusions = []
    schema_errors = []
    schema_by_namespace = {
        "assistant": {
            tool["name"]: tool for tool in tool_catalog[domain]["agent"]
        },
        "user": {tool["name"]: tool for tool in tool_catalog[domain]["user"]},
    }
    for event in events:
        if event["requestor"] == "assistant" and event["name"] in user_tools:
            namespace_confusions.append(event["turn_index"])
        if event["requestor"] == "user" and event["name"] in agent_tools:
            namespace_confusions.append(event["turn_index"])
        schema = schema_by_namespace.get(event["requestor"], {}).get(event["name"])
        if schema:
            required = {
                argument["name"]
                for argument in schema.get("arguments", [])
                if argument.get("required")
            }
            allowed = {
                argument["name"] for argument in schema.get("arguments", [])
            }
            supplied = set(event["arguments"])
            missing = sorted(required - supplied)
            extra = sorted(supplied - allowed)
            if event["arguments_parse_error"] or missing or extra:
                schema_errors.append(
                    {
                        "turn_index": event["turn_index"],
                        "tool": event["name"],
                        "parse_error": event["arguments_parse_error"],
                        "missing_required": missing,
                        "unexpected": extra,
                    }
                )

    write_confirmations = []
    for event in observed_writes:
        candidate = _preceding_affirmative(messages, event["message_index"])
        write_confirmations.append(
            {
                "write_turn": event["turn_index"],
                "tool": event["name"],
                "candidate_affirmative": candidate,
            }
        )

    false_success_candidate_turns = []
    for event in events:
        if not event["error"]:
            continue
        for message in messages[event["message_index"] + 1 :]:
            if message.get("role") == "assistant" and message.get("content"):
                if SUCCESS_CLAIM_RE.search(str(message["content"])):
                    false_success_candidate_turns.append(
                        message.get("turn_idx", messages.index(message))
                    )
                break

    progressive_candidate = None
    if domain == "telecom":
        stage_tools = (
            {
                "check_status_bar",
                "check_network_status",
                "check_sim_status",
                "toggle_airplane_mode",
                "reseat_sim_card",
            },
            {
                "check_network_mode_preference",
                "check_data_restriction_status",
                "check_apn_settings",
                "check_vpn_status",
                "get_data_usage",
                "run_speed_test",
            },
            {"can_send_mms", "check_app_status", "check_app_permissions"},
        )
        first_indices = [
            next(
                (
                    index
                    for index, event in enumerate(events)
                    if event["name"] in names
                ),
                None,
            )
            for names in stage_tools
        ]
        applicable = [index for index in first_indices if index is not None]
        progressive_candidate = len(applicable) >= 2 and applicable == sorted(applicable)

    action_checks = reward_info.get("action_checks") or []
    expected_names = {action.get("name") for action in expected_writes}
    observed_names = {event["name"] for event in observed_agent}
    reward = reward_info.get("reward")
    success = reward is not None and float(reward) >= 1.0 - 1e-6
    db_check = reward_info.get("db_check") or {}
    reward_basis = reward_info.get("reward_basis") or criteria.get("reward_basis") or []
    if not reward_basis:
        reward_basis = list((reward_info.get("reward_breakdown") or {}).keys())

    return {
        "trajectory_id": trajectory_id(model, domain, sim.get("task_id"), sim.get("trial")),
        "model": model,
        "domain": domain,
        "task_id": str(sim.get("task_id")),
        "trial": sim.get("trial"),
        "reward": reward,
        "success": success,
        "reward_basis": reward_basis,
        "termination_reason": sim.get("termination_reason"),
        "num_messages": len(messages),
        "valid_turn_indices": [
            message.get("turn_idx", index)
            for index, message in enumerate(messages)
            if isinstance(message.get("turn_idx", index), int)
        ],
        "num_assistant_turns": sum(m.get("role") == "assistant" for m in messages),
        "num_tool_calls": len(events),
        "tool_names": [event["name"] for event in events],
        "tool_errors": sum(event["error"] for event in events),
        "false_success_candidate_turns": sorted(set(false_success_candidate_turns)),
        "telecom_progressive_order_candidate": progressive_candidate,
        "invalid_tool_names": invalid,
        "invalid_tool_calls": sum(event["name"] not in valid_tools for event in events),
        "tool_argument_schema_errors": schema_errors,
        "namespace_confusion_turns": sorted(set(namespace_confusions)),
        "adjacent_repeated_calls": adjacent_repeats,
        "nonadjacent_repeated_calls": nonadjacent_repeats,
        "fanout_names": fanout_names,
        "fanout_calls": fanout_calls,
        "multitool_turns": len(multitool_messages),
        "multitool_turn_indices": [
            message.get("turn_idx", messages.index(message))
            for message in multitool_messages
        ],
        "multitool_policy_violation": bool(multitool_messages),
        "multiwrite_turns": multiwrite_turns,
        "gold_requires_agent_write": bool(expected_writes),
        "workflow_type": "state_changing" if expected_writes else "read_only",
        "expected_agent_actions": expected_agent_actions,
        "expected_write_actions": expected_writes,
        "expected_write_names": sorted(expected_names),
        "observed_agent_writes": [
            {
                "turn_index": event["turn_index"],
                "name": event["name"],
                "arguments": event["arguments"],
                "error": event["error"],
            }
            for event in observed_writes
        ],
        "num_observed_writes": len(observed_writes),
        "expected_write_name_called": bool(expected_names & observed_names),
        "expected_write_name_called_failure": bool(expected_names & observed_names) and not success,
        "golden_total": len(action_checks),
        "golden_matched": sum(bool(check.get("action_match")) for check in action_checks),
        "exact_action_match": bool(action_checks)
        and all(bool(check.get("action_match")) for check in action_checks),
        "db_match": db_check.get("db_match") if isinstance(db_check, dict) else None,
        "env_assertions": criteria.get("env_assertions")
        or reward_info.get("env_assertions")
        or [],
        "nl_assertions": criteria.get("nl_assertions")
        or reward_info.get("nl_assertions")
        or [],
        "write_confirmation_candidates": write_confirmations,
    }


def build_judge_payload(
    cell: dict[str, Any],
    tools: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Build blinded judge evidence. Model stage and reward are intentionally absent."""
    task = cell["task"]
    sim = cell["simulation"]
    criteria = task.get("evaluation_criteria") or {}
    messages = []
    for index, message in enumerate(sim.get("messages") or []):
        item = {
            "index": index,
            "turn_index": message.get("turn_idx", index),
            "role": message.get("role"),
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            item["tool_calls"] = message["tool_calls"]
        if message.get("role") == "tool":
            item["requestor"] = message.get("requestor")
            item["error"] = message.get("error")
        messages.append(item)
    real_id = trajectory_id(
        cell["model"], cell["domain"], sim.get("task_id"), sim.get("trial")
    )
    features = extract_features(
        sim, task, "blind", cell["domain"], tools
    )
    tool_error_turns = [
        event["turn_index"]
        for event in _tool_events(sim.get("messages") or [])
        if event["error"]
    ]
    multi_tool_turns = [
        {
            "turn_index": message.get("turn_idx", index),
            "tool_names": [
                call.get("name") for call in message.get("tool_calls") or []
            ],
        }
        for index, message in enumerate(sim.get("messages") or [])
        if message.get("role") == "assistant"
        and len(message.get("tool_calls") or []) > 1
    ]
    blind_id = hashlib.sha256(
        f"{cell['domain']}|{sim.get('task_id')}|{sim.get('trial')}".encode("utf-8")
    ).hexdigest()[:20]
    return {
        "_trajectory_id": real_id,
        "trajectory_id": f"blind-{blind_id}",
        "prompt_version": PROMPT_VERSION,
        "domain": cell["domain"],
        "policy": sim.get("policy"),
        "agent_tools": tools[cell["domain"]]["agent"],
        "user_tools": tools[cell["domain"]]["user"],
        "task": {
            "description": task.get("description"),
            "user_scenario": task.get("user_scenario"),
            "reference_actions": criteria.get("actions"),
            "environment_assertions": criteria.get("env_assertions"),
            "natural_language_assertions": criteria.get("nl_assertions"),
        },
        "mechanical_audit_facts": {
            "multi_tool_turns": multi_tool_turns,
            "invalid_tool_names": features["invalid_tool_names"],
            "tool_argument_schema_errors": features[
                "tool_argument_schema_errors"
            ],
            "namespace_confusion_turns": features["namespace_confusion_turns"],
            "adjacent_repeated_calls": features["adjacent_repeated_calls"],
            "nonadjacent_repeated_calls": features[
                "nonadjacent_repeated_calls"
            ],
            "fanout_names": features["fanout_names"],
            "tool_error_turns": tool_error_turns,
            "false_success_candidate_turns": features[
                "false_success_candidate_turns"
            ],
            "write_confirmation_candidates": features[
                "write_confirmation_candidates"
            ],
            "telecom_progressive_order_candidate": features[
                "telecom_progressive_order_candidate"
            ],
        },
        "conversation": messages,
    }


def validate_review(review: dict[str, Any], valid_turns: set[int]) -> list[str]:
    errors = []
    if review.get("prompt_version") != PROMPT_VERSION:
        errors.append("wrong prompt_version")
    patterns = review.get("patterns")
    if not isinstance(patterns, dict):
        return errors + ["patterns must be an object"]
    missing = set(PATTERNS) - set(patterns)
    if missing:
        errors.append(f"missing patterns: {sorted(missing)}")
    for name, finding in patterns.items():
        if not isinstance(finding, dict):
            errors.append(f"{name}: finding must be an object")
            continue
        if finding.get("status") not in {"pass", "fail", "not_applicable", "uncertain"}:
            errors.append(f"{name}: invalid status")
        if finding.get("severity") not in {"critical", "minor", "none"}:
            errors.append(f"{name}: invalid severity")
        turns = finding.get("turn_indices")
        if not isinstance(turns, list) or any(
            not isinstance(turn, int) or turn not in valid_turns for turn in turns
        ):
            errors.append(f"{name}: invalid turn_indices")
    if review.get("primary_failure_mode") not in PRIMARY_FAILURE_MODES:
        errors.append("invalid primary_failure_mode")
    user_error = review.get("user_simulator_error")
    if not isinstance(user_error, dict) or user_error.get("status") not in {
        "none",
        "helped_agent",
        "hindered_agent",
        "minor",
    }:
        errors.append("invalid user_simulator_error")
    elif any(
        not isinstance(turn, int) or turn not in valid_turns
        for turn in user_error.get("turn_indices", [])
    ):
        errors.append("user_simulator_error: invalid turn_indices")
    return errors


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def duplicate_ids(records: Iterable[dict[str, Any]]) -> list[str]:
    counts = Counter(record.get("trajectory_id") for record in records)
    return sorted(str(key) for key, count in counts.items() if count > 1)
