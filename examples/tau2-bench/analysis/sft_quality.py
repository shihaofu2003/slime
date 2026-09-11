#!/usr/bin/env python3
"""Core utilities for auditing and filtering tau2 SFT trajectories.

The AReaL SFT file stores one progressively longer prefix per assistant turn.
This module reconstructs one canonical training view per source dialog while
also counting the repeated assistant-turn exposures seen by the SFT loss.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from protocol_profiles import (  # noqa: E402
    PROTOCOL_CURRENT_SINGLE,
    PROTOCOL_DEPENDENCY_SAFE_MULTI,
    PROTOCOL_DESCRIPTIONS,
    domain_policy_for_profile,
    validate_protocol_profile as _validate_protocol_profile,
)

PROMPT_VERSION = "tau2-sft-quality-v2"
LABEL_GROUPS = (
    "consensus_success",
    "consensus_failure",
    "reward_only",
    "correct_only",
)
DOMAINS = ("airline", "retail", "telecom")
SOURCE_METADATA_FIELDS = (
    "seed_pattern_task_id",
    "task_id",
    "difficulty",
    "num_subtasks",
    "scenario_id",
)
DIMENSIONS = (
    "task_completion",
    "policy_compliance",
    "tool_choice",
    "argument_grounding",
    "workflow_and_dependencies",
    "authorization_and_confirmation",
    "recovery",
    "communication_and_closeout",
)
CRITICAL_ISSUES = {
    "wrong_tool",
    "wrong_arguments",
    "missing_precondition",
    "confirmation_violation",
    "unauthorized_or_extra_write",
    "multi_tool_policy_violation",
    "tool_namespace_confusion",
    "repeated_or_fanout_calls",
    "failed_recovery",
    "false_success_claim",
    "incomplete_task",
    "policy_violation",
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
ERROR_RE = re.compile(
    r"(?i)\b(error|invalid|not found|does not exist|missing|required|failed|cannot|unable)\b"
)


def stable_json_dumps(value: Any) -> str:
    """Serialize JSON-compatible audit data canonically for cache keys."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def payload_hash(value: Any) -> str:
    """Hash audit content, excluding a payload object's self-declared hashes."""

    if not isinstance(value, dict):
        return stable_json_hash(value)
    unhashed = {
        key: item
        for key, item in value.items()
        if key not in {"payload_hash", "payload_sha256"}
    }
    return stable_json_hash(unhashed)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def parse_arguments(value: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(value, dict):
        return value, False
    if isinstance(value, str):
        try:
            parsed = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError:
            return {}, True
        return (parsed, False) if isinstance(parsed, dict) else ({}, True)
    return ({}, False) if value is None else ({}, True)


def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments_value = function.get("arguments")
    else:
        name = call.get("name")
        arguments_value = call.get("arguments")
    arguments, parse_error = parse_arguments(arguments_value)
    return {
        "id": call.get("id"),
        "name": name,
        "arguments": arguments,
        "arguments_parse_error": parse_error,
        "requestor": call.get("requestor", "assistant"),
    }


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "role": message.get("role"),
        "content": message.get("content") or "",
    }
    calls = message.get("tool_calls") or []
    if calls:
        normalized["tool_calls"] = [normalize_tool_call(call) for call in calls]
    for key in ("id", "tool_call_id", "name", "requestor", "error"):
        if key in message:
            normalized[key] = message[key]
    return normalized


def row_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [normalize_message(message) for message in row.get("messages") or []]
    answer = row.get("answer")
    if isinstance(answer, dict):
        messages.append(normalize_message(answer))
    return messages


def message_fingerprint(message: dict[str, Any]) -> str:
    visible = {
        "role": message.get("role"),
        "content": message.get("content") or "",
        "tool_calls": message.get("tool_calls") or [],
    }
    return hashlib.sha256(
        json.dumps(visible, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def infer_domain(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    domain = str(metadata.get("domain") or "").lower()
    if domain in DOMAINS:
        return domain
    dialog_id = str(metadata.get("source_dialog_id") or "").lower()
    system = " ".join(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if message.get("role") == "system"
    ).lower()
    for candidate in DOMAINS:
        if candidate in dialog_id or candidate in system:
            return candidate
    return "unknown"


def extract_policy(messages: list[dict[str, Any]]) -> str:
    system = next(
        (str(message.get("content") or "") for message in messages if message.get("role") == "system"),
        "",
    )
    lowered = system.lower()
    policy_start = lowered.rfind("<policy>")
    policy_end = lowered.rfind("</policy>")
    if 0 <= policy_start < policy_end:
        return system[policy_start + len("<policy>") : policy_end].strip()
    for marker in ("<tools>", "## Available Tools"):
        if marker in system:
            system = system.split(marker, 1)[0]
    return system.strip()


def policy_for_profile(policy: str, protocol_profile: str) -> tuple[str, bool]:
    """Return the policy text shown to a judge under the selected counterfactual."""

    return domain_policy_for_profile(policy, protocol_profile)


def assistant_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    if message.get("role") != "assistant":
        return []
    return list(message.get("tool_calls") or [])


def exposure_counts(messages: list[dict[str, Any]], domain: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    writes = WRITE_TOOLS.get(domain, set())
    for message in messages:
        if message.get("role") != "assistant":
            continue
        counts["assistant_turn_exposures"] += 1
        calls = assistant_calls(message)
        counts["tool_call_exposures"] += len(calls)
        counts["write_call_exposures"] += sum(call.get("name") in writes for call in calls)
        counts["multitool_turn_exposures"] += int(len(calls) > 1)
        counts["mixed_content_tool_exposures"] += int(
            bool(calls) and bool(str(message.get("content") or "").strip())
        )
    return counts


def _canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)


def _tool_result_error(message: dict[str, Any]) -> bool:
    return bool(message.get("error")) or bool(
        ERROR_RE.search(str(message.get("content") or ""))
    )


def _multitool_turn_details(
    messages: list[dict[str, Any]],
    turn_index: int,
    calls: list[dict[str, Any]],
    writes: set[str],
) -> dict[str, Any]:
    result_messages = []
    for result_index in range(turn_index + 1, len(messages)):
        result = messages[result_index]
        if result.get("role") != "tool":
            break
        result_messages.append(
            {
                "turn_index": result_index,
                "id": result.get("id"),
                "tool_call_id": result.get("tool_call_id"),
                "name": result.get("name"),
                "error": _tool_result_error(result),
                "content_hash": stable_json_hash(str(result.get("content") or "")),
            }
        )

    available_results = set(range(len(result_messages)))
    associations = []
    normalized_calls = []
    for call_index, call in enumerate(calls):
        name = call.get("name")
        signature = f"{name}|{_canonical_arguments(call.get('arguments') or {})}"
        normalized_calls.append(
            {
                "call_index": call_index,
                "id": call.get("id"),
                "name": name,
                "arguments": call.get("arguments") or {},
                "arguments_parse_error": bool(call.get("arguments_parse_error")),
                "requestor": call.get("requestor"),
                "is_write": name in writes,
                "signature": signature,
            }
        )

        matching_index = None
        association = "missing"
        call_id = call.get("id")
        if call_id is not None:
            matching_index = next(
                (
                    index
                    for index in sorted(available_results)
                    if call_id
                    in {
                        result_messages[index].get("id"),
                        result_messages[index].get("tool_call_id"),
                    }
                ),
                None,
            )
            if matching_index is not None:
                association = "id"
        if matching_index is None and name is not None:
            matching_index = next(
                (
                    index
                    for index in sorted(available_results)
                    if result_messages[index].get("name") == name
                ),
                None,
            )
            if matching_index is not None:
                association = "name"
        if matching_index is None and available_results:
            matching_index = min(available_results)
            association = "position"
        result_turn_indices = []
        result_errors = []
        if matching_index is not None:
            available_results.remove(matching_index)
            matched_result = result_messages[matching_index]
            result_turn_indices.append(matched_result["turn_index"])
            result_errors.append(matched_result["error"])
        associations.append(
            {
                "call_index": call_index,
                "call_id": call_id,
                "tool": name,
                "association": association,
                "result_turn_indices": result_turn_indices,
                "result_errors": result_errors,
            }
        )

    signatures: dict[str, list[int]] = defaultdict(list)
    for call in normalized_calls:
        signatures[call["signature"]].append(call["call_index"])
    duplicate_groups = [
        {"signature": signature, "call_indices": call_indices}
        for signature, call_indices in signatures.items()
        if len(call_indices) > 1
    ]
    return {
        "turn_index": turn_index,
        "batch_size": len(calls),
        "read_call_count": sum(call.get("name") not in writes for call in calls),
        "write_call_count": sum(call.get("name") in writes for call in calls),
        "calls": normalized_calls,
        "duplicate_groups": duplicate_groups,
        "duplicate_call_indices": sorted(
            index
            for group in duplicate_groups
            for index in group["call_indices"][1:]
        ),
        "result_messages": result_messages,
        "result_associations": associations,
        "unassociated_result_turns": [
            result_messages[index]["turn_index"] for index in sorted(available_results)
        ],
    }


def deterministic_features(
    messages: list[dict[str, Any]],
    domain: str,
    tool_catalog: dict[str, dict[str, list[dict[str, Any]]]],
    protocol_profile: str = PROTOCOL_CURRENT_SINGLE,
) -> dict[str, Any]:
    _validate_protocol_profile(protocol_profile)
    assistant_tools = {
        tool["name"]: tool for tool in tool_catalog.get(domain, {}).get("agent", [])
    }
    user_tools = {
        tool["name"]: tool for tool in tool_catalog.get(domain, {}).get("user", [])
    }
    valid_names = set(assistant_tools) | {"done"}
    writes = WRITE_TOOLS.get(domain, set())
    events = []
    mixed_turns = []
    multitool_turns = []
    multitool_batch_sizes = []
    multitool_details = []
    multiwrite_turns = []
    for index, message in enumerate(messages):
        calls = assistant_calls(message)
        if calls and str(message.get("content") or "").strip():
            mixed_turns.append(index)
        if len(calls) > 1:
            multitool_turns.append(index)
            multitool_batch_sizes.append(len(calls))
            multitool_details.append(
                _multitool_turn_details(messages, index, calls, writes)
            )
            if any(call.get("name") in writes for call in calls):
                multiwrite_turns.append(index)
        for call in calls:
            events.append({"turn_index": index, **call})

    invalid_names = sorted(
        {str(event.get("name")) for event in events if event.get("name") not in valid_names}
    )
    namespace_turns = sorted(
        {
            event["turn_index"]
            for event in events
            if event.get("name") in user_tools and event.get("name") not in assistant_tools
        }
    )
    schema_errors = []
    for event in events:
        schema = assistant_tools.get(event.get("name"))
        if not schema:
            continue
        required = {
            argument["name"]
            for argument in schema.get("arguments", [])
            if argument.get("required")
        }
        allowed = {argument["name"] for argument in schema.get("arguments", [])}
        supplied = set(event.get("arguments") or {})
        missing = sorted(required - supplied)
        unexpected = sorted(supplied - allowed)
        if event.get("arguments_parse_error") or missing or unexpected:
            schema_errors.append(
                {
                    "turn_index": event["turn_index"],
                    "tool": event.get("name"),
                    "parse_error": bool(event.get("arguments_parse_error")),
                    "missing_required": missing,
                    "unexpected": unexpected,
                }
            )

    signatures = [
        (str(event.get("name")), _canonical_arguments(event.get("arguments") or {}))
        for event in events
    ]
    adjacent_repeats = sum(first == second for first, second in zip(signatures, signatures[1:]))
    seen: dict[tuple[str, str], int] = {}
    nonadjacent_repeats = 0
    for index, signature in enumerate(signatures):
        if signature in seen and index - seen[signature] > 1:
            nonadjacent_repeats += 1
        seen[signature] = index
    arguments_by_name: dict[str, set[str]] = defaultdict(set)
    for name, arguments in signatures:
        arguments_by_name[name].add(arguments)
    fanout_names = sorted(name for name, arguments in arguments_by_name.items() if len(arguments) >= 3)

    tool_error_turns = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "tool" and _tool_result_error(message)
    ]
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    final_assistant = assistant_messages[-1] if assistant_messages else {}
    final_has_text = bool(str(final_assistant.get("content") or "").strip())
    final_has_tool = bool(final_assistant.get("tool_calls"))
    mechanical_hard_issues = []
    if mixed_turns:
        mechanical_hard_issues.append("content_and_tool_same_turn")
    if invalid_names:
        mechanical_hard_issues.append("invalid_tool_name")
    if namespace_turns:
        mechanical_hard_issues.append("tool_namespace_confusion")
    if schema_errors:
        mechanical_hard_issues.append("tool_argument_schema_error")
    policy_hard_issues = []
    if multitool_turns and protocol_profile == PROTOCOL_CURRENT_SINGLE:
        policy_hard_issues.append("multi_tool_policy_violation")
    hard_issues = mechanical_hard_issues + policy_hard_issues

    return {
        "protocol_profile": protocol_profile,
        "num_messages": len(messages),
        "num_assistant_turns": len(assistant_messages),
        "num_tool_calls": len(events),
        "num_write_calls": sum(event.get("name") in writes for event in events),
        "num_read_calls": sum(event.get("name") not in writes for event in events),
        "tool_names": [event.get("name") for event in events],
        "mixed_content_tool_turns": mixed_turns,
        "has_multitool_turn": bool(multitool_turns),
        "multitool_turns": multitool_turns,
        "multitool_batch_sizes": multitool_batch_sizes,
        "multitool_details": multitool_details,
        "max_calls_in_turn": max(multitool_batch_sizes, default=1 if events else 0),
        "multiwrite_turns": multiwrite_turns,
        "invalid_tool_names": invalid_names,
        "namespace_confusion_turns": namespace_turns,
        "tool_argument_schema_errors": schema_errors,
        "adjacent_repeated_calls": adjacent_repeats,
        "nonadjacent_repeated_calls": nonadjacent_repeats,
        "fanout_names": fanout_names,
        "tool_error_turns": tool_error_turns,
        "final_has_text": final_has_text,
        "final_has_tool": final_has_tool,
        "mechanical_hard_issues": mechanical_hard_issues,
        "policy_hard_issues": policy_hard_issues,
        "hard_issues": hard_issues,
    }


def _view_rank(row: dict[str, Any]) -> tuple[int, int]:
    metadata = row.get("metadata") or {}
    turn_index = metadata.get("turn_index")
    return (turn_index if isinstance(turn_index, int) else -1, len(row.get("messages") or []))


def _is_success_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return math.isclose(float(value), 1.0)
    except (TypeError, ValueError):
        return False


def label_group(correct: Any, reward: Any, labels_consistent: bool = True) -> str:
    if not labels_consistent:
        return "inconsistent"
    correct_success = _is_success_label(correct)
    reward_success = _is_success_label(reward)
    if correct_success and reward_success:
        return "consensus_success"
    if not correct_success and not reward_success:
        return "consensus_failure"
    return "reward_only" if reward_success else "correct_only"


def load_training_index(path: Path) -> dict[str, Any]:
    included_source_rows: set[int] = set()
    dialogs: Counter[str] = Counter()
    protocol_allows_multi: dict[str, bool] = defaultdict(bool)
    rows_by_source: dict[int, dict[str, Any]] = {}
    duplicate_source_rows = []
    for row in read_jsonl(path):
        metadata = row.get("metadata") or {}
        source_row = metadata.get("source_row")
        dialog_id = str(metadata.get("source_dialog_id") or "")
        if not isinstance(source_row, int) or not dialog_id:
            raise ValueError(f"{path}: every row must preserve source_row and source_dialog_id")
        if source_row in included_source_rows:
            duplicate_source_rows.append(source_row)
        included_source_rows.add(source_row)
        rows_by_source[source_row] = row
        dialogs[dialog_id] += 1
        system = next(
            (
                str(message.get("content") or "")
                for message in row.get("messages") or []
                if message.get("role") == "system"
            ),
            "",
        )
        protocol_allows_multi[dialog_id] |= "one or more tool calls" in system.lower()
    if duplicate_source_rows:
        raise ValueError(f"{path}: duplicate source_row values: {duplicate_source_rows[:5]}")
    return {
        "source_rows": included_source_rows,
        "dialog_row_counts": dialogs,
        "protocol_allows_multi": dict(protocol_allows_multi),
        "rows_by_source": rows_by_source,
    }


def _judge_conversation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conversation = []
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            continue
        item = {
            "turn_index": index,
            "role": message.get("role"),
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            item["tool_calls"] = message["tool_calls"]
        for key in ("id", "tool_call_id", "name", "requestor"):
            if key in message:
                item[key] = message[key]
        if message.get("role") == "tool":
            item["error"] = _tool_result_error(message)
        conversation.append(item)
    return conversation


def _mechanical_audit_facts(static: dict[str, Any]) -> dict[str, Any]:
    return {
        key: static[key]
        for key in (
            "protocol_profile",
            "has_multitool_turn",
            "multitool_turns",
            "multitool_batch_sizes",
            "multitool_details",
            "max_calls_in_turn",
            "multiwrite_turns",
            "invalid_tool_names",
            "namespace_confusion_turns",
            "tool_argument_schema_errors",
            "adjacent_repeated_calls",
            "nonadjacent_repeated_calls",
            "fanout_names",
            "tool_error_turns",
            "final_has_text",
            "final_has_tool",
            "mechanical_hard_issues",
            "policy_hard_issues",
            "hard_issues",
        )
    }


def _expected_converted_non_system(
    source_messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Reproduce the strict no-thinking AReaL message conversion semantically."""

    converted = []
    for message in source_messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            converted.append({"role": "user", "content": str(message.get("content") or "")})
            continue
        if role == "tool":
            content = str(message.get("content") or "").strip() or "[no_observation]"
            converted.append({"role": "user", "content": f"Tool result:\n{content}"})
            continue
        if role != "assistant":
            continue
        pieces = []
        content = str(message.get("content") or "").strip()
        if content:
            pieces.append(content)
        for call in assistant_calls(message):
            pieces.append(
                "<tool_call>"
                + json.dumps(
                    {"name": call.get("name"), "arguments": call.get("arguments") or {}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "</tool_call>"
            )
        rendered = "\n\n".join(pieces).strip()
        if rendered:
            converted.append({"role": "assistant", "content": rendered})
    return converted


def audit_source_dialogs(
    source_path: Path,
    training_path: Path,
    tool_catalog: dict[str, dict[str, list[dict[str, Any]]]],
    protocol_profile: str = PROTOCOL_CURRENT_SINGLE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _validate_protocol_profile(protocol_profile)
    training = load_training_index(training_path)
    included = training["source_rows"]
    accumulators: dict[str, dict[str, Any]] = {}
    seen_included: set[int] = set()

    for source_row, row in enumerate(read_jsonl(source_path)):
        metadata = row.get("metadata") or {}
        dialog_id = str(metadata.get("source_dialog_id") or "")
        if not dialog_id:
            raise ValueError(f"{source_path}:{source_row + 1}: missing source_dialog_id")
        domain = infer_domain(row)
        accumulator = accumulators.setdefault(
            dialog_id,
            {
                "dialog_id": dialog_id,
                "domain": domain,
                "source_row_count": 0,
                "included_source_rows": [],
                "correct_values": set(),
                "reward_values": set(),
                "best_source_row": None,
                "best_training_row": None,
                "best_converted_row": None,
                "included_targets": [],
                "conversion_checks": [],
                "exposures": Counter(),
            },
        )
        if accumulator["domain"] != domain:
            raise ValueError(f"{dialog_id}: inconsistent domains")
        accumulator["source_row_count"] += 1
        accumulator["correct_values"].add(metadata.get("correct"))
        accumulator["reward_values"].add(metadata.get("reward"))
        if accumulator["best_source_row"] is None or _view_rank(row) > _view_rank(accumulator["best_source_row"]):
            accumulator["best_source_row"] = row
        if source_row in included:
            seen_included.add(source_row)
            accumulator["included_source_rows"].append(source_row)
            converted_row = training["rows_by_source"][source_row]
            converted_metadata = converted_row.get("metadata") or {}
            if (
                str(converted_metadata.get("source_dialog_id") or "") != dialog_id
                or converted_metadata.get("source_row") != source_row
            ):
                raise ValueError(
                    f"{training_path}: source row {source_row} metadata does not "
                    f"match {dialog_id}"
                )
            source_messages_for_row = row_messages(row)
            converted_messages = row_messages(converted_row)
            expected_converted_non_system = _expected_converted_non_system(
                source_messages_for_row
            )
            converted_non_system = [
                {
                    "role": message.get("role"),
                    "content": str(message.get("content") or ""),
                }
                for message in converted_messages
                if message.get("role") != "system"
            ]
            source_system = next(
                (
                    str(message.get("content") or "")
                    for message in source_messages_for_row
                    if message.get("role") == "system"
                ),
                "",
            )
            converted_system = next(
                (
                    str(message.get("content") or "")
                    for message in converted_messages
                    if message.get("role") == "system"
                ),
                "",
            )
            answer = row.get("answer")
            source_target_fingerprint = None
            if isinstance(answer, dict):
                source_target_fingerprint = message_fingerprint(
                    normalize_message(answer)
                )
            converted_target = next(
                (
                    message
                    for message in reversed(converted_messages)
                    if message.get("role") == "assistant"
                ),
                None,
            )
            expected_target_content = (
                _expected_converted_non_system([normalize_message(answer)])[0]["content"]
                if isinstance(answer, dict)
                else None
            )
            converted_target_content = (
                str(converted_target.get("content") or "")
                if converted_target is not None
                else None
            )
            accumulator["conversion_checks"].append(
                {
                    "source_row": source_row,
                    "turn_index": metadata.get("turn_index"),
                    "non_system_messages_match": (
                        stable_json_hash(expected_converted_non_system)
                        == stable_json_hash(converted_non_system)
                    ),
                    "source_system_preserved": (
                        bool(source_system) and source_system in converted_system
                    ),
                    "target_fingerprint_match": (
                        expected_target_content == converted_target_content
                    ),
                    "source_non_system_hash": stable_json_hash(
                        expected_converted_non_system
                    ),
                    "converted_non_system_hash": stable_json_hash(converted_non_system),
                    "source_target_fingerprint": source_target_fingerprint,
                    "expected_converted_target_hash": stable_json_hash(
                        expected_target_content
                    ),
                    "converted_target_hash": stable_json_hash(
                        converted_target_content
                    ),
                    "converted_messages_hash": stable_json_hash(converted_messages),
                    "converted_system_hash": stable_json_hash(converted_system),
                }
            )
            row_exposures = exposure_counts(source_messages_for_row, domain)
            accumulator["exposures"].update(row_exposures)
            accumulator["exposures"]["training_rows_with_multitool_prefix"] += int(
                row_exposures["multitool_turn_exposures"] > 0
            )
            if isinstance(answer, dict):
                normalized_answer = normalize_message(answer)
                target_calls = assistant_calls(normalized_answer)
                accumulator["exposures"]["target_assistant_rows"] += 1
                accumulator["exposures"]["target_tool_calls"] += len(target_calls)
                accumulator["exposures"]["target_multitool_rows"] += int(len(target_calls) > 1)
                accumulator["exposures"]["target_write_calls"] += sum(
                    call.get("name") in WRITE_TOOLS.get(domain, set()) for call in target_calls
                )
                accumulator["included_targets"].append(
                    {
                        "source_row": source_row,
                        "turn_index": metadata.get("turn_index"),
                        "conversation_turn_index": len(source_messages_for_row) - 1,
                        "fingerprint": message_fingerprint(normalized_answer),
                        "num_tool_calls": len(target_calls),
                        "is_multitool": len(target_calls) > 1,
                        "tool_names": [call.get("name") for call in target_calls],
                        "num_write_calls": sum(
                            call.get("name") in WRITE_TOOLS.get(domain, set())
                            for call in target_calls
                        ),
                    }
                )
            if accumulator["best_training_row"] is None or _view_rank(row) > _view_rank(
                accumulator["best_training_row"]
            ):
                accumulator["best_training_row"] = row
                accumulator["best_converted_row"] = converted_row

    missing = sorted(included - seen_included)
    if missing:
        raise ValueError(f"{training_path}: source rows absent from {source_path}: {missing[:5]}")

    features = []
    payloads = []
    for dialog_id, accumulator in sorted(accumulators.items()):
        training_row = accumulator["best_training_row"]
        if training_row is None:
            continue
        converted_training_row = accumulator["best_converted_row"]
        if converted_training_row is None:
            raise ValueError(f"{dialog_id}: missing converted training row")
        correct_values = accumulator["correct_values"]
        reward_values = accumulator["reward_values"]
        labels_consistent = len(correct_values) == 1 and len(reward_values) == 1
        correct = next(iter(correct_values)) if len(correct_values) == 1 else None
        reward = next(iter(reward_values)) if len(reward_values) == 1 else None
        source_row = accumulator["best_source_row"]
        source_messages = row_messages(source_row)
        training_messages = row_messages(training_row)
        actual_training_messages = row_messages(converted_training_row)
        full_source_static = deterministic_features(
            source_messages,
            accumulator["domain"],
            tool_catalog,
            protocol_profile,
        )
        training_visible_static = deterministic_features(
            training_messages,
            accumulator["domain"],
            tool_catalog,
            protocol_profile,
        )
        visible_assistant_fingerprints = {
            message_fingerprint(message)
            for message in training_messages
            if message.get("role") == "assistant"
        }
        missing_prefix_targets = [
            {
                "source_row": target["source_row"],
                "turn_index": target["turn_index"],
            }
            for target in accumulator["included_targets"]
            if target["fingerprint"] not in visible_assistant_fingerprints
        ]
        training_visible_static["missing_prefix_targets"] = missing_prefix_targets
        full_source_static["missing_prefix_targets"] = []
        if missing_prefix_targets:
            training_visible_static["mechanical_hard_issues"].append(
                "inconsistent_dialog_prefixes"
            )
            training_visible_static["hard_issues"].append("inconsistent_dialog_prefixes")
        label_issues = []
        if not labels_consistent:
            label_issues.append("inconsistent_source_labels")
        if not _is_success_label(correct):
            label_issues.append("source_correct_not_one")
        if not _is_success_label(reward):
            label_issues.append("source_reward_not_one")
        hard_issues = label_issues + training_visible_static["hard_issues"]
        full_source_hard_issues = label_issues + full_source_static["hard_issues"]
        warnings = []
        if (
            protocol_profile == PROTOCOL_CURRENT_SINGLE
            and training["protocol_allows_multi"].get(dialog_id)
        ):
            warnings.append("training_prompt_allows_multi_but_tau2_policy_forbids_it")
        if (
            full_source_static["adjacent_repeated_calls"]
            or full_source_static["nonadjacent_repeated_calls"]
        ):
            warnings.append("repeated_tool_calls")
        if full_source_static["fanout_names"]:
            warnings.append("tool_fanout")
        if (
            training_visible_static["final_has_tool"]
            and not training_visible_static["final_has_text"]
        ):
            warnings.append("training_view_ends_at_tool_call")
        training_metadata = converted_training_row.get("metadata") or {}
        source_metadata = source_row.get("metadata") or {}
        conversion_checks = accumulator["conversion_checks"]
        conversion_mismatch_rows = [
            check["source_row"]
            for check in conversion_checks
            if not (
                check["non_system_messages_match"]
                and check["source_system_preserved"]
                and check["target_fingerprint_match"]
            )
        ]
        training_system = next(
            (
                str(message.get("content") or "")
                for message in actual_training_messages
                if message.get("role") == "system"
            ),
            "",
        )
        source_training_system = next(
            (
                str(message.get("content") or "")
                for message in training_messages
                if message.get("role") == "system"
            ),
            "",
        )
        preserved_metadata = {
            key: (
                source_metadata.get(key)
                if source_metadata.get(key) is not None
                else training_metadata.get(key)
            )
            for key in SOURCE_METADATA_FIELDS
        }
        reason_for_call = (
            source_metadata.get("reason_for_call")
            if source_metadata.get("reason_for_call") is not None
            else training_metadata.get("reason_for_call")
        )
        group = label_group(correct, reward, labels_consistent)
        has_full_multitool = full_source_static["has_multitool_turn"]
        has_training_multitool = training_visible_static["has_multitool_turn"]
        record = {
            "dialog_id": dialog_id,
            "domain": accumulator["domain"],
            "protocol_profile": protocol_profile,
            "source_row_count": accumulator["source_row_count"],
            "included_row_count": len(accumulator["included_source_rows"]),
            "included_source_rows": sorted(accumulator["included_source_rows"]),
            "included_targets": accumulator["included_targets"],
            "conversion_checks": conversion_checks,
            "conversion_mismatch_source_rows": conversion_mismatch_rows,
            "actual_training_messages_hash": stable_json_hash(actual_training_messages),
            "actual_training_system_hash": stable_json_hash(training_system),
            "source_derived_training_messages_hash": stable_json_hash(
                training_messages
            ),
            "source_derived_training_system_hash": stable_json_hash(
                source_training_system
            ),
            "full_source_messages_hash": stable_json_hash(source_messages),
            "max_source_turn_index": source_metadata.get("turn_index"),
            "max_included_turn_index": training_metadata.get("turn_index"),
            "source_correct": correct,
            "source_reward": reward,
            "labels_consistent": labels_consistent,
            "label_group": group,
            "reason_for_call": reason_for_call,
            **preserved_metadata,
            "source_metadata": preserved_metadata,
            "protocol_allows_multi": training["protocol_allows_multi"].get(dialog_id, False),
            "exposures": dict(sorted(accumulator["exposures"].items())),
            # Keep ``static`` as the training-visible view for legacy reports/filters.
            "static": training_visible_static,
            "training_visible_static": training_visible_static,
            "full_source_static": full_source_static,
            "has_training_visible_multitool": has_training_multitool,
            "has_full_source_multitool": has_full_multitool,
            "multitool_view_alignment": (
                "same"
                if has_full_multitool == has_training_multitool
                else "missing_from_training"
                if has_full_multitool
                else "training_only"
            ),
            "hard_issues": hard_issues,
            "training_visible_hard_issues": hard_issues,
            "full_source_hard_issues": full_source_hard_issues,
            "warnings": warnings,
            "deterministic_clean_success": not hard_issues,
            "full_source_deterministic_clean_success": not full_source_hard_issues,
            "workflow_type": (
                "state_changing" if training_visible_static["num_write_calls"] else "read_only"
            ),
            "full_source_workflow_type": (
                "state_changing" if full_source_static["num_write_calls"] else "read_only"
            ),
            "length_bucket": (
                "short"
                if training_visible_static["num_messages"] <= 16
                else "medium"
                if training_visible_static["num_messages"] <= 32
                else "long"
            ),
            "full_source_length_bucket": (
                "short"
                if full_source_static["num_messages"] <= 16
                else "medium"
                if full_source_static["num_messages"] <= 32
                else "long"
            ),
        }
        features.append(record)
        blind_id = hashlib.sha256(dialog_id.encode("utf-8")).hexdigest()[:20]
        original_policy = extract_policy(source_messages)
        candidate_policy, policy_changed = policy_for_profile(
            original_policy, protocol_profile
        )
        agent_tools = tool_catalog.get(accumulator["domain"], {}).get("agent", [])
        user_tools = tool_catalog.get(accumulator["domain"], {}).get("user", [])
        full_conversation = _judge_conversation(source_messages)
        training_conversation = _judge_conversation(training_messages)
        review_full_source = protocol_profile == PROTOCOL_DEPENDENCY_SAFE_MULTI
        review_conversation = (
            full_conversation if review_full_source else training_conversation
        )
        review_static = (
            full_source_static if review_full_source else training_visible_static
        )
        payload = {
            "_dialog_id": dialog_id,
            "dialog_id": f"blind-{blind_id}",
            "prompt_version": PROMPT_VERSION,
            "protocol_profile": protocol_profile,
            "domain": accumulator["domain"],
            "reason_for_call": reason_for_call,
            **preserved_metadata,
            "label_group": group,
            "official_protocol": PROTOCOL_DESCRIPTIONS[protocol_profile],
            "domain_policy": candidate_policy,
            "original_domain_policy": original_policy,
            "counterfactual_policy_applied": policy_changed,
            "agent_tools": agent_tools,
            "user_tools": user_tools,
            # Preserve the legacy current-single baseline on the actual retained
            # training prefix.  The counterfactual audit explicitly opts into the
            # complete source dialog so outcome and failure attribution are not
            # censored by prefix conversion.
            "conversation_view": (
                "full_source" if review_full_source else "training_visible"
            ),
            "conversation": review_conversation,
            "mechanical_audit_facts": _mechanical_audit_facts(review_static),
            "training_visible_view": {
                "max_turn_index": training_metadata.get("turn_index"),
                "conversation_hash": stable_json_hash(training_conversation),
                "source_derived_messages_hash": stable_json_hash(training_messages),
                "actual_converted_messages_hash": stable_json_hash(
                    actual_training_messages
                ),
                "actual_converted_system_hash": stable_json_hash(training_system),
                "mechanical_audit_facts": _mechanical_audit_facts(training_visible_static),
            },
            "view_alignment": {
                "full_source_has_multitool": has_full_multitool,
                "training_visible_has_multitool": has_training_multitool,
                "status": record["multitool_view_alignment"],
            },
            "content_hashes": {
                "original_policy": stable_json_hash(original_policy),
                "candidate_policy": stable_json_hash(candidate_policy),
                "tool_catalog": stable_json_hash(
                    {"agent": agent_tools, "user": user_tools}
                ),
                "full_source_conversation": stable_json_hash(full_conversation),
                "training_visible_conversation": stable_json_hash(training_conversation),
                "actual_training_messages": stable_json_hash(actual_training_messages),
                "actual_training_system": stable_json_hash(training_system),
                "source_derived_training_messages": stable_json_hash(
                    training_messages
                ),
                "source_derived_training_system": stable_json_hash(
                    source_training_system
                ),
            },
        }
        payload["payload_sha256"] = payload_hash(payload)
        payloads.append(payload)

    source_rows_by_domain: Counter[str] = Counter()
    for accumulator in accumulators.values():
        source_rows_by_domain[accumulator["domain"]] += accumulator["source_row_count"]
    inventory = {
        "source_path": str(source_path),
        "training_path": str(training_path),
        "protocol_profile": protocol_profile,
        "source_rows": sum(
            accumulator["source_row_count"] for accumulator in accumulators.values()
        ),
        "source_dialogs": len(accumulators),
        "training_rows": len(included),
        "training_dialogs": len(features),
        "omitted_source_dialogs": len(accumulators) - len(features),
        "source_rows_by_domain": dict(sorted(source_rows_by_domain.items())),
        "source_dialogs_by_domain": dict(
            sorted(Counter(accumulator["domain"] for accumulator in accumulators.values()).items())
        ),
        "label_groups": dict(sorted(Counter(record["label_group"] for record in features).items())),
        "full_source_multitool_dialogs": sum(
            record["has_full_source_multitool"] for record in features
        ),
        "training_visible_multitool_dialogs": sum(
            record["has_training_visible_multitool"] for record in features
        ),
        "multitool_view_mismatches": sum(
            record["multitool_view_alignment"] != "same" for record in features
        ),
        "conversion_mismatch_rows": sum(
            len(record["conversion_mismatch_source_rows"]) for record in features
        ),
        "conversion_mismatch_dialogs": sum(
            bool(record["conversion_mismatch_source_rows"]) for record in features
        ),
    }
    return features, payloads, inventory


def _stable_shuffle(rows: list[dict[str, Any]], seed: int, salt: str) -> list[dict[str, Any]]:
    shuffled = list(rows)
    digest = hashlib.sha256(f"{seed}|{salt}".encode("utf-8")).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(shuffled)
    return shuffled


def stratified_sample(
    rows: list[dict[str, Any]],
    limit: int,
    seed: int,
    strata: tuple[str, ...],
) -> list[dict[str, Any]]:
    if limit <= 0 or not rows:
        return []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in strata)].append(row)
    queues = {
        key: _stable_shuffle(group, seed, json.dumps(key, default=str))
        for key, group in sorted(groups.items(), key=lambda item: str(item[0]))
    }
    selected = []
    while len(selected) < min(limit, len(rows)):
        made_progress = False
        for key in sorted(queues, key=str):
            if queues[key]:
                selected.append(queues[key].pop())
                made_progress = True
                if len(selected) == min(limit, len(rows)):
                    break
        if not made_progress:
            break
    return selected


def _record_label_group(record: dict[str, Any]) -> str:
    return record.get("label_group") or label_group(
        record.get("source_correct"),
        record.get("source_reward"),
        record.get("labels_consistent", True),
    )


def _full_source_static(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("full_source_static") or record.get("static") or {}


def _reason_tokens(record: dict[str, Any]) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(record.get("reason_for_call") or "").lower()))


def _control_match(
    single: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    domain = single.get("domain")
    exact_fields = [
        field
        for field in SOURCE_METADATA_FIELDS
        if single.get(field) is not None and single.get(field) == control.get(field)
    ]
    single_tokens = _reason_tokens(single)
    control_tokens = _reason_tokens(control)
    union = single_tokens | control_tokens
    reason_similarity = len(single_tokens & control_tokens) / len(union) if union else 0.0
    weights = {
        "seed_pattern_task_id": 8.0 if domain == "airline" else 1.0,
        "task_id": 6.0,
        "difficulty": 4.0 if domain == "telecom" else 2.0,
        "num_subtasks": 4.0 if domain == "telecom" else 2.0,
        "scenario_id": 2.0,
    }
    metadata_score = sum(weights[field] for field in exact_fields)
    reason_weight = 5.0 if domain == "retail" else 2.0
    return {
        "score": round(metadata_score + reason_weight * reason_similarity, 6),
        "exact_metadata_fields": exact_fields,
        "reason_similarity": round(reason_similarity, 6),
    }


def build_counterfactual_samples(
    features: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    max_multi_controls_per_single: int = 3,
    seed: int = 20260802,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the protocol-neutral full audit set and its matched-pair manifest.

    The review set contains every non-consensus-success dialog, every full-source
    single-call consensus success, and up to ``max_multi_controls_per_single``
    unique same-domain full-source multi-call consensus-success controls per
    single-call success. The second return value records every match and explicit
    unmatched singles.
    """

    if max_multi_controls_per_single < 0:
        raise ValueError("max_multi_controls_per_single must be non-negative")
    payload_by_id = {payload["_dialog_id"]: payload for payload in payloads}
    non_consensus = [
        record
        for record in features
        if _record_label_group(record) != "consensus_success"
    ]
    single_successes = [
        record
        for record in features
        if _record_label_group(record) == "consensus_success"
        and not _full_source_static(record).get("has_multitool_turn", False)
        and not _full_source_static(record).get("multitool_turns")
    ]
    controls_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in features:
        full_static = _full_source_static(record)
        if (
            _record_label_group(record) == "consensus_success"
            and (
                full_static.get("has_multitool_turn", False)
                or bool(full_static.get("multitool_turns"))
            )
        ):
            controls_by_domain[str(record.get("domain"))].append(record)

    selected_controls: dict[str, dict[str, Any]] = {}
    matched_control_ids: dict[str, list[str]] = defaultdict(list)
    matches = []
    for single in sorted(single_successes, key=lambda record: str(record["dialog_id"])):
        candidates = []
        for control in controls_by_domain.get(str(single.get("domain")), []):
            match = _control_match(single, control)
            tie_break = stable_json_hash(
                [seed, single["dialog_id"], control["dialog_id"]]
            )
            candidates.append((match, tie_break, control))
        candidates.sort(
            key=lambda item: (-item[0]["score"], item[1], str(item[2]["dialog_id"]))
        )
        chosen = candidates[:max_multi_controls_per_single]
        if not chosen:
            matches.append(
                {
                    "single_dialog_id": single["dialog_id"],
                    "control_dialog_id": None,
                    "domain": single.get("domain"),
                    "match_status": "unmatched",
                    "reason": (
                        "controls_disabled"
                        if max_multi_controls_per_single == 0
                        else "no_unused_same_domain_multi_success"
                    ),
                }
            )
            continue
        for rank, (match, _, control) in enumerate(chosen, start=1):
            control_id = control["dialog_id"]
            selected_controls[control_id] = control
            matched_control_ids[single["dialog_id"]].append(control_id)
            matches.append(
                {
                    "single_dialog_id": single["dialog_id"],
                    "control_dialog_id": control_id,
                    "domain": single.get("domain"),
                    "match_status": "matched",
                    "match_rank": rank,
                    **match,
                }
            )

    selected = []
    for role, records in (
        ("non_consensus_success", non_consensus),
        ("single_consensus_success", single_successes),
        ("multi_consensus_success_control", list(selected_controls.values())),
    ):
        for record in sorted(records, key=lambda item: str(item["dialog_id"])):
            enriched = dict(record)
            enriched["counterfactual_sample_role"] = role
            if role == "single_consensus_success":
                enriched["matched_control_dialog_ids"] = matched_control_ids.get(
                    record["dialog_id"], []
                )
            elif role == "multi_consensus_success_control":
                enriched["matched_single_dialog_ids"] = sorted(
                    match["single_dialog_id"]
                    for match in matches
                    if match.get("control_dialog_id") == record["dialog_id"]
                )
            selected.append(
                {
                    "features": enriched,
                    "judge_payload": payload_by_id[record["dialog_id"]],
                }
            )
    return selected, matches


def build_samples(
    features: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    success_limit: int = 48,
    calibration_limit: int = 72,
    seed: int = 20260802,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload_by_id = {payload["_dialog_id"]: payload for payload in payloads}
    clean = [record for record in features if record["deterministic_clean_success"]]
    success_features = stratified_sample(
        clean,
        success_limit,
        seed,
        ("domain", "workflow_type", "length_bucket"),
    )

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in features:
        if record["deterministic_clean_success"]:
            category = "positive_clean"
        elif record["source_correct"] == 1 and record["source_reward"] == 1:
            category = (
                "positive_multitool"
                if "multi_tool_policy_violation" in record["hard_issues"]
                else "positive_static_issue"
            )
        elif (record["source_correct"] == 1) != (record["source_reward"] == 1):
            category = "label_conflict"
        else:
            category = "negative"
        enriched = dict(record)
        enriched["calibration_category"] = category
        categories[category].append(enriched)

    targets = {
        "positive_clean": 24,
        "positive_multitool": 12,
        "positive_static_issue": 12,
        "label_conflict": 12,
        "negative": 12,
    }
    calibration_features = []
    for category, target in targets.items():
        calibration_features.extend(
            stratified_sample(
                categories.get(category, []),
                target,
                seed,
                ("domain", "workflow_type", "length_bucket"),
            )
        )
    if len(calibration_features) < min(calibration_limit, len(features)):
        chosen = {record["dialog_id"] for record in calibration_features}
        remainder = []
        for record in features:
            if record["dialog_id"] in chosen:
                continue
            enriched = dict(record)
            enriched["calibration_category"] = "stratified_fill"
            remainder.append(enriched)
        calibration_features.extend(
            stratified_sample(
                remainder,
                min(calibration_limit, len(features)) - len(calibration_features),
                seed,
                ("domain", "workflow_type", "length_bucket"),
            )
        )
    calibration_features = calibration_features[:calibration_limit]

    def attach(record: dict[str, Any]) -> dict[str, Any]:
        return {"features": record, "judge_payload": payload_by_id[record["dialog_id"]]}

    return [attach(record) for record in success_features], [attach(record) for record in calibration_features]


def validate_review(review: dict[str, Any], valid_turns: set[int]) -> list[str]:
    errors = []
    if review.get("prompt_version") != PROMPT_VERSION:
        errors.append("wrong prompt_version")
    if review.get("verdict") not in {"keep", "review", "drop"}:
        errors.append("invalid verdict")
    if review.get("apparent_task_outcome") not in {"success", "partial", "failure", "uncertain"}:
        errors.append("invalid apparent_task_outcome")
    confidence = review.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        errors.append("invalid confidence")
    dimensions = review.get("dimensions")
    if not isinstance(dimensions, dict):
        errors.append("dimensions must be an object")
    else:
        missing = set(DIMENSIONS) - set(dimensions)
        extra = set(dimensions) - set(DIMENSIONS)
        if missing:
            errors.append(f"missing dimensions: {sorted(missing)}")
        if extra:
            errors.append(f"unknown dimensions: {sorted(extra)}")
        for name, finding in dimensions.items():
            if not isinstance(finding, dict):
                errors.append(f"{name}: finding must be an object")
                continue
            score = finding.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
                errors.append(f"{name}: invalid score")
            turns = finding.get("evidence_turns")
            if not isinstance(turns, list) or any(
                isinstance(turn, bool) or not isinstance(turn, int) or turn not in valid_turns
                for turn in turns
            ):
                errors.append(f"{name}: invalid evidence_turns")
            if not isinstance(finding.get("rationale"), str):
                errors.append(f"{name}: rationale must be a string")
    issues = review.get("critical_issues")
    if not isinstance(issues, list) or any(issue not in CRITICAL_ISSUES for issue in issues):
        errors.append("invalid critical_issues")
    for field in ("strengths", "weaknesses"):
        findings = review.get(field)
        if not isinstance(findings, list):
            errors.append(f"{field} must be an array")
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                errors.append(f"{field}: entries must be objects")
                continue
            if not isinstance(finding.get("pattern"), str):
                errors.append(f"{field}: pattern must be a string")
            if not isinstance(finding.get("explanation"), str):
                errors.append(f"{field}: explanation must be a string")
            turns = finding.get("evidence_turns")
            if not isinstance(turns, list) or any(
                isinstance(turn, bool) or not isinstance(turn, int) or turn not in valid_turns
                for turn in turns
            ):
                errors.append(f"{field}: invalid evidence_turns")
    if not isinstance(review.get("summary"), str):
        errors.append("summary must be a string")
    return errors


def review_score(review: dict[str, Any]) -> float:
    dimensions = review.get("dimensions") or {}
    scores = [dimensions.get(name, {}).get("score") for name in DIMENSIONS]
    valid = [score for score in scores if isinstance(score, (int, float))]
    return statistics.mean(valid) if len(valid) == len(DIMENSIONS) else 0.0


def classify_dialog(
    feature: dict[str, Any],
    judge_record: dict[str, Any] | None,
    confidence_threshold: float = 0.75,
    score_threshold: float = 3.25,
) -> dict[str, Any]:
    reasons = []
    if feature["hard_issues"]:
        return {
            "tier": "drop",
            "quality_score": 0.0,
            "reasons": list(feature["hard_issues"]),
            "selected_provider": None,
        }
    if not judge_record or not isinstance(judge_record.get("selected_review"), dict):
        return {
            "tier": "review",
            "quality_score": None,
            "reasons": ["missing_model_review"],
            "selected_provider": None,
        }
    review = judge_record["selected_review"]
    score = review_score(review)
    if judge_record.get("unresolved_escalation"):
        reasons.append("unresolved_judge_escalation")
    if review.get("confidence", 0) < confidence_threshold:
        reasons.append("low_judge_confidence")
    if review.get("critical_issues"):
        reasons.extend(f"judge:{issue}" for issue in review["critical_issues"])
    core_scores = [
        review.get("dimensions", {}).get(name, {}).get("score", 0)
        for name in (
            "task_completion",
            "policy_compliance",
            "tool_choice",
            "argument_grounding",
            "workflow_and_dependencies",
        )
    ]
    if review.get("verdict") == "drop" or review.get("apparent_task_outcome") == "failure":
        reasons.append("judge_drop")
        tier = "drop"
    elif (
        review.get("verdict") == "keep"
        and (
            review.get("apparent_task_outcome") == "success"
            or (
                review.get("apparent_task_outcome") == "uncertain"
                and feature.get("static", {}).get("final_has_tool")
            )
        )
        and score >= score_threshold
        and min(core_scores) >= 3
        and not reasons
    ):
        tier = "keep"
    else:
        reasons.append("judge_review_threshold")
        tier = "review"
    return {
        "tier": tier,
        "quality_score": round(score, 4),
        "reasons": sorted(set(reasons)),
        "selected_provider": judge_record.get("selected_provider"),
    }
