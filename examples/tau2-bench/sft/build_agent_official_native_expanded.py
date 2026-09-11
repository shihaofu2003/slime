#!/usr/bin/env python3
"""Convert successful raw AReaL Tau2 SFT rows to official-native targets."""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
EXPERIMENT_DIR = (
    SERVICE_AGENT_ROOT
    / "slime/output/experiments/tau2-sft-official-native-expanded"
)
DEFAULT_INPUT = (
    SERVICE_AGENT_ROOT / "datasets/AReaL-tau2-data/tau2_sft_train.jsonl"
)
DEFAULT_TOKENIZER = SERVICE_AGENT_ROOT / "models/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "data/agent_official_native_expanded.jsonl"
DEFAULT_SMOKE_OUTPUT = (
    EXPERIMENT_DIR / "data/agent_official_native_expanded_longest32.jsonl"
)
DEFAULT_GREETING = "Hi! How can I help you today?"
DOMAINS = ("airline", "retail", "telecom")
MAX_TOTAL_TOKENS = 16384

_POLICY_RE = re.compile(r"<policy>\s*(?P<policy>.*?)\s*</policy>", re.DOTALL)
_TELECOM_MARKERS = (
    "mms_issue",
    "data_usage",
    "mobile_data",
    "wifi_calling",
    "apn",
    "roaming",
    "sim_card",
)

CallKey = tuple


@dataclass
class DomainRuntime:
    domain: str
    policy: str
    system_prompt: str
    agent_tools: tuple[Any, ...]
    user_tools: tuple[Any, ...]
    tool_schemas: tuple[dict[str, Any], ...]
    greeting: str
    agent_tools_by_name: dict[str, Any] = field(init=False, repr=False)
    user_tools_by_name: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.agent_tools_by_name = {tool.name: tool for tool in self.agent_tools}
        self.user_tools_by_name = {tool.name: tool for tool in self.user_tools}


@dataclass
class DialogInfo:
    domain: str | None = None
    labels: tuple[int, int] | None = None
    invalid_reasons: set[str] = field(default_factory=set)
    tool_results: dict[CallKey, str] = field(default_factory=dict)
    call_definitions: dict[CallKey, tuple[str, str]] = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return not self.invalid_reasons and self.labels == (1, 1) and self.domain in DOMAINS


@dataclass(frozen=True)
class ParsedCall:
    key: CallKey
    source_id: str | None
    name: str
    arguments: dict[str, Any]


class RowDrop(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as file:
        for row_index, line in enumerate(file):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{row_index + 1}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{row_index + 1}: expected a JSON object")
            yield row_index, row


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _dialog_id(row: Mapping[str, Any]) -> str | None:
    value = _metadata(row).get("source_dialog_id")
    if value is None or not str(value).strip():
        return None
    return str(value)


def _binary_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    return None


def _row_labels(row: Mapping[str, Any]) -> tuple[int, int] | None:
    metadata = _metadata(row)
    correct = _binary_label(metadata.get("correct"))
    reward = _binary_label(metadata.get("reward"))
    if correct is None or reward is None:
        return None
    return correct, reward


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _source_policy(row: Mapping[str, Any]) -> str | None:
    for message in row.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = _POLICY_RE.search(content)
        if match:
            return match.group("policy")
    return None


def _raw_call_name(call: Any) -> str | None:
    if not isinstance(call, Mapping):
        return None
    function = call.get("function")
    value = function.get("name") if isinstance(function, Mapping) else call.get("name")
    return value if isinstance(value, str) and value else None


def infer_domain(
    row: Mapping[str, Any], runtimes: Mapping[str, DomainRuntime]
) -> str | None:
    metadata = _metadata(row)
    explicit = str(metadata.get("domain") or "").lower()
    if explicit in runtimes:
        return explicit

    dialog_id = str(metadata.get("source_dialog_id") or "").lower()
    dialog_matches = [domain for domain in runtimes if domain in dialog_id]
    if len(dialog_matches) == 1:
        return dialog_matches[0]

    policy = _source_policy(row)
    if policy:
        normalized = _normalized_text(policy)
        policy_matches = [
            domain
            for domain, runtime in runtimes.items()
            if normalized == _normalized_text(runtime.policy)
        ]
        if len(policy_matches) == 1:
            return policy_matches[0]

    metadata_text = " ".join(str(value) for value in metadata.values()).lower()
    if any(marker in metadata_text for marker in _TELECOM_MARKERS) and "telecom" in runtimes:
        return "telecom"

    tool_names = {
        name
        for message in list(row.get("messages") or []) + [row.get("answer") or {}]
        if isinstance(message, Mapping)
        for name in (_raw_call_name(call) for call in message.get("tool_calls") or [])
        if name
    }
    scores = {
        domain: len(
            tool_names
            & (set(runtime.agent_tools_by_name) | set(runtime.user_tools_by_name))
        )
        for domain, runtime in runtimes.items()
    }
    best = max(scores.values(), default=0)
    matches = [domain for domain, score in scores.items() if score == best and score > 0]
    return matches[0] if len(matches) == 1 else None


def _parse_arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments are not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("tool arguments must be an object")
    return copy.deepcopy(dict(value))


def _parse_call(
    call: Any,
    *,
    participant: str,
    participant_ordinal: int,
    call_index: int,
) -> ParsedCall:
    if not isinstance(call, Mapping):
        raise ValueError("tool call must be an object")
    function = call.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
        top_level_name = call.get("name")
        if top_level_name not in (None, name):
            raise ValueError("tool call names disagree")
    else:
        name = call.get("name")
        arguments = call.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError("tool call is missing its name")

    source_id_value = call.get("id")
    source_id = str(source_id_value) if source_id_value not in (None, "") else None
    key: CallKey
    if source_id is not None:
        key = ("id", source_id)
    else:
        key = (participant, participant_ordinal, call_index)
    return ParsedCall(
        key=key,
        source_id=source_id,
        name=name,
        arguments=_parse_arguments(arguments),
    )


def _result_id(message: Mapping[str, Any]) -> str | None:
    tool_call_id = message.get("tool_call_id")
    identifier = message.get("id")
    if tool_call_id not in (None, "") and identifier not in (None, ""):
        if str(tool_call_id) != str(identifier):
            raise ValueError("tool result ids disagree")
    value = tool_call_id if tool_call_id not in (None, "") else identifier
    return str(value) if value not in (None, "") else None


def _result_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("tool result content must be a string")
    return content


def _record_agent_result(
    info: DialogInfo,
    call: ParsedCall,
    result: Mapping[str, Any],
) -> None:
    definition = (
        call.name,
        json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    previous_definition = info.call_definitions.get(call.key)
    if previous_definition is not None and previous_definition != definition:
        raise RowDrop("conflicting_tool_call")
    info.call_definitions[call.key] = definition

    content = _result_content(result)
    previous_result = info.tool_results.get(call.key)
    if previous_result is not None and previous_result != content:
        raise RowDrop("conflicting_tool_result")
    info.tool_results[call.key] = content


def _index_row_tool_results(row: Mapping[str, Any], info: DialogInfo) -> None:
    messages = row.get("messages") or []
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise RowDrop("invalid_messages")

    assistant_ordinal = -1
    user_ordinal = -1
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, Mapping):
            raise RowDrop("invalid_messages")
        role = message.get("role")
        if role == "assistant":
            assistant_ordinal += 1
            participant_ordinal = assistant_ordinal
        elif role == "user":
            user_ordinal += 1
            participant_ordinal = user_ordinal
        else:
            participant_ordinal = -1

        calls = message.get("tool_calls") or []
        if calls:
            if role not in {"assistant", "user"} or not isinstance(calls, Sequence):
                raise RowDrop("unpaired_tool_result")
            participant = str(role)
            try:
                parsed_calls = [
                    _parse_call(
                        call,
                        participant=participant,
                        participant_ordinal=participant_ordinal,
                        call_index=call_index,
                    )
                    for call_index, call in enumerate(calls)
                ]
            except ValueError as exc:
                raise RowDrop("unpaired_tool_result") from exc
            if len({call.key for call in parsed_calls}) != len(parsed_calls):
                raise RowDrop("unpaired_tool_result")

            result_index = index + 1
            results: list[Mapping[str, Any]] = []
            while result_index < len(messages):
                result = messages[result_index]
                if not isinstance(result, Mapping) or result.get("role") != "tool":
                    break
                results.append(result)
                result_index += 1
            if len(results) != len(parsed_calls):
                raise RowDrop("unpaired_tool_result")

            unmatched = list(parsed_calls)
            pairs: list[tuple[ParsedCall, Mapping[str, Any]]] = []
            for result in results:
                try:
                    identifier = _result_id(result)
                except ValueError as exc:
                    raise RowDrop("unpaired_tool_result") from exc
                if identifier is None:
                    if not unmatched:
                        raise RowDrop("unpaired_tool_result")
                    call = unmatched[0]
                else:
                    matches = [call for call in unmatched if call.source_id == identifier]
                    if len(matches) != 1:
                        raise RowDrop("unpaired_tool_result")
                    call = matches[0]
                requestor = result.get("requestor")
                if requestor not in (None, participant):
                    raise RowDrop("unpaired_tool_result")
                unmatched.remove(call)
                pairs.append((call, result))
            if unmatched:
                raise RowDrop("unpaired_tool_result")
            if role == "assistant":
                for call, result in pairs:
                    _record_agent_result(info, call, result)
            index = result_index
            continue

        if role == "tool":
            raise RowDrop("unpaired_tool_result")
        index += 1


def index_source_dialogs(
    path: Path,
    runtimes: Mapping[str, DomainRuntime],
) -> tuple[dict[str, DialogInfo], Counter[str]]:
    """First pass: collect dialog labels/domains and stable real tool results."""

    dialogs: dict[str, DialogInfo] = {}
    stats: Counter[str] = Counter()
    for _, row in read_jsonl(path):
        stats["source_rows"] += 1
        dialog_id = _dialog_id(row)
        if dialog_id is None:
            stats["rows_missing_dialog_id"] += 1
            continue
        info = dialogs.setdefault(dialog_id, DialogInfo())

        domain = infer_domain(row, runtimes)
        if domain is not None:
            if info.domain is None:
                info.domain = domain
            elif info.domain != domain:
                info.invalid_reasons.add("conflicting_domain")
        else:
            stats["rows_unknown_domain"] += 1

        labels = _row_labels(row)
        if labels is None:
            info.invalid_reasons.add("invalid_labels")
        elif info.labels is None:
            info.labels = labels
        elif info.labels != labels:
            info.invalid_reasons.add("conflicting_labels")

        try:
            _index_row_tool_results(row, info)
        except RowDrop as exc:
            info.invalid_reasons.add(exc.reason)

    for info in dialogs.values():
        if info.domain is None:
            info.invalid_reasons.add("unknown_domain")
        if info.labels is None:
            info.invalid_reasons.add("invalid_labels")
        stats["dialogs"] += 1
        if info.eligible:
            stats["eligible_dialogs"] += 1
        elif info.labels != (1, 1):
            stats["non_double_success_dialogs"] += 1
        for reason in info.invalid_reasons:
            stats[f"invalid_dialog_{reason}"] += 1
    return dialogs, stats


def _validate_tool_arguments(tool: Any, arguments: Mapping[str, Any]) -> None:
    fields = set(tool.params.model_fields)
    if set(arguments) - fields:
        raise ValueError("unknown tool argument")
    tool.params.model_validate_json(json.dumps(arguments), strict=True)


def _validated_agent_call(call: ParsedCall, runtime: DomainRuntime) -> ParsedCall:
    tool = runtime.agent_tools_by_name.get(call.name)
    if tool is None:
        raise RowDrop("invalid_agent_tool")
    try:
        _validate_tool_arguments(tool, call.arguments)
    except (TypeError, ValueError) as exc:
        raise RowDrop("invalid_tool_arguments") from exc
    return call


def _validate_user_calls(
    calls: Sequence[Any],
    *,
    runtime: DomainRuntime,
    user_ordinal: int,
) -> None:
    for call_index, raw_call in enumerate(calls):
        try:
            call = _parse_call(
                raw_call,
                participant="user",
                participant_ordinal=user_ordinal,
                call_index=call_index,
            )
        except ValueError as exc:
            raise RowDrop("invalid_user_tool") from exc
        tool = runtime.user_tools_by_name.get(call.name)
        if tool is None:
            raise RowDrop("invalid_user_tool")
        try:
            _validate_tool_arguments(tool, call.arguments)
        except (TypeError, ValueError) as exc:
            raise RowDrop("invalid_user_tool_arguments") from exc


def _text_message(role: str, content: str, *, loss_mask: int = 0) -> dict[str, Any]:
    return {"role": role, "content": content, "step_loss_mask": loss_mask}


def _internal_call_message(call: ParsedCall, *, loss_mask: int) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "_call": call,
        "step_loss_mask": loss_mask,
    }


def _internal_result_message(call: ParsedCall, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": content,
        "_call_key": call.key,
        "step_loss_mask": 0,
    }


def _finalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    output_ids: dict[CallKey, str] = {}
    call_number = 0
    for message in messages:
        call = message.get("_call")
        if isinstance(call, ParsedCall):
            call_number += 1
            output_id = f"call_{call_number:04d}"
            if call.key in output_ids:
                raise RowDrop("duplicate_source_call")
            output_ids[call.key] = output_id
            finalized.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": output_id,
                            "name": call.name,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                    "step_loss_mask": message["step_loss_mask"],
                }
            )
            continue

        call_key = message.get("_call_key")
        if call_key is not None:
            if call_key not in output_ids:
                raise RowDrop("result_before_call")
            finalized.append(
                {
                    "role": "tool",
                    "content": message.get("content"),
                    "tool_call_id": output_ids[call_key],
                    "step_loss_mask": 0,
                }
            )
            continue
        finalized.append(copy.deepcopy(dict(message)))
    return finalized


def _prefix_messages(
    row: Mapping[str, Any],
    *,
    info: DialogInfo,
    runtime: DomainRuntime,
) -> tuple[list[dict[str, Any]], int]:
    source_messages = row.get("messages") or []
    if not isinstance(source_messages, Sequence) or isinstance(source_messages, (str, bytes)):
        raise RowDrop("invalid_messages")

    messages = [
        _text_message("system", runtime.system_prompt),
        _text_message("assistant", runtime.greeting),
    ]
    assistant_ordinal = -1
    user_ordinal = -1
    before_first_user = True
    removed_source_greeting = False
    awaiting_user_report = False

    for source_message in source_messages:
        if not isinstance(source_message, Mapping):
            raise RowDrop("invalid_messages")
        role = source_message.get("role")
        calls = source_message.get("tool_calls") or []
        if role == "system":
            continue
        if role == "assistant":
            assistant_ordinal += 1
            if awaiting_user_report:
                raise RowDrop("missing_user_report")
            if calls:
                for call_index, raw_call in enumerate(calls):
                    try:
                        call = _parse_call(
                            raw_call,
                            participant="assistant",
                            participant_ordinal=assistant_ordinal,
                            call_index=call_index,
                        )
                    except ValueError as exc:
                        raise RowDrop("invalid_agent_tool") from exc
                    _validated_agent_call(call, runtime)
                    if call.key not in info.tool_results:
                        raise RowDrop("missing_prefix_result")
                    messages.append(_internal_call_message(call, loss_mask=0))
                    messages.append(
                        _internal_result_message(call, info.tool_results[call.key])
                    )
                continue

            content = source_message.get("content")
            if not isinstance(content, str):
                content = "" if content is None else str(content)
            if (
                before_first_user
                and not removed_source_greeting
                and content.strip() == runtime.greeting.strip()
            ):
                removed_source_greeting = True
                continue
            if content.strip():
                messages.append(_text_message("assistant", content))
            continue

        if role == "user":
            user_ordinal += 1
            before_first_user = False
            if calls:
                if not isinstance(calls, Sequence):
                    raise RowDrop("invalid_user_tool")
                _validate_user_calls(calls, runtime=runtime, user_ordinal=user_ordinal)
                awaiting_user_report = True
                continue
            content = source_message.get("content")
            if isinstance(content, str) and content.strip():
                messages.append(_text_message("user", content))
                awaiting_user_report = False
            continue

        if role == "tool":
            continue
        raise RowDrop("unsupported_role")

    if awaiting_user_report:
        raise RowDrop("missing_user_report")
    return messages, assistant_ordinal + 1


def _metadata_for_target(
    row: Mapping[str, Any],
    *,
    source_row: int,
    domain: str,
    target_type: str,
    expanded_call_index: int | None = None,
    expanded_call_count: int | None = None,
) -> dict[str, Any]:
    source_metadata = _metadata(row)
    metadata = {
        "domain": domain,
        "source_dialog_id": str(source_metadata["source_dialog_id"]),
        "source_row": source_row,
        "source_turn_index": source_metadata.get("turn_index"),
        "target_type": target_type,
        "correct": 1,
        "reward": 1,
    }
    if expanded_call_count is not None and expanded_call_count > 1:
        metadata["expanded_call_index"] = expanded_call_index
        metadata["expanded_call_count"] = expanded_call_count
    return metadata


def _converted_row(
    messages: Sequence[Mapping[str, Any]],
    *,
    runtime: DomainRuntime,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "messages": _finalize_messages(messages),
        "tools": copy.deepcopy(list(runtime.tool_schemas)),
        "metadata": metadata,
    }


def _convert_source_row(
    row: Mapping[str, Any],
    *,
    source_row: int,
    info: DialogInfo,
    runtime: DomainRuntime,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    try:
        prefix, target_ordinal = _prefix_messages(row, info=info, runtime=runtime)
    except RowDrop as exc:
        stats[f"drop_{exc.reason}"] += 1
        return []

    target = row.get("answer")
    if not isinstance(target, Mapping) or target.get("role") != "assistant":
        stats["drop_invalid_target"] += 1
        return []
    raw_calls = target.get("tool_calls") or []
    if raw_calls:
        if not isinstance(raw_calls, Sequence):
            stats["drop_invalid_agent_tool"] += 1
            return []
        converted: list[dict[str, Any]] = []
        target_context = list(prefix)
        call_count = len(raw_calls)
        for call_index, raw_call in enumerate(raw_calls):
            try:
                call = _parse_call(
                    raw_call,
                    participant="assistant",
                    participant_ordinal=target_ordinal,
                    call_index=call_index,
                )
                _validated_agent_call(call, runtime)
                target_messages = target_context + [
                    _internal_call_message(call, loss_mask=1)
                ]
                converted.append(
                    _converted_row(
                        target_messages,
                        runtime=runtime,
                        metadata=_metadata_for_target(
                            row,
                            source_row=source_row,
                            domain=runtime.domain,
                            target_type="tool_call",
                            expanded_call_index=call_index,
                            expanded_call_count=call_count,
                        ),
                    )
                )
            except RowDrop as exc:
                stats[f"drop_{exc.reason}"] += 1
                break
            except ValueError:
                stats["drop_invalid_agent_tool"] += 1
                break

            if call_index + 1 < call_count:
                result = info.tool_results.get(call.key)
                if result is None:
                    stats["drop_missing_target_result"] += 1
                    break
                target_context.append(_internal_call_message(call, loss_mask=0))
                target_context.append(_internal_result_message(call, result))
        return converted

    content = target.get("content")
    if not isinstance(content, str) or not content.strip():
        stats["drop_empty_target"] += 1
        return []
    try:
        return [
            _converted_row(
                prefix + [_text_message("assistant", content, loss_mask=1)],
                runtime=runtime,
                metadata=_metadata_for_target(
                    row,
                    source_row=source_row,
                    domain=runtime.domain,
                    target_type="text",
                ),
            )
        ]
    except RowDrop as exc:
        stats[f"drop_{exc.reason}"] += 1
        return []


def iter_converted_source_rows(
    path: Path,
    *,
    dialogs: Mapping[str, DialogInfo],
    runtimes: Mapping[str, DomainRuntime],
    stats: Counter[str],
) -> Iterable[dict[str, Any]]:
    """Second pass: stream eligible raw rows and yield expanded native targets."""

    for source_row, row in read_jsonl(path):
        stats["source_rows"] += 1
        dialog_id = _dialog_id(row)
        if dialog_id is None or dialog_id not in dialogs:
            stats["drop_missing_dialog_id"] += 1
            continue
        info = dialogs[dialog_id]
        if info.invalid_reasons:
            stats["drop_invalid_dialog"] += 1
            for reason in info.invalid_reasons:
                stats[f"drop_dialog_{reason}"] += 1
            continue
        if info.labels != (1, 1):
            stats["drop_not_double_success"] += 1
            continue
        if info.domain not in runtimes:
            stats["drop_unknown_domain"] += 1
            continue

        stats["double_success_rows"] += 1
        converted = _convert_source_row(
            row,
            source_row=source_row,
            info=info,
            runtime=runtimes[info.domain],
            stats=stats,
        )
        stats["expanded_rows"] += len(converted)
        yield from converted


def load_domain_runtimes() -> dict[str, DomainRuntime]:
    from tau2.agent.llm_agent import LLMAgent
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
    from tau2.registry import registry

    greeting = DEFAULT_FIRST_AGENT_MESSAGE.content or DEFAULT_GREETING
    runtimes: dict[str, DomainRuntime] = {}
    for domain in DOMAINS:
        environment = registry.get_env_constructor(domain)()
        policy = environment.get_policy()
        agent_tools = tuple(environment.get_tools())
        try:
            user_tools = tuple(environment.get_user_tools())
        except ValueError:
            user_tools = ()
        system_prompt = LLMAgent(
            tools=list(agent_tools),
            domain_policy=policy,
            llm="unused",
        ).system_prompt
        runtimes[domain] = DomainRuntime(
            domain=domain,
            policy=policy,
            system_prompt=system_prompt,
            agent_tools=agent_tools,
            user_tools=user_tools,
            tool_schemas=tuple(copy.deepcopy(tool.openai_schema) for tool in agent_tools),
            greeting=greeting,
        )
    return runtimes


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ValueError("tokenizer did not return a token ID list")
    return value


def _render_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    generator: Any,
) -> tuple[list[int], list[int]]:
    token_ids, loss_mask = generator.get_loss_mask(
        row["messages"], tools=row["tools"]
    )
    token_ids = _token_ids(token_ids)
    native_ids = _token_ids(
        tokenizer.apply_chat_template(
            row["messages"],
            tools=row["tools"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
    )
    if token_ids != native_ids:
        raise ValueError("qwen3_full token IDs differ from the native chat template")
    if len(token_ids) != len(loss_mask) or not any(loss_mask):
        raise ValueError("invalid qwen3_full token/loss mask")
    return token_ids, loss_mask


def validate_converted_row(
    row: Mapping[str, Any],
    *,
    runtime: DomainRuntime,
    tokenizer: Any,
    generator: Any,
    max_tokens: int,
) -> int:
    if set(row) != {"messages", "tools", "metadata"}:
        raise ValueError("row has unexpected top-level fields")
    if row.get("tools") != list(runtime.tool_schemas):
        raise ValueError("row tools differ from the current runtime order or schemas")

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("row metadata is missing")
    required_metadata = {
        "domain",
        "source_dialog_id",
        "source_row",
        "source_turn_index",
        "target_type",
        "correct",
        "reward",
    }
    expanded_metadata = {"expanded_call_index", "expanded_call_count"}
    if set(metadata) not in (required_metadata, required_metadata | expanded_metadata):
        raise ValueError("row metadata contains unexpected fields")
    if metadata.get("domain") != runtime.domain or runtime.domain not in DOMAINS:
        raise ValueError("row has an invalid domain")
    if metadata.get("correct") != 1 or metadata.get("reward") != 1:
        raise ValueError("row is not double-success")
    if expanded_metadata <= set(metadata):
        count = metadata["expanded_call_count"]
        index = metadata["expanded_call_index"]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 1
            or not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < count
        ):
            raise ValueError("invalid expanded-call metadata")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("row messages are missing")
    systems = [message for message in messages if message.get("role") == "system"]
    if len(systems) != 1 or messages[0] != {
        "role": "system",
        "content": runtime.system_prompt,
        "step_loss_mask": 0,
    }:
        raise ValueError("row does not contain exactly the native system prompt")
    if messages[1] != {
        "role": "assistant",
        "content": runtime.greeting,
        "step_loss_mask": 0,
    }:
        raise ValueError("row does not contain the native greeting in the expected position")
    if sum(
        message.get("role") == "assistant"
        and message.get("content") == runtime.greeting
        and not message.get("tool_calls")
        for message in messages
    ) != 1:
        raise ValueError("row does not contain exactly one greeting")
    if messages[-1].get("role") != "assistant":
        raise ValueError("row target is not the final Assistant message")
    if messages[-1].get("step_loss_mask") != 1 or any(
        message.get("step_loss_mask") != 0 for message in messages[:-1]
    ):
        raise ValueError("row does not use target-only loss masking")

    seen_call_ids: list[str] = []
    result_ids: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("message must be an object")
        if "reasoning" in message or "thinking" in message:
            raise ValueError("row retained reasoning or thinking")
        role = message.get("role")
        if role in {"system", "user"}:
            if set(message) != {"role", "content", "step_loss_mask"}:
                raise ValueError(f"invalid {role} message shape")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"empty {role} message")
            continue
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                if set(message) != {"role", "content", "tool_calls", "step_loss_mask"}:
                    raise ValueError("invalid Assistant tool-call shape")
                if message.get("content") is not None or len(calls) != 1:
                    raise ValueError("Assistant message has mixed or multiple outputs")
                call = calls[0]
                if not isinstance(call, Mapping) or set(call) != {
                    "id",
                    "name",
                    "type",
                    "function",
                }:
                    raise ValueError("invalid native tool-call wire shape")
                function = call.get("function")
                if not isinstance(function, Mapping) or set(function) != {"name", "arguments"}:
                    raise ValueError("invalid native tool-call function shape")
                if call.get("type") != "function" or call.get("name") != function.get("name"):
                    raise ValueError("native tool-call names disagree")
                expected_id = f"call_{len(seen_call_ids) + 1:04d}"
                if call.get("id") != expected_id:
                    raise ValueError("tool-call ids are not sequential")
                try:
                    arguments = json.loads(function["arguments"])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError("tool-call arguments are not JSON") from exc
                parsed = ParsedCall(
                    key=("id", expected_id),
                    source_id=expected_id,
                    name=str(call.get("name") or ""),
                    arguments=arguments,
                )
                _validated_agent_call(parsed, runtime)
                seen_call_ids.append(expected_id)
            else:
                if set(message) != {"role", "content", "step_loss_mask"}:
                    raise ValueError("invalid Assistant text shape")
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise ValueError("empty Assistant text")
            continue
        if role == "tool":
            if set(message) != {"role", "content", "tool_call_id", "step_loss_mask"}:
                raise ValueError("invalid tool-result shape")
            result_id = message.get("tool_call_id")
            if result_id not in seen_call_ids or result_id in result_ids:
                raise ValueError("tool result does not pair with one preceding call")
            if not isinstance(message.get("content"), str):
                raise ValueError("tool result content must be a string")
            result_ids.append(result_id)
            continue
        raise ValueError(f"unsupported message role: {role!r}")

    last_has_call = bool(messages[-1].get("tool_calls"))
    unmatched = set(seen_call_ids) - set(result_ids)
    expected_unmatched = {seen_call_ids[-1]} if last_has_call else set()
    if unmatched != expected_unmatched:
        raise ValueError("prefix call/result pairing is incomplete")
    expected_target_type = "tool_call" if last_has_call else "text"
    if metadata.get("target_type") != expected_target_type:
        raise ValueError("target_type does not match the final Assistant message")

    token_ids, _ = _render_row(row, tokenizer=tokenizer, generator=generator)
    if len(token_ids) > max_tokens:
        raise ValueError(f"row exceeds the no-truncation cap: {len(token_ids)} > {max_tokens}")
    return len(token_ids)


def _runtime_and_tokenizer(tokenizer_path: Path) -> tuple[Any, Any, dict[str, DomainRuntime]]:
    from transformers import AutoTokenizer

    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    return tokenizer, generator, load_domain_runtimes()


def _percentiles(lengths: Sequence[int]) -> dict[str, int]:
    if not lengths:
        return {}
    ordered = sorted(lengths)
    summary = {
        label: ordered[round((len(ordered) - 1) * fraction)]
        for label, fraction in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
    }
    summary["max"] = ordered[-1]
    return summary


def build(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0 or args.smoke_size <= 0:
        raise ValueError("--max-tokens and --smoke-size must be positive")
    if args.output == args.smoke_output:
        raise ValueError("full and smoke outputs must use different paths")

    tokenizer, generator, runtimes = _runtime_and_tokenizer(args.tokenizer)
    dialogs, first_pass = index_source_dialogs(args.input, runtimes)
    second_pass: Counter[str] = Counter()
    kept_by_domain: Counter[str] = Counter()
    target_types: Counter[str] = Counter()
    dropped_over_cap: Counter[str] = Counter()
    token_lengths: list[int] = []
    longest: list[tuple[int, int, dict[str, Any]]] = []
    kept = 0
    sequence = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_smoke = args.smoke_output.with_suffix(args.smoke_output.suffix + ".tmp")
    try:
        with temporary_output.open("w", encoding="utf-8") as output_file:
            for row in iter_converted_source_rows(
                args.input,
                dialogs=dialogs,
                runtimes=runtimes,
                stats=second_pass,
            ):
                domain = str(row["metadata"]["domain"])
                total_tokens = validate_converted_row(
                    row,
                    runtime=runtimes[domain],
                    tokenizer=tokenizer,
                    generator=generator,
                    max_tokens=max(args.max_tokens, 1 << 60),
                )
                if total_tokens > args.max_tokens:
                    dropped_over_cap[domain] += 1
                    continue
                output_file.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                kept += 1
                kept_by_domain[domain] += 1
                target_types[str(row["metadata"]["target_type"])] += 1
                token_lengths.append(total_tokens)
                item = (total_tokens, sequence, row)
                sequence += 1
                if len(longest) < args.smoke_size:
                    heapq.heappush(longest, item)
                elif item[:2] > longest[0][:2]:
                    heapq.heapreplace(longest, item)

        with temporary_smoke.open("w", encoding="utf-8") as smoke_file:
            for _, _, row in sorted(longest, reverse=True):
                smoke_file.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        temporary_output.replace(args.output)
        temporary_smoke.replace(args.smoke_output)
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_smoke.unlink(missing_ok=True)

    summary = {
        "source_rows": first_pass["source_rows"],
        "double_success_rows": second_pass["double_success_rows"],
        "expanded_rows": second_pass["expanded_rows"],
        "kept_rows": kept,
        "kept_by_domain": dict(sorted(kept_by_domain.items())),
        "target_types": dict(sorted(target_types.items())),
        "dropped_over_token_cap": sum(dropped_over_cap.values()),
        "dropped_over_token_cap_by_domain": dict(sorted(dropped_over_cap.items())),
        "drop_reasons": dict(
            sorted((key, value) for key, value in second_pass.items() if key.startswith("drop_"))
        ),
        "invalid_dialog_reasons": dict(
            sorted((key, value) for key, value in first_pass.items() if key.startswith("invalid_dialog_"))
        ),
        "token_lengths": _percentiles(token_lengths),
        "max_tokens": args.max_tokens,
        "smoke_rows": len(longest),
        "output": str(args.output),
        "smoke_output": str(args.smoke_output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def validate(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    tokenizer, generator, runtimes = _runtime_and_tokenizer(args.tokenizer)
    rows = 0
    domains: Counter[str] = Counter()
    target_types: Counter[str] = Counter()
    token_lengths: list[int] = []
    for row_index, row in read_jsonl(args.input):
        metadata = row.get("metadata") or {}
        domain = metadata.get("domain")
        if domain not in runtimes:
            raise ValueError(f"{args.input}:{row_index + 1}: invalid domain {domain!r}")
        try:
            total_tokens = validate_converted_row(
                row,
                runtime=runtimes[domain],
                tokenizer=tokenizer,
                generator=generator,
                max_tokens=args.max_tokens,
            )
        except ValueError as exc:
            raise ValueError(f"{args.input}:{row_index + 1}: {exc}") from exc
        rows += 1
        domains[str(domain)] += 1
        target_types[str(metadata.get("target_type"))] += 1
        token_lengths.append(total_tokens)
    print(
        json.dumps(
            {
                "rows": rows,
                "rows_by_domain": dict(sorted(domains.items())),
                "target_types": dict(sorted(target_types.items())),
                "token_lengths": _percentiles(token_lengths),
                "max_tokens": args.max_tokens,
                "input": str(args.input),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument("--smoke-output", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    build_parser.add_argument("--max-tokens", type=int, default=MAX_TOTAL_TOKENS)
    build_parser.add_argument("--smoke-size", type=int, default=32)
    build_parser.set_defaults(func=build)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    validate_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    validate_parser.add_argument("--max-tokens", type=int, default=MAX_TOTAL_TOKENS)
    validate_parser.set_defaults(func=validate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
