"""One signed tau2 Agent contract shared by SFT, official eval, and RL.

The contract intentionally keeps User/device tools out of the Agent's native
``tools=`` payload.  They are retained only as ownership metadata so source
trajectories can be converted to the official Agent view without teaching the
Agent to execute User actions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from protocol_profiles import (
    AGENT_TOOL_OWNERSHIP_RULES,
    DEPENDENCY_SAFE_MULTI_RULE,
    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    domain_policy_for_profile,
    validate_protocol_profile,
)


AGENT_CONTRACT_IMPLEMENTATION_VERSION = "tau2-agent-owned-contract-v2"
OFFICIAL_AGENT_VIEW_VERSION = "tau2-official-agent-view-v2"
BOUNDARY_ANCHOR_VIEW = "boundary_anchor"
OFFICIAL_AGENT_VIEW = "official_agent"
_BANNED_SINGLE_CALL_RE = re.compile(
    r"(?:only|at most) make one tool call at a time|"
    r"exactly ONE tool call|CANNOT make multiple tool calls",
    re.IGNORECASE,
)

def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = _value(tool, "openai_schema")
    if callable(schema):
        schema = schema()
    if schema is None and isinstance(tool, Mapping):
        schema = tool
    if not isinstance(schema, Mapping):
        raise ValueError(f"tool does not expose an OpenAI schema: {tool!r}")
    result = copy.deepcopy(dict(schema))
    function = result.get("function")
    if not isinstance(function, Mapping):
        # Accept the compact fixtures used by the analysis tests.
        name = result.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tool schema is missing function.name: {result!r}")
        result = {
            "type": "function",
            "function": {
                "name": name,
                "description": result.get("description") or name,
                "parameters": result.get("parameters")
                or {"type": "object", "properties": {}},
            },
        }
    return result


def _schema_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    name = function.get("name") if isinstance(function, Mapping) else None
    if not isinstance(name, str) or not name:
        raise ValueError(f"tool schema is missing function.name: {schema!r}")
    return name


def openai_tool_schemas(tools: Iterable[Any], *, drop_done: bool = True) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool in tools:
        schema = _tool_schema(tool)
        name = _schema_name(schema)
        if drop_done and name == "done":
            continue
        if name in seen:
            raise ValueError(f"duplicate tool schema name: {name}")
        seen.add(name)
        schemas.append(schema)
    return schemas


def _normalize_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("tool call arguments are not valid JSON") from exc
    if not isinstance(arguments, Mapping):
        raise ValueError("tool call arguments must be an object")
    return copy.deepcopy(dict(arguments))


def normalize_tool_call(
    call: Any,
    *,
    default_requestor: str,
    fallback_id: str,
) -> dict[str, Any]:
    function = _value(call, "function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = _value(call, "name")
        arguments = _value(call, "arguments")
    if not isinstance(name, str) or not name:
        raise ValueError("tool call is missing a name")
    call_id = _value(call, "id") or fallback_id
    requestor = _value(call, "requestor", default_requestor) or default_requestor
    if requestor not in {"assistant", "user"}:
        raise ValueError(f"invalid tool requestor: {requestor!r}")
    return {
        "id": str(call_id),
        "name": name,
        "arguments": _normalize_arguments(arguments),
        "requestor": requestor,
    }


def structured_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(call["id"]),
        "type": "function",
        "function": {
            "name": str(call["name"]),
            "arguments": copy.deepcopy(dict(call.get("arguments") or {})),
        },
    }


def native_agent_message_to_chat(
    message: Any,
    *,
    agent_tool_names: set[str] | None = None,
    validate_assistant_tools: bool = True,
) -> dict[str, Any] | None:
    """Convert one tau2 message to the native Agent-visible chat shape."""

    role = _value(message, "role")
    content = _value(message, "content") or ""
    calls = list(_value(message, "tool_calls") or [])
    if role == "user":
        if calls:
            return None
        return {"role": "user", "content": str(content)}
    if role == "assistant":
        chat: dict[str, Any] = {"role": "assistant", "content": str(content)}
        if calls:
            normalized = [
                normalize_tool_call(
                    call,
                    default_requestor="assistant",
                    fallback_id=f"assistant_call_{index}",
                )
                for index, call in enumerate(calls)
            ]
            if validate_assistant_tools and agent_tool_names is not None:
                wrong = [
                    call["name"]
                    for call in normalized
                    if call["name"] not in agent_tool_names
                ]
                if wrong:
                    raise ValueError(
                        f"Assistant called tools outside Agent schemas: {wrong}"
                    )
            chat["tool_calls"] = [
                structured_tool_call(call) for call in normalized
            ]
        return chat
    if role == "tool":
        requestor = _value(message, "requestor", "assistant") or "assistant"
        if requestor != "assistant":
            return None
        call_id = _value(message, "tool_call_id") or _value(message, "id")
        if not call_id:
            raise ValueError("Agent tool result is missing its call id")
        return {
            "role": "tool",
            "content": str(content),
            "tool_call_id": str(call_id),
        }
    return None


@dataclass(frozen=True)
class AgentView:
    messages: list[dict[str, Any]]
    user_tool_events: list[dict[str, Any]]


@dataclass(frozen=True)
class AgentContract:
    """Prompt, schemas, view rules, and content signature for one domain."""

    domain: str
    profile: str
    policy: str
    tools: list[dict[str, Any]]
    user_tools: list[dict[str, Any]]
    chat_template: str
    system_prompt: str
    policy_hash: str
    schema_hash: str
    user_schema_hash: str
    chat_template_hash: str
    agent_contract_signature: str

    @classmethod
    def create(
        cls,
        *,
        domain: str,
        domain_policy: str,
        agent_tools: Iterable[Any],
        user_tools: Iterable[Any] = (),
        chat_template: str,
        profile: str = PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    ) -> "AgentContract":
        validate_protocol_profile(profile)
        if profile != PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI:
            raise ValueError(
                "AgentContract is only defined for "
                f"{PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI!r}"
            )
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError("AgentContract requires the exact non-empty chat template")

        policy, _ = domain_policy_for_profile(domain_policy, profile)
        # The contract owns the protocol paragraph. Keep the policy/manual
        # complete, but do not repeat the same multi-call rule inside it.
        policy = policy.replace(DEPENDENCY_SAFE_MULTI_RULE, "").strip()
        if _BANNED_SINGLE_CALL_RE.search(policy):
            raise ValueError("single-call-only language remains in the Agent policy")
        if domain == "telecom" and "<tech_support_policy>" not in policy:
            raise ValueError("telecom Agent policy is missing the full tech-support manual")

        schemas = openai_tool_schemas(agent_tools)
        user_schemas = openai_tool_schemas(user_tools)
        if not schemas:
            raise ValueError("AgentContract requires at least one Agent tool schema")
        overlap = {_schema_name(item) for item in schemas} & {
            _schema_name(item) for item in user_schemas
        }
        if overlap:
            raise ValueError(f"Agent/User tool schemas overlap: {sorted(overlap)}")

        system_prompt = (
            "You are a customer service agent. Complete the user's task while following "
            "the policy exactly.\n\n"
            "In each turn, either send one plain-text message to the user or make one or "
            "more native tool calls; never do both in the same turn.\n"
            f"{DEPENDENCY_SAFE_MULTI_RULE}\n\n"
            f"{AGENT_TOOL_OWNERSHIP_RULES}\n\n"
            "<policy>\n"
            f"{policy.strip()}\n"
            "</policy>"
        )
        policy_hash = sha256_text(policy)
        schema_hash = sha256_text(_stable_json(schemas))
        user_schema_hash = sha256_text(_stable_json(user_schemas))
        chat_template_hash = sha256_text(chat_template)
        signature_payload = {
            "implementation_version": AGENT_CONTRACT_IMPLEMENTATION_VERSION,
            "view_version": OFFICIAL_AGENT_VIEW_VERSION,
            "domain": domain,
            "profile": profile,
            "system_prompt_hash": sha256_text(system_prompt),
            "policy_hash": policy_hash,
            "schema_hash": schema_hash,
            "user_schema_hash": user_schema_hash,
            "chat_template_hash": chat_template_hash,
        }
        return cls(
            domain=domain,
            profile=profile,
            policy=policy,
            tools=schemas,
            user_tools=user_schemas,
            chat_template=chat_template,
            system_prompt=system_prompt,
            policy_hash=policy_hash,
            schema_hash=schema_hash,
            user_schema_hash=user_schema_hash,
            chat_template_hash=chat_template_hash,
            agent_contract_signature=sha256_text(_stable_json(signature_payload)),
        )

    @property
    def agent_tool_names(self) -> set[str]:
        return {_schema_name(schema) for schema in self.tools}

    @property
    def user_tool_names(self) -> set[str]:
        return {_schema_name(schema) for schema in self.user_tools}

    def metadata(self, *, view: str, anchor_type: str | None = None) -> dict[str, Any]:
        if view not in {OFFICIAL_AGENT_VIEW, BOUNDARY_ANCHOR_VIEW}:
            raise ValueError(f"unsupported Agent view: {view!r}")
        result = {
            "agent_contract_profile": self.profile,
            "agent_contract_signature": self.agent_contract_signature,
            "policy_hash": self.policy_hash,
            "schema_hash": self.schema_hash,
            "user_schema_hash": self.user_schema_hash,
            "chat_template_hash": self.chat_template_hash,
            "view": view,
        }
        if anchor_type is not None:
            result["anchor_type"] = anchor_type
        return result

    def message_to_chat(
        self,
        message: Any,
        *,
        validate_assistant_tools: bool = True,
    ) -> dict[str, Any] | None:
        """Convert one official tau2 message to the Agent-visible native view."""

        role = _value(message, "role")
        content = _value(message, "content") or ""
        calls_value = _value(message, "tool_calls")
        calls = list(calls_value or [])
        if role == "user":
            if calls:
                return None
            if not str(content).strip():
                raise ValueError("empty User turn is not valid in the Agent view")
            return {"role": "user", "content": str(content)}
        if role == "assistant":
            chat: dict[str, Any] = {"role": "assistant", "content": str(content)}
            if calls:
                normalized = [
                    normalize_tool_call(
                        call,
                        default_requestor="assistant",
                        fallback_id=f"assistant_call_{index}",
                    )
                    for index, call in enumerate(calls)
                ]
                wrong = [call["name"] for call in normalized if call["name"] not in self.agent_tool_names]
                if validate_assistant_tools and wrong:
                    raise ValueError(f"Assistant called tools outside Agent schemas: {wrong}")
                chat["tool_calls"] = [structured_tool_call(call) for call in normalized]
            if not calls and not str(content).strip():
                raise ValueError("empty Assistant turn is not valid in the Agent view")
            return chat
        if role == "tool":
            requestor = _value(message, "requestor", "assistant") or "assistant"
            if requestor != "assistant":
                return None
            call_id = _value(message, "tool_call_id") or _value(message, "id")
            if not call_id:
                raise ValueError("Agent tool result is missing its call id")
            return {
                "role": "tool",
                "content": str(content),
                "tool_call_id": str(call_id),
            }
        if role == "system":
            return {"role": "system", "content": self.system_prompt}
        return None

    def build_source_view(
        self,
        source_messages: Sequence[Mapping[str, Any]],
        *,
        allow_final_assistant_call_without_result: bool = True,
    ) -> AgentView:
        """Build a strict, loss-ready Agent view from raw AReaL/tau2 messages.

        User calls and their results are hidden.  They remain in
        ``user_tool_events`` with source indices and must be followed by a
        non-empty natural-language User report.  Tool results without an
        unambiguous ID or positional association are rejected.
        """

        visible: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        user_events: list[dict[str, Any]] = []
        pending: dict[str, Any] | None = None
        awaiting_report_event_indices: list[int] = []
        seen_system = False

        def finish_pending_before_non_tool(
            source_index: int,
            *,
            at_end: bool = False,
        ) -> None:
            nonlocal pending, awaiting_report_event_indices
            if pending is None:
                return
            unmatched = pending["unmatched"]
            if unmatched:
                if (
                    at_end
                    and allow_final_assistant_call_without_result
                    and pending["requestor"] == "assistant"
                ):
                    pending = None
                    return
                raise ValueError(
                    f"source message {source_index}: tool calls are missing results"
                )
            if pending["requestor"] == "user":
                awaiting_report_event_indices = list(pending["event_indices"])
            pending = None

        for source_index, raw_message in enumerate(source_messages):
            role = raw_message.get("role")
            content = raw_message.get("content") or ""
            raw_calls = list(raw_message.get("tool_calls") or [])
            content_consumed_as_report = False

            if role != "tool":
                finish_pending_before_non_tool(source_index)

            if role == "system":
                if seen_system or source_index != 0:
                    raise ValueError("source trajectory must contain one leading system message")
                seen_system = True
                continue

            # A tau2 User message may report the preceding device operation and
            # start the next one in the same source message.  The native Agent
            # view keeps that natural-language report while hiding the new User
            # call below.  An empty call message cannot satisfy the report gate.
            if role == "user" and awaiting_report_event_indices:
                if not str(content).strip():
                    raise ValueError(
                        f"source message {source_index}: User operation has no natural report"
                    )
                for event_index in awaiting_report_event_indices:
                    user_events[event_index]["source_report_message_index"] = source_index
                awaiting_report_event_indices = []
                visible.append({"role": "user", "content": str(content)})
                content_consumed_as_report = True

            if role in {"assistant", "user"} and raw_calls:
                if awaiting_report_event_indices:
                    raise ValueError(
                        f"source message {source_index}: User operation has no natural report"
                    )
                normalized = [
                    normalize_tool_call(
                        call,
                        default_requestor=role,
                        fallback_id=f"source_{source_index}_call_{call_index}",
                    )
                    for call_index, call in enumerate(raw_calls)
                ]
                if any(call["requestor"] != role for call in normalized):
                    raise ValueError(
                        f"source message {source_index}: role/requestor mismatch"
                    )
                allowed = self.agent_tool_names if role == "assistant" else self.user_tool_names
                wrong = [call["name"] for call in normalized if call["name"] not in allowed]
                if wrong:
                    raise ValueError(
                        f"source message {source_index}: {role} called tools outside its schemas: {wrong}"
                    )
                if role == "assistant":
                    visible.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [structured_tool_call(call) for call in normalized],
                        }
                    )
                    event_indices: list[int] = []
                else:
                    event_indices = []
                    for call in normalized:
                        event_indices.append(len(user_events))
                        user_events.append(
                            {
                                "requestor": "user",
                                "call_id": call["id"],
                                "name": call["name"],
                                "arguments": call["arguments"],
                                "result": None,
                                "source_call_message_index": source_index,
                                "source_result_message_index": None,
                                "source_report_message_index": None,
                            }
                        )
                pending = {
                    "requestor": role,
                    "calls": normalized,
                    "unmatched": list(range(len(normalized))),
                    "event_indices": event_indices,
                }
                continue

            if role == "tool":
                if pending is None or not pending["unmatched"]:
                    raise ValueError(f"source message {source_index}: orphan tool result")
                result_requestor = raw_message.get("requestor")
                if result_requestor is not None and result_requestor != pending["requestor"]:
                    raise ValueError(
                        f"source message {source_index}: tool result requestor mismatch"
                    )
                result_id = raw_message.get("tool_call_id") or raw_message.get("id")
                if result_id:
                    matches = [
                        call_index
                        for call_index in pending["unmatched"]
                        if pending["calls"][call_index]["id"] == str(result_id)
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"source message {source_index}: tool result id is not uniquely paired"
                        )
                    call_index = matches[0]
                else:
                    # Raw AReaL rows commonly omit IDs on result messages.  A
                    # contiguous result batch has one unambiguous positional
                    # association with its immediately preceding call batch.
                    call_index = pending["unmatched"][0]
                pending["unmatched"].remove(call_index)
                call = pending["calls"][call_index]
                result_name = raw_message.get("name")
                if result_name is not None and result_name != call["name"]:
                    raise ValueError(
                        f"source message {source_index}: tool result name mismatch"
                    )
                if pending["requestor"] == "assistant":
                    visible.append(
                        {
                            "role": "tool",
                            "content": str(content),
                            "tool_call_id": call["id"],
                        }
                    )
                else:
                    event_index = pending["event_indices"][call_index]
                    user_events[event_index]["result"] = str(content)
                    user_events[event_index]["source_result_message_index"] = source_index
                continue

            if role == "user":
                if not str(content).strip():
                    raise ValueError(f"source message {source_index}: empty User turn")
                if not content_consumed_as_report:
                    visible.append({"role": "user", "content": str(content)})
                continue

            if role == "assistant":
                if awaiting_report_event_indices:
                    raise ValueError(
                        f"source message {source_index}: User operation has no natural report"
                    )
                if not str(content).strip():
                    raise ValueError(f"source message {source_index}: empty Assistant turn")
                visible.append({"role": "assistant", "content": str(content)})
                continue

            raise ValueError(f"source message {source_index}: unsupported role {role!r}")

        finish_pending_before_non_tool(len(source_messages), at_end=True)
        if awaiting_report_event_indices:
            raise ValueError("User operation at end of source trajectory has no natural report")
        if not seen_system:
            raise ValueError("source trajectory is missing its leading system message")
        if any(event["result"] is None for event in user_events):
            raise ValueError("User tool event is missing its paired result")
        if any(
            message["role"] == "user" and not str(message.get("content") or "").strip()
            for message in visible
        ):
            raise ValueError("Agent view contains an empty User turn")
        return AgentView(messages=visible, user_tool_events=user_events)


def contract_from_environment(
    environment: Any,
    *,
    domain: str,
    chat_template: str,
    profile: str = PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
) -> AgentContract:
    """Construct the shared contract from tau2's official environment API."""

    try:
        user_tools = environment.get_user_tools() or []
    except (AttributeError, ValueError):
        user_tools = []
    return AgentContract.create(
        domain=domain,
        domain_policy=environment.get_policy(),
        agent_tools=environment.get_tools(),
        user_tools=user_tools,
        chat_template=chat_template,
        profile=profile,
    )
