"""Tau2 action parsing and observation formatting.

Adapted from examples/tau-bench/tau-bench-example/tau2/actions.py. Domain-agnostic
and shared between evaluation (eval.py) and the upcoming GRPO rollout.

We standardize on Qwen3 native function calling:

  <tool_call>{"name": "...", "arguments": {...}}</tool_call>

The tau2 AgentGymEnv accepts the functional form ``tool_name(kw=value)`` for tool
calls or a plain string for a user-facing message (see tau2-bench gym README), so
``env_action_from_parsed_action`` emits exactly those two shapes.

Thinking models (Qwen3.5) and base instruct models often answer in plain prose
instead of a `respond` tool call. ``parse_action(..., allow_prose_respond=True)``
lifts such prose (text after `</think>`) into a `respond` action — the eval opts
in; the GRPO rollout uses the default (strict) so a missing tool call stays a
parse error for the training signal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
# qwen3_coder tool-call body (Qwen3.5 family): <function=name>...<parameter=k>v</parameter>...</function>
_FUNC_RE = re.compile(r"<function=(?P<name>[^>\s]+)>(?P<body>.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=(?P<key>[^>\s]+)>(?P<val>.*?)</parameter>", re.S)


@dataclass(frozen=True, slots=True)
class ParsedAction:
    name: str
    arguments: dict[str, Any]
    raw_action_call: str


def _to_py_literal(value: Any) -> str:
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_to_py_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f"{_to_py_literal(k)}: {_to_py_literal(v)}")
        return "{" + ", ".join(items) + "}"
    return repr(value)


def _to_functional_call(name: str, arguments: dict[str, Any]) -> str:
    if not arguments:
        return f"{name}()"
    parts = [f"{k}={_to_py_literal(v)}" for k, v in arguments.items()]
    return f"{name}({', '.join(parts)})"


def _find_tool_call_blocks(text: str) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    cursor = 0
    while True:
        start = text.find(_TOOL_CALL_START, cursor)
        if start == -1:
            break
        end = text.find(_TOOL_CALL_END, start + len(_TOOL_CALL_START))
        if end == -1:
            raise ValueError("Missing </tool_call> for <tool_call> block")
        content = text[start + len(_TOOL_CALL_START) : end].strip()
        blocks.append((content, start, end + len(_TOOL_CALL_END)))
        cursor = end + len(_TOOL_CALL_END)
    return blocks


def _strip_think_prose(text: str) -> str | None:
    """User-facing prose from a plain reply: the text after the first
    ``</think>`` (thinking models) or the whole text (non-thinking models).

    Returns it stripped, or ``None`` when there is no usable reply:
    - an unclosed ``<think>`` (open tag, no ``</think>``) is non-convergent
      thinking — never send raw reasoning to the user;
    - an empty remainder is not a reply.
    """
    if "<think>" in text and "</think>" not in text:
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    return text or None


def _coerce_param_value(value: str) -> Any:
    """The qwen3_coder ``<parameter>`` format is string-only; coerce a value to a
    Python scalar/container when it parses as JSON (int/float/bool/null/list/
    object), otherwise keep it as a string."""
    v = value.strip()
    try:
        return json.loads(v)
    except Exception:
        return v


def _parse_function_format(content: str) -> tuple[str, dict[str, Any]] | None:
    """Parse the qwen3_coder tool-call body
    ``<function=name>...<parameter=key>value</parameter>...</function>``.
    Returns ``(name, arguments)`` or ``None`` when no ``<function>`` tag is found.
    """
    fmatch = _FUNC_RE.search(content)
    if not fmatch:
        return None
    name = fmatch.group("name").strip()
    arguments: dict[str, Any] = {}
    for pm in _PARAM_RE.finditer(fmatch.group("body")):
        arguments[pm.group("key").strip()] = _coerce_param_value(pm.group("val"))
    return name, arguments


def _parse_tool_call_content(content: str) -> tuple[str, dict[str, Any]]:
    """Parse the inside of one ``<tool_call>...</tool_call>`` block. Accepts the
    Qwen3 JSON form ``{"name": ..., "arguments": {...}}`` and the Qwen3.5
    qwen3_coder ``<function>/<parameter>`` form."""
    # JSON form first (Qwen3-Instruct family).
    try:
        data = json.loads(content)
    except Exception:
        data = None
    if isinstance(data, dict):
        name = data.get("name")
        arguments = data.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        if isinstance(name, str) and name and isinstance(arguments, dict):
            return name, arguments

    # qwen3_coder <function>/<parameter> form (Qwen3.5 family).
    parsed = _parse_function_format(content)
    if parsed and parsed[0]:
        return parsed

    raise ValueError(
        'tool_call content is neither JSON {"name":..., "arguments":...} '
        "nor <function>/<parameter> format"
    )


def parse_action(text: str, *, allow_prose_respond: bool = False) -> ParsedAction:
    blocks = _find_tool_call_blocks(text)
    if not blocks:
        # Thinking models (e.g. Qwen3.5) and base instruct models often answer
        # in plain prose — after </think> for thinkers — instead of wrapping the
        # reply in a `respond` tool call. tau2's gym accepts a plain string as a
        # user-facing message, so optionally lift that prose into a `respond`.
        if allow_prose_respond:
            prose = _strip_think_prose(text)
            if prose:
                return ParsedAction(
                    name="respond",
                    arguments={"content": prose},
                    raw_action_call=_to_functional_call("respond", {"content": prose}),
                )
        raise ValueError("Missing <tool_call>...</tool_call> block")
    if len(blocks) > 1:
        raise ValueError("Multiple <tool_call> blocks found; expected exactly one")

    # Thinking models (e.g. Qwen3.5) emit <think>/reasoning prose before/after
    # the tool call; ignore any text outside the single <tool_call> block.
    content, _, _ = blocks[0]
    name, arguments = _parse_tool_call_content(content)

    if not isinstance(name, str) or not name:
        raise ValueError("Tool call missing non-empty 'name'")
    if not isinstance(arguments, dict):
        raise ValueError("Tool call 'arguments' must be an object")

    return ParsedAction(name=name, arguments=arguments, raw_action_call=_to_functional_call(name, arguments))


def _strip_role_prefix(line: str) -> tuple[str | None, str]:
    line = line.strip()
    if ": " not in line:
        return None, line
    role, content = line.split(": ", 1)
    return role.strip().lower(), content


@dataclass(frozen=True, slots=True)
class ParsedObservation:
    user: str
    tool: str
    other: str


def split_observation(observation: str) -> ParsedObservation:
    user_lines: list[str] = []
    tool_lines: list[str] = []
    other_lines: list[str] = []

    for raw_line in (observation or "").splitlines():
        role, content = _strip_role_prefix(raw_line)
        content = content.strip()
        if not content:
            continue
        if role == "user":
            user_lines.append(content)
        elif role == "tool":
            tool_lines.append(content)
        else:
            other_lines.append(content if role is None else f"{role}: {content}")

    return ParsedObservation(
        user="\n".join(user_lines).strip(),
        tool="\n".join(tool_lines).strip(),
        other="\n".join(other_lines).strip(),
    )


def followup_messages_for_observation(
    *,
    observation: str,
    last_action_call: str,
    last_action_was_tool: bool,
) -> list[dict[str, str]]:
    parsed = split_observation(observation)
    messages: list[dict[str, str]] = []

    if last_action_was_tool:
        tool_payload = parsed.tool or parsed.other or "[no_observation]"
        messages.append({"role": "user", "content": f"Tool result for {last_action_call}:\n{tool_payload}"})
        if parsed.user:
            messages.append({"role": "user", "content": parsed.user})
        return messages

    user_payload = parsed.user or parsed.other or "[no_observation]"
    messages.append({"role": "user", "content": user_payload})
    return messages


def env_action_from_parsed_action(action: ParsedAction) -> str:
    if action.name == "respond":
        content = action.arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("respond requires a non-empty content string")
        return content
    return action.raw_action_call
