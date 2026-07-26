#!/usr/bin/env python3
"""Prepare tau2-bench SFT data for slime.

The output format is JSONL with one training example per line:

    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}],
     "metadata": {...}}

`slime.rollout.sft_rollout.generate_rollout` consumes the `messages` column
directly and masks assistant turns for SFT loss.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
DEFAULT_APIGEN_PATH = SERVICE_AGENT_ROOT / "datasets/APIGen-MT-5k/apigen-mt_5k.json"
DEFAULT_AREAL_PATH = SERVICE_AGENT_ROOT / "datasets/AReaL-tau2-data/tau2_sft_train.jsonl"
DEFAULT_OUTPUT_DIR = SERVICE_AGENT_ROOT / "datasets/tau2-bench-sft"

OUTPUT_FORMAT = """## Output Format
In each turn you can either:
- Send one plain-text message to the user.
- Make one or more tool calls in this format:
<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>

If you make multiple tool calls, emit one <tool_call> block for each call and no
other content. You cannot send a user message and make a tool call in the same
turn.
When the task is complete, call:
<tool_call>{"name": "done", "arguments": {}}</tool_call>
"""


def read_jsonl(path: Path, limit: int | None = None):
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def tool_call_payload(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments") or {}
    else:
        name = tool_call.get("name")
        arguments = tool_call.get("arguments") or {}

    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    if not isinstance(name, str) or not name:
        raise ValueError(f"tool call missing name: {tool_call!r}")
    if not isinstance(arguments, dict):
        raise ValueError(f"tool call arguments must be an object: {tool_call!r}")
    return {"name": name, "arguments": arguments}


def render_tool_call(tool_call: dict[str, Any]) -> str:
    payload = tool_call_payload(tool_call)
    return (
        "<tool_call>"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "</tool_call>"
    )


def with_output_format(system_content: str) -> str:
    content = (system_content or "").strip()
    if "<tool_call>" in content:
        return content
    if content:
        return OUTPUT_FORMAT + "\n---\n\n" + content
    return OUTPUT_FORMAT


def render_tools(tools: Any) -> str:
    if tools is None:
        return "[]"
    if isinstance(tools, str):
        tools = json.loads(tools) if tools.strip() else []
    return json.dumps(tools, ensure_ascii=False, indent=2)


def system_with_tools(system_content: str, tools: Any) -> str:
    return with_output_format(
        f"{(system_content or '').strip()}\n\n"
        f"## Available Tools\n{render_tools(tools)}".strip()
    )


def assistant_content(
    message: dict[str, Any],
    *,
    include_thinking: bool,
    thinking_key: str,
) -> str:
    pieces: list[str] = []
    thinking = (message.get(thinking_key) or "").strip()
    content = (message.get("content") or "").strip()
    tool_calls = message.get("tool_calls") or []

    if include_thinking and thinking:
        pieces.append(f"<think>\n{thinking}\n</think>")
    if content:
        pieces.append(content)
    for call in tool_calls:
        pieces.append(render_tool_call(call))
    return "\n\n".join(piece for piece in pieces if piece).strip()


def assistant_shape_error(message: dict[str, Any]) -> str | None:
    tool_calls = message.get("tool_calls") or []
    content = (message.get("content") or "").strip()
    if tool_calls and content:
        return "content_and_tool"
    return None


def assistant_multi_tool_turns(message: dict[str, Any]) -> int:
    return int(len(message.get("tool_calls") or []) > 1)


def areal_domain(metadata: dict[str, Any], messages: list[dict[str, str]]) -> str:
    joined = " ".join(str(value) for value in metadata.values()).lower()
    joined += " " + " ".join(
        message["content"] for message in messages if message["role"] == "system"
    ).lower()
    for domain in ("airline", "retail", "telecom"):
        if domain in joined:
            return domain
    return "unknown"


def areal_shape_error(row: dict[str, Any]) -> str | None:
    for message in row.get("messages") or []:
        if message.get("role") == "assistant":
            error = assistant_shape_error(message)
            if error:
                return error
    return assistant_shape_error(row.get("answer") or {})


def areal_multi_tool_turns(row: dict[str, Any]) -> int:
    count = 0
    for message in row.get("messages") or []:
        if message.get("role") == "assistant":
            count += assistant_multi_tool_turns(message)
    count += assistant_multi_tool_turns(row.get("answer") or {})
    return count


def convert_areal_row(
    row: dict[str, Any],
    *,
    include_thinking: bool,
    strict: bool,
    row_index: int,
    stats: Counter,
) -> dict[str, Any] | None:
    messages: list[dict[str, str]] = []
    drop_reason: str | None = None

    for message in row.get("messages") or []:
        role = message.get("role")
        if role == "system":
            messages.append(
                {
                    "role": "system",
                    "content": with_output_format(message.get("content") or ""),
                }
            )
        elif role == "user":
            messages.append({"role": "user", "content": message.get("content") or ""})
        elif role == "tool":
            payload = (message.get("content") or "").strip() or "[no_observation]"
            messages.append({"role": "user", "content": f"Tool result:\n{payload}"})
        elif role == "assistant":
            drop_reason = drop_reason or assistant_shape_error(message)
            content = assistant_content(
                message,
                include_thinking=include_thinking,
                thinking_key="reasoning",
            )
            if content:
                messages.append({"role": "assistant", "content": content})
        else:
            stats[f"areal_unsupported_role_{role}"] += 1
            return None

    answer = row.get("answer") or {}
    drop_reason = drop_reason or assistant_shape_error(answer)
    answer_content = assistant_content(answer, include_thinking=include_thinking, thinking_key="thinking")
    if not answer_content:
        stats["areal_empty_answer"] += 1
        return None
    if strict and drop_reason:
        return None

    messages.append({"role": "assistant", "content": answer_content})
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "source_dataset": "AReaL-tau2-data",
            "source_row": row_index,
            "domain": areal_domain(metadata, messages),
            "strict": strict,
            "thinking": include_thinking,
        }
    )
    return {"messages": messages, "metadata": metadata}


def convert_apigen_row(
    row: dict[str, Any],
    *,
    include_thinking: bool,
    row_index: int,
    stats: Counter,
) -> dict[str, Any] | None:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_with_tools(row.get("system") or "", row.get("tools"))}
    ]

    for turn in row.get("conversations") or []:
        role = turn.get("from")
        value = turn.get("value") or ""
        if role == "human":
            messages.append({"role": "user", "content": value})
        elif role == "gpt":
            content = value.strip()
            if include_thinking:
                content = content
            if content:
                messages.append({"role": "assistant", "content": content})
        elif role == "function_call":
            try:
                messages.append({"role": "assistant", "content": render_tool_call(json.loads(value))})
            except Exception:
                stats["apigen_bad_function_call"] += 1
                return None
        elif role == "observation":
            payload = value.strip() or "[no_observation]"
            messages.append({"role": "user", "content": f"Tool result:\n{payload}"})
        else:
            stats[f"apigen_unsupported_role_{role}"] += 1
            return None

    if not any(message["role"] == "assistant" for message in messages):
        stats["apigen_no_assistant"] += 1
        return None
    return {
        "messages": messages,
        "metadata": {
            "source_dataset": "APIGen-MT-5k",
            "source_row": row_index,
            "thinking": include_thinking,
        },
    }


def validate_row(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return "missing_messages"
    if not any(message.get("role") == "assistant" for message in messages):
        return "missing_assistant"
    for index, message in enumerate(messages):
        if message.get("role") not in {"system", "user", "assistant"}:
            return f"bad_role_{message.get('role')}"
        if not isinstance(message.get("content"), str):
            return "non_string_content"
        if message.get("role") == "system" and index != 0:
            return "system_not_first"
        if message.get("role") == "assistant" and not message.get("content").strip():
            return "empty_assistant"
    return None


def validate_rows(rows: list[dict[str, Any]], stats: Counter, prefix: str) -> None:
    for row in rows:
        error = validate_row(row)
        if error:
            stats[f"{prefix}_validation_{error}"] += 1


def convert_areal(path: Path, output_dir: Path, limit: int | None, stats: Counter) -> dict[str, Path]:
    rows_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_count = 0
    for row_index, row in enumerate(read_jsonl(path, limit=limit)):
        source_count += 1
        drop_reason = areal_shape_error(row)
        if drop_reason:
            stats[f"areal_strict_drop_{drop_reason}"] += 1
        stats["areal_multi_tool_turns"] += areal_multi_tool_turns(row)
        for strict in (True, False):
            for include_thinking in (False, True):
                converted = convert_areal_row(
                    row,
                    include_thinking=include_thinking,
                    strict=strict,
                    row_index=row_index,
                    stats=stats,
                )
                if converted is None:
                    continue
                mode = "strict" if strict else "loose"
                thinking = "with_thinking" if include_thinking else "no_thinking"
                rows_by_name[f"areal_tau2_sft_{mode}_{thinking}.jsonl"].append(converted)

    stats["areal_source_rows"] = source_count
    paths: dict[str, Path] = {}
    for name, rows in rows_by_name.items():
        validate_rows(rows, stats, name)
        path = output_dir / name
        write_jsonl(path, rows)
        stats[f"{name}_rows"] = len(rows)
        paths[name] = path
    return paths


def convert_apigen(path: Path, output_dir: Path, limit: int | None, stats: Counter) -> dict[str, Path]:
    source_rows = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        source_rows = source_rows[:limit]

    paths: dict[str, Path] = {}
    for include_thinking in (False, True):
        rows = []
        for row_index, row in enumerate(source_rows):
            converted = convert_apigen_row(
                row,
                include_thinking=include_thinking,
                row_index=row_index,
                stats=stats,
            )
            if converted is not None:
                rows.append(converted)
        thinking = "with_thinking" if include_thinking else "no_thinking"
        name = f"apigen_mt_5k_{thinking}.jsonl"
        validate_rows(rows, stats, name)
        out_path = output_dir / name
        write_jsonl(out_path, rows)
        stats[f"{name}_rows"] = len(rows)
        paths[name] = out_path

    stats["apigen_source_rows"] = len(source_rows)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apigen-path", type=Path, default=DEFAULT_APIGEN_PATH)
    parser.add_argument("--areal-path", type=Path, default=DEFAULT_AREAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per source for smoke tests.")
    parser.add_argument("--skip-apigen", action="store_true")
    parser.add_argument("--skip-areal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    outputs: dict[str, str] = {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_areal:
        for name, path in convert_areal(args.areal_path, args.output_dir, args.limit, stats).items():
            outputs[name] = str(path)
    if not args.skip_apigen:
        for name, path in convert_apigen(args.apigen_path, args.output_dir, args.limit, stats).items():
            outputs[name] = str(path)

    stats_path = args.output_dir / "stats.json"
    payload = {"outputs": outputs, "stats": dict(sorted(stats.items()))}
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
