#!/usr/bin/env python3
"""Unified Agent/User tool-namespace accounting for tau2 trajectories."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


_TOOL_CALL_START_RE = re.compile(r"<tool_call(?:\s[^>]*)?>", re.IGNORECASE)
_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"(?P<name>[^"\\]+)"')
_FUNCTION_NAME_RE = re.compile(r"<function=(?P<name>[^>\s]+)>")


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _as_mapping(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    raise TypeError(f"trajectory is not serializable: {type(item).__name__}")


def raw_tool_names(text: str) -> list[str]:
    """Return attributable tool attempts from the model's unparsed text."""

    names: list[str] = []
    text = text or ""
    starts = list(_TOOL_CALL_START_RE.finditer(text))
    for index, match in enumerate(starts):
        next_start = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        closing = text.find("</tool_call>", match.end(), next_start)
        body_end = closing if closing >= 0 else next_start
        body = text[match.end() : body_end]
        name_match = _JSON_NAME_RE.search(body) or _FUNCTION_NAME_RE.search(body)
        if name_match:
            names.append(name_match.group("name"))
    return names


def _structured_calls(messages: Iterable[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for turn_index, message in enumerate(messages):
        if _value(message, "role") != "assistant":
            continue
        for call in _value(message, "tool_calls") or []:
            function = _value(call, "function")
            name = _value(function, "name") if function is not None else _value(call, "name")
            if not isinstance(name, str) or not name:
                continue
            calls.append(
                {
                    "id": str(_value(call, "id") or ""),
                    "name": name,
                    "turn_index": turn_index,
                }
            )
    return calls


def _raw_attempts(messages: Iterable[Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for turn_index, message in enumerate(messages):
        if _value(message, "role") != "assistant":
            continue
        raw_data = _value(message, "raw_data") or {}
        text = _value(raw_data, "text") if raw_data else None
        names = raw_tool_names(text) if isinstance(text, str) else []
        if not names:
            # Historical artifacts do not always retain raw model text.  The
            # structured call is then the narrowest available attempt count.
            names = [call["name"] for call in _structured_calls([message])]
        attempts.extend({"name": name, "turn_index": turn_index} for name in names)
    return attempts


def analyze_namespace_trajectory(
    trajectory: Any,
    *,
    user_tool_names: set[str],
) -> dict[str, Any]:
    record = _as_mapping(trajectory)
    simulation = record.get("simulation")
    if simulation is not None:
        record = _as_mapping(simulation)
    messages = list(record.get("messages") or [])

    raw_attempts = [
        item for item in _raw_attempts(messages) if item["name"] in user_tool_names
    ]
    parsed_calls = [
        item for item in _structured_calls(messages) if item["name"] in user_tool_names
    ]
    parsed_ids = {item["id"] for item in parsed_calls if item["id"]}
    result_ids = {
        str(_value(message, "tool_call_id") or _value(message, "id") or "")
        for message in messages
        if _value(message, "role") == "tool"
        and (_value(message, "requestor", "assistant") or "assistant") == "assistant"
    }
    executed_calls = [item for item in parsed_calls if item["id"] in result_ids]
    termination_reason = record.get("termination_reason")
    affected = bool(raw_attempts or parsed_calls)
    return {
        "task_id": record.get("task_id") or record.get("id"),
        "raw_attempt_count": len(raw_attempts),
        "parsed_call_count": len(parsed_calls),
        "executed_call_count": len(executed_calls),
        "raw_names": dict(sorted(Counter(item["name"] for item in raw_attempts).items())),
        "parsed_names": dict(sorted(Counter(item["name"] for item in parsed_calls).items())),
        "executed_names": dict(sorted(Counter(item["name"] for item in executed_calls).items())),
        "affected_turns": sorted(
            {item["turn_index"] for item in [*raw_attempts, *parsed_calls]}
        ),
        "affected": affected,
        "termination_reason": termination_reason,
        "namespace_attributed_termination": bool(
            affected and termination_reason == "too_many_errors"
        ),
        "unpaired_parsed_call_ids": sorted(parsed_ids - result_ids),
    }


def analyze_namespace_trajectories(
    trajectories: Iterable[Any],
    *,
    user_tool_names: set[str],
) -> dict[str, Any]:
    # ``tau2.runner.run_domain`` returns a pydantic ``Results`` object.  Iterating
    # that object directly yields ``(field_name, value)`` tuples, not simulation
    # records.  Accept the live object and its persisted JSON shape at the same
    # boundary so callers cannot accidentally analyze the container itself.
    if isinstance(trajectories, Mapping):
        simulations = trajectories.get("simulations")
    else:
        simulations = getattr(trajectories, "simulations", None)
    if simulations is not None:
        trajectories = simulations
    rows = [
        analyze_namespace_trajectory(item, user_tool_names=user_tool_names)
        for item in trajectories
    ]
    affected = [row for row in rows if row["affected"]]
    raw_names: Counter[str] = Counter()
    parsed_names: Counter[str] = Counter()
    executed_names: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    for row in rows:
        raw_names.update(row["raw_names"])
        parsed_names.update(row["parsed_names"])
        executed_names.update(row["executed_names"])
        if row["namespace_attributed_termination"]:
            terminations[str(row["termination_reason"])] += 1
    intensities = [row["raw_attempt_count"] for row in affected]
    return {
        "trajectory_count": len(rows),
        "affected_trajectory_count": len(affected),
        "raw_attempt_count": sum(row["raw_attempt_count"] for row in rows),
        "parsed_call_count": sum(row["parsed_call_count"] for row in rows),
        "executed_call_count": sum(row["executed_call_count"] for row in rows),
        "raw_names": dict(sorted(raw_names.items())),
        "parsed_names": dict(sorted(parsed_names.items())),
        "executed_names": dict(sorted(executed_names.items())),
        "attempts_per_affected_trajectory": {
            "mean": sum(intensities) / len(intensities) if intensities else 0.0,
            "max": max(intensities, default=0),
            "values": intensities,
        },
        "namespace_attributed_termination_count": sum(terminations.values()),
        "namespace_attributed_terminations": dict(sorted(terminations.items())),
        "trajectories": rows,
    }


def analyze_single_call_attempts(trajectories: Iterable[Any]) -> dict[str, Any]:
    """Count raw and parsed assistant turns that attempted multiple calls."""

    if isinstance(trajectories, Mapping):
        simulations = trajectories.get("simulations")
    else:
        simulations = getattr(trajectories, "simulations", None)
    if simulations is not None:
        trajectories = simulations

    trajectory_count = 0
    affected_trajectories = 0
    raw_multi_turns = 0
    raw_calls_in_multi_turns = 0
    parsed_multi_turns = 0
    protocol_error_turns = 0
    for trajectory in trajectories:
        trajectory_count += 1
        record = _as_mapping(trajectory)
        simulation = record.get("simulation")
        if simulation is not None:
            record = _as_mapping(simulation)
        affected = False
        for message in record.get("messages") or []:
            if _value(message, "role") != "assistant":
                continue
            structured_count = len(_value(message, "tool_calls") or [])
            raw_data = _value(message, "raw_data") or {}
            text = _value(raw_data, "text") if raw_data else None
            raw_count = (
                len(_TOOL_CALL_START_RE.findall(text))
                if isinstance(text, str)
                else structured_count
            )
            reported_count = int(
                _value(raw_data, "tau2_multi_tool_attempt_count", 0) or 0
            )
            raw_count = max(raw_count, reported_count)
            if raw_count > 1:
                raw_multi_turns += 1
                raw_calls_in_multi_turns += raw_count
                affected = True
            if structured_count > 1:
                parsed_multi_turns += 1
                affected = True
            if _value(raw_data, "tau2_single_call_protocol_error", False):
                protocol_error_turns += 1
        affected_trajectories += int(affected)
    return {
        "trajectory_count": trajectory_count,
        "trajectories_with_multi_call_output": affected_trajectories,
        "multi_call_output_turns": raw_multi_turns,
        "attempted_calls_in_multi_call_outputs": raw_calls_in_multi_turns,
        "parsed_multi_call_turns": parsed_multi_turns,
        "single_call_protocol_error_turns": protocol_error_turns,
    }


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("namespace analyzer input must be a JSON object or list")
    for key in ("simulations", "results", "trajectories"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if "messages" in payload or "simulation" in payload:
        return [payload]
    raise ValueError("could not locate trajectories in analyzer input")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tau2.registry import registry

    environment = registry.get_env_constructor(args.domain)()
    try:
        user_tool_names = {tool.name for tool in environment.get_user_tools()}
    except ValueError:
        user_tool_names = set()
    summary = analyze_namespace_trajectories(
        _load_trajectories(args.input),
        user_tool_names=user_tool_names,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
