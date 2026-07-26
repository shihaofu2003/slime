#!/usr/bin/env python3
"""Prepare user-model SFT data for tau2-bench.

The output JSONL uses slime SFT's `messages` input, with the simulated user as
the assistant role because `sft_rollout` masks assistant turns. This mirrors
tau2's `UserState.flip_roles()` behavior: agent text becomes user input, and
user utterances or user tool calls become assistant targets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
DEFAULT_APIGEN_PATH = SERVICE_AGENT_ROOT / "datasets/APIGen-MT-5k/apigen-mt_5k.json"
DEFAULT_AREAL_PATH = SERVICE_AGENT_ROOT / "datasets/AReaL-tau2-data/tau2_sft_train.jsonl"
DEFAULT_OUTPUT_DIR = SERVICE_AGENT_ROOT / "datasets/tau2-bench-user-sft"
DEFAULT_USER_GUIDELINES_PATH = (
    SERVICE_AGENT_ROOT
    / "tau2-bench/data/tau2/user_simulator/simulation_guidelines.md"
)
DEFAULT_USER_GUIDELINES_TOOLS_PATH = (
    SERVICE_AGENT_ROOT
    / "tau2-bench/data/tau2/user_simulator/simulation_guidelines_tools.md"
)

STOP_TOKEN = "###STOP###"


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


def read_user_guidelines(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


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


def render_user_message(message: dict[str, Any]) -> str:
    pieces: list[str] = []
    content = (message.get("content") or "").strip()
    if content:
        pieces.append(content)
    for tool_call in message.get("tool_calls") or []:
        pieces.append(render_tool_call(tool_call))
    return "\n\n".join(pieces).strip()


def user_target_kind(message: dict[str, Any]) -> str:
    has_text = bool((message.get("content") or "").strip())
    has_tool = bool(message.get("tool_calls") or [])
    if has_text and has_tool:
        return "content_and_tool"
    if has_tool:
        return "tool"
    if has_text:
        return "text"
    return "empty"


def tool_result_message(content: str) -> dict[str, str]:
    payload = content.strip() or "[no_observation]"
    return {"role": "user", "content": f"Tool result:\n{payload}"}


def areal_domain(metadata: dict[str, Any]) -> str:
    joined = " ".join(str(value) for value in metadata.values()).lower()
    for domain in ("airline", "retail", "telecom"):
        if domain in joined:
            return domain
    telecom_markers = ("mms_issue", "data_usage", "apn", "roaming", "sim_card")
    if any(marker in joined for marker in telecom_markers):
        return "telecom"
    return "unknown"


def areal_scenario(metadata: dict[str, Any]) -> str:
    reason = (metadata.get("reason_for_call") or "").strip()
    if reason:
        return reason
    return "Continue as the customer in this service conversation."


def guidelines_for_domain(domain: str, guidelines: dict[str, str]) -> str:
    """Pick the user-simulator guidelines that tau2 uses at eval time.

    telecom is the only tau2 domain whose environment exposes user tools, so its
    user simulator runs with the tools-aware guidelines; airline/retail use the
    no-tools guidelines. Mirrors ``UserSimulator.global_simulation_guidelines``
    (``use_tools = self.tools is not None``).
    """
    if domain == "telecom":
        return guidelines["tools"]
    return guidelines["no_tools"]


def system_prompt(guidelines: str, scenario: str) -> str:
    guideline_text = guidelines.replace("<PERSONA_GUIDELINES>", "").strip()
    pieces = [guideline_text, f"<scenario>\n{scenario.strip()}\n</scenario>"]
    return "\n\n".join(piece for piece in pieces if piece).strip()


def append_tool_result_if_for_user(
    state: list[dict[str, str]],
    message: dict[str, Any],
    pending_requestor: str | None,
    pending_count: int,
) -> tuple[str | None, int]:
    if pending_requestor == "user":
        state.append(tool_result_message(message.get("content") or ""))
    if pending_count > 0:
        pending_count -= 1
    if pending_count == 0:
        pending_requestor = None
    return pending_requestor, pending_count


def areal_terminal_candidate_key(row: dict[str, Any], row_index: int) -> tuple[int, int, int]:
    metadata = dict(row.get("metadata") or {})
    turn_index = metadata.get("turn_index")
    if not isinstance(turn_index, int):
        turn_index = -1
    return (len(row.get("messages") or []), turn_index, row_index)


def is_successful_areal_row(row: dict[str, Any]) -> bool:
    metadata = dict(row.get("metadata") or {})
    correct = metadata.get("correct")
    reward = metadata.get("reward")
    return correct == 1 or correct is True or reward == 1.0


def convert_areal_terminal_stop_row(
    row: dict[str, Any],
    *,
    row_index: int,
    guidelines: dict[str, str],
    seen: set[str],
    stats: Counter,
) -> dict[str, Any] | None:
    if not is_successful_areal_row(row):
        stats["areal_terminal_stop_skip_unsuccessful"] += 1
        return None

    answer = dict(row.get("answer") or {})
    answer_content = (answer.get("content") or "").strip()
    if not answer_content:
        stats["areal_terminal_stop_skip_empty_answer"] += 1
        return None
    if answer.get("tool_calls"):
        stats["areal_terminal_stop_skip_tool_answer"] += 1
        return None

    metadata = dict(row.get("metadata") or {})
    source_dialog_id = metadata.get("source_dialog_id") or f"row_{row_index}"
    domain = areal_domain(metadata)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt(
                guidelines_for_domain(domain, guidelines), areal_scenario(metadata)
            ),
        }
    ]
    pending_requestor: str | None = None
    pending_count = 0

    for message in [*(row.get("messages") or []), answer]:
        role = message.get("role")
        if role == "system":
            continue
        if role == "assistant":
            content = (message.get("content") or "").strip()
            if content:
                messages.append({"role": "user", "content": content})
            tool_calls = message.get("tool_calls") or []
            pending_requestor = "assistant" if tool_calls else None
            pending_count = len(tool_calls)
            continue
        if role == "tool":
            pending_requestor, pending_count = append_tool_result_if_for_user(
                messages,
                message,
                pending_requestor,
                pending_count,
            )
            continue
        if role != "user":
            stats[f"areal_terminal_stop_unsupported_role_{role}"] += 1
            continue

        target = render_user_message(message)
        if not target:
            stats["areal_terminal_stop_empty_user_message"] += 1
            continue
        messages.append({"role": "assistant", "content": target})
        tool_calls = message.get("tool_calls") or []
        pending_requestor = "user" if tool_calls else None
        pending_count = len(tool_calls)

    row_messages = messages + [{"role": "assistant", "content": STOP_TOKEN}]
    dedup_key = json.dumps(row_messages, ensure_ascii=False, separators=(",", ":"))
    if dedup_key in seen:
        stats["areal_duplicate_terminal_stop"] += 1
        return None
    seen.add(dedup_key)
    stats["areal_target_stop"] += 1

    row_metadata = {
        "source_dataset": "AReaL-tau2-data",
        "source_row": row_index,
        "source_dialog_id": source_dialog_id,
        "source_message_index": "terminal",
        "domain": domain,
        "target_kind": "stop",
        "thinking": False,
    }
    row_metadata.update(
        {
            key: metadata[key]
            for key in ("correct", "reward", "difficulty", "num_subtasks")
            if key in metadata
        }
    )
    return {"messages": row_messages, "metadata": row_metadata}


def convert_areal_row(
    row: dict[str, Any],
    *,
    row_index: int,
    guidelines: dict[str, str],
    seen: set[str],
    stats: Counter,
) -> list[dict[str, Any]]:
    metadata = dict(row.get("metadata") or {})
    source_dialog_id = metadata.get("source_dialog_id") or f"row_{row_index}"
    domain = areal_domain(metadata)
    scenario = areal_scenario(metadata)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt(guidelines_for_domain(domain, guidelines), scenario)}
    ]
    rows: list[dict[str, Any]] = []
    pending_requestor: str | None = None
    pending_count = 0

    for message_index, message in enumerate(row.get("messages") or []):
        role = message.get("role")
        if role == "system":
            continue
        if role == "assistant":
            content = (message.get("content") or "").strip()
            if content:
                messages.append({"role": "user", "content": content})
            tool_calls = message.get("tool_calls") or []
            pending_requestor = "assistant" if tool_calls else None
            pending_count = len(tool_calls)
            continue
        if role == "tool":
            pending_requestor, pending_count = append_tool_result_if_for_user(
                messages,
                message,
                pending_requestor,
                pending_count,
            )
            continue
        if role != "user":
            stats[f"areal_unsupported_role_{role}"] += 1
            continue

        target = render_user_message(message)
        kind = user_target_kind(message)
        if not target:
            stats["areal_empty_user_target"] += 1
            continue

        if kind == "content_and_tool":
            # tau2's user protocol forbids mixing a message and a tool call in
            # one turn, and these are the turns the sglang parser most often
            # fails to extract. Drop the training target but keep the rendered
            # turn in history so later turns stay well-conditioned.
            stats["areal_skip_content_and_tool"] += 1
        else:
            row_messages = messages + [{"role": "assistant", "content": target}]
            dedup_key = json.dumps(row_messages, ensure_ascii=False, separators=(",", ":"))
            if dedup_key not in seen:
                seen.add(dedup_key)
                row_metadata = {
                    "source_dataset": "AReaL-tau2-data",
                    "source_row": row_index,
                    "source_dialog_id": source_dialog_id,
                    "source_message_index": message_index,
                    "domain": domain,
                    "target_kind": kind,
                    "thinking": False,
                }
                row_metadata.update(
                    {
                        key: metadata[key]
                        for key in ("correct", "reward", "difficulty", "num_subtasks")
                        if key in metadata
                    }
                )
                rows.append({"messages": row_messages, "metadata": row_metadata})
                stats[f"areal_target_{kind}"] += 1
            else:
                stats["areal_duplicate_user_turn"] += 1

        messages.append({"role": "assistant", "content": target})
        tool_calls = message.get("tool_calls") or []
        pending_requestor = "user" if tool_calls else None
        pending_count = len(tool_calls)

    return rows


def convert_apigen_row(
    row: dict[str, Any],
    *,
    row_index: int,
    guidelines: dict[str, str],
    seen: set[str],
    stats: Counter,
) -> list[dict[str, Any]]:
    scenario = (
        "No hidden scenario was provided for this generic conversation. "
        "Continue naturally and stay consistent with the prior user messages."
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt(guidelines["no_tools"], scenario)}
    ]
    rows: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(row.get("conversations") or []):
        role = turn.get("from")
        value = (turn.get("value") or "").strip()
        if role == "human":
            if not value:
                stats["apigen_empty_human_target"] += 1
                continue
            if "<tool_call>" in value or '"function"' in value or '"type":"function"' in value:
                stats["apigen_skip_tool_markup_human_target"] += 1
                continue
            row_messages = messages + [{"role": "assistant", "content": value}]
            dedup_key = json.dumps(row_messages, ensure_ascii=False, separators=(",", ":"))
            if dedup_key not in seen:
                seen.add(dedup_key)
                rows.append(
                    {
                        "messages": row_messages,
                        "metadata": {
                            "source_dataset": "APIGen-MT-5k",
                            "source_row": row_index,
                            "source_turn_index": turn_index,
                            "domain": "unknown",
                            "target_kind": "text",
                            "thinking": False,
                        },
                    }
                )
                stats["apigen_target_text"] += 1
            else:
                stats["apigen_duplicate_user_turn"] += 1
            messages.append({"role": "assistant", "content": value})
        elif role == "gpt":
            if value:
                messages.append({"role": "user", "content": value})
        elif role in {"function_call", "observation"}:
            continue
        else:
            stats[f"apigen_unsupported_role_{role}"] += 1

    return rows


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


def convert_areal(
    path: Path,
    output_dir: Path,
    limit: int | None,
    guidelines: dict[str, str],
    stats: Counter,
) -> tuple[Path, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    terminal_candidates: dict[str, tuple[tuple[int, int, int], int, dict[str, Any]]] = {}
    source_count = 0
    for row_index, row in enumerate(read_jsonl(path, limit=limit)):
        source_count += 1
        metadata = dict(row.get("metadata") or {})
        source_dialog_id = metadata.get("source_dialog_id") or f"row_{row_index}"
        candidate_key = areal_terminal_candidate_key(row, row_index)
        if (
            source_dialog_id not in terminal_candidates
            or candidate_key > terminal_candidates[source_dialog_id][0]
        ):
            terminal_candidates[source_dialog_id] = (candidate_key, row_index, row)
        rows.extend(
            convert_areal_row(
                row,
                row_index=row_index,
                guidelines=guidelines,
                seen=seen,
                stats=stats,
            )
        )

    if limit is None:
        for _, row_index, row in sorted(
            terminal_candidates.values(), key=lambda item: item[0]
        ):
            terminal_row = convert_areal_terminal_stop_row(
                row,
                row_index=row_index,
                guidelines=guidelines,
                seen=seen,
                stats=stats,
            )
            if terminal_row is not None:
                rows.append(terminal_row)
    else:
        stats["areal_terminal_stop_skipped_for_limited_input"] = len(
            terminal_candidates
        )

    name = "areal_tau2_user_sft_no_thinking.jsonl"
    validate_rows(rows, stats, name)
    out_path = output_dir / name
    write_jsonl(out_path, rows)
    stats["areal_source_rows"] = source_count
    stats["areal_source_dialogs"] = len(terminal_candidates)
    stats[f"{name}_rows"] = len(rows)
    return out_path, rows


def convert_apigen(
    path: Path,
    output_dir: Path,
    limit: int | None,
    guidelines: dict[str, str],
    stats: Counter,
) -> tuple[Path, list[dict[str, Any]]]:
    source_rows = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        source_rows = source_rows[:limit]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, row in enumerate(source_rows):
        rows.extend(
            convert_apigen_row(
                row,
                row_index=row_index,
                guidelines=guidelines,
                seen=seen,
                stats=stats,
            )
        )

    name = "apigen_mt_5k_user_sft_no_thinking.jsonl"
    validate_rows(rows, stats, name)
    out_path = output_dir / name
    write_jsonl(out_path, rows)
    stats["apigen_source_rows"] = len(source_rows)
    stats[f"{name}_rows"] = len(rows)
    return out_path, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apigen-path", type=Path, default=DEFAULT_APIGEN_PATH)
    parser.add_argument("--areal-path", type=Path, default=DEFAULT_AREAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--user-guidelines-path", type=Path, default=DEFAULT_USER_GUIDELINES_PATH)
    parser.add_argument(
        "--user-guidelines-tools-path",
        type=Path,
        default=DEFAULT_USER_GUIDELINES_TOOLS_PATH,
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per source for smoke tests.")
    parser.add_argument("--skip-apigen", action="store_true")
    parser.add_argument("--skip-areal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    outputs: dict[str, str] = {}
    mixed_rows: list[dict[str, Any]] = []
    guidelines = {
        "no_tools": read_user_guidelines(args.user_guidelines_path),
        "tools": read_user_guidelines(args.user_guidelines_tools_path),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_areal:
        path, rows = convert_areal(args.areal_path, args.output_dir, args.limit, guidelines, stats)
        outputs[path.name] = str(path)
        mixed_rows.extend(rows)
    if not args.skip_apigen:
        path, rows = convert_apigen(args.apigen_path, args.output_dir, args.limit, guidelines, stats)
        outputs[path.name] = str(path)
        mixed_rows.extend(rows)

    if mixed_rows:
        name = "mixed_user_sft_no_thinking.jsonl"
        mixed_path = args.output_dir / name
        validate_rows(mixed_rows, stats, name)
        write_jsonl(mixed_path, mixed_rows)
        outputs[name] = str(mixed_path)
        stats[f"{name}_rows"] = len(mixed_rows)

    stats_path = args.output_dir / "stats.json"
    payload = {"outputs": outputs, "stats": dict(sorted(stats.items()))}
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
