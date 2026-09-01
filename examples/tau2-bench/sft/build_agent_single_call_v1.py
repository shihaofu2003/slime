#!/usr/bin/env python3
"""Expand the selected boundary-v2 SFT data into strict single-call targets."""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from protocol_profiles import strict_single_system_prompt  # noqa: E402


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
EXPERIMENT_DIR = (
    SERVICE_AGENT_ROOT / "slime/output/experiments/tau2-agent-single-call-v1"
)
DEFAULT_INPUT = (
    SERVICE_AGENT_ROOT
    / "slime/output/experiments/tau2-sft-agent-user-boundary-v2/contract_boundary_19318.jsonl"
)
DEFAULT_SOURCE = (
    SERVICE_AGENT_ROOT / "datasets/AReaL-tau2-data/tau2_sft_train.jsonl"
)
DEFAULT_TOKENIZER = SERVICE_AGENT_ROOT / "models/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "data/single_call_v1_25708.jsonl"
DEFAULT_SMOKE_OUTPUT = EXPERIMENT_DIR / "data/single_call_v1_longest32_smoke.jsonl"

DATASET_VERSION = "tau2-agent-single-call-sft-v1"
MAX_TOTAL_TOKENS = 16384
EXPECTED_EXPANDED = 25761
EXPECTED_DROPPED = 53
EXPECTED_OUTPUT = 25708

_POLICY_RE = re.compile(r"<policy>\s*(?P<policy>.*)\s*</policy>\s*$", re.S)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def _call_id(call: Mapping[str, Any], fallback: str) -> str:
    return str(call.get("id") or fallback)


def index_real_tool_results(
    source_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], str]:
    """Index observed Agent tool results by source dialog and original call id."""

    results: dict[tuple[str, str], str] = {}
    for row in source_rows:
        dialog_id = str((row.get("metadata") or {}).get("source_dialog_id"))
        messages = list(row.get("messages") or [])
        for message_index, message in enumerate(messages):
            calls = list(message.get("tool_calls") or [])
            if message.get("role") != "assistant" or not calls:
                continue
            pending = [
                _call_id(call, f"source_{message_index}_call_{call_index}")
                for call_index, call in enumerate(calls)
            ]
            result_index = message_index + 1
            while (
                result_index < len(messages)
                and messages[result_index].get("role") == "tool"
                and pending
            ):
                result = messages[result_index]
                result_id = result.get("tool_call_id") or result.get("id")
                if result_id is not None and str(result_id) in pending:
                    call_id = str(result_id)
                else:
                    call_id = pending[0]
                key = (dialog_id, call_id)
                content = str(result.get("content") or "")
                if key in results and results[key] != content:
                    raise ValueError(
                        f"conflicting real results for dialog={dialog_id!r}, call={call_id!r}"
                    )
                results[key] = content
                pending.remove(call_id)
                result_index += 1
    return results


def _clean_message(message: Mapping[str, Any], *, loss_mask: int) -> dict[str, Any]:
    role = str(message.get("role") or "")
    cleaned: dict[str, Any] = {
        "role": role,
        "content": str(message.get("content") or ""),
        "step_loss_mask": loss_mask,
    }
    if message.get("tool_calls"):
        cleaned["tool_calls"] = copy.deepcopy(list(message["tool_calls"]))
    if role == "tool":
        call_id = message.get("tool_call_id") or message.get("id")
        if not call_id:
            raise ValueError("Agent tool result is missing its call id")
        cleaned["tool_call_id"] = str(call_id)
    return cleaned


def _single_call_message(
    message: Mapping[str, Any],
    call: Mapping[str, Any],
    *,
    loss_mask: int,
) -> dict[str, Any]:
    if str(message.get("content") or "").strip():
        raise ValueError("Assistant message mixes text with tool calls")
    cleaned = _clean_message(message, loss_mask=loss_mask)
    cleaned["tool_calls"] = [copy.deepcopy(dict(call))]
    return cleaned


def split_prefix_multi_calls(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Split completed prefix batches into call/result/call/result order."""

    converted: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = list(message.get("tool_calls") or [])
        if message.get("role") != "assistant" or len(calls) <= 1:
            converted.append(_clean_message(message, loss_mask=0))
            index += 1
            continue

        results: dict[str, Mapping[str, Any]] = {}
        result_index = index + 1
        while (
            result_index < len(messages)
            and messages[result_index].get("role") == "tool"
        ):
            result = messages[result_index]
            call_id = result.get("tool_call_id") or result.get("id")
            if not call_id:
                raise ValueError("prefix tool result is missing its call id")
            results[str(call_id)] = result
            result_index += 1
        call_ids = [_call_id(call, f"prefix_{index}_call_{i}") for i, call in enumerate(calls)]
        if set(call_ids) != set(results):
            raise ValueError("prefix multi-call does not have one real result per call")
        for call, call_id in zip(calls, call_ids):
            converted.append(_single_call_message(message, call, loss_mask=0))
            converted.append(_clean_message(results[call_id], loss_mask=0))
        index = result_index
    return converted


def _policy_from_system_prompt(prompt: str) -> str:
    match = _POLICY_RE.search(prompt)
    if not match:
        raise ValueError("boundary row system prompt has no trailing <policy> block")
    return match.group("policy").strip()


def _clean_metadata(
    metadata: Mapping[str, Any],
    *,
    target_call_count: int,
    target_call_index: int | None,
) -> dict[str, Any]:
    cleaned = {
        "dataset_version": DATASET_VERSION,
        "domain": metadata.get("domain"),
        "source_dataset": metadata.get("source_dataset"),
        "source_dialog_id": metadata.get("source_dialog_id"),
        "source_row": metadata.get("source_row"),
        "turn_index": metadata.get("turn_index"),
        "target_type": "tool_call" if target_call_count else "text",
        "target_call_count": target_call_count,
    }
    if target_call_index is not None:
        cleaned["target_call_index"] = target_call_index
    return cleaned


def convert_boundary_row(
    row: Mapping[str, Any],
    *,
    real_tool_results: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    """Return one target-only row per original target call, in source order."""

    source_messages = list(row.get("messages") or [])
    if not source_messages or source_messages[-1].get("role") != "assistant":
        raise ValueError("boundary row must end with an Assistant target")
    if not isinstance(row.get("tools"), list) or not row["tools"]:
        raise ValueError("boundary row is missing native Agent tool schemas")

    prefix = split_prefix_multi_calls(source_messages[:-1])
    if not prefix or prefix[0].get("role") != "system":
        raise ValueError("boundary row is missing its leading system message")
    prefix[0]["content"] = strict_single_system_prompt(
        _policy_from_system_prompt(prefix[0]["content"])
    )

    target = source_messages[-1]
    target_calls = list(target.get("tool_calls") or [])
    metadata = row.get("metadata") or {}
    dialog_id = str(metadata.get("source_dialog_id"))
    converted: list[dict[str, Any]] = []

    if not target_calls:
        messages = copy.deepcopy(prefix)
        messages.append(_clean_message(target, loss_mask=1))
        converted.append(
            {
                "messages": messages,
                "tools": copy.deepcopy(row["tools"]),
                "metadata": _clean_metadata(
                    metadata,
                    target_call_count=0,
                    target_call_index=None,
                ),
            }
        )
        return converted

    for call_index, call in enumerate(target_calls):
        messages = copy.deepcopy(prefix)
        for previous_index, previous_call in enumerate(target_calls[:call_index]):
            previous_id = _call_id(
                previous_call,
                f"target_call_{previous_index}",
            )
            result_key = (dialog_id, previous_id)
            if result_key not in real_tool_results:
                raise ValueError(
                    f"missing real result for dialog={dialog_id!r}, call={previous_id!r}"
                )
            messages.append(
                _single_call_message(target, previous_call, loss_mask=0)
            )
            messages.append(
                {
                    "role": "tool",
                    "content": real_tool_results[result_key],
                    "tool_call_id": previous_id,
                    "step_loss_mask": 0,
                }
            )
        messages.append(_single_call_message(target, call, loss_mask=1))
        converted.append(
            {
                "messages": messages,
                "tools": copy.deepcopy(row["tools"]),
                "metadata": _clean_metadata(
                    metadata,
                    target_call_count=len(target_calls),
                    target_call_index=call_index,
                ),
            }
        )
    return converted


def validate_single_call_row(row: Mapping[str, Any]) -> None:
    messages = list(row.get("messages") or [])
    if not messages or messages[0].get("role") != "system":
        raise ValueError("single-call row is missing its system message")
    if messages[-1].get("role") != "assistant":
        raise ValueError("single-call row does not end with an Assistant target")
    for message in messages:
        if message.get("role") == "assistant" and len(message.get("tool_calls") or []) > 1:
            raise ValueError("single-call row retained a multi-call Assistant message")
    if messages[-1].get("step_loss_mask") != 1 or any(
        message.get("step_loss_mask") != 0 for message in messages[:-1]
    ):
        raise ValueError("single-call row changed target-only loss masking")
    metadata_text = json.dumps(row.get("metadata") or {}, ensure_ascii=False).lower()
    if "sha256" in metadata_text or "hash" in metadata_text or "signature" in metadata_text:
        raise ValueError("single-call metadata retained hash or signature fields")


def _token_count(row: Mapping[str, Any], tokenizer: Any) -> int:
    token_ids = tokenizer.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    elif hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    return len(token_ids)


def build(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=False,
    )
    real_tool_results = index_real_tool_results(read_jsonl(args.source))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_smoke = args.smoke_output.with_suffix(args.smoke_output.suffix + ".tmp")

    input_rows = 0
    expanded = 0
    dropped = 0
    kept = 0
    kept_by_domain: Counter[str] = Counter()
    dropped_by_domain: Counter[str] = Counter()
    target_types: Counter[str] = Counter()
    longest: list[tuple[int, int, dict[str, Any]]] = []
    sequence = 0

    with temporary_output.open("w", encoding="utf-8") as output_file:
        for input_rows, source_row in enumerate(read_jsonl(args.input), start=1):
            for converted in convert_boundary_row(
                source_row,
                real_tool_results=real_tool_results,
            ):
                expanded += 1
                validate_single_call_row(converted)
                total_tokens = _token_count(converted, tokenizer)
                domain = str(converted["metadata"].get("domain"))
                if total_tokens > args.max_tokens:
                    dropped += 1
                    dropped_by_domain[domain] += 1
                    continue
                output_file.write(
                    json.dumps(converted, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                kept += 1
                kept_by_domain[domain] += 1
                target_types[str(converted["metadata"]["target_type"])] += 1
                item = (total_tokens, sequence, converted)
                sequence += 1
                if len(longest) < args.smoke_size:
                    heapq.heappush(longest, item)
                elif item[:2] > longest[0][:2]:
                    heapq.heapreplace(longest, item)

    if expanded != args.expected_expanded:
        raise ValueError(
            f"expanded target count changed: {expanded} != {args.expected_expanded}"
        )
    if dropped != args.expected_dropped:
        raise ValueError(
            f"over-cap target count changed: {dropped} != {args.expected_dropped}"
        )
    if kept != args.expected_output:
        raise ValueError(f"output target count changed: {kept} != {args.expected_output}")

    with temporary_smoke.open("w", encoding="utf-8") as smoke_file:
        for _, _, row in sorted(longest, reverse=True):
            smoke_file.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary_output.replace(args.output)
    temporary_smoke.replace(args.smoke_output)

    summary = {
        "input_rows": input_rows,
        "expanded_targets": expanded,
        "dropped_over_token_cap": dropped,
        "output_rows": kept,
        "max_tokens": args.max_tokens,
        "output_rows_by_domain": dict(sorted(kept_by_domain.items())),
        "dropped_rows_by_domain": dict(sorted(dropped_by_domain.items())),
        "target_types": dict(sorted(target_types.items())),
        "longest_smoke_rows": len(longest),
        "longest_smoke_max_tokens": max((item[0] for item in longest), default=0),
        "output": str(args.output),
        "smoke_output": str(args.smoke_output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def validate(args: argparse.Namespace) -> None:
    rows = 0
    domains: Counter[str] = Counter()
    for rows, row in enumerate(read_jsonl(args.input), start=1):
        validate_single_call_row(row)
        domains[str((row.get("metadata") or {}).get("domain"))] += 1
    if rows != args.expected_rows:
        raise ValueError(f"row count changed: {rows} != {args.expected_rows}")
    print(json.dumps({"rows": rows, "rows_by_domain": dict(sorted(domains.items()))}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    build_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument(
        "--smoke-output",
        type=Path,
        default=DEFAULT_SMOKE_OUTPUT,
    )
    build_parser.add_argument("--max-tokens", type=int, default=MAX_TOTAL_TOKENS)
    build_parser.add_argument("--smoke-size", type=int, default=32)
    build_parser.add_argument(
        "--expected-expanded",
        type=int,
        default=EXPECTED_EXPANDED,
    )
    build_parser.add_argument(
        "--expected-dropped",
        type=int,
        default=EXPECTED_DROPPED,
    )
    build_parser.add_argument(
        "--expected-output",
        type=int,
        default=EXPECTED_OUTPUT,
    )
    build_parser.set_defaults(func=build)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    validate_parser.add_argument(
        "--expected-rows",
        type=int,
        default=EXPECTED_OUTPUT,
    )
    validate_parser.set_defaults(func=validate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
