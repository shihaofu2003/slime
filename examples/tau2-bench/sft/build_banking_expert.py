#!/usr/bin/env python3
"""Convert Banking synthetic expert trajectories into native Qwen SFT rows.

The saved Banking trajectories contain a complete conversation, including
User-owned tool calls and a terminal ``###STOP###`` message.  This converter
keeps the Agent view, expands every Agent Assistant turn into a target row,
and validates the result with the same native Qwen3 chat template used by the
processed AReaL SFT data.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
PROJECT_ROOT = SERVICE_AGENT_ROOT / "slime"
BANKING_EXPERIMENT_DIR = PROJECT_ROOT / "output/experiments/tau2-banking-independent-synthetic-v1"
FINAL_DIR = BANKING_EXPERIMENT_DIR / "final-minimal"

DEFAULT_INPUT = FINAL_DIR / "successful_trajectories.jsonl"
DEFAULT_TASKS = FINAL_DIR / "train_tasks.json"
DEFAULT_CONTRACTS = FINAL_DIR / "train_contracts.jsonl"
DEFAULT_ALLOWED_DOCUMENTS = BANKING_EXPERIMENT_DIR / "allowed_documents.json"
DEFAULT_DOCUMENTS_DIR = (
    SERVICE_AGENT_ROOT / "tau2-bench/data/tau2/domains/banking_knowledge/documents"
)
DEFAULT_TOKENIZER = SERVICE_AGENT_ROOT / "models/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/experiments/tau2-banking-expert-sft/data"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "banking_expert_sft.jsonl"
DEFAULT_STATS = DEFAULT_OUTPUT_DIR / "banking_expert_sft.stats.json"

MAX_TOTAL_TOKENS = 16_384
STOP_MARKER = "###STOP###"


class ConversionError(ValueError):
    """Raised when a trajectory cannot be represented as native Agent SFT."""


@dataclass(frozen=True)
class BankingRuntime:
    """Native prompt and Agent tool boundary for one Banking task."""

    system_prompt: str
    tools: tuple[dict[str, Any], ...]
    agent_tool_names: frozenset[str]
    retrieval_variant: str


def _copy_json(value: Any) -> Any:
    return copy.deepcopy(value)


def _json_object(value: Any, *, field: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"{field} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ConversionError(f"{field} must be a JSON object")
    return _copy_json(dict(value))


def _call_name(raw_call: Mapping[str, Any]) -> str:
    function = raw_call.get("function")
    function_name = function.get("name") if isinstance(function, Mapping) else None
    top_level_name = raw_call.get("name")
    if function_name and top_level_name and str(function_name) != str(top_level_name):
        raise ConversionError("tool call has conflicting names")
    name = function_name or top_level_name
    if not isinstance(name, str) or not name.strip():
        raise ConversionError("tool call is missing a name")
    return name.strip()


def _call_arguments(raw_call: Mapping[str, Any]) -> dict[str, Any]:
    function = raw_call.get("function")
    value = function.get("arguments") if isinstance(function, Mapping) else raw_call.get("arguments")
    return _json_object(value, field="tool-call arguments")


def _call_requestor(raw_call: Mapping[str, Any], default: str) -> str:
    requestor = raw_call.get("requestor") or default
    if requestor not in {"assistant", "user"}:
        raise ConversionError(f"unsupported tool-call requestor: {requestor!r}")
    return str(requestor)


def _source_call_id(raw_call: Mapping[str, Any], ordinal: int) -> str:
    value = raw_call.get("id")
    if value not in (None, ""):
        return str(value)
    return f"implicit_call_{ordinal}"


def _native_call(
    *,
    output_id: str,
    name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": output_id,
        "name": name,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                dict(arguments)
            ),
        },
    }


def _clean_user_content(content: Any) -> str | None:
    if not isinstance(content, str):
        return None
    # The marker is a simulator control message, not Agent-visible training
    # content.  Only remove it when it is a terminal suffix.
    cleaned = re.sub(rf"(?:\s*{re.escape(STOP_MARKER)}\s*)+$", "", content)
    cleaned = cleaned.strip()
    return cleaned or None


def _message(role: str, content: Any, *, loss_mask: int = 0) -> dict[str, Any]:
    return {"role": role, "content": content, "step_loss_mask": loss_mask}


def _result_identifier(message: Mapping[str, Any]) -> str | None:
    tool_call_id = message.get("tool_call_id")
    message_id = message.get("id")
    if tool_call_id not in (None, "") and message_id not in (None, ""):
        if str(tool_call_id) != str(message_id):
            raise ConversionError("tool result has conflicting ids")
    value = tool_call_id if tool_call_id not in (None, "") else message_id
    return str(value) if value not in (None, "") else None


def _result_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _consume_results(
    raw_messages: Sequence[Mapping[str, Any]],
    start: int,
    calls: Sequence[Mapping[str, Any]],
    *,
    requestor: str,
) -> tuple[list[str | None], int]:
    """Pair contiguous tool results with calls and return the next index."""

    results: list[Mapping[str, Any]] = []
    index = start
    while index < len(raw_messages) and raw_messages[index].get("role") == "tool":
        result = raw_messages[index]
        result_requestor = result.get("requestor") or requestor
        if result_requestor != requestor:
            raise ConversionError("tool result requestor does not match its call")
        results.append(result)
        index += 1

    if len(results) != len(calls):
        raise ConversionError(
            f"expected {len(calls)} {requestor} tool results, found {len(results)}"
        )

    source_ids = [_source_call_id(call, ordinal) for ordinal, call in enumerate(calls)]
    unmatched = list(range(len(calls)))
    paired: list[str | None] = [None] * len(calls)
    for result in results:
        result_id = _result_identifier(result)
        if result_id is None:
            if not unmatched:
                raise ConversionError("too many unpaired tool results")
            call_index = unmatched.pop(0)
        else:
            matches = [idx for idx in unmatched if source_ids[idx] == result_id]
            if len(matches) != 1:
                raise ConversionError("tool result does not match a preceding call")
            call_index = matches[0]
            unmatched.remove(call_index)
        paired[call_index] = _result_content(result)
    if unmatched:
        raise ConversionError("tool result pairing is incomplete")
    return paired, index


def _agent_tool_names(tools: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    names: set[str] = set()
    for schema in tools:
        function = schema.get("function") if isinstance(schema, Mapping) else None
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            raise ConversionError("invalid top-level Agent tool schema")
        names.add(str(function["name"]))
    return frozenset(names)


def _parse_agent_calls(
    raw_calls: Sequence[Any],
    *,
    agent_tool_names: frozenset[str],
    next_call_number: int,
) -> tuple[list[dict[str, Any]], int]:
    native_calls: list[dict[str, Any]] = []
    for offset, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, Mapping):
            raise ConversionError("tool call must be an object")
        requestor = _call_requestor(raw_call, "assistant")
        if requestor != "assistant":
            raise ConversionError("Assistant message contains a User-owned tool call")
        name = _call_name(raw_call)
        if name not in agent_tool_names:
            raise ConversionError(f"unknown Agent tool: {name}")
        arguments = _call_arguments(raw_call)
        native_calls.append(
            _native_call(
                output_id=f"call_{next_call_number + offset:04d}",
                name=name,
                arguments=arguments,
            )
        )
    return native_calls, next_call_number + len(raw_calls)


def _canonical_events(
    raw_messages: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str,
    agent_tool_names: frozenset[str],
) -> list[dict[str, Any]]:
    """Return the complete Agent-visible event stream.

    Internal events carry only the fields needed to expand targets.  They are
    converted to public SFT message dictionaries by ``expand_agent_targets``.
    """

    events: list[dict[str, Any]] = [_message("system", system_prompt)]
    next_call_number = 1
    index = 0
    while index < len(raw_messages):
        raw = raw_messages[index]
        if not isinstance(raw, Mapping):
            raise ConversionError("message must be an object")
        role = raw.get("role")
        calls = raw.get("tool_calls") or []
        if calls and not isinstance(calls, Sequence):
            raise ConversionError("tool_calls must be a sequence")

        if role == "system":
            index += 1
            continue

        if role == "user":
            if calls:
                # User-owned execution is hidden from the Agent.  Consume its
                # result messages so they cannot leak into a later prefix.
                for raw_call in calls:
                    if not isinstance(raw_call, Mapping):
                        raise ConversionError("User tool call must be an object")
                    if _call_requestor(raw_call, "user") != "user":
                        raise ConversionError("User message contains an Agent-owned call")
                _, index = _consume_results(
                    raw_messages, index + 1, calls, requestor="user"
                )
                continue
            content = _clean_user_content(raw.get("content"))
            if content is not None:
                events.append(_message("user", content))
            index += 1
            continue

        if role == "assistant":
            if calls:
                native_calls, next_call_number = _parse_agent_calls(
                    list(calls),
                    agent_tool_names=agent_tool_names,
                    next_call_number=next_call_number,
                )
                result_contents, index = _consume_results(
                    raw_messages, index + 1, calls, requestor="assistant"
                )
                call_count = len(native_calls)
                if len(result_contents) != call_count:
                    raise ConversionError("tool-call/result counts are misaligned")
                for call_index, (native_call, result_content) in enumerate(
                    zip(native_calls, result_contents)
                ):
                    events.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [native_call],
                            "_target_call_index": call_index,
                            "_target_call_count": call_count,
                        }
                    )
                    if result_content is not None:
                        events.append(
                            {
                                "role": "tool",
                                "content": result_content,
                                "tool_call_id": native_call["id"],
                            }
                        )
                # ``index`` already points after the contiguous results.
                continue

            content = raw.get("content")
            if isinstance(content, str) and content.strip():
                # Native Qwen messages cannot mix prose and structured calls;
                # call turns above intentionally discard whitespace/prose from
                # the simulator's mixed response.
                events.append(_message("assistant", content.strip()))
            index += 1
            continue

        if role == "tool":
            raise ConversionError("unpaired tool result")
        raise ConversionError(f"unsupported message role: {role!r}")

    return events


def _public_message(message: Mapping[str, Any], *, loss_mask: int) -> dict[str, Any]:
    result = _copy_json(dict(message))
    result.pop("_target_call_index", None)
    result.pop("_target_call_count", None)
    result["step_loss_mask"] = loss_mask
    return result


def _validate_row_shape(row: Mapping[str, Any]) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ConversionError("converted row has no messages")
    if messages[0].get("role") != "system":
        raise ConversionError("converted row does not start with system")
    if messages[-1].get("role") != "assistant":
        raise ConversionError("converted row target is not Assistant")
    masks = [message.get("step_loss_mask") for message in messages]
    if masks[-1] != 1 or any(mask != 0 for mask in masks[:-1]):
        raise ConversionError("converted row does not use target-only loss masking")
    if not isinstance(row.get("tools"), list) or not row["tools"]:
        raise ConversionError("converted row has no Agent tools")

    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role in {"system", "user"}:
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ConversionError(f"empty {role} message")
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                if message.get("content") is not None or len(calls) != 1:
                    raise ConversionError("Assistant target must contain one call or text")
                call = calls[0]
                function = call.get("function") if isinstance(call, Mapping) else None
                if not isinstance(function, Mapping):
                    raise ConversionError("invalid native Agent tool call")
                if call.get("id") in seen_calls:
                    raise ConversionError("duplicate native tool-call id")
                seen_calls.add(str(call["id"]))
                try:
                    _json_object(function.get("arguments"), field="native tool arguments")
                except ConversionError:
                    raise
            elif not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ConversionError("empty Assistant target")
        elif role == "tool":
            result_id = message.get("tool_call_id")
            if not result_id or result_id not in seen_calls or result_id in seen_results:
                raise ConversionError("tool result is not paired with one preceding call")
            if not isinstance(message.get("content"), str):
                raise ConversionError("tool result content must be a string")
            seen_results.add(str(result_id))
        else:
            raise ConversionError(f"unsupported output role: {role!r}")


def expand_agent_targets(
    raw_messages: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str,
    tools: Sequence[Mapping[str, Any]],
    agent_tool_names: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand a complete expert trajectory into target-only SFT rows."""

    tool_list = [_copy_json(dict(tool)) for tool in tools]
    allowed_names = (
        frozenset(str(name) for name in agent_tool_names)
        if agent_tool_names is not None
        else _agent_tool_names(tool_list)
    )
    events = _canonical_events(
        raw_messages,
        system_prompt=system_prompt,
        agent_tool_names=allowed_names,
    )
    assistant_indices = [
        index for index, message in enumerate(events) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ConversionError("trajectory has no Agent Assistant targets")

    base_metadata = _copy_json(dict(metadata or {}))
    rows: list[dict[str, Any]] = []
    for target_number, target_index in enumerate(assistant_indices):
        target = events[target_index]
        target_messages = [
            _public_message(message, loss_mask=0) for message in events[: target_index + 1]
        ]
        target_messages[-1]["step_loss_mask"] = 1
        target_metadata = _copy_json(base_metadata)
        target_metadata.update(
            {
                "target_index": target_number,
                "target_type": "tool_call" if target.get("tool_calls") else "text",
            }
        )
        if target.get("tool_calls") and target.get("_target_call_count", 0) > 1:
            target_metadata.update(
                {
                    "target_call_index": target.get("_target_call_index"),
                    "target_call_count": target.get("_target_call_count"),
                }
            )
        row = {
            "messages": target_messages,
            "tools": _copy_json(tool_list),
            "metadata": target_metadata,
        }
        _validate_row_shape(row)
        rows.append(row)
    return rows


def validate_tokenized_row(row: Mapping[str, Any], generator: Any, max_total_tokens: int) -> int:
    """Validate native Qwen token IDs and target loss mask for one row."""

    token_ids, loss_mask = generator.get_loss_mask(row["messages"], tools=row["tools"])
    expected = generator.tokenizer.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        tokenize=True,
        return_dict=False,
    )
    if token_ids != expected:
        raise ConversionError("Qwen3 full token IDs differ from native chat template")
    if len(token_ids) != len(loss_mask) or not any(loss_mask):
        raise ConversionError("Qwen3 full token/loss mask is empty or misaligned")
    if len(token_ids) > max_total_tokens:
        raise ConversionError(
            f"row has {len(token_ids)} tokens, above cap {max_total_tokens}"
        )
    return len(token_ids)


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def _load_runtime_factory(
    *,
    documents_dir: Path,
    allowed_document_ids: Sequence[str],
) -> Any:
    """Build a lazy runtime factory so CPU-only formatter tests need no tau2 import."""

    try:
        from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
        from tau2.data_model.tasks import Task
        from tau2.domains.banking_knowledge.data_model import Document, KnowledgeBase, TransactionalDB
        from tau2.domains.banking_knowledge.retrieval import build_policy, build_tools, resolve_variant
    except ImportError as exc:
        raise RuntimeError(
            "Banking runtime dependencies are unavailable; run this formatter in the tau2 setup environment"
        ) from exc

    if len(allowed_document_ids) != 480 or len(set(allowed_document_ids)) != 480:
        raise ValueError("expected exactly 480 unique allowlisted document IDs")
    documents = {
        document_id: Document.model_validate(
            json.loads((documents_dir / f"{document_id}.json").read_text(encoding="utf-8"))
        )
        for document_id in allowed_document_ids
    }
    knowledge_base = KnowledgeBase(documents=documents)
    empty_db_path = PROJECT_ROOT / "examples/tau2-bench/analysis/banking_synthetic/assets/empty_db.json"
    empty_db = json.loads(empty_db_path.read_text(encoding="utf-8"))
    variant_cache: dict[str, tuple[Any, tuple[dict[str, Any], ...], frozenset[str]]] = {}

    def get_runtime(task_data: Mapping[str, Any], retrieval_variant: str) -> BankingRuntime:
        task = Task.model_validate(dict(task_data))
        if retrieval_variant not in variant_cache:
            variant = resolve_variant(retrieval_variant)
            database = TransactionalDB.model_validate(empty_db)
            toolkit = build_tools(variant, database, knowledge_base)
            schemas = tuple(_copy_json(tool.openai_schema) for tool in toolkit.get_tools().values())
            variant_cache[retrieval_variant] = (variant, schemas, _agent_tool_names(schemas))
        variant, schemas, names = variant_cache[retrieval_variant]
        policy = build_policy(variant, knowledge_base, task)
        return BankingRuntime(
            system_prompt=SYSTEM_PROMPT.format(
                agent_instruction=AGENT_INSTRUCTION,
                domain_policy=policy,
            ),
            tools=schemas,
            agent_tool_names=names,
            retrieval_variant=retrieval_variant,
        )

    return get_runtime


def _load_tokenizer_generator(tokenizer_path: Path) -> Any:
    from transformers import AutoTokenizer

    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--allowed-documents", type=Path, default=DEFAULT_ALLOWED_DOCUMENTS)
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--limit-trajectories", type=int, default=None)
    parser.add_argument("--max-total-tokens", type=int, default=MAX_TOTAL_TOKENS)
    parser.add_argument(
        "--skip-token-validation",
        action="store_true",
        help="Only for formatter development; normal SFT data must use native token validation.",
    )
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("--tasks must contain a JSON array")
    tasks_by_id = {str(task["id"]): task for task in tasks}
    contracts_by_id = {
        str(row["task_id"]): row for _, row in _read_jsonl(args.contracts)
    }
    train_ids = {
        task_id
        for task_id, contract in contracts_by_id.items()
        if contract.get("split") == "train"
    }
    allowed_manifest = json.loads(args.allowed_documents.read_text(encoding="utf-8"))
    allowed_ids = list(allowed_manifest.get("allowed_document_ids") or [])

    runtime_for = _load_runtime_factory(
        documents_dir=args.documents_dir,
        allowed_document_ids=allowed_ids,
    )
    generator = None if args.skip_token_validation else _load_tokenizer_generator(args.tokenizer)
    stats: Counter[str] = Counter()
    runtime_cache: dict[tuple[str, str], BankingRuntime] = {}

    # Full Banking conversion expands each trajectory into multiple target rows.
    # Write rows as they are validated so the complete expanded dataset is not
    # kept in memory until the end of the run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    written = 0
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for trajectory_index, (_, trajectory) in enumerate(_read_jsonl(args.input)):
                if args.limit_trajectories is not None and trajectory_index >= args.limit_trajectories:
                    break
                stats["input_trajectories"] += 1
                if stats["input_trajectories"] % 100 == 0:
                    print(
                        f"[banking-expert-sft] processed {stats['input_trajectories']} trajectories",
                        flush=True,
                    )
                task_id = str(trajectory.get("task_id") or "")
                if task_id not in train_ids or task_id not in tasks_by_id:
                    stats["drop_non_train_or_unknown_task"] += 1
                    continue
                if trajectory.get("data_origin") != "independent_synthetic":
                    stats["drop_non_synthetic_origin"] += 1
                    continue
                if trajectory.get("training_eligible") is not True:
                    stats["drop_not_training_eligible"] += 1
                    continue
                reward = (trajectory.get("reward_info") or {}).get("reward")
                if float(reward or 0.0) != 1.0:
                    stats["drop_non_success"] += 1
                    continue
                contract = contracts_by_id[task_id]
                if contract.get("data_origin") != "independent_synthetic":
                    stats["drop_contract_non_synthetic_origin"] += 1
                    continue
                variant_name = str(contract.get("retrieval_variant") or "")
                cache_key = (task_id, variant_name)
                runtime = runtime_cache.get(cache_key)
                if runtime is None:
                    runtime = runtime_for(tasks_by_id[task_id], variant_name)
                    runtime_cache[cache_key] = runtime
                metadata = {
                    "dataset_version": "tau2-banking-expert-sft-v1",
                    "source_dataset": "tau2-banking-independent-synthetic-v1",
                    "source_trajectory_id": trajectory.get("id"),
                    "task_id": task_id,
                    "domain": "banking_knowledge",
                    "retrieval_variant": runtime.retrieval_variant,
                    "reward": 1,
                }
                try:
                    rows = expand_agent_targets(
                        trajectory.get("messages") or [],
                        system_prompt=runtime.system_prompt,
                        tools=runtime.tools,
                        agent_tool_names=runtime.agent_tool_names,
                        metadata=metadata,
                    )
                except ConversionError as exc:
                    stats[f"drop_conversion_{str(exc)}"] += 1
                    continue

                stats["candidate_rows"] += len(rows)
                accepted_for_trajectory = 0
                for row in rows:
                    try:
                        if generator is not None:
                            row["metadata"]["total_tokens"] = validate_tokenized_row(
                                row, generator, args.max_total_tokens
                            )
                        row["metadata"]["max_total_tokens"] = args.max_total_tokens
                    except ConversionError as exc:
                        message = str(exc)
                        cap_match = re.fullmatch(
                            r"row has (\d+) tokens, above cap (\d+)", message
                        )
                        if cap_match:
                            stats["drop_target_over_token_cap"] += 1
                            stats["max_rejected_target_tokens"] = max(
                                stats["max_rejected_target_tokens"],
                                int(cap_match.group(1)),
                            )
                        else:
                            stats[f"drop_target_{message}"] += 1
                        continue
                    output_file.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    written += 1
                    accepted_for_trajectory += 1

                if not accepted_for_trajectory:
                    stats["drop_trajectory_without_valid_target"] += 1
                    continue
                stats["accepted_trajectories"] += 1
                stats["output_rows"] += accepted_for_trajectory
        temporary.replace(args.output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    stats["written_rows"] = written
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(
        json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2))
    return dict(stats)


def main() -> None:
    build_dataset(_parse_args())


if __name__ == "__main__":
    main()
