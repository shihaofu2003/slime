#!/usr/bin/env python3
"""Review and filter official-native Tau2 SFT rows one target turn at a time."""

from __future__ import annotations

import argparse
import heapq
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "output/experiments/tau2-sft-official-native-expanded/data"
    / "agent_official_native_expanded.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "output/experiments/tau2-sft-turn-quality-qwen38-xhigh-v1"
)
DEFAULT_MODEL_PATH = Path("/mnt/afs/models/Qwen3.8-27B")
DEFAULT_MODEL_NAME = "Qwen3.8-27B-turn-quality"
PROMPT_VERSION = "tau2-turn-quality-v5"
PASS_SEEDS = {"a": 300, "b": 301}
CONTEXT_LENGTH = 32768
MAX_COMPLETION_TOKENS = 8192
DECODE_BALANCE_TOKENS = 2048

VERDICTS = {"keep", "review", "drop"}
PREFIX_RELATIONS = {
    "clean",
    "recovers_prior_error",
    "depends_on_prior_error",
    "prior_error_irrelevant",
    "uncertain",
}
ISSUE_TYPES = {
    "factual_grounding",
    "capability",
    "policy_workflow",
    "tool_choice",
    "tool_arguments",
    "confirmation",
    "state_outcome",
    "recovery_dependency",
    "response_quality",
}

# These are the manually established cases discussed before this filter was
# designed. They gate the pilot; retain cases also guard the full-data merge
# against false-positive deletion from stochastic judge drift.
DEFAULT_CALIBRATION = {
    4768: "drop",
    6581: "drop",
    12485: "drop",
    14336: "retain",
    21429: "drop",
    21750: "drop",
    23976: "retain",
    25985: "drop",
    29239: "drop",
}

REVIEW_SYSTEM = """You are an evidence-grounded reviewer of exactly one Tau2 Agent SFT target.
Judge only the Assistant message whose step_loss_mask is 1, using the visible
prefix, system policy, Agent tool schemas, User messages, and tool observations
contained in the supplied record. Use no outside facts.

Treat prior Assistant statements as untrusted claims. First establish the
current state and User intent, identify the governing policy and available
capabilities, derive the permissible next behavior, and then compare the target
against it.

Check material factual and capability claims, policy and end-to-end workflow,
tool choice and arguments, required confirmation, and claims about resulting
state. Read complete tool schemas rather than inferring capability from tool
names. A successful API call would not make a policy-violating workflow valid.
Do not approve a sequence that changes an intermediate state merely to evade an
explicit restriction.

Treat every concrete action that the target offers, recommends, or presents as
an alternative as a claim that the action is currently feasible under the
supplied facts. Validate every offered option independently. Asking whether the
User wants an action does not postpone its policy or eligibility check until a
later confirmation or tool call. If the known state rules an offered action
out, that is a material error even when other parts of the target are correct.
Distinguish this from a genuinely conditional explanation: saying an action is
possible "if eligible" does not assert that eligibility is met when the correct
entity or required facts are still unresolved. Do not drop a target merely for
mentioning such a condition. If the exact entity and disqualifying facts are
already established, however, superficial hedging does not make a known-invalid
option acceptable.

For Airline, interpret "Basic economy flights cannot be modified" as an
end-to-end restriction on changing those flights. The separate permission to
change cabin allows a genuine cabin-change request; it does not authorize
changing cabin merely as an intermediate step to unlock an otherwise forbidden
flight change. Proposing or continuing that two-step workaround is policy
circumvention and is a material error.

For Airline cancellation, do not present cancellation as an available option
unless the supplied state satisfies at least one cancellation condition in the
policy. When the record already establishes booking time, cabin, insurance,
and airline cancellation status, use those facts now; a duplicate booking or
ordinary change of plan does not by itself create cancellation eligibility.
For the "within the last 24 hrs" condition, explicitly calculate
current_time - created_at and require that elapsed time to be at most 24 hours;
a span of multiple calendar days cannot satisfy this condition.

Do not penalize a final tool-call target because its future tool result is
absent. Do not reject a correct correction or recovery merely because an
earlier Assistant turn was wrong. If the target endorses or causally depends on
an earlier invalid action, treat that as a material error. Ignore harmless style
preferences.

keep: no material error.
drop: at least one clear material error directly supported by the supplied evidence.
review: the supplied evidence is genuinely ambiguous or insufficient.

Return exactly one submit_turn_review call. Do not propose, repair, or rewrite
a replacement target."""

REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_turn_review",
        "description": "Submit the evidence-grounded quality decision for this target.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_id": {"type": "integer"},
                "verdict": {"type": "string", "enum": sorted(VERDICTS)},
                "prefix_relation": {
                    "type": "string",
                    "enum": sorted(PREFIX_RELATIONS),
                },
                "issues": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": sorted(ISSUE_TYPES)},
                            "evidence": {
                                "type": "array",
                                "maxItems": 3,
                                "items": {"type": "string"},
                            },
                            "explanation": {"type": "string"},
                        },
                        "required": ["type", "evidence", "explanation"],
                        "additionalProperties": False,
                    },
                },
                "reason": {"type": "string"},
            },
            "required": [
                "target_id",
                "verdict",
                "prefix_relation",
                "issues",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class RowIndex:
    input_line: int
    offset: int
    byte_length: int
    prompt_tokens: int
    domain: str
    dialog_id: str
    source_turn_index: int
    expanded_call_index: int
    target_type: str

    @property
    def group(self) -> tuple[str, str]:
        return self.domain, self.dialog_id

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.domain,
            self.dialog_id,
            self.source_turn_index,
            self.expanded_call_index,
            self.input_line,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "source_dialog_id": self.dialog_id,
            "source_turn_index": self.source_turn_index,
            "expanded_call_index": (
                None if self.expanded_call_index < 0 else self.expanded_call_index
            ),
            "target_type": self.target_type,
        }


class ReviewFormatError(ValueError):
    """The server responded, but not with a valid turn review."""


class JudgeTransportError(RuntimeError):
    """The local inference endpoint could not complete a request."""


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _int_metadata(metadata: Mapping[str, Any], key: str, default: int) -> int:
    value = metadata.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def validate_training_row(row: Mapping[str, Any], input_line: int) -> None:
    messages = row.get("messages")
    tools = row.get("tools")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"input line {input_line}: messages must be a non-empty list")
    if not isinstance(tools, list):
        raise ValueError(f"input line {input_line}: tools must be a list")

    target_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, Mapping)
        and message.get("step_loss_mask") == 1
    ]
    final_message = messages[-1]
    if (
        target_indices != [len(messages) - 1]
        or not isinstance(final_message, Mapping)
        or final_message.get("role") != "assistant"
    ):
        raise ValueError(
            f"input line {input_line}: expected only the final Assistant message "
            "to have step_loss_mask=1"
        )


def evidence_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact messages/tools view while blinding source reward labels."""

    return {"tools": row["tools"], "messages": row["messages"]}


def judge_messages(input_line: int, row: Mapping[str, Any]) -> list[dict[str, str]]:
    evidence = compact_json(evidence_record(row))
    user_content = (
        "Review the following evidence record. The sole candidate is the final "
        "Assistant message marked step_loss_mask=1.\n\n"
        f"<evidence_json>\n{evidence}\n</evidence_json>\n\n"
        f"Return the decision for target_id {input_line}."
    )
    return [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def judge_request_body(
    input_line: int,
    row: Mapping[str, Any],
    *,
    model: str,
    seed: int,
    max_tokens: int = MAX_COMPLETION_TOKENS,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": judge_messages(input_line, row),
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        },
        "tools": [REVIEW_TOOL],
        "tool_choice": {
            "type": "function",
            "function": {"name": REVIEW_TOOL["function"]["name"]},
        },
    }


def prompt_token_count(tokenizer: Any, input_line: int, row: Mapping[str, Any]) -> int:
    rendered = tokenizer.apply_chat_template(
        judge_messages(input_line, row),
        tools=[REVIEW_TOOL],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        preserve_thinking=True,
        reasoning_effort="xhigh",
    )
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if not isinstance(rendered, Sequence):
        raise TypeError("tokenizer.apply_chat_template did not return token ids")
    return len(rendered)


def _row_index(
    row: Mapping[str, Any],
    *,
    input_line: int,
    offset: int,
    byte_length: int,
    prompt_tokens: int,
) -> RowIndex:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    domain = str(metadata.get("domain") or "unknown")
    dialog_value = metadata.get("source_dialog_id")
    dialog_id = str(dialog_value) if dialog_value is not None else f"line-{input_line}"
    target = row["messages"][-1]
    target_type = str(
        metadata.get("target_type")
        or ("tool_call" if target.get("tool_calls") else "text")
    )
    return RowIndex(
        input_line=input_line,
        offset=offset,
        byte_length=byte_length,
        prompt_tokens=prompt_tokens,
        domain=domain,
        dialog_id=dialog_id,
        source_turn_index=_int_metadata(metadata, "source_turn_index", input_line),
        expanded_call_index=_int_metadata(metadata, "expanded_call_index", -1),
        target_type=target_type,
    )


def assign_shards(rows: Sequence[RowIndex], num_shards: int) -> dict[tuple[str, str], int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    group_weights: Counter[tuple[str, str]] = Counter()
    for row in rows:
        group_weights[row.group] += row.prompt_tokens + DECODE_BALANCE_TOKENS

    heap = [(0, shard_id) for shard_id in range(num_shards)]
    heapq.heapify(heap)
    assignment: dict[tuple[str, str], int] = {}
    for group, weight in sorted(
        group_weights.items(), key=lambda item: (-item[1], item[0])
    ):
        current_weight, shard_id = heapq.heappop(heap)
        assignment[group] = shard_id
        heapq.heappush(heap, (current_weight + weight, shard_id))
    return assignment


def select_pilot_rows(
    rows: Sequence[RowIndex],
    *,
    pilot_size: int,
    seed: int,
    calibration: Mapping[int, str],
) -> list[RowIndex]:
    if pilot_size <= 0:
        raise ValueError("pilot_size must be positive")
    by_line = {row.input_line: row for row in rows}
    selected = {
        line: by_line[line] for line in calibration if line in by_line
    }
    target_size = min(max(pilot_size, len(selected)), len(rows))
    longest_count = min(32, target_size - len(selected))
    for row in sorted(rows, key=lambda item: (-item.prompt_tokens, item.input_line))[
        :longest_count
    ]:
        selected.setdefault(row.input_line, row)

    randomizer = random.Random(seed)
    buckets: dict[tuple[str, str], list[RowIndex]] = defaultdict(list)
    for row in rows:
        if row.input_line not in selected:
            buckets[(row.domain, row.target_type)].append(row)
    for bucket in buckets.values():
        randomizer.shuffle(bucket)

    bucket_keys = sorted(buckets)
    while len(selected) < target_size:
        progressed = False
        for key in bucket_keys:
            if buckets[key] and len(selected) < target_size:
                row = buckets[key].pop()
                selected[row.input_line] = row
                progressed = True
        if not progressed:
            break
    calibration_rows = [
        selected[line] for line in calibration if line in selected
    ]
    calibration_ids = {row.input_line for row in calibration_rows}
    remaining_rows = sorted(
        (row for row in selected.values() if row.input_line not in calibration_ids),
        key=lambda row: row.sort_key,
    )
    return calibration_rows + remaining_rows


def _read_indexed_row(source: Any, index: RowIndex) -> dict[str, Any]:
    source.seek(index.offset)
    raw_line = source.read(index.byte_length)
    row = json.loads(raw_line)
    validate_training_row(row, index.input_line)
    return row


def _write_payload(file: Any, index: RowIndex, row: Mapping[str, Any]) -> None:
    payload = {
        "input_line": index.input_line,
        "prompt_version": PROMPT_VERSION,
        "prompt_tokens": index.prompt_tokens,
        "metadata": index.metadata(),
        "tools": row["tools"],
        "messages": row["messages"],
    }
    file.write(compact_json(payload) + "\n")


def prepare_dataset(
    *,
    input_path: Path,
    output_root: Path,
    tokenizer: Any,
    model_path: Path,
    num_shards: int,
    pilot_size: int,
    pilot_seed: int,
    calibration: Mapping[int, str],
    context_length: int = CONTEXT_LENGTH,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    rows: list[RowIndex] = []
    started = time.monotonic()
    with input_path.open("rb") as source:
        input_line = 0
        while True:
            offset = source.tell()
            raw_line = source.readline()
            if not raw_line:
                break
            input_line += 1
            if not raw_line.strip():
                raise ValueError(f"input line {input_line}: blank lines are unsupported")
            row = json.loads(raw_line)
            if not isinstance(row, Mapping):
                raise ValueError(f"input line {input_line}: expected a JSON object")
            validate_training_row(row, input_line)
            tokens = prompt_token_count(tokenizer, input_line, row)
            if tokens + max_completion_tokens > context_length:
                raise ValueError(
                    f"input line {input_line}: {tokens} prompt tokens plus "
                    f"{max_completion_tokens} completion tokens exceeds "
                    f"context length {context_length}; turns are never truncated"
                )
            rows.append(
                _row_index(
                    row,
                    input_line=input_line,
                    offset=offset,
                    byte_length=len(raw_line),
                    prompt_tokens=tokens,
                )
            )
            if input_line % 250 == 0:
                print(
                    f"PREPARE_PROGRESS indexed={input_line} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
    if not rows:
        raise ValueError("input dataset is empty")

    missing_calibration = sorted(set(calibration) - {row.input_line for row in rows})
    if missing_calibration:
        raise ValueError(f"calibration lines are absent: {missing_calibration}")

    assignment = assign_shards(rows, num_shards)
    shard_rows: list[list[RowIndex]] = [[] for _ in range(num_shards)]
    for row in rows:
        shard_rows[assignment[row.group]].append(row)
    for shard in shard_rows:
        shard.sort(key=lambda row: row.sort_key)
    pilot_rows = select_pilot_rows(
        rows,
        pilot_size=pilot_size,
        seed=pilot_seed,
        calibration=calibration,
    )

    shard_dir = output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open("rb") as source:
        for shard_id, indexed_rows in enumerate(shard_rows):
            shard_path = shard_dir / f"shard_{shard_id:02d}.jsonl"
            with shard_path.open("w", encoding="utf-8") as output:
                for index in indexed_rows:
                    _write_payload(output, index, _read_indexed_row(source, index))
            print(
                f"PREPARE_SHARD shard={shard_id:02d} rows={len(indexed_rows)}",
                flush=True,
            )

        with (shard_dir / "pilot.jsonl").open("w", encoding="utf-8") as output:
            for index in pilot_rows:
                _write_payload(output, index, _read_indexed_row(source, index))

    shard_summaries = []
    for shard_id, indexed_rows in enumerate(shard_rows):
        shard_summaries.append(
            {
                "shard_id": shard_id,
                "rows": len(indexed_rows),
                "estimated_tokens_per_pass": sum(
                    row.prompt_tokens + DECODE_BALANCE_TOKENS for row in indexed_rows
                ),
            }
        )
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "input": str(input_path),
        "model_path": str(model_path),
        "rows": len(rows),
        "num_shards": num_shards,
        "pilot_rows": len(pilot_rows),
        "pilot_seed": pilot_seed,
        "context_length": context_length,
        "max_completion_tokens": max_completion_tokens,
        "max_prompt_tokens": max(row.prompt_tokens for row in rows),
        "mean_prompt_tokens": round(
            sum(row.prompt_tokens for row in rows) / len(rows), 2
        ),
        "shards": shard_summaries,
    }
    (shard_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pilot_manifest = {
        "prompt_version": PROMPT_VERSION,
        "rows": [row.input_line for row in pilot_rows],
        "calibration": {
            str(line): expected
            for line, expected in calibration.items()
            if line in {row.input_line for row in pilot_rows}
        },
    }
    (shard_dir / "pilot_manifest.json").write_text(
        json.dumps(pilot_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _post_json(url: str, body: Mapping[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local-turn-review",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")[:1000]
        raise JudgeTransportError(f"HTTP {exc.code}: {body_text}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise JudgeTransportError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise JudgeTransportError("server response was not JSON") from exc
    if not isinstance(payload, dict):
        raise JudgeTransportError("server response was not an object")
    return payload


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewFormatError(f"{field} must be a non-empty string")
    return value.strip()


def validate_review(value: Mapping[str, Any], expected_target_id: int) -> dict[str, Any]:
    expected_keys = {"target_id", "verdict", "prefix_relation", "issues", "reason"}
    if set(value) != expected_keys:
        raise ReviewFormatError(
            f"review fields must be {sorted(expected_keys)}, got {sorted(value)}"
        )
    target_id = value["target_id"]
    if isinstance(target_id, bool) or target_id != expected_target_id:
        raise ReviewFormatError(
            f"target_id must equal {expected_target_id}, got {target_id!r}"
        )
    verdict = value["verdict"]
    if verdict not in VERDICTS:
        raise ReviewFormatError(f"invalid verdict: {verdict!r}")
    prefix_relation = value["prefix_relation"]
    if prefix_relation not in PREFIX_RELATIONS:
        raise ReviewFormatError(f"invalid prefix_relation: {prefix_relation!r}")
    issues = value["issues"]
    if not isinstance(issues, list) or len(issues) > 3:
        raise ReviewFormatError("issues must be a list of at most three items")
    normalized_issues = []
    for issue in issues:
        if not isinstance(issue, Mapping) or set(issue) != {
            "type",
            "evidence",
            "explanation",
        }:
            raise ReviewFormatError("each issue must contain type/evidence/explanation")
        issue_type = issue["type"]
        if issue_type not in ISSUE_TYPES:
            raise ReviewFormatError(f"invalid issue type: {issue_type!r}")
        evidence = issue["evidence"]
        if not isinstance(evidence, list) or len(evidence) > 3:
            raise ReviewFormatError("issue evidence must contain at most three strings")
        normalized_evidence = [
            _nonempty_string(item, "issue evidence") for item in evidence
        ]
        normalized_issues.append(
            {
                "type": issue_type,
                "evidence": normalized_evidence,
                "explanation": _nonempty_string(
                    issue["explanation"], "issue explanation"
                ),
            }
        )
    if verdict == "drop" and not normalized_issues:
        raise ReviewFormatError("drop requires at least one evidenced issue")
    return {
        "target_id": expected_target_id,
        "verdict": verdict,
        "prefix_relation": prefix_relation,
        "issues": normalized_issues,
        "reason": _nonempty_string(value["reason"], "reason"),
    }


def parse_review_response(
    response: Mapping[str, Any], expected_target_id: int
) -> tuple[dict[str, Any], str, bool, dict[str, Any]]:
    try:
        choice = response["choices"][0]
        message = choice["message"]
        calls = message.get("tool_calls") or []
    except (KeyError, IndexError, TypeError) as exc:
        raise ReviewFormatError("missing completion message/tool call") from exc
    if len(calls) != 1:
        raise ReviewFormatError(f"expected one tool call, got {len(calls)}")
    function = calls[0].get("function") or {}
    if function.get("name") != REVIEW_TOOL["function"]["name"]:
        raise ReviewFormatError(f"unexpected tool name: {function.get('name')!r}")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ReviewFormatError("tool arguments are not valid JSON") from exc
    if not isinstance(arguments, Mapping):
        raise ReviewFormatError("tool arguments must be an object")
    review = validate_review(arguments, expected_target_id)
    finish_reason = str(choice.get("finish_reason") or "")
    if finish_reason == "length":
        raise ReviewFormatError("completion reached max_tokens")
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = response.get("usage")
    usage = dict(usage) if isinstance(usage, Mapping) else {}
    return review, finish_reason, bool(str(reasoning).strip()), usage


def _synthetic_review(target_id: int, error: str) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "verdict": "review",
        "prefix_relation": "uncertain",
        "issues": [
            {
                "type": "response_quality",
                "evidence": [],
                "explanation": "The judge did not return a valid structured decision.",
            }
        ],
        "reason": "Retained for review because the judge response was invalid.",
        "judge_error": error,
    }


def request_review(
    record: Mapping[str, Any],
    *,
    pass_id: str,
    endpoint: str,
    model: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    input_line = record.get("input_line")
    if isinstance(input_line, bool) or not isinstance(input_line, int):
        raise ValueError("judge record requires an integer input_line")
    if pass_id not in PASS_SEEDS:
        raise ValueError(f"unknown pass: {pass_id}")
    row = {"messages": record.get("messages"), "tools": record.get("tools")}
    validate_training_row(row, input_line)

    started = time.monotonic()
    format_errors: list[str] = []
    last_transport_error: Exception | None = None
    for attempt in range(retries + 1):
        body = judge_request_body(
            input_line,
            row,
            model=model,
            seed=PASS_SEEDS[pass_id] + attempt * 1000,
        )
        try:
            response = _post_json(endpoint, body, timeout)
            review, finish_reason, had_reasoning, usage = parse_review_response(
                response, input_line
            )
            return {
                "input_line": input_line,
                "pass_id": pass_id,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "reasoning_effort": "xhigh",
                "seed": PASS_SEEDS[pass_id],
                "attempts": attempt + 1,
                "finish_reason": finish_reason,
                "had_reasoning": had_reasoning,
                "usage": usage,
                "latency_seconds": round(time.monotonic() - started, 3),
                "metadata": record.get("metadata") or {},
                **review,
            }
        except ReviewFormatError as exc:
            format_errors.append(str(exc))
        except JudgeTransportError as exc:
            last_transport_error = exc
        if attempt < retries:
            time.sleep(min(8, 2**attempt))

    if last_transport_error is not None and not format_errors:
        raise JudgeTransportError(
            f"target {input_line} pass {pass_id} failed after {retries + 1} "
            f"attempts: {last_transport_error}"
        ) from last_transport_error
    error = "; ".join(format_errors) or str(last_transport_error)
    review = _synthetic_review(input_line, error)
    return {
        "input_line": input_line,
        "pass_id": pass_id,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "reasoning_effort": "xhigh",
        "seed": PASS_SEEDS[pass_id],
        "attempts": retries + 1,
        "finish_reason": "invalid",
        "had_reasoning": False,
        "usage": {},
        "latency_seconds": round(time.monotonic() - started, 3),
        "metadata": record.get("metadata") or {},
        **review,
    }


def _completed_reviews(path: Path, pass_id: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        input_line = row.get("input_line")
        if row.get("prompt_version") != PROMPT_VERSION or row.get("pass_id") != pass_id:
            raise ValueError(f"{path}: incompatible cached review")
        if isinstance(input_line, bool) or not isinstance(input_line, int):
            raise ValueError(f"{path}: invalid input_line")
        if input_line in completed:
            raise ValueError(f"{path}: duplicate input_line {input_line}")
        completed[input_line] = row
    return completed


def judge_shard(
    *,
    input_path: Path,
    output_a: Path,
    output_b: Path,
    base_url: str,
    model: str,
    concurrency: int,
    timeout: int,
    retries: int,
) -> dict[str, int]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    records = read_jsonl(input_path)
    record_ids = [row.get("input_line") for row in records]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in record_ids):
        raise ValueError(f"{input_path}: every record needs an integer input_line")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{input_path}: duplicate input_line")
    for record in records:
        if record.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"{input_path}: incompatible prompt version")
    records.sort(
        key=lambda record: (
            record["metadata"]["domain"],
            record["metadata"]["source_turn_index"],
            record["prompt_tokens"],
            record["input_line"],
        )
    )

    completed = {
        "a": _completed_reviews(output_a, "a"),
        "b": _completed_reviews(output_b, "b"),
    }
    outputs = {"a": output_a, "b": output_b}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        (record, pass_id)
        for record in records
        for pass_id in ("a", "b")
        if record["input_line"] not in completed[pass_id]
    ]
    endpoint = base_url.rstrip("/") + (
        "/chat/completions"
        if base_url.rstrip("/").endswith("/v1")
        else "/v1/chat/completions"
    )
    handles = {
        pass_id: outputs[pass_id].open("a", encoding="utf-8")
        for pass_id in ("a", "b")
    }
    completed_now = Counter()
    task_iter = iter(tasks)
    pending: dict[Future[dict[str, Any]], tuple[int, str]] = {}

    def submit_next(pool: ThreadPoolExecutor) -> bool:
        try:
            record, pass_id = next(task_iter)
        except StopIteration:
            return False
        future = pool.submit(
            request_review,
            record,
            pass_id=pass_id,
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            retries=retries,
        )
        pending[future] = (record["input_line"], pass_id)
        return True

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _ in range(min(concurrency, len(tasks))):
                submit_next(pool)
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    input_line, pass_id = pending.pop(future)
                    result = future.result()
                    if result["input_line"] != input_line or result["pass_id"] != pass_id:
                        raise RuntimeError("judge result identity mismatch")
                    handles[pass_id].write(compact_json(result) + "\n")
                    handles[pass_id].flush()
                    completed_now[pass_id] += 1
                    total_done = completed_now["a"] + completed_now["b"]
                    if total_done % 25 == 0 or total_done == len(tasks):
                        print(
                            f"JUDGE_PROGRESS completed={total_done}/{len(tasks)} "
                            f"pass_a={completed_now['a']} pass_b={completed_now['b']}",
                            flush=True,
                        )
                    submit_next(pool)
    finally:
        for handle in handles.values():
            handle.close()
    return {"pass_a": completed_now["a"], "pass_b": completed_now["b"]}


def merge_verdicts(verdict_a: str, verdict_b: str) -> tuple[str, bool, str]:
    if verdict_a not in VERDICTS or verdict_b not in VERDICTS:
        raise ValueError("cannot merge invalid verdicts")
    if verdict_a == verdict_b == "drop":
        return "drop", False, "unanimous_drop"
    if verdict_a == verdict_b == "keep":
        return "keep", True, "unanimous_keep"
    return "review", True, "review_or_disagreement"


def _load_review_directory(
    review_root: Path, pass_id: str, expected_shards: int
) -> dict[int, dict[str, Any]]:
    reviews: dict[int, dict[str, Any]] = {}
    for shard_id in range(expected_shards):
        path = review_root / f"pass_{pass_id}" / f"shard_{shard_id:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            input_line = row.get("input_line")
            if row.get("prompt_version") != PROMPT_VERSION or row.get("pass_id") != pass_id:
                raise ValueError(f"{path}: incompatible review")
            if isinstance(input_line, bool) or not isinstance(input_line, int):
                raise ValueError(f"{path}: invalid input_line")
            if input_line in reviews:
                raise ValueError(f"duplicate pass {pass_id} review for line {input_line}")
            reviews[input_line] = row
    return reviews


def _decision(
    input_line: int, review_a: Mapping[str, Any], review_b: Mapping[str, Any]
) -> dict[str, Any]:
    final_verdict, retained, basis = merge_verdicts(
        str(review_a.get("verdict")), str(review_b.get("verdict"))
    )
    metadata = review_a.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}

    def pass_view(review: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: review.get(key)
            for key in (
                "verdict",
                "prefix_relation",
                "issues",
                "reason",
                "judge_error",
                "attempts",
                "finish_reason",
                "had_reasoning",
                "usage",
                "latency_seconds",
            )
            if key in review
        }

    return {
        "input_line": input_line,
        "metadata": metadata,
        "pass_a": pass_view(review_a),
        "pass_b": pass_view(review_b),
        "final_verdict": final_verdict,
        "retained": retained,
        "decision_basis": basis,
    }


def _review_metrics(reviews: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    usage = Counter()
    latency = 0.0
    judge_errors = 0
    missing_reasoning = 0
    for review in reviews:
        for key, value in (review.get("usage") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] += value
        latency += float(review.get("latency_seconds") or 0.0)
        judge_errors += int(bool(review.get("judge_error")))
        missing_reasoning += int(not review.get("had_reasoning"))
    return {
        "usage": dict(usage),
        "summed_request_latency_seconds": round(latency, 3),
        "judge_errors": judge_errors,
        "missing_reasoning": missing_reasoning,
    }


def check_pilot(*, output_root: Path) -> dict[str, Any]:
    pilot_manifest_path = output_root / "shards/pilot_manifest.json"
    manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    expected_ids = set(manifest["rows"])
    reviews_a = _completed_reviews(output_root / "reviews/pilot/pass_a.jsonl", "a")
    reviews_b = _completed_reviews(output_root / "reviews/pilot/pass_b.jsonl", "b")
    if set(reviews_a) != expected_ids or set(reviews_b) != expected_ids:
        raise ValueError("pilot reviews are incomplete")

    decisions = [
        _decision(input_line, reviews_a[input_line], reviews_b[input_line])
        for input_line in sorted(expected_ids)
    ]
    decision_path = output_root / "reviews/pilot/decisions.jsonl"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    with decision_path.open("w", encoding="utf-8") as file:
        for decision in decisions:
            file.write(compact_json(decision) + "\n")

    failures = []
    by_line = {decision["input_line"]: decision for decision in decisions}
    for line_text, expected in manifest.get("calibration", {}).items():
        input_line = int(line_text)
        actual = by_line[input_line]["final_verdict"]
        if expected == "drop" and actual != "drop":
            failures.append(f"line {input_line}: expected drop, got {actual}")
        if expected == "retain" and not by_line[input_line]["retained"]:
            failures.append(f"line {input_line}: expected retained, got {actual}")

    summary = {
        "prompt_version": PROMPT_VERSION,
        "rows": len(decisions),
        "final_verdicts": dict(Counter(row["final_verdict"] for row in decisions)),
        "agreement": dict(
            Counter(
                f"{row['pass_a']['verdict']}/{row['pass_b']['verdict']}"
                for row in decisions
            )
        ),
        "pass_a": _review_metrics(reviews_a.values()),
        "pass_b": _review_metrics(reviews_b.values()),
        "calibration_failures": failures,
    }
    (output_root / "reviews/pilot/summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise ValueError("pilot calibration failed: " + "; ".join(failures))
    return summary


def finalize_dataset(
    *, input_path: Path, output_root: Path, output_path: Path | None = None
) -> dict[str, Any]:
    manifest = json.loads(
        (output_root / "shards/manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("incompatible shard manifest")
    expected_rows = int(manifest["rows"])
    num_shards = int(manifest["num_shards"])
    pilot_manifest_path = output_root / "shards/pilot_manifest.json"
    manual_retain_lines: set[int] = set()
    if pilot_manifest_path.is_file():
        pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
        manual_retain_lines = {
            int(line)
            for line, expected in pilot_manifest.get("calibration", {}).items()
            if expected == "retain"
        }
    reviews_a = _load_review_directory(output_root / "reviews", "a", num_shards)
    reviews_b = _load_review_directory(output_root / "reviews", "b", num_shards)
    expected_ids = set(range(1, expected_rows + 1))
    if set(reviews_a) != expected_ids:
        missing = sorted(expected_ids - set(reviews_a))[:20]
        extra = sorted(set(reviews_a) - expected_ids)[:20]
        raise ValueError(f"pass A coverage mismatch: missing={missing} extra={extra}")
    if set(reviews_b) != expected_ids:
        missing = sorted(expected_ids - set(reviews_b))[:20]
        extra = sorted(set(reviews_b) - expected_ids)[:20]
        raise ValueError(f"pass B coverage mismatch: missing={missing} extra={extra}")

    output_path = output_path or (
        output_root / "data/agent_official_native_expanded_turn_filtered.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path = output_root / "reviews/turn_decisions.jsonl"
    review_path = output_root / "reviews/review_turns.jsonl"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)

    final_counts = Counter()
    agreement = Counter()
    domain_target_counts: Counter[tuple[str, str, str]] = Counter()
    issue_counts = Counter()
    manual_retain_overrides = []
    with input_path.open("rb") as source, output_path.open("wb") as filtered, \
        decisions_path.open("w", encoding="utf-8") as decisions_file, \
        review_path.open("w", encoding="utf-8") as review_file:
        input_line = 0
        for raw_line in source:
            input_line += 1
            if input_line > expected_rows:
                raise ValueError("source has more rows than the prepared manifest")
            decision = _decision(input_line, reviews_a[input_line], reviews_b[input_line])
            if input_line in manual_retain_lines and not decision["retained"]:
                decision["final_verdict"] = "review"
                decision["retained"] = True
                decision["decision_basis"] = "manual_calibration_retain"
                manual_retain_overrides.append(input_line)
            decisions_file.write(compact_json(decision) + "\n")
            if decision["final_verdict"] == "review":
                review_file.write(compact_json(decision) + "\n")
            if decision["retained"]:
                filtered.write(raw_line)

            final_verdict = decision["final_verdict"]
            final_counts[final_verdict] += 1
            agreement[
                f"{decision['pass_a']['verdict']}/{decision['pass_b']['verdict']}"
            ] += 1
            metadata = decision["metadata"]
            domain_target_counts[
                (
                    str(metadata.get("domain") or "unknown"),
                    str(metadata.get("target_type") or "unknown"),
                    final_verdict,
                )
            ] += 1
            for pass_name in ("pass_a", "pass_b"):
                for issue in decision[pass_name].get("issues") or []:
                    issue_counts[str(issue.get("type") or "unknown")] += 1
        if input_line != expected_rows:
            raise ValueError(
                f"source row count mismatch: expected {expected_rows}, got {input_line}"
            )

    summary = {
        "prompt_version": PROMPT_VERSION,
        "source": str(input_path),
        "output": str(output_path),
        "input_rows": expected_rows,
        "retained_rows": final_counts["keep"] + final_counts["review"],
        "dropped_rows": final_counts["drop"],
        "final_verdicts": dict(final_counts),
        "agreement": dict(agreement),
        "manual_retain_overrides": manual_retain_overrides,
        "by_domain_target_verdict": [
            {
                "domain": domain,
                "target_type": target_type,
                "verdict": verdict,
                "count": count,
            }
            for (domain, target_type, verdict), count in sorted(
                domain_target_counts.items()
            )
        ],
        "issue_types_across_passes": dict(issue_counts),
        "pass_a": _review_metrics(reviews_a.values()),
        "pass_b": _review_metrics(reviews_b.values()),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def validate_filtered_dataset(
    *, input_path: Path, output_path: Path, decisions_path: Path
) -> dict[str, int]:
    decisions = read_jsonl(decisions_path)
    decision_map = {row.get("input_line"): row for row in decisions}
    if len(decision_map) != len(decisions):
        raise ValueError("decision file contains duplicate input_line values")

    retained = 0
    input_rows = 0
    with input_path.open("rb") as source, output_path.open("rb") as filtered:
        for input_rows, raw_line in enumerate(source, 1):
            decision = decision_map.get(input_rows)
            if decision is None:
                raise ValueError(f"missing decision for input line {input_rows}")
            if decision.get("retained"):
                retained += 1
                output_line = filtered.readline()
                if output_line != raw_line:
                    raise ValueError(
                        f"filtered output diverges from source at retained row {retained} "
                        f"(input line {input_rows})"
                    )
        if filtered.readline():
            raise ValueError("filtered output contains extra rows")
    if set(decision_map) != set(range(1, input_rows + 1)):
        raise ValueError("decision coverage does not match the source")
    return {"input_rows": input_rows, "retained_rows": retained}


def _load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )


def _prepare_command(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    calibration = (
        DEFAULT_CALIBRATION
        if input_path.resolve() == DEFAULT_INPUT.resolve()
        else {}
    )
    manifest = prepare_dataset(
        input_path=input_path,
        output_root=Path(args.output_root),
        tokenizer=_load_tokenizer(Path(args.model_path)),
        model_path=Path(args.model_path),
        num_shards=args.num_shards,
        pilot_size=args.pilot_size,
        pilot_seed=args.pilot_seed,
        calibration=calibration,
        context_length=args.context_length,
        max_completion_tokens=args.max_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _judge_command(args: argparse.Namespace) -> None:
    result = judge_shard(
        input_path=Path(args.input),
        output_a=Path(args.output_a),
        output_b=Path(args.output_b),
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps(result, sort_keys=True))


def _check_pilot_command(args: argparse.Namespace) -> None:
    print(json.dumps(check_pilot(output_root=Path(args.output_root)), indent=2))


def _finalize_command(args: argparse.Namespace) -> None:
    output_path = Path(args.output) if args.output else None
    summary = finalize_dataset(
        input_path=Path(args.input),
        output_root=Path(args.output_root),
        output_path=output_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _validate_command(args: argparse.Namespace) -> None:
    result = validate_filtered_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        decisions_path=Path(args.decisions),
    )
    print(json.dumps(result, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare pilot and 16 review shards")
    prepare.add_argument("--input", default=str(DEFAULT_INPUT))
    prepare.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    prepare.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    prepare.add_argument("--num-shards", type=int, default=16)
    prepare.add_argument("--pilot-size", type=int, default=128)
    prepare.add_argument("--pilot-seed", type=int, default=20260906)
    prepare.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    prepare.add_argument("--max-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    prepare.set_defaults(func=_prepare_command)

    judge = subparsers.add_parser("judge", help="run both independent passes for one shard")
    judge.add_argument("--input", required=True)
    judge.add_argument("--output-a", required=True)
    judge.add_argument("--output-b", required=True)
    judge.add_argument("--base-url", required=True)
    judge.add_argument("--model", default=DEFAULT_MODEL_NAME)
    judge.add_argument("--concurrency", type=int, default=2)
    judge.add_argument("--timeout", type=int, default=900)
    judge.add_argument("--retries", type=int, default=1)
    judge.set_defaults(func=_judge_command)

    pilot = subparsers.add_parser("check-pilot", help="merge and gate pilot decisions")
    pilot.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    pilot.set_defaults(func=_check_pilot_command)

    finalize = subparsers.add_parser("finalize", help="merge reviews and copy retained lines")
    finalize.add_argument("--input", default=str(DEFAULT_INPUT))
    finalize.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    finalize.add_argument("--output")
    finalize.set_defaults(func=_finalize_command)

    validate = subparsers.add_parser("validate", help="verify byte-identical retained rows")
    validate.add_argument("--input", default=str(DEFAULT_INPUT))
    validate.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_ROOT
            / "data/agent_official_native_expanded_turn_filtered.jsonl"
        ),
    )
    validate.add_argument(
        "--decisions",
        default=str(DEFAULT_OUTPUT_ROOT / "reviews/turn_decisions.jsonl"),
    )
    validate.set_defaults(func=_validate_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
