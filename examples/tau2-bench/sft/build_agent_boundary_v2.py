#!/usr/bin/env python3
"""Build the two signed tau2 Agent/User-boundary SFT v2 training arms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from agent_contract import (  # noqa: E402
    BOUNDARY_ANCHOR_VIEW,
    OFFICIAL_AGENT_VIEW,
    AgentContract,
    contract_from_environment,
    sha256_text,
)
from protocol_profiles import (  # noqa: E402
    PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
    protocol_signature,
)


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
DEFAULT_SOURCE = SERVICE_AGENT_ROOT / "datasets/AReaL-tau2-data/tau2_sft_train.jsonl"
DEFAULT_BASE_TRAINING = (
    SERVICE_AGENT_ROOT
    / "slime/output/experiments/tau2-sft-local-first-relaxed/sft_train_budget_matched_19318.jsonl"
)
DEFAULT_LEGACY_TRAINING = (
    SERVICE_AGENT_ROOT
    / "slime/output/datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking_max8192.jsonl"
)
DEFAULT_FILTER_DECISIONS = (
    SERVICE_AGENT_ROOT
    / "slime/output/experiments/tau2-sft-local-first-relaxed/filter_decisions.jsonl"
)
DEFAULT_TOKENIZER = SERVICE_AGENT_ROOT / "models/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT = (
    SERVICE_AGENT_ROOT / "slime/output/experiments/tau2-sft-agent-user-boundary-v2"
)

DATASET_VERSION = "tau2-agent-user-boundary-sft-v2"
MAX_TOTAL_TOKENS = 16384
CONTRACT_ONLY_COUNTS = {"airline": 8873, "retail": 10092, "telecom": 353}
BOUNDARY_COUNTS = {"airline": 8783, "retail": 8595, "telecom": 1940}
TELECOM_RECIPE_COUNTS = {
    "ordinary": 970,
    "natural_handoff": 580,
    "ownership_repair": 390,
}
MIN_ORDINARY_UNIQUE_TARGETS = 485
MIN_ORDINARY_DIALOGS = 100
MIN_HANDOFF_DIALOGS = 100
EXPECTED_HANDOFF_CANDIDATES = 762
EXPECTED_NATURAL_USER_TOOLS = 25
EXPECTED_TELECOM_USER_TOOLS = 30
OWNERSHIP_CONTEXTS_PER_TOOL = 13
EXPECTED_LEGACY_TELECOM_ROWS = 1940
EXPECTED_LEGACY_TELECOM_DIALOGS = 497

_OWNERSHIP_CONTEXTS = (
    "The policy suggests this next action: {candidate}. What should you do?",
    "A troubleshooting note proposes {candidate}. Continue safely.",
    "The next diagnostic is recorded as {candidate}. Explain the next step.",
    "Review this proposed operation before acting: {candidate}.",
    "The device manual names this candidate action: {candidate}.",
    "A prior agent left this proposed next action: {candidate}.",
    "The troubleshooting checklist currently points to {candidate}.",
    "Decide how to handle this tool ownership record: {candidate}.",
    "The user asks you to take this device-side action: {candidate}.",
    "The workflow says the following action belongs next: {candidate}.",
    "Handle this explicit requestor annotation correctly: {candidate}.",
    "The current candidate call is {candidate}. Respond under the tool boundary.",
    "Before continuing, interpret this ownership-tagged action: {candidate}.",
)


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(stable_json(row) + "\n")
            count += 1
    temporary.replace(path)
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def infer_domain(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    domain = str(metadata.get("domain") or "").lower()
    if domain in {"airline", "retail", "telecom"}:
        return domain
    system = " ".join(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if message.get("role") == "system"
    ).lower()
    for candidate in ("airline", "retail", "telecom"):
        if candidate in system:
            return candidate
    raise ValueError("could not infer tau2 domain")


def source_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = copy.deepcopy(list(row.get("messages") or []))
    answer = row.get("answer")
    if isinstance(answer, Mapping):
        messages.append(copy.deepcopy(dict(answer)))
    if not messages:
        raise ValueError("source row has no messages")
    return messages


def _mask_only_final_assistant(messages: list[dict[str, Any]]) -> None:
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices or assistant_indices[-1] != len(messages) - 1:
        raise ValueError("SFT target must be the final Assistant message")
    for message in messages:
        message["step_loss_mask"] = 0
    messages[assistant_indices[-1]]["step_loss_mask"] = 1


def _tokenize_and_validate_row(
    row: dict[str, Any],
    *,
    generator,
    max_total_tokens: int,
) -> int:
    token_ids, loss_mask = generator.get_loss_mask(
        row["messages"],
        tools=row["tools"],
    )
    expected = generator.tokenizer.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        tokenize=True,
        return_dict=False,
    )
    if token_ids != expected:
        raise ValueError("qwen3_full token IDs differ from the complete native template")
    if len(token_ids) != len(loss_mask) or not any(loss_mask):
        raise ValueError("invalid qwen3_full token/loss mask")
    if len(token_ids) > max_total_tokens:
        raise ValueError(
            f"row has {len(token_ids)} tokens, above the no-truncation cap {max_total_tokens}"
        )
    return len(token_ids)


def convert_source_target(
    source_row: Mapping[str, Any],
    *,
    source_row_index: int,
    contract: AgentContract,
    generator,
    inherited_metadata: Mapping[str, Any] | None = None,
    view: str = OFFICIAL_AGENT_VIEW,
    anchor_type: str | None = None,
    source_message_override: Sequence[Mapping[str, Any]] | None = None,
    provenance_events: Sequence[Mapping[str, Any]] | None = None,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> dict[str, Any]:
    messages = list(
        source_messages(source_row)
        if source_message_override is None
        else source_message_override
    )
    agent_view = contract.build_source_view(messages)
    visible = copy.deepcopy(agent_view.messages)
    _mask_only_final_assistant(visible)
    metadata = copy.deepcopy(dict(inherited_metadata or source_row.get("metadata") or {}))
    metadata.update(
        {
            "dataset_version": DATASET_VERSION,
            "source_dataset": "AReaL-tau2-data",
            "source_row": source_row_index,
            "domain": contract.domain,
            "protocol_hash": protocol_signature(contract.profile),
            **contract.metadata(view=view, anchor_type=anchor_type),
            "user_tool_events": copy.deepcopy(
                list(provenance_events)
                if provenance_events is not None
                else agent_view.user_tool_events
            ),
        }
    )
    row = {
        "messages": visible,
        "tools": copy.deepcopy(contract.tools),
        "metadata": metadata,
    }
    total_tokens = _tokenize_and_validate_row(
        row,
        generator=generator,
        max_total_tokens=max_total_tokens,
    )
    row["metadata"]["total_tokens"] = total_tokens
    row["metadata"]["max_total_tokens"] = max_total_tokens
    return row


def rebuild_contract_only_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    base_rows: Sequence[Mapping[str, Any]],
    contracts: Mapping[str, AgentContract],
    generator,
    replacement_candidates: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
    expected_counts: Mapping[str, int] = CONTRACT_ONLY_COUNTS,
) -> list[dict[str, Any]]:
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    replacement_info: dict[tuple[str, int], dict[str, Any]] = {}
    replacement_cursor = Counter()
    used_replacement_sources: set[tuple[str, int]] = set()
    base_source_keys = {
        (str((row.get("metadata") or {}).get("domain")),
         (row.get("metadata") or {}).get("source_row"))
        for row in base_rows
    }
    rebuilt: list[dict[str, Any]] = []
    for base_index, base_row in enumerate(base_rows):
        metadata = base_row.get("metadata") or {}
        source_row_index = metadata.get("source_row")
        domain = metadata.get("domain")
        if not isinstance(source_row_index, int) or isinstance(source_row_index, bool):
            raise ValueError(f"base row {base_index}: missing integer source_row")
        if domain not in contracts:
            raise ValueError(f"base row {base_index}: invalid domain {domain!r}")
        cache_key = (str(domain), source_row_index)
        if cache_key not in cache:
            try:
                cache[cache_key] = convert_source_target(
                    source_rows[source_row_index],
                    source_row_index=source_row_index,
                    contract=contracts[str(domain)],
                    generator=generator,
                    inherited_metadata=metadata,
                    max_total_tokens=max_total_tokens,
                )
            except ValueError as original_error:
                domain_candidates = list(
                    (replacement_candidates or {}).get(str(domain), ())
                )
                replacement_row = None
                while replacement_cursor[str(domain)] < len(domain_candidates):
                    candidate = domain_candidates[replacement_cursor[str(domain)]]
                    replacement_cursor[str(domain)] += 1
                    candidate_source_row = candidate.get("source_row")
                    if (
                        not isinstance(candidate_source_row, int)
                        or isinstance(candidate_source_row, bool)
                        or not 0 <= candidate_source_row < len(source_rows)
                    ):
                        raise ValueError(
                            f"invalid {domain} replacement source_row: "
                            f"{candidate_source_row!r}"
                        )
                    candidate_key = (str(domain), candidate_source_row)
                    if (
                        candidate_key in base_source_keys
                        or candidate_key in used_replacement_sources
                    ):
                        continue
                    converted_row = candidate.get("converted_row")
                    if converted_row is not None:
                        replacement_row = copy.deepcopy(dict(converted_row))
                    else:
                        try:
                            replacement_row = convert_source_target(
                                source_rows[candidate_source_row],
                                source_row_index=candidate_source_row,
                                contract=contracts[str(domain)],
                                generator=generator,
                                max_total_tokens=max_total_tokens,
                            )
                        except ValueError:
                            continue
                    used_replacement_sources.add(candidate_key)
                    decision_sha256 = candidate.get("decision_sha256")
                    replacement_info[cache_key] = {
                        "original_source_row": source_row_index,
                        "original_rejection": str(original_error),
                        "replacement_source_row": candidate_source_row,
                        "replacement_filter_decision_sha256": decision_sha256,
                    }
                    replacement_row["metadata"]["quality_filter"] = {
                        "version": candidate.get("filter_version"),
                        "protocol_profile": candidate.get("protocol_profile"),
                        "review_tier": candidate.get("review_tier"),
                        "decision_sha256": decision_sha256,
                        "target_only_loss": True,
                    }
                    if candidate.get("ordinary_review_sha256"):
                        replacement_row["metadata"][
                            "contract_only_replacement_review_sha256"
                        ] = candidate["ordinary_review_sha256"]
                    break
                if replacement_row is None:
                    raise ValueError(
                        f"base row {base_index}: rejected by the Agent view "
                        f"({original_error}) and no strict same-domain replacement remains"
                    ) from original_error
                cache[cache_key] = replacement_row
        row = copy.deepcopy(cache[cache_key])
        if cache_key in replacement_info:
            info = copy.deepcopy(replacement_info[cache_key])
            info["original_schedule_index"] = base_index
            row["metadata"]["contract_only_replacement"] = info
            schedule = copy.deepcopy(metadata.get("training_schedule") or {})
            if schedule:
                schedule["filter_decision_sha256"] = info[
                    "replacement_filter_decision_sha256"
                ]
                row["metadata"]["training_schedule"] = schedule
        else:
            # Preserve each repeated row's deterministic schedule provenance.
            for key, value in metadata.items():
                if key not in {
                    "agent_contract_signature",
                    "policy_hash",
                    "schema_hash",
                    "user_schema_hash",
                    "chat_template_hash",
                    "view",
                    "user_tool_events",
                    "total_tokens",
                    "max_total_tokens",
                }:
                    row["metadata"][key] = copy.deepcopy(value)
        row["metadata"]["contract_only_schedule_index"] = base_index
        rebuilt.append(row)

    counts = Counter(row["metadata"]["domain"] for row in rebuilt)
    if dict(counts) != dict(expected_counts):
        raise ValueError(
            f"contract-only distribution changed: {dict(counts)} != {dict(expected_counts)}"
        )
    return rebuilt


def contract_only_replacement_candidates(
    filter_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic, previously approved same-domain replacement targets."""

    candidates: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for decision in filter_decisions:
        domain = decision.get("domain")
        target = decision.get("target") or {}
        source_row_index = target.get("source_row")
        if (
            domain not in CONTRACT_ONLY_COUNTS
            or decision.get("label_group") != "consensus_success"
            or decision.get("keep") is not True
            or not isinstance(source_row_index, int)
            or isinstance(source_row_index, bool)
        ):
            continue
        candidates[str(domain)][source_row_index] = {
            "source_row": source_row_index,
            "filter_version": decision.get("filter_version"),
            "protocol_profile": decision.get("protocol_profile"),
            "review_tier": decision.get("review_tier"),
            "decision_sha256": decision.get("decision_sha256"),
        }
    return {
        domain: sorted(
            rows.values(),
            key=lambda row: stable_hash(
                [
                    DATASET_VERSION,
                    "contract-only-strict-replacement",
                    domain,
                    row["source_row"],
                    row["decision_sha256"],
                ]
            ),
        )
        for domain, rows in candidates.items()
    }


def reviewed_contract_only_replacement_candidates(
    approved_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Use only cryptographically validated boundary-v2 reviews for new targets."""

    candidates = []
    for row in approved_rows:
        metadata = row.get("metadata") or {}
        source_row_index = metadata.get("source_row")
        review_sha256 = metadata.get("ordinary_review_sha256")
        if (
            metadata.get("domain") != "telecom"
            or not isinstance(source_row_index, int)
            or isinstance(source_row_index, bool)
            or not isinstance(review_sha256, str)
            or not review_sha256.startswith("sha256:")
        ):
            raise ValueError("approved ordinary row has invalid replacement provenance")
        candidates.append(
            {
                "source_row": source_row_index,
                "filter_version": DATASET_VERSION,
                "protocol_profile": PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
                "review_tier": metadata.get("ordinary_review_model"),
                "decision_sha256": review_sha256,
                "ordinary_review_sha256": review_sha256,
                "converted_row": copy.deepcopy(dict(row)),
            }
        )
    return sorted(
        candidates,
        key=lambda row: stable_hash(
            [
                DATASET_VERSION,
                "contract-only-reviewed-replacement",
                row["source_row"],
                row["decision_sha256"],
            ]
        ),
    )


def _tool_name(call: Mapping[str, Any]) -> str | None:
    function = call.get("function")
    if isinstance(function, Mapping):
        return function.get("name")
    return call.get("name")


def _longest_telecom_dialogs(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    source_row_indices: Iterable[int] | None = None,
) -> dict[str, tuple[int, list[dict[str, Any]], Mapping[str, Any]]]:
    longest: dict[str, tuple[int, list[dict[str, Any]], Mapping[str, Any]]] = {}
    indices = range(len(source_rows)) if source_row_indices is None else source_row_indices
    for source_row_index in indices:
        if not isinstance(source_row_index, int) or isinstance(source_row_index, bool):
            raise ValueError("legacy training metadata contains a non-integer source_row")
        if not 0 <= source_row_index < len(source_rows):
            raise ValueError(f"legacy source_row is out of range: {source_row_index}")
        row = source_rows[source_row_index]
        if infer_domain(row) != "telecom":
            continue
        metadata = row.get("metadata") or {}
        dialog_id = str(metadata.get("source_dialog_id") or "")
        if not dialog_id:
            raise ValueError(f"source row {source_row_index} has no telecom dialog id")
        messages = source_messages(row)
        current = longest.get(dialog_id)
        if current is None or len(messages) > len(current[1]):
            longest[dialog_id] = (source_row_index, messages, metadata)
    return longest


def _handoff_shape(
    messages: Sequence[Mapping[str, Any]],
    call_index: int,
) -> tuple[int, int] | None:
    """Return the plain report and following Assistant target indices."""

    call_message = messages[call_index]
    if call_message.get("role") != "user" or not call_message.get("tool_calls"):
        return None
    report_index = call_index + 1
    while report_index < len(messages) and messages[report_index].get("role") == "tool":
        report_index += 1
    if report_index == call_index + 1 or report_index + 1 >= len(messages):
        return None
    report = messages[report_index]
    target = messages[report_index + 1]
    if (
        report.get("role") != "user"
        or report.get("tool_calls")
        or not str(report.get("content") or "").strip()
        or target.get("role") != "assistant"
    ):
        return None
    return report_index, report_index + 1


def _legacy_handoff_inventory(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    legacy_training_rows: Sequence[Mapping[str, Any]],
) -> tuple[set[tuple[str, int, int]], set[str]]:
    telecom_rows = [
        row
        for row in legacy_training_rows
        if (row.get("metadata") or {}).get("domain") == "telecom"
    ]
    if len(telecom_rows) != EXPECTED_LEGACY_TELECOM_ROWS:
        raise ValueError(
            "legacy telecom row count changed: "
            f"{len(telecom_rows)} != {EXPECTED_LEGACY_TELECOM_ROWS}"
        )
    legacy_longest = _longest_telecom_dialogs(
        source_rows,
        source_row_indices=[
            (row.get("metadata") or {}).get("source_row") for row in telecom_rows
        ],
    )
    if len(legacy_longest) != EXPECTED_LEGACY_TELECOM_DIALOGS:
        raise ValueError(
            "legacy telecom dialog count changed: "
            f"{len(legacy_longest)} != {EXPECTED_LEGACY_TELECOM_DIALOGS}"
        )

    inventory: set[tuple[str, int, int]] = set()
    observed_tools: set[str] = set()
    for dialog_id, (_, messages, _) in legacy_longest.items():
        for call_index, message in enumerate(messages):
            if message.get("role") != "user" or not message.get("tool_calls"):
                continue
            names = {_tool_name(call) for call in message.get("tool_calls") or []}
            if None in names:
                raise ValueError("legacy User call is missing a tool name")
            observed_tools.update(str(name) for name in names)
            shape = _handoff_shape(messages, call_index)
            if shape is not None:
                _, target_index = shape
                inventory.add((dialog_id, call_index, target_index))

    if len(inventory) != EXPECTED_HANDOFF_CANDIDATES:
        raise ValueError(
            "legacy natural-handoff inventory changed: "
            f"{len(inventory)} != {EXPECTED_HANDOFF_CANDIDATES}"
        )
    if len(observed_tools) != EXPECTED_NATURAL_USER_TOOLS:
        raise ValueError(
            "legacy naturally occurring User-tool count changed: "
            f"{len(observed_tools)} != {EXPECTED_NATURAL_USER_TOOLS}"
        )
    return inventory, observed_tools


def extract_natural_handoff_candidates(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    legacy_training_rows: Sequence[Mapping[str, Any]],
    contract: AgentContract,
    generator,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> list[dict[str, Any]]:
    legacy_inventory, observed_tools = _legacy_handoff_inventory(
        source_rows=source_rows,
        legacy_training_rows=legacy_training_rows,
    )
    candidates: dict[str, dict[str, Any]] = {}
    for dialog_id, (source_row_index, messages, source_metadata) in (
        _longest_telecom_dialogs(source_rows).items()
    ):
        for call_index, user_call in enumerate(messages):
            shape = _handoff_shape(messages, call_index)
            if shape is None:
                continue
            report_index, target_index = shape
            names = {_tool_name(call) for call in user_call.get("tool_calls") or []}
            if None in names or not names <= observed_tools:
                continue
            try:
                target_view = contract.build_source_view(messages[: target_index + 1])
            except ValueError:
                continue
            handoff_events = [
                event
                for event in target_view.user_tool_events
                if event["source_call_message_index"] == call_index
            ]
            if len(handoff_events) != len(user_call.get("tool_calls") or []):
                continue
            visible = copy.deepcopy(target_view.messages)
            _mask_only_final_assistant(visible)
            all_events = []
            anchor_events = []
            for event in target_view.user_tool_events:
                annotated = {
                    **copy.deepcopy(event),
                    "anchor_event": event["source_call_message_index"] == call_index,
                }
                all_events.append(annotated)
                if annotated["anchor_event"]:
                    anchor_events.append(copy.deepcopy(annotated))
            metadata = copy.deepcopy(dict(source_metadata))
            metadata.update(
                {
                    "dataset_version": DATASET_VERSION,
                    "source_dataset": "AReaL-tau2-data",
                    "source_row": source_row_index,
                    "source_dialog_id": dialog_id,
                    "source_target_message_index": target_index,
                    "source_user_call_message_index": call_index,
                    "source_user_report_message_index": report_index,
                    "domain": "telecom",
                    "protocol_hash": protocol_signature(contract.profile),
                    **contract.metadata(
                        view=BOUNDARY_ANCHOR_VIEW,
                        anchor_type="natural_handoff",
                    ),
                    "user_tool_events": all_events,
                    "anchor_user_tool_events": anchor_events,
                    "legacy_handoff_inventory_member": (
                        dialog_id,
                        call_index,
                        target_index,
                    )
                    in legacy_inventory,
                }
            )
            row = {
                "messages": visible,
                "tools": copy.deepcopy(contract.tools),
                "metadata": metadata,
            }
            try:
                total_tokens = _tokenize_and_validate_row(
                    row,
                    generator=generator,
                    max_total_tokens=max_total_tokens,
                )
            except ValueError:
                continue
            row["metadata"]["total_tokens"] = total_tokens
            row["metadata"]["max_total_tokens"] = max_total_tokens
            fingerprint = stable_hash(
                [
                    dialog_id,
                    call_index,
                    report_index,
                    target_index,
                    messages[target_index],
                    handoff_events,
                ]
            )
            row["metadata"]["anchor_fingerprint"] = fingerprint
            candidates[fingerprint] = row

    ordered = sorted(
        candidates.values(),
        key=lambda row: (
            not row["metadata"]["legacy_handoff_inventory_member"],
            stable_hash(
                [
                    DATASET_VERSION,
                    "sealed-natural-handoff-pool",
                    row["metadata"]["anchor_fingerprint"],
                ]
            ),
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    covered_tools: set[str] = set()
    covered_dialogs: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        fingerprint = row["metadata"]["anchor_fingerprint"]
        if fingerprint in selected_ids:
            return
        selected.append(copy.deepcopy(row))
        selected_ids.add(fingerprint)
        covered_tools.update(_candidate_tool_names(row))
        covered_dialogs.add(str(row["metadata"]["source_dialog_id"]))

    while covered_tools != observed_tools:
        missing = observed_tools - covered_tools
        choice = next(
            (row for row in ordered if _candidate_tool_names(row) & missing),
            None,
        )
        if choice is None:
            raise ValueError(
                f"strict handoff candidates cannot cover User tools: {sorted(missing)}"
            )
        add(choice)
    for row in ordered:
        if len(covered_dialogs) >= MIN_HANDOFF_DIALOGS:
            break
        if str(row["metadata"]["source_dialog_id"]) not in covered_dialogs:
            add(row)
    for row in ordered:
        if len(selected) == EXPECTED_HANDOFF_CANDIDATES:
            break
        add(row)
    if len(selected) != EXPECTED_HANDOFF_CANDIDATES:
        raise ValueError(
            f"need {EXPECTED_HANDOFF_CANDIDATES} strict handoff candidates, "
            f"found {len(selected)}"
        )
    if covered_tools != observed_tools or len(covered_dialogs) < MIN_HANDOFF_DIALOGS:
        raise ValueError("sealed handoff pool does not satisfy coverage constraints")
    for pool_index, row in enumerate(selected):
        row["metadata"]["handoff_candidate_pool_index"] = pool_index
    return selected


def _candidate_tool_names(row: Mapping[str, Any]) -> set[str]:
    metadata = row.get("metadata") or {}
    return {
        str(event["name"])
        for event in (
            metadata.get("anchor_user_tool_events")
            or metadata.get("user_tool_events")
            or []
        )
    }


def select_natural_handoffs(
    candidates: Sequence[dict[str, Any]],
    *,
    count: int = TELECOM_RECIPE_COUNTS["natural_handoff"],
    min_dialogs: int = MIN_HANDOFF_DIALOGS,
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"need {count} unique handoff anchors, found {len(candidates)}")
    ordered = sorted(
        candidates,
        key=lambda row: stable_hash(
            [DATASET_VERSION, "natural_handoff", row["metadata"]["anchor_fingerprint"]]
        ),
    )
    all_tools = set().union(*(_candidate_tool_names(row) for row in ordered))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    covered_tools: set[str] = set()
    covered_dialogs: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        fingerprint = row["metadata"]["anchor_fingerprint"]
        if fingerprint in selected_ids:
            return
        selected.append(copy.deepcopy(row))
        selected_ids.add(fingerprint)
        covered_tools.update(_candidate_tool_names(row))
        covered_dialogs.add(str(row["metadata"]["source_dialog_id"]))

    while covered_tools != all_tools:
        missing = all_tools - covered_tools
        choice = next((row for row in ordered if _candidate_tool_names(row) & missing), None)
        if choice is None:
            raise ValueError(f"cannot cover natural User tools: {sorted(missing)}")
        add(choice)
    for row in ordered:
        if len(covered_dialogs) >= min_dialogs:
            break
        if str(row["metadata"]["source_dialog_id"]) not in covered_dialogs:
            add(row)
    for row in ordered:
        if len(selected) == count:
            break
        add(row)
    if len(selected) != count or len(covered_dialogs) < min_dialogs:
        raise ValueError("natural handoff coverage constraints are not satisfiable")
    return selected


def _empty_arguments(schema: Mapping[str, Any]) -> dict[str, Any]:
    function = schema["function"]
    properties = (function.get("parameters") or {}).get("properties") or {}
    return {name: f"<{name}>" for name in properties}


def build_ownership_repair_anchors(
    *,
    contract: AgentContract,
    generator,
    contexts_per_tool: int = OWNERSHIP_CONTEXTS_PER_TOOL,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> list[dict[str, Any]]:
    if contexts_per_tool != len(_OWNERSHIP_CONTEXTS):
        raise ValueError(
            f"ownership contexts must remain fixed at {len(_OWNERSHIP_CONTEXTS)}"
        )
    rows: list[dict[str, Any]] = []
    for schema in sorted(contract.user_tools, key=lambda item: item["function"]["name"]):
        name = schema["function"]["name"]
        description = str(schema["function"].get("description") or name).split("\n", 1)[0]
        candidate = stable_json(
            {
                "requestor": "user",
                "name": name,
                "arguments": _empty_arguments(schema),
            }
        )
        target = (
            f"Please perform the `{name}` action on your device ({description}). "
            "When it finishes, tell me in your own words what happened. I'll wait for "
            "your report before continuing."
        )
        for context_index, template in enumerate(_OWNERSHIP_CONTEXTS):
            messages = [
                {"role": "system", "content": contract.system_prompt, "step_loss_mask": 0},
                {
                    "role": "user",
                    "content": template.format(candidate=candidate),
                    "step_loss_mask": 0,
                },
                {"role": "assistant", "content": target, "step_loss_mask": 1},
            ]
            row = {
                "messages": messages,
                "tools": copy.deepcopy(contract.tools),
                "metadata": {
                    "dataset_version": DATASET_VERSION,
                    "domain": "telecom",
                    "protocol_hash": protocol_signature(contract.profile),
                    **contract.metadata(
                        view=BOUNDARY_ANCHOR_VIEW,
                        anchor_type="ownership_repair",
                    ),
                    "ownership_requestor": "user",
                    "ownership_tool_name": name,
                    "ownership_context_index": context_index,
                    "user_tool_events": [],
                },
            }
            total_tokens = _tokenize_and_validate_row(
                row,
                generator=generator,
                max_total_tokens=max_total_tokens,
            )
            row["metadata"]["total_tokens"] = total_tokens
            row["metadata"]["max_total_tokens"] = max_total_tokens
            rows.append(row)
    return rows


def _quality_approved(
    record: Mapping[str, Any],
    *,
    original_messages: Sequence[Mapping[str, Any]],
    contract: AgentContract,
) -> bool:
    review = record.get("review") or {}
    model = str(review.get("model") or review.get("reviewer_model") or "")
    declared_hash = record.get("review_sha256")
    repaired_messages = record.get("repaired_source_messages")
    reviewed_messages = repaired_messages if repaired_messages is not None else original_messages
    return (
        record.get("approved") is True
        and review.get("quality_verdict") == "keep"
        and review.get("all_quality_gates_passed") is True
        and "Qwen3.6-27B" in model
        and review.get("agent_contract_signature")
        == contract.agent_contract_signature
        and review.get("original_source_sha256")
        == sha256_text(stable_json(original_messages))
        and review.get("reviewed_source_sha256")
        == sha256_text(stable_json(reviewed_messages))
        and declared_hash == sha256_text(stable_json(review))
    )


def load_approved_ordinary_rows(
    review_records: Sequence[Mapping[str, Any]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    contract: AgentContract,
    generator,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in review_records:
        source_row_index = record.get("source_row")
        if not isinstance(source_row_index, int) or isinstance(source_row_index, bool):
            raise ValueError("approved ordinary record is missing integer source_row")
        if not 0 <= source_row_index < len(source_rows):
            raise ValueError(f"approved ordinary source_row is out of range: {source_row_index}")
        original_messages = source_messages(source_rows[source_row_index])
        if not _quality_approved(
            record,
            original_messages=original_messages,
            contract=contract,
        ):
            continue
        source_override = record.get("repaired_source_messages")
        row = convert_source_target(
            source_rows[source_row_index],
            source_row_index=source_row_index,
            contract=contract,
            generator=generator,
            inherited_metadata=source_rows[source_row_index].get("metadata") or {},
            source_message_override=source_override,
            max_total_tokens=max_total_tokens,
        )
        metadata = row["metadata"]
        metadata["ordinary_review_sha256"] = record["review_sha256"]
        metadata["ordinary_review_model"] = (
            (record.get("review") or {}).get("model")
            or (record.get("review") or {}).get("reviewer_model")
        )
        metadata["ordinary_repaired"] = source_override is not None
        target_key = str(record.get("target_id") or source_row_index)
        metadata["ordinary_target_id"] = target_key
        rows[target_key] = row
    return sorted(rows.values(), key=lambda row: stable_hash(row["metadata"]["ordinary_target_id"]))


def select_ordinary_rows(
    rows: Sequence[dict[str, Any]],
    *,
    count: int = TELECOM_RECIPE_COUNTS["ordinary"],
) -> list[dict[str, Any]]:
    if len(rows) < MIN_ORDINARY_UNIQUE_TARGETS:
        raise ValueError(
            f"need at least {MIN_ORDINARY_UNIQUE_TARGETS} approved telecom targets, found {len(rows)}"
        )
    ordered = sorted(
        rows,
        key=lambda row: stable_hash(
            [DATASET_VERSION, "ordinary", row["metadata"]["ordinary_target_id"]]
        ),
    )
    selected_unique = ordered[: min(len(ordered), count)]
    dialogs = {
        str(row["metadata"].get("source_dialog_id"))
        for row in selected_unique
        if row["metadata"].get("source_dialog_id")
    }
    if len(dialogs) < MIN_ORDINARY_DIALOGS:
        raise ValueError(
            f"ordinary targets cover {len(dialogs)} dialogs; need {MIN_ORDINARY_DIALOGS}"
        )
    selected = [copy.deepcopy(row) for row in selected_unique]
    duplicate_count = count - len(selected)
    if duplicate_count > len(selected_unique):
        raise ValueError("ordinary recipe would sample a target more than twice")
    selected.extend(copy.deepcopy(row) for row in selected_unique[:duplicate_count])
    for schedule_index, row in enumerate(selected):
        row["metadata"]["ordinary_schedule_index"] = schedule_index
    return selected


def _take_domain_rows(
    contract_only: Sequence[dict[str, Any]],
    *,
    domain: str,
    count: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in contract_only if row["metadata"]["domain"] == domain]
    if len(candidates) < count:
        raise ValueError(f"not enough {domain} rows: {len(candidates)} < {count}")
    return [copy.deepcopy(row) for row in candidates[:count]]


def assemble_boundary_arm(
    *,
    contract_only: Sequence[dict[str, Any]],
    ordinary: Sequence[dict[str, Any]],
    natural_handoffs: Sequence[dict[str, Any]],
    ownership_repairs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    telecom_rows = [*ordinary, *natural_handoffs, *ownership_repairs]
    anchor_counts = Counter(
        row["metadata"].get("anchor_type") or "ordinary" for row in telecom_rows
    )
    if dict(anchor_counts) != TELECOM_RECIPE_COUNTS:
        raise ValueError(
            f"telecom recipe changed: {dict(anchor_counts)} != {TELECOM_RECIPE_COUNTS}"
        )
    rows = [
        *_take_domain_rows(
            contract_only,
            domain="airline",
            count=BOUNDARY_COUNTS["airline"],
        ),
        *_take_domain_rows(
            contract_only,
            domain="retail",
            count=BOUNDARY_COUNTS["retail"],
        ),
        *(copy.deepcopy(row) for row in telecom_rows),
    ]
    rows.sort(
        key=lambda row: stable_hash(
            [
                DATASET_VERSION,
                "contract_boundary",
                row["metadata"].get("source_row"),
                row["metadata"].get("anchor_fingerprint"),
                row["metadata"].get("ownership_tool_name"),
                row["metadata"].get("ownership_context_index"),
                row["metadata"].get("ordinary_schedule_index"),
                row["metadata"].get("contract_only_schedule_index"),
            ]
        )
    )
    counts = Counter(row["metadata"]["domain"] for row in rows)
    if dict(counts) != BOUNDARY_COUNTS:
        raise ValueError(f"boundary distribution changed: {dict(counts)} != {BOUNDARY_COUNTS}")
    for schedule_index, row in enumerate(rows):
        row["metadata"]["boundary_schedule_index"] = schedule_index
    return rows


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    counts = Counter()
    views = Counter()
    anchors = Counter()
    signatures: dict[str, set[str]] = defaultdict(set)
    max_tokens = 0
    replacement_rows = 0
    replacement_sources: set[int] = set()
    rejected_sources: set[int] = set()
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row.get("tools"), list) or not row["tools"]:
            raise ValueError(f"line {line_number}: missing top-level tools")
        messages = row.get("messages") or []
        if not messages or messages[0].get("role") != "system":
            raise ValueError(f"line {line_number}: missing system message")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"line {line_number}: target is not Assistant")
        masks = [message.get("step_loss_mask") for message in messages]
        if masks[-1] != 1 or any(mask != 0 for mask in masks[:-1]):
            raise ValueError(f"line {line_number}: target-only mask changed")
        if any(
            message.get("role") == "user"
            and not str(message.get("content") or "").strip()
            for message in messages
        ):
            raise ValueError(f"line {line_number}: empty User turn")
        metadata = row.get("metadata") or {}
        domain = metadata.get("domain")
        if domain not in expected_counts:
            raise ValueError(f"line {line_number}: invalid domain {domain!r}")
        counts[domain] += 1
        views[metadata.get("view")] += 1
        if metadata.get("anchor_type"):
            anchors[metadata["anchor_type"]] += 1
        signature = metadata.get("agent_contract_signature")
        if not isinstance(signature, str) or not signature.startswith("sha256:"):
            raise ValueError(f"line {line_number}: missing Agent contract signature")
        signatures[str(domain)].add(signature)
        if metadata.get("view") not in {OFFICIAL_AGENT_VIEW, BOUNDARY_ANCHOR_VIEW}:
            raise ValueError(f"line {line_number}: invalid Agent view")
        if not isinstance(metadata.get("user_tool_events"), list):
            raise ValueError(f"line {line_number}: missing User-tool provenance")
        replacement = metadata.get("contract_only_replacement")
        if replacement is not None:
            if not isinstance(replacement, Mapping):
                raise ValueError(f"line {line_number}: malformed contract-only replacement")
            replacement_rows += 1
            replacement_sources.add(int(replacement["replacement_source_row"]))
            rejected_sources.add(int(replacement["original_source_row"]))
        total_tokens = metadata.get("total_tokens")
        if not isinstance(total_tokens, int) or total_tokens > MAX_TOTAL_TOKENS:
            raise ValueError(f"line {line_number}: invalid total_tokens")
        max_tokens = max(max_tokens, total_tokens)
        if any(schema["function"]["name"] == "done" for schema in row["tools"]):
            raise ValueError(f"line {line_number}: fake done tool remains")
        agent_tool_names = {schema["function"]["name"] for schema in row["tools"]}
        for message in messages:
            if message.get("role") == "tool" and not message.get("tool_call_id"):
                raise ValueError(
                    f"line {line_number}: Agent tool result is missing its call id"
                )
            if message.get("role") != "assistant":
                continue
            target_names = {
                call.get("function", {}).get("name")
                for call in message.get("tool_calls") or []
            }
            if not target_names <= agent_tool_names:
                raise ValueError(
                    f"line {line_number}: Assistant target uses non-Agent tools "
                    f"{sorted(target_names - agent_tool_names)}"
                )
        if (
            metadata.get("anchor_type") == "ownership_repair"
            and messages[-1].get("tool_calls")
        ):
            raise ValueError(
                f"line {line_number}: ownership-repair target must be plain text"
            )
    if dict(counts) != dict(expected_counts):
        raise ValueError(f"domain counts changed: {dict(counts)} != {dict(expected_counts)}")
    if any(len(values) != 1 for values in signatures.values()):
        raise ValueError("a domain contains multiple Agent contract signatures")
    return {
        "rows": len(rows),
        "rows_by_domain": dict(sorted(counts.items())),
        "views": dict(sorted(views.items())),
        "anchors": dict(sorted(anchors.items())),
        "max_total_tokens": max_tokens,
        "contract_only_replacement_rows": replacement_rows,
        "contract_only_replacement_targets": len(replacement_sources),
        "contract_only_rejected_targets": len(rejected_sources),
        "agent_contract_signatures": {
            domain: next(iter(values)) for domain, values in sorted(signatures.items())
        },
    }


def longest_smoke_rows(rows: Sequence[dict[str, Any]], count: int = 32) -> list[dict[str, Any]]:
    return sorted(
        (copy.deepcopy(row) for row in rows),
        key=lambda row: (
            -int(row["metadata"]["total_tokens"]),
            stable_hash(row["metadata"]),
        ),
    )[:count]


def consensus_review_candidates(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    filter_decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for decision in filter_decisions:
        if decision.get("domain") != "telecom" or decision.get("label_group") != "consensus_success":
            continue
        target = decision.get("target") or {}
        source_row_index = target.get("source_row")
        if not isinstance(source_row_index, int) or isinstance(source_row_index, bool):
            raise ValueError("consensus-success decision is missing source_row")
        row = source_rows[source_row_index]
        candidates.append(
            {
                "source_row": source_row_index,
                "target_id": f"source_row:{source_row_index}",
                "source_dialog_id": (row.get("metadata") or {}).get("source_dialog_id"),
                "source_messages": source_messages(row),
                "previous_decision": decision.get("decision"),
                "previous_decision_reasons": decision.get("decision_reasons") or [],
                "required_reviewer_model": "Qwen3.6-27B",
                "required_quality_gate": "unchanged-local-first-quality-gate",
            }
        )
    candidates.sort(key=lambda row: row["source_row"])
    return candidates


def _load_contracts(tokenizer) -> dict[str, AgentContract]:
    from tau2.registry import registry

    contracts = {}
    for domain in ("airline", "retail", "telecom"):
        contracts[domain] = contract_from_environment(
            registry.get_env_constructor(domain)(),
            domain=domain,
            chat_template=tokenizer.chat_template,
            profile=PROTOCOL_AGENT_OWNED_DEPENDENCY_SAFE_MULTI,
        )
    return contracts


def _runtime(tokenizer_path: Path):
    from transformers import AutoTokenizer
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    return tokenizer, generator, _load_contracts(tokenizer)


def prepare(args: argparse.Namespace) -> None:
    tokenizer, generator, contracts = _runtime(args.tokenizer)
    del tokenizer
    source_rows = list(read_jsonl(args.source))
    legacy_training_rows = list(read_jsonl(args.legacy_training))
    decisions = list(read_jsonl(args.filter_decisions))
    ordinary_candidates = consensus_review_candidates(
        source_rows=source_rows,
        filter_decisions=decisions,
    )
    if len(ordinary_candidates) != args.expected_ordinary_candidates:
        raise ValueError(
            f"expected {args.expected_ordinary_candidates} ordinary candidates, "
            f"found {len(ordinary_candidates)}"
        )
    handoffs = extract_natural_handoff_candidates(
        source_rows,
        legacy_training_rows=legacy_training_rows,
        contract=contracts["telecom"],
        generator=generator,
    )
    if len(handoffs) != args.expected_handoff_candidates:
        raise ValueError(
            f"expected {args.expected_handoff_candidates} handoff candidates, found {len(handoffs)}"
        )
    natural_tools = set().union(*(_candidate_tool_names(row) for row in handoffs))
    if len(natural_tools) != args.expected_natural_user_tools:
        raise ValueError(
            f"expected {args.expected_natural_user_tools} naturally occurring User tools, "
            f"found {len(natural_tools)}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    ordinary_path = args.output / "ordinary_telecom_review_candidates.jsonl"
    handoff_path = args.output / "natural_handoff_candidates.jsonl"
    write_jsonl(ordinary_path, ordinary_candidates)
    write_jsonl(handoff_path, handoffs)
    write_json(
        args.output / "prepare_manifest.json",
        {
            "dataset_version": DATASET_VERSION,
            "source": str(args.source),
            "source_sha256": file_sha256(args.source),
            "filter_decisions": str(args.filter_decisions),
            "filter_decisions_sha256": file_sha256(args.filter_decisions),
            "legacy_training": str(args.legacy_training),
            "legacy_training_sha256": file_sha256(args.legacy_training),
            "ordinary_candidates": len(ordinary_candidates),
            "handoff_candidates": len(handoffs),
            "handoff_dialogs": len(
                {row["metadata"]["source_dialog_id"] for row in handoffs}
            ),
            "natural_user_tools": sorted(natural_tools),
            "contract_signatures": {
                domain: contract.agent_contract_signature
                for domain, contract in contracts.items()
            },
        },
    )


def build(args: argparse.Namespace) -> None:
    tokenizer, generator, contracts = _runtime(args.tokenizer)
    del tokenizer
    source_rows = list(read_jsonl(args.source))
    base_rows = list(read_jsonl(args.base_training))
    filter_decisions = list(read_jsonl(args.filter_decisions))
    review_records = (
        list(read_jsonl(args.ordinary_reviewed))
        if args.ordinary_reviewed is not None
        else []
    )
    approved_ordinary = (
        load_approved_ordinary_rows(
            review_records,
            source_rows=source_rows,
            contract=contracts["telecom"],
            generator=generator,
        )
        if review_records
        else []
    )
    replacements = contract_only_replacement_candidates(filter_decisions)
    replacements.setdefault("telecom", []).extend(
        reviewed_contract_only_replacement_candidates(approved_ordinary)
    )
    contract_only = rebuild_contract_only_rows(
        source_rows=source_rows,
        base_rows=base_rows,
        contracts=contracts,
        generator=generator,
        replacement_candidates=replacements,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    contract_only_path = args.output / "contract_only_19318.jsonl"
    write_jsonl(contract_only_path, contract_only)
    write_jsonl(
        args.output / "contract_only_longest32_smoke.jsonl",
        longest_smoke_rows(contract_only),
    )
    summaries = {
        "contract_only": validate_rows(
            contract_only,
            expected_counts=CONTRACT_ONLY_COUNTS,
        )
    }

    if args.arm in {"both", "contract-boundary"}:
        if args.ordinary_reviewed is None:
            raise ValueError("contract-boundary build requires --ordinary-reviewed")
        ordinary = select_ordinary_rows(approved_ordinary)
        if args.handoff_candidates is None:
            handoff_candidates = extract_natural_handoff_candidates(
                source_rows,
                legacy_training_rows=list(read_jsonl(args.legacy_training)),
                contract=contracts["telecom"],
                generator=generator,
            )
        else:
            handoff_candidates = list(read_jsonl(args.handoff_candidates))
        natural_handoffs = select_natural_handoffs(handoff_candidates)
        if len(set().union(*(_candidate_tool_names(row) for row in natural_handoffs))) != EXPECTED_NATURAL_USER_TOOLS:
            raise ValueError("selected natural anchors do not cover all 25 observed User tools")
        ownership_repairs = build_ownership_repair_anchors(
            contract=contracts["telecom"],
            generator=generator,
        )
        if len(contracts["telecom"].user_tools) != EXPECTED_TELECOM_USER_TOOLS:
            raise ValueError("official telecom User schema count changed from 30")
        boundary = assemble_boundary_arm(
            contract_only=contract_only,
            ordinary=ordinary,
            natural_handoffs=natural_handoffs,
            ownership_repairs=ownership_repairs,
        )
        boundary_path = args.output / "contract_boundary_19318.jsonl"
        write_jsonl(boundary_path, boundary)
        write_jsonl(
            args.output / "contract_boundary_longest32_smoke.jsonl",
            longest_smoke_rows(boundary),
        )
        summaries["contract_boundary"] = validate_rows(
            boundary,
            expected_counts=BOUNDARY_COUNTS,
        )

    artifacts = {}
    for path in sorted(args.output.glob("*.jsonl")):
        artifacts[path.name] = file_sha256(path)
    write_json(
        args.output / "build_manifest.json",
        {
            "dataset_version": DATASET_VERSION,
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "loss_mask_type": "qwen3_full",
            "source_sha256": file_sha256(args.source),
            "base_training_sha256": file_sha256(args.base_training),
            "filter_decisions_sha256": file_sha256(args.filter_decisions),
            "legacy_training_sha256": file_sha256(args.legacy_training),
            "summaries": summaries,
            "artifacts": artifacts,
        },
    )


def validate(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.input))
    if args.smoke:
        if len(rows) != 32:
            raise ValueError(f"smoke file must contain 32 rows, found {len(rows)}")
        expected = dict(Counter((row.get("metadata") or {}).get("domain") for row in rows))
    else:
        expected = CONTRACT_ONLY_COUNTS if args.arm == "contract-only" else BOUNDARY_COUNTS
    print(json.dumps(validate_rows(rows, expected_counts=expected), indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    common.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    common.add_argument(
        "--legacy-training",
        type=Path,
        default=DEFAULT_LEGACY_TRAINING,
    )

    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    prepare_parser.add_argument(
        "--filter-decisions",
        type=Path,
        default=DEFAULT_FILTER_DECISIONS,
    )
    prepare_parser.add_argument("--expected-ordinary-candidates", type=int, default=715)
    prepare_parser.add_argument(
        "--expected-handoff-candidates",
        type=int,
        default=EXPECTED_HANDOFF_CANDIDATES,
    )
    prepare_parser.add_argument(
        "--expected-natural-user-tools",
        type=int,
        default=EXPECTED_NATURAL_USER_TOOLS,
    )
    prepare_parser.set_defaults(function=prepare)

    build_parser = subparsers.add_parser("build", parents=[common])
    build_parser.add_argument("--base-training", type=Path, default=DEFAULT_BASE_TRAINING)
    build_parser.add_argument(
        "--filter-decisions",
        type=Path,
        default=DEFAULT_FILTER_DECISIONS,
    )
    build_parser.add_argument(
        "--arm",
        choices=("contract-only", "contract-boundary", "both"),
        default="both",
    )
    build_parser.add_argument("--ordinary-reviewed", type=Path, default=None)
    build_parser.add_argument("--handoff-candidates", type=Path, default=None)
    build_parser.set_defaults(function=build)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--input", type=Path, required=True)
    validate_parser.add_argument(
        "--arm",
        choices=("contract-only", "contract-boundary"),
        required=True,
    )
    validate_parser.add_argument("--smoke", action="store_true")
    validate_parser.set_defaults(function=validate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
