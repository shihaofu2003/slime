#!/usr/bin/env python3
"""Build a local-first, target-level filtered tau2 Agent SFT dataset.

The preparation stage reconstructs every consensus-success source dialog under
the dependency-safe-multi protocol and emits local Qwen judge inputs.  The build
stage combines those local reviews with the already sealed Luna reviews where
available, applies target-prefix mechanical and semantic gates, amends the
training prompt to the same protocol, and masks every assistant turn except the
approved final target.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from protocol_profiles import PROTOCOL_DEPENDENCY_SAFE_MULTI  # noqa: E402
from audit_sft_multitool import (
    CANDIDATE_POLICY_RULE,
    _assessment_is_strict_safe,
    _counterfactual_payload,
    _manifest_map,
    _review_reliability,
    _selected_judge_identity,
    _selected_review,
    _split_payloads,
    _validated_review_map,
    build_payload_manifest,
    file_sha256,
    validate_prepare_manifest,
)
from judge_sft_quality import (
    DEPENDENCY_SAFE_QUALITY_DIMENSIONS,
    MULTITOOL_PROMPT_VERSION,
)
from sft_quality import (
    audit_source_dialogs,
    deterministic_features,
    read_jsonl,
    row_messages,
    stable_json_hash,
    write_jsonl,
)
from trajectory_patterns import load_tool_catalog


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
DEFAULT_SOURCE = SERVICE_AGENT_ROOT / "datasets/AReaL-tau2-data/tau2_sft_train.jsonl"
DEFAULT_TRAINING = (
    SERVICE_AGENT_ROOT
    / "slime/output/datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking_max8192.jsonl"
)
DEFAULT_TAU2_ROOT = SERVICE_AGENT_ROOT / "tau2-bench"
DEFAULT_TOKENIZER = SERVICE_AGENT_ROOT / "models/Qwen3.6-27B"
DEFAULT_TRAINING_TOKENIZER = SERVICE_AGENT_ROOT / "models/Qwen3-4B-Instruct-2507"
DEFAULT_REFERENCE_AUDIT = (
    SERVICE_AGENT_ROOT
    / "slime/output/experiments/tau2-sft-multitool-quality-audit"
)
DEFAULT_OUT = (
    SERVICE_AGENT_ROOT
    / "slime/output/experiments/tau2-sft-local-first-relaxed"
)

FILTER_VERSION = "local-first-target-prefix-v1"
TRAINING_SCHEDULE_VERSION = "matched-row-budget-v1"
DEFAULT_BUDGET_ROWS = 19318
DEFAULT_BUDGET_SEED = 20260803

CORE_QUALITY_DIMENSIONS = (
    "business_policy_compliance",
    "task_completion",
    "tool_choice",
    "argument_grounding",
    "dependency_safety",
    "authorization_and_confirmation",
)
SUPPORT_QUALITY_DIMENSIONS = (
    "call_necessity",
    "recovery",
    "communication",
)
if set(CORE_QUALITY_DIMENSIONS) | set(SUPPORT_QUALITY_DIMENSIONS) != set(
    DEPENDENCY_SAFE_QUALITY_DIMENSIONS
):
    raise RuntimeError("local-first quality dimensions do not cover the judge schema")

AIRLINE_SINGLE_CALL_RULES = """CRITICAL RULES (you MUST follow these):
1. In each turn, you can ONLY do ONE of these:
   - Send a message to the user, OR
   - Make exactly ONE tool call
2. You CANNOT make multiple tool calls in a single turn - only ONE tool call per turn!
3. You CANNOT send a message and make a tool call at the same time."""
AIRLINE_DEPENDENCY_SAFE_RULES = """CRITICAL TOOL-CALL RULES (you MUST follow these):
1. In each turn, either send a message to the user OR make one or more tool calls.
2. Multiple calls must satisfy the dependency-safe multi-tool protocol stated above.
3. You CANNOT send a message and make tool calls at the same time."""
PROTOCOL_INSERTION = (
    "## Dependency-safe multi-tool protocol\n"
    + CANDIDATE_POLICY_RULE
)
SYSTEM_SECTION_MARKER = "\n\n---\n\n<instructions>"
SINGLE_CALL_POLICY_RE = re.compile(
    r"You should (?:only|at most) make one tool call at a time, and if you "
    r"(?:make|take) a tool call, you should not respond to the user simultaneously\. "
    r"If you respond to the user, you should not make a tool call at the same time\.",
    re.IGNORECASE,
)
BANNED_SINGLE_CALL_RE = re.compile(
    r"exactly ONE tool call|CANNOT make multiple tool calls|"
    r"(?:only|at most) make one tool call at a time",
    re.IGNORECASE,
)
TOOL_TEXT_EXCLUSIVITY_RULE = (
    "If you make tool calls, you should not respond to the user simultaneously. "
    "If you respond to the user, you should not make tool calls at the same time."
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def amend_training_system(system: str) -> tuple[str, dict[str, bool]]:
    """Apply the reviewed dependency-safe protocol to one converted prompt."""

    if SYSTEM_SECTION_MARKER not in system:
        raise ValueError("training system prompt is missing the instructions boundary")
    amended = system
    inserted_protocol = PROTOCOL_INSERTION not in amended
    if inserted_protocol:
        amended = amended.replace(
            SYSTEM_SECTION_MARKER,
            "\n\n" + PROTOCOL_INSERTION + SYSTEM_SECTION_MARKER,
            1,
        )
    replaced_instruction_rules = AIRLINE_SINGLE_CALL_RULES in amended
    amended = amended.replace(
        AIRLINE_SINGLE_CALL_RULES,
        AIRLINE_DEPENDENCY_SAFE_RULES,
        1,
    )
    amended, policy_replacements = SINGLE_CALL_POLICY_RE.subn(
        TOOL_TEXT_EXCLUSIVITY_RULE,
        amended,
    )
    if BANNED_SINGLE_CALL_RE.search(amended):
        raise ValueError("single-call-only language remains in amended training prompt")
    if amended.count(PROTOCOL_INSERTION) != 1:
        raise ValueError("dependency-safe protocol must appear exactly once")
    if "Make one or more tool calls" not in amended:
        raise ValueError("amended prompt no longer exposes the multi-call output format")
    return amended, {
        "protocol_inserted": inserted_protocol,
        "airline_instruction_rules_replaced": replaced_instruction_rules,
        "domain_policy_clauses_replaced": policy_replacements > 0,
    }


def quality_gate_reasons(
    record: dict[str, Any] | None,
    *,
    side: str,
) -> list[str]:
    """Return failures under the calibrated relaxed semantic gate.

    A reliable ``review`` is accepted when all numeric dimensions are adequate.
    Critical turns remain diagnostic evidence represented by those scores and
    the verdict; the dual-judge calibration did not validate them as a separate
    hard gate.  A determinate ``drop`` remains a hard veto.
    """

    if record is None:
        return [f"{side}:missing_review"]
    reasons = []
    reliability = _review_reliability(record)
    if not reliability["reliable"]:
        reasons.extend(
            f"{side}:unreliable:{reason}" for reason in reliability["reasons"]
        )
    review = _selected_review(record) or {}
    verdict = review.get("quality_verdict")
    if verdict not in {"keep", "review"}:
        reasons.append(f"{side}:quality_verdict:{verdict or 'missing'}")
    dimensions = review.get("dimensions") or {}
    for name in CORE_QUALITY_DIMENSIONS:
        score = (dimensions.get(name) or {}).get("score")
        if isinstance(score, bool) or not isinstance(score, int) or score < 3:
            reasons.append(f"{side}:core_dimension:{name}:{score}")
    for name in SUPPORT_QUALITY_DIMENSIONS:
        score = (dimensions.get(name) or {}).get("score")
        if isinstance(score, bool) or not isinstance(score, int) or score < 2:
            reasons.append(f"{side}:support_dimension:{name}:{score}")
    return reasons


def multitool_prefix_reasons(
    record: dict[str, Any] | None,
    static: dict[str, Any],
    *,
    side: str,
) -> list[str]:
    """Require every multi-call turn in the dependency prefix to be strict-safe."""

    multi_turns = {
        turn
        for turn in static.get("multitool_turns") or []
        if isinstance(turn, int) and not isinstance(turn, bool)
    }
    if not multi_turns:
        return []
    review = _selected_review(record) or {}
    assessments = {
        item.get("turn_index"): item
        for item in review.get("multi_turn_assessments") or []
        if isinstance(item, dict)
        and isinstance(item.get("turn_index"), int)
        and not isinstance(item.get("turn_index"), bool)
    }
    write_turns = {
        turn
        for turn in static.get("multiwrite_turns") or []
        if isinstance(turn, int) and not isinstance(turn, bool)
    }
    reasons = []
    for turn in sorted(multi_turns):
        assessment = assessments.get(turn)
        if assessment is None:
            reasons.append(f"{side}:multi_turn:{turn}:missing_assessment")
        elif not _assessment_is_strict_safe(assessment, write_turns):
            reasons.append(f"{side}:multi_turn:{turn}:not_strict_safe")
    return reasons


def apply_target_only_loss_mask(
    row: dict[str, Any],
    amended_system: str,
) -> dict[str, Any]:
    """Copy a converted row and supervise only its final assistant target."""

    result = copy.deepcopy(row)
    messages = result.get("messages") or []
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices or assistant_indices[-1] != len(messages) - 1:
        raise ValueError("converted SFT row must end at its assistant target")
    for message in messages:
        message["step_loss_mask"] = 0
    messages[assistant_indices[-1]]["step_loss_mask"] = 1
    system_messages = [message for message in messages if message.get("role") == "system"]
    if len(system_messages) != 1:
        raise ValueError("converted SFT row must contain exactly one system message")
    system_messages[0]["content"] = amended_system
    return result


def matched_budget_rows(
    rows: list[dict[str, Any]],
    budget_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically repeat selected rows to match the old row/update budget."""

    if budget_rows < len(rows):
        raise ValueError(
            f"budget_rows={budget_rows} is smaller than canonical selected rows={len(rows)}"
        )
    if not rows and budget_rows:
        raise ValueError("cannot build a non-empty budget from zero selected rows")
    scheduled = []
    cycle = 0
    while len(scheduled) < budget_rows:
        order = sorted(
            rows,
            key=lambda row: stable_json_hash(
                [
                    TRAINING_SCHEDULE_VERSION,
                    seed,
                    cycle,
                    (row.get("metadata") or {}).get("source_row"),
                    (row.get("metadata") or {}).get("source_dialog_id"),
                ]
            ),
        )
        for base_row in order:
            if len(scheduled) == budget_rows:
                break
            item = copy.deepcopy(base_row)
            metadata = item.setdefault("metadata", {})
            quality_filter = metadata.get("quality_filter") or {}
            metadata["training_schedule"] = {
                "version": TRAINING_SCHEDULE_VERSION,
                "seed": seed,
                "cycle": cycle,
                "schedule_index": len(scheduled),
                "filter_decision_sha256": quality_filter.get("decision_sha256"),
            }
            scheduled.append(item)
        cycle += 1
    return scheduled


def artifact_record(path: Path, rows: int | None) -> dict[str, Any]:
    return {"rows": rows, "sha256": file_sha256(path)}


def tokenizer_file_hashes(tokenizer: Path) -> dict[str, str]:
    hashes = {}
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    ):
        path = tokenizer / filename
        if path.is_file():
            hashes[filename] = file_sha256(path)
    return hashes


def validate_reference_inputs(
    reference_audit: Path,
    source: Path,
    training: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_manifest = validate_prepare_manifest(
        reference_audit / "prepare_manifest.json",
        reference_audit,
    )
    actual_hashes = {
        "source": file_sha256(source),
        "training": file_sha256(training),
    }
    if actual_hashes != reference_manifest.get("input_sha256"):
        raise ValueError(
            "source/training inputs differ from the sealed dual-judge reference audit"
        )
    reference_features = list(read_jsonl(reference_audit / "features.jsonl"))
    return reference_manifest, reference_features


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    _, reference_features = validate_reference_inputs(
        args.reference_audit,
        args.source,
        args.training,
    )
    catalog = load_tool_catalog(args.tau2_root)
    features, payloads, inventory = audit_source_dialogs(
        args.source,
        args.training,
        catalog,
        protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
    )
    if stable_json_hash(features) != stable_json_hash(reference_features):
        raise ValueError("reconstructed features differ from the sealed reference audit")
    if inventory.get("conversion_mismatch_rows"):
        raise ValueError(
            f"training conversion mismatches={inventory['conversion_mismatch_rows']}"
        )
    success_features = [
        feature for feature in features if feature.get("label_group") == "consensus_success"
    ]
    success_ids = {str(feature["dialog_id"]) for feature in success_features}
    payload_by_id = {str(payload["_dialog_id"]): payload for payload in payloads}
    success_payloads = [
        _counterfactual_payload(
            feature,
            payload_by_id[str(feature["dialog_id"])],
            review_mode="quality",
        )
        for feature in success_features
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    local_input_limit = args.local_context_limit - args.judge_max_tokens
    local_payloads, long_payloads, local_windows = _split_payloads(
        success_payloads,
        tokenizer=tokenizer,
        review_mode="quality",
        local_input_limit=local_input_limit,
    )
    payload_manifest = build_payload_manifest(
        "quality",
        local_payloads,
        long_payloads,
        local_windows,
    )
    if {str(row["dialog_id"]) for row in payload_manifest} != success_ids:
        raise ValueError("success payload manifest does not cover the selected dialogs")

    args.out.mkdir(parents=True, exist_ok=True)
    prepared = {
        "success_features.jsonl": success_features,
        "success_judge_payloads_local.jsonl": local_payloads,
        "success_judge_payloads_long.jsonl": long_payloads,
        "success_local_windows.jsonl": local_windows,
        "success_payload_manifest.jsonl": payload_manifest,
    }
    for filename, rows in prepared.items():
        write_jsonl(args.out / filename, rows)
    prepared_inventory = {
        "filter_version": FILTER_VERSION,
        "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
        "prompt_version": MULTITOOL_PROMPT_VERSION,
        "source_rows": inventory["source_rows"],
        "training_rows": inventory["training_rows"],
        "training_dialogs": inventory["training_dialogs"],
        "consensus_success_dialogs": len(success_features),
        "success_local_full_dialogs": len(local_payloads),
        "success_local_window_dialogs": len(local_windows),
        "success_long_full_dialogs": len(long_payloads),
        "success_dialogs_by_domain": dict(
            sorted(Counter(str(row["domain"]) for row in success_features).items())
        ),
        "success_payload_manifest_sha256": stable_json_hash(payload_manifest),
        "local_context_limit": args.local_context_limit,
        "judge_max_tokens": args.judge_max_tokens,
        "local_input_limit": local_input_limit,
        "input_sha256": {
            "source": file_sha256(args.source),
            "training": file_sha256(args.training),
        },
        "reference_audit": str(args.reference_audit),
        "reference_features_sha256": stable_json_hash(reference_features),
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_file_sha256": tokenizer_file_hashes(args.tokenizer),
    }
    write_json(args.out / "prepare_inventory.json", prepared_inventory)
    artifacts = {
        filename: artifact_record(args.out / filename, len(rows))
        for filename, rows in prepared.items()
    }
    artifacts["prepare_inventory.json"] = artifact_record(
        args.out / "prepare_inventory.json", None
    )
    analysis_dir = Path(__file__).resolve().parent
    code_paths = {
        "build_local_first_sft.py": Path(__file__).resolve(),
        "audit_sft_multitool.py": analysis_dir / "audit_sft_multitool.py",
        "judge_sft_quality.py": analysis_dir / "judge_sft_quality.py",
        "sft_quality.py": analysis_dir / "sft_quality.py",
    }
    write_json(
        args.out / "local_first_prepare_manifest.json",
        {
            "filter_version": FILTER_VERSION,
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "input_sha256": prepared_inventory["input_sha256"],
            "reference_audit": str(args.reference_audit),
            "reference_features_sha256": prepared_inventory[
                "reference_features_sha256"
            ],
            "tokenizer_path": str(args.tokenizer),
            "tokenizer_file_sha256": prepared_inventory[
                "tokenizer_file_sha256"
            ],
            "runner_script_path": str(args.runner_script.resolve()),
            "runner_script_sha256": file_sha256(args.runner_script),
            "code_sha256": {
                name: file_sha256(path) for name, path in code_paths.items()
            },
            "artifacts": artifacts,
        },
    )
    print(json.dumps(prepared_inventory, ensure_ascii=False, indent=2, sort_keys=True))


def validate_local_first_prepare(
    out: Path,
    source: Path,
    training: Path,
) -> dict[str, Any]:
    path = out / "local_first_prepare_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for filename, expected in (manifest.get("artifacts") or {}).items():
        artifact = out / filename
        if not artifact.is_file() or file_sha256(artifact) != expected.get("sha256"):
            raise ValueError(f"{path}: prepared artifact hash mismatch: {filename}")
    actual_inputs = {
        "source": file_sha256(source),
        "training": file_sha256(training),
    }
    if actual_inputs != manifest.get("input_sha256"):
        raise ValueError(f"{path}: source/training input hash mismatch")
    analysis_dir = Path(__file__).resolve().parent
    code_paths = {
        "build_local_first_sft.py": Path(__file__).resolve(),
        "audit_sft_multitool.py": analysis_dir / "audit_sft_multitool.py",
        "judge_sft_quality.py": analysis_dir / "judge_sft_quality.py",
        "sft_quality.py": analysis_dir / "sft_quality.py",
    }
    for name, code_path in code_paths.items():
        if file_sha256(code_path) != (manifest.get("code_sha256") or {}).get(name):
            raise ValueError(f"{path}: code hash mismatch: {name}")
    return manifest


def load_converted_rows(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    by_source_row = {}
    for row in rows:
        source_row = (row.get("metadata") or {}).get("source_row")
        if not isinstance(source_row, int) or source_row in by_source_row:
            raise ValueError(f"{path}: invalid or duplicate source_row={source_row!r}")
        by_source_row[source_row] = row
    return rows, by_source_row


def load_source_rows(path: Path, selected: set[int]) -> dict[int, dict[str, Any]]:
    rows = {
        index: row
        for index, row in enumerate(read_jsonl(path))
        if index in selected
    }
    missing = selected - set(rows)
    if missing:
        raise ValueError(f"{path}: missing selected source rows {sorted(missing)[:5]}")
    return rows


def build(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    prepare_manifest = validate_local_first_prepare(
        args.out,
        args.source,
        args.training,
    )
    success_features = list(read_jsonl(args.out / "success_features.jsonl"))
    feature_by_id = {str(row["dialog_id"]): row for row in success_features}
    local_manifest = _manifest_map(
        args.out / "success_payload_manifest.jsonl",
        "quality",
    )
    local_reviews = _validated_review_map(
        args.local_reviews,
        local_manifest,
        review_mode="quality",
        side="local",
    )
    for dialog_id, record in local_reviews.items():
        _, model = _selected_judge_identity(record)
        if model != args.expected_local_model.lower():
            raise ValueError(
                f"{dialog_id}: selected local judge model {model!r} does not match "
                f"{args.expected_local_model!r}"
            )

    reference_manifest, reference_features = validate_reference_inputs(
        args.reference_audit,
        args.source,
        args.training,
    )
    if stable_json_hash(reference_features) != prepare_manifest.get(
        "reference_features_sha256"
    ):
        raise ValueError("reference audit changed after local-first prepare")
    reference_quality_manifest = _manifest_map(
        args.reference_audit / "payload_manifest.jsonl",
        "quality",
    )
    known_luna_manifest = {
        dialog_id: manifest
        for dialog_id, manifest in reference_quality_manifest.items()
        if dialog_id in feature_by_id
    }
    luna_reviews = _validated_review_map(
        args.reference_audit / "quality_fallback_reviews.jsonl",
        known_luna_manifest,
        review_mode="quality",
        side="fallback",
        allow_additional=True,
    )
    for dialog_id, record in luna_reviews.items():
        _, model = _selected_judge_identity(record)
        if not model.startswith(args.expected_fallback_model.lower()):
            raise ValueError(
                f"{dialog_id}: selected fallback model {model!r} is not "
                f"{args.expected_fallback_model!r}"
            )

    converted_rows, converted_by_source = load_converted_rows(args.training)
    source_rows = load_source_rows(args.source, set(converted_by_source))
    catalog = load_tool_catalog(args.tau2_root)
    training_tokenizer = AutoTokenizer.from_pretrained(
        args.training_tokenizer,
        trust_remote_code=True,
    )
    loss_mask_generator = MultiTurnLossMaskGenerator(
        training_tokenizer,
        tokenizer_type="qwen3",
    )
    target_by_source = {}
    conversion_by_source = {}
    for feature in success_features:
        for target in feature.get("included_targets") or []:
            target_by_source[target["source_row"]] = (feature, target)
        for conversion in feature.get("conversion_checks") or []:
            conversion_by_source[conversion["source_row"]] = conversion

    decisions = []
    selected_rows = []
    selected_dialogs: dict[str, dict[str, Any]] = {}
    reason_counts: Counter[str] = Counter()
    prompt_amendments: Counter[str] = Counter()
    kept_by_domain: Counter[str] = Counter()
    kept_by_tier: Counter[str] = Counter()
    kept_multi_by_domain: Counter[str] = Counter()
    all_targets_by_domain: Counter[str] = Counter()
    success_targets_by_domain: Counter[str] = Counter()
    selected_source_rows = set()
    max_selected_total_tokens = 0
    max_rejected_total_tokens = 0

    for converted_row in converted_rows:
        metadata = converted_row.get("metadata") or {}
        source_row = metadata["source_row"]
        dialog_id = str(metadata.get("source_dialog_id") or "")
        domain = str(metadata.get("domain") or "")
        all_targets_by_domain[domain] += 1
        reasons = []
        feature_target = target_by_source.get(source_row)
        feature = feature_target[0] if feature_target else None
        target = feature_target[1] if feature_target else None
        tier = "not_consensus_success"
        static: dict[str, Any] = {}
        if feature is None or target is None:
            reasons.append("label_group:not_consensus_success")
        else:
            domain = str(feature["domain"])
            success_targets_by_domain[domain] += 1
            target_turn = target.get("conversation_turn_index")
            if not isinstance(target_turn, int):
                reasons.append("target:invalid_conversation_turn_index")
                target_turn = -1
            raw_messages = row_messages(source_rows[source_row])
            static = deterministic_features(
                raw_messages,
                domain,
                catalog,
                protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
            )
            reasons.extend(
                f"mechanical_hard_issue:{issue}"
                for issue in static.get("mechanical_hard_issues") or []
            )
            conversion = conversion_by_source.get(source_row) or {}
            for field in (
                "non_system_messages_match",
                "source_system_preserved",
                "target_fingerprint_match",
            ):
                if conversion.get(field) is not True:
                    reasons.append(f"conversion:{field}")
            if stable_json_hash(row_messages(converted_row)) != conversion.get(
                "converted_messages_hash"
            ):
                reasons.append("conversion:converted_messages_hash")

            local_record = local_reviews.get(dialog_id)
            reasons.extend(
                quality_gate_reasons(
                    local_record,
                    side="local",
                )
            )
            reasons.extend(
                multitool_prefix_reasons(local_record, static, side="local")
            )
            luna_record = luna_reviews.get(dialog_id)
            if luna_record is not None:
                tier = "dual_calibrated"
                reasons.extend(
                    quality_gate_reasons(
                        luna_record,
                        side="luna",
                    )
                )
                reasons.extend(
                    multitool_prefix_reasons(luna_record, static, side="luna")
                )
            else:
                tier = "local_only"

        training_row = None
        amendment: dict[str, bool] = {}
        system = None
        amended_system = None
        training_total_tokens = None
        if not reasons:
            system = next(
                str(message.get("content") or "")
                for message in converted_row["messages"]
                if message.get("role") == "system"
            )
            amended_system, amendment = amend_training_system(system)
            training_row = apply_target_only_loss_mask(converted_row, amended_system)
            token_ids, loss_mask = loss_mask_generator.get_loss_mask(
                training_row["messages"],
                tools=None,
            )
            if len(token_ids) != len(loss_mask):
                raise ValueError(
                    f"source_row={source_row}: token/loss-mask length mismatch"
                )
            training_total_tokens = len(token_ids)
            if not any(loss_mask):
                reasons.append("training_loss_mask:no_supervised_tokens")
            if training_total_tokens > args.max_total_tokens:
                reasons.append(
                    f"amended_training_tokens:over_cap:{training_total_tokens}"
                )
                max_rejected_total_tokens = max(
                    max_rejected_total_tokens,
                    training_total_tokens,
                )

        unique_reasons = sorted(set(reasons))
        keep = not unique_reasons
        target_summary = {
            "source_row": source_row,
            "training_turn_index": metadata.get("turn_index"),
            "conversation_turn_index": (
                target.get("conversation_turn_index") if target else None
            ),
            "is_multitool": bool(target and target.get("is_multitool")),
            "num_tool_calls": target.get("num_tool_calls") if target else None,
            "tool_names": target.get("tool_names") if target else [],
            "prefix_multitool_turns": static.get("multitool_turns") or [],
            "prefix_mechanical_hard_issues": static.get("mechanical_hard_issues") or [],
            "amended_training_total_tokens": training_total_tokens,
        }
        decision = {
            "artifact_kind": "provenance_only_local_first_filter_decision",
            "contains_training_messages": False,
            "filter_version": FILTER_VERSION,
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "dialog_id": dialog_id,
            "domain": domain,
            "label_group": feature.get("label_group") if feature else "not_consensus_success",
            "review_tier": tier,
            "local_input_payload_sha256": (
                local_manifest.get(dialog_id, {}).get("local_input_payload_sha256")
            ),
            "local_judge_config_sha256": (
                local_reviews.get(dialog_id, {}).get("judge_config_sha256")
            ),
            "luna_input_payload_sha256": (
                known_luna_manifest.get(dialog_id, {}).get(
                    "fallback_input_payload_sha256"
                )
            ),
            "luna_judge_config_sha256": (
                luna_reviews.get(dialog_id, {}).get("judge_config_sha256")
            ),
            "target": target_summary,
            "decision": "keep" if keep else "reject",
            "keep": keep,
            "decision_reasons": unique_reasons,
        }
        decision["decision_sha256"] = stable_json_hash(decision)
        decisions.append(decision)
        reason_counts.update(unique_reasons)
        if not keep:
            continue

        assert training_row is not None
        assert system is not None
        assert amended_system is not None
        assert training_total_tokens is not None
        if source_row in selected_source_rows:
            raise ValueError(f"duplicate selected source row {source_row}")
        selected_source_rows.add(source_row)
        prompt_amendments.update(
            name for name, applied in amendment.items() if applied
        )
        training_metadata = training_row.setdefault("metadata", {})
        training_metadata["quality_filter"] = {
            "version": FILTER_VERSION,
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "review_tier": tier,
            "decision_sha256": decision["decision_sha256"],
            "original_system_sha256": stable_json_hash(system),
            "amended_system_sha256": stable_json_hash(amended_system),
            "target_only_loss": True,
            "total_tokens": training_total_tokens,
            "max_total_tokens": args.max_total_tokens,
        }
        selected_rows.append(training_row)
        max_selected_total_tokens = max(
            max_selected_total_tokens,
            training_total_tokens,
        )
        kept_by_domain[domain] += 1
        kept_by_tier[tier] += 1
        if target and target.get("is_multitool"):
            kept_multi_by_domain[domain] += 1
        selected_dialogs[dialog_id] = feature

    selected_rows.sort(key=lambda row: (row.get("metadata") or {})["source_row"])
    decisions.sort(key=lambda row: row["target"]["source_row"])
    if len(decisions) != len(converted_rows):
        raise ValueError("filter decisions do not cover the exact converted training file")
    canonical_path = args.out / "sft_selected_target_only.jsonl"
    decisions_path = args.out / "filter_decisions.jsonl"
    write_jsonl(canonical_path, selected_rows)
    write_jsonl(decisions_path, decisions)
    budget_rows = matched_budget_rows(
        selected_rows,
        args.budget_rows,
        args.budget_seed,
    )
    budget_path = args.out / f"sft_train_budget_matched_{args.budget_rows}.jsonl"
    write_jsonl(budget_path, budget_rows)

    summary = {
        "filter_version": FILTER_VERSION,
        "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
        "selection_rule": {
            "labels": "correct=1 and reward=1",
            "judge_verdicts_allowed": ["keep", "review"],
            "core_dimension_minimum": 3,
            "support_dimension_minimum": 2,
            "critical_turns": "diagnostic; represented by verdict and dimension gates",
            "multitool_dependency_prefix": "strict-safe for every local assessment",
            "known_luna_reviews": "same gate; conservative veto",
            "mechanical_prefix_hard_issues_allowed": False,
            "loss_mask": "final approved assistant target only",
            "max_total_tokens": args.max_total_tokens,
        },
        "input": {
            "training_rows": len(converted_rows),
            "training_rows_by_domain": dict(sorted(all_targets_by_domain.items())),
            "consensus_success_dialogs": len(success_features),
            "consensus_success_targets": sum(success_targets_by_domain.values()),
            "consensus_success_targets_by_domain": dict(
                sorted(success_targets_by_domain.items())
            ),
            "locally_reviewed_success_dialogs": len(local_reviews),
            "luna_calibrated_success_dialogs": len(luna_reviews),
        },
        "selected": {
            "canonical_rows": len(selected_rows),
            "dialogs": len(selected_dialogs),
            "rows_by_domain": dict(sorted(kept_by_domain.items())),
            "dialogs_by_domain": dict(
                sorted(
                    Counter(str(feature["domain"]) for feature in selected_dialogs.values()).items()
                )
            ),
            "rows_by_review_tier": dict(sorted(kept_by_tier.items())),
            "multitool_target_rows_by_domain": dict(
                sorted(kept_multi_by_domain.items())
            ),
            "budget_rows": len(budget_rows),
            "budget_seed": args.budget_seed,
            "max_total_tokens": max_selected_total_tokens,
            "max_rejected_total_tokens": max_rejected_total_tokens,
        },
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "prompt_amendment_counts": dict(sorted(prompt_amendments.items())),
        "artifacts": {
            canonical_path.name: artifact_record(canonical_path, len(selected_rows)),
            budget_path.name: artifact_record(budget_path, len(budget_rows)),
            decisions_path.name: artifact_record(decisions_path, len(decisions)),
            "local_reviews": {
                "path": str(args.local_reviews),
                "rows": len(local_reviews),
                "sha256": file_sha256(args.local_reviews),
            },
            "reference_luna_reviews": {
                "path": str(
                    args.reference_audit / "quality_fallback_reviews.jsonl"
                ),
                "used_rows": len(luna_reviews),
                "sha256": file_sha256(
                    args.reference_audit / "quality_fallback_reviews.jsonl"
                ),
            },
        },
        "input_sha256": reference_manifest["input_sha256"],
        "prepare_manifest_sha256": file_sha256(
            args.out / "local_first_prepare_manifest.json"
        ),
    }
    write_json(args.out / "filter_summary.json", summary)
    report_lines = [
        "# Local-first relaxed SFT filter",
        "",
        f"筛选版本：`{FILTER_VERSION}`。全量语义复审只使用本地 Qwen3.6-27B；已有",
        f"Luna 结果覆盖 {len(luna_reviews)} 个成功 dialog，并作为保守否决证据复用，未发起新 API 请求。",
        "",
        "## Yield",
        "",
        f"- 原训练文件：{len(converted_rows):,} targets；consensus-success：{sum(success_targets_by_domain.values()):,}。",
        f"- 最终 canonical：{len(selected_rows):,} targets / {len(selected_dialogs):,} dialogs。",
        f"- 领域 target：{dict(sorted(kept_by_domain.items()))}。",
        f"- 多工具 target：{sum(kept_multi_by_domain.values()):,}，分领域 {dict(sorted(kept_multi_by_domain.items()))}。",
        f"- 训练预算文件：{len(budget_rows):,} rows，和旧 SFT 的 row/update budget 对齐。",
        "",
        "## Rule",
        "",
        "只自动保留 consensus-success。可靠 judge 的 keep/review 均可候选，但 drop、核心维度",
        "低于 3、支持维度低于 2、目标依赖前缀中的机械 hard issue，或任一",
        "multi turn 未通过 dependency/grounding/authorization/necessity strict-safe gate 时拒绝。",
        "已有 Luna 覆盖的样本必须同时通过相同门槛。所有历史 assistant turn 的",
        "`step_loss_mask=0`，只监督获准的最后目标；训练 system 已同步为 dependency-safe multi。",
        "",
        "## Artifacts",
        "",
        f"- `{canonical_path.name}`：无重复 canonical targets。",
        f"- `{budget_path.name}`：确定性重复到旧训练的 {args.budget_rows:,}-row budget。",
        "- `filter_decisions.jsonl`：覆盖原文件每个 target 的无文本 provenance 决策。",
        "- `filter_summary.json`：完整计数、reason 分布和 SHA256。",
        "",
    ]
    (args.out / "LOCAL_FIRST_FILTER.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare local judge inputs for every consensus-success dialog",
    )
    prepare_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare_parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    prepare_parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    prepare_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    prepare_parser.add_argument(
        "--reference-audit", type=Path, default=DEFAULT_REFERENCE_AUDIT
    )
    prepare_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    prepare_parser.add_argument("--local-context-limit", type=int, default=32768)
    prepare_parser.add_argument("--judge-max-tokens", type=int, default=2048)
    prepare_parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path(__file__).resolve().parent / "run_local_first_sft_filter.sh",
    )
    prepare_parser.set_defaults(func=prepare)

    build_parser = subparsers.add_parser(
        "build",
        help="Apply target-level gates and emit actual SFT JSONL",
    )
    build_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    build_parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    build_parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    build_parser.add_argument(
        "--training-tokenizer",
        type=Path,
        default=DEFAULT_TRAINING_TOKENIZER,
    )
    build_parser.add_argument(
        "--reference-audit", type=Path, default=DEFAULT_REFERENCE_AUDIT
    )
    build_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build_parser.add_argument("--local-reviews", type=Path, required=True)
    build_parser.add_argument("--expected-local-model", default="Qwen3.6-27B")
    build_parser.add_argument("--expected-fallback-model", default="gpt-5.6-luna")
    build_parser.add_argument("--budget-rows", type=int, default=DEFAULT_BUDGET_ROWS)
    build_parser.add_argument("--budget-seed", type=int, default=DEFAULT_BUDGET_SEED)
    build_parser.add_argument("--max-total-tokens", type=int, default=8192)
    build_parser.set_defaults(func=build)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
