#!/usr/bin/env python3
"""Protocol-neutral comparison of single- and multi-tool tau2 SFT trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from protocol_profiles import (  # noqa: E402
    DEPENDENCY_SAFE_MULTI_RULE,
    PROTOCOL_DEPENDENCY_SAFE_MULTI,
    domain_policy_for_profile,
)
from sft_quality import (
    audit_source_dialogs,
    payload_hash,
    read_jsonl,
    stable_json_hash,
    write_jsonl,
)
from judge_sft_quality import (
    DEPENDENCY_SAFE_QUALITY_DIMENSIONS,
    MULTITOOL_PROMPT_VERSION,
    request_messages,
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
DEFAULT_OUT = (
    SERVICE_AGENT_ROOT / "slime/output/experiments/tau2-sft-multitool-quality-audit"
)

CANDIDATE_POLICY_RULE = DEPENDENCY_SAFE_MULTI_RULE
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "customer",
    "for",
    "from",
    "in",
    "multiple",
    "of",
    "on",
    "the",
    "their",
    "to",
    "user",
    "wants",
    "with",
}

PRIMARY_FAILURE_CAUSES = (
    "wrong_workflow",
    "wrong_arguments",
    "missing_precondition",
    "confirmation_violation",
    "batch_dependency",
    "unrelated_or_extra_action",
    "tool_namespace_or_nonexistent",
    "failed_recovery_or_repetition",
    "incomplete_task",
    "false_success_claim",
    "environment_or_user",
    "unknown",
)
CAUSAL_CALL_MODES = (
    "multi_turn",
    "single_call_turn",
    "non_tool",
    "environment_or_user",
    "unknown",
)
MULTI_CAUSAL_ROLES = (
    "caused",
    "contributed",
    "irrelevant",
    "beneficial",
    "uncertain",
    "not_applicable",
)

FAILURE_CALIBRATION_LIMIT = 72
FAILURE_CALIBRATION_PILOT_SEED = 20260802
FAILURE_CALIBRATION_PILOT_VERSION = "failure-calibration-v1"
FAILURE_CALIBRATION_SEED = 20260803
FAILURE_CALIBRATION_VERSION = "failure-luna-evidence-validation-v2"
QUALITY_CALIBRATION_LIMIT = 72
QUALITY_CALIBRATION_PILOT_SEED = 20260802
QUALITY_CALIBRATION_V1_SEED = 20260803
QUALITY_CALIBRATION_VALIDATION_SEED = 20260804
QUALITY_CALIBRATION_VERSION = "quality-asymmetric-intersection-validation-v2"
FAILURE_UNKNOWN_VALUES = {
    "primary_failure_cause": "unknown",
    "causal_call_mode": "unknown",
    "multi_causal_role": "uncertain",
}
FAILURE_RESOLUTION_STATUSES = (
    "agreed_resolved",
    "agreed_unknown",
    "disagreed",
    "unreliable",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def label_group(record: dict[str, Any]) -> str:
    existing = record.get("label_group")
    if existing:
        return str(existing)
    correct = record.get("source_correct")
    reward = record.get("source_reward")
    if correct == 1 and reward == 1:
        return "consensus_success"
    if correct == 0 and reward == 0:
        return "consensus_failure"
    if correct == 0 and reward == 1:
        return "reward_only"
    if correct == 1 and reward == 0:
        return "correct_only"
    return "unknown"


def full_static(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("full_source_static") or record.get("static") or {}


def training_static(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("training_visible_static") or record.get("static") or {}


def has_multi(record: dict[str, Any], *, view: str = "full") -> bool:
    static = full_static(record) if view == "full" else training_static(record)
    return bool(static.get("multitool_turns"))


def amend_policy(policy: str) -> tuple[str, bool]:
    return domain_policy_for_profile(policy, PROTOCOL_DEPENDENCY_SAFE_MULTI)


def _task_tokens(record: dict[str, Any]) -> set[str]:
    if record.get("domain") == "telecom":
        raw = str(record.get("task_id") or "").split("[PERSONA", 1)[0]
    else:
        raw = str(record.get("reason_for_call") or "")
    return {token for token in _TOKEN_RE.findall(raw.lower()) if token not in _STOPWORDS}


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _eligible_control(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if anchor["domain"] != candidate["domain"]:
        return False
    domain = anchor["domain"]
    if domain == "airline":
        family = anchor.get("seed_pattern_task_id")
        return bool(family) and family == candidate.get("seed_pattern_task_id")
    if domain == "telecom":
        return (
            anchor.get("difficulty") == candidate.get("difficulty")
            and anchor.get("num_subtasks") == candidate.get("num_subtasks")
            and _jaccard(_task_tokens(anchor), _task_tokens(candidate)) >= 0.5
        )
    return _jaccard(_task_tokens(anchor), _task_tokens(candidate)) >= 0.5


def _control_score(anchor: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, str]:
    similarity = _jaccard(_task_tokens(anchor), _task_tokens(candidate))
    if anchor["domain"] == "telecom":
        similarity += 0.25 * int(anchor.get("difficulty") == candidate.get("difficulty"))
        similarity += 0.25 * int(anchor.get("num_subtasks") == candidate.get("num_subtasks"))
    return (-similarity, str(candidate["dialog_id"]))


def build_comparison_selection(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select all non-consensus dialogs plus matched consensus-success comparisons."""

    by_id = {str(record["dialog_id"]): record for record in features}
    membership: dict[str, dict[str, Any]] = {}

    def add(dialog_id: str, role: str, match_set: str | None = None) -> None:
        item = membership.setdefault(
            dialog_id,
            {
                "dialog_id": dialog_id,
                "domain": by_id[dialog_id]["domain"],
                "label_group": label_group(by_id[dialog_id]),
                "selection_roles": [],
                "match_sets": [],
            },
        )
        if role not in item["selection_roles"]:
            item["selection_roles"].append(role)
        if match_set and match_set not in item["match_sets"]:
            item["match_sets"].append(match_set)

    for record in features:
        if label_group(record) != "consensus_success":
            add(str(record["dialog_id"]), "non_consensus")

    anchors = sorted(
        (
            record
            for record in features
            if label_group(record) == "consensus_success" and not has_multi(record)
        ),
        key=lambda item: (item["domain"], str(item["dialog_id"])),
    )
    controls = [
        record
        for record in features
        if label_group(record) == "consensus_success" and has_multi(record)
    ]
    candidates_by_anchor = {}
    for anchor in anchors:
        anchor_id = str(anchor["dialog_id"])
        match_set = f"{anchor['domain']}:{anchor_id}"
        add(anchor_id, "single_success_anchor", match_set)
        candidates_by_anchor[anchor_id] = sorted(
            (candidate for candidate in controls if _eligible_control(anchor, candidate)),
            key=lambda candidate: _control_score(anchor, candidate),
        )
    used_controls = set()
    matched_counts = Counter()
    allocation_order = sorted(
        anchors,
        key=lambda anchor: (
            len(candidates_by_anchor[str(anchor["dialog_id"])]),
            anchor["domain"],
            str(anchor["dialog_id"]),
        ),
    )
    for _ in range(3):
        for anchor in allocation_order:
            anchor_id = str(anchor["dialog_id"])
            candidate = next(
                (
                    candidate
                    for candidate in candidates_by_anchor[anchor_id]
                    if str(candidate["dialog_id"]) not in used_controls
                ),
                None,
            )
            if candidate is None:
                continue
            control_id = str(candidate["dialog_id"])
            used_controls.add(control_id)
            matched_counts[anchor_id] += 1
            add(
                control_id,
                "multi_success_control",
                f"{anchor['domain']}:{anchor_id}",
            )
    for anchor in anchors:
        membership[str(anchor["dialog_id"])]["matched_control_count"] = matched_counts[
            str(anchor["dialog_id"])
        ]

    for item in membership.values():
        item.setdefault("matched_control_count", None)
        item["selection_roles"].sort()
        item["match_sets"].sort()
    return sorted(membership.values(), key=lambda item: (item["domain"], item["dialog_id"]))


def build_comparison_pairs(
    features: list[dict[str, Any]], selection: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    feature_by_id = {str(record["dialog_id"]): record for record in features}
    controls_by_match_set: dict[str, list[str]] = defaultdict(list)
    for item in selection:
        if "multi_success_control" not in item["selection_roles"]:
            continue
        for match_set in item["match_sets"]:
            controls_by_match_set[match_set].append(item["dialog_id"])
    pairs = []
    for anchor in selection:
        if "single_success_anchor" not in anchor["selection_roles"]:
            continue
        anchor_id = anchor["dialog_id"]
        match_set = f"{anchor['domain']}:{anchor_id}"
        control_ids = sorted(
            controls_by_match_set.get(match_set, []),
            key=lambda candidate_id: _control_score(
                feature_by_id[anchor_id], feature_by_id[candidate_id]
            ),
        )
        if not control_ids:
            pairs.append(
                {
                    "match_set": match_set,
                    "domain": anchor["domain"],
                    "single_dialog_id": anchor_id,
                    "multi_dialog_id": None,
                    "match_status": "unmatched",
                    "match_rule": "domain-specific exact/common-support rule",
                }
            )
            continue
        for rank, control_id in enumerate(control_ids, start=1):
            single = feature_by_id[anchor_id]
            control = feature_by_id[control_id]
            pairs.append(
                {
                    "match_set": match_set,
                    "domain": anchor["domain"],
                    "single_dialog_id": anchor_id,
                    "multi_dialog_id": control_id,
                    "match_status": "matched",
                    "match_rank": rank,
                    "reason_similarity": round(
                        _jaccard(_task_tokens(single), _task_tokens(control)), 6
                    ),
                    "exact_metadata_fields": [
                        field
                        for field in (
                            "seed_pattern_task_id",
                            "task_id",
                            "difficulty",
                            "num_subtasks",
                            "scenario_id",
                        )
                        if single.get(field) is not None
                        and single.get(field) == control.get(field)
                    ],
                    "match_rule": (
                        "same seed_pattern_task_id"
                        if anchor["domain"] == "airline"
                        else "same difficulty/subtasks and reason similarity >= 0.5"
                        if anchor["domain"] == "telecom"
                        else "reason similarity >= 0.5"
                    ),
                }
            )
    return pairs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prepare_manifest(path: Path, artifact_root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for filename, expected in (manifest.get("artifacts") or {}).items():
        artifact = artifact_root / filename
        if not artifact.is_file() or file_sha256(artifact) != expected.get("sha256"):
            raise ValueError(f"{path}: prepared artifact hash mismatch: {filename}")
    analysis_dir = Path(__file__).resolve().parent
    current_code = {
        "audit_sft_multitool.py": Path(__file__).resolve(),
        "sft_quality.py": analysis_dir / "sft_quality.py",
        "judge_sft_quality.py": analysis_dir / "judge_sft_quality.py",
        "run_multitool_quality_audit.sh": analysis_dir
        / "run_multitool_quality_audit.sh",
    }
    for filename, code_path in current_code.items():
        if file_sha256(code_path) != (manifest.get("code_sha256") or {}).get(filename):
            raise ValueError(f"{path}: audit code hash mismatch: {filename}")
    return manifest


def validate_selected_artifacts(
    manifest: dict[str, Any], artifacts: dict[str, Path]
) -> None:
    """Bind explicitly supplied CLI paths to the sealed prepare artifacts."""

    declared = manifest.get("artifacts") or {}
    for canonical_name, path in artifacts.items():
        expected = declared.get(canonical_name)
        if (
            not isinstance(expected, dict)
            or not path.is_file()
            or file_sha256(path) != expected.get("sha256")
        ):
            raise ValueError(
                f"{path}: does not match sealed prepare artifact {canonical_name}"
            )


def build_payload_manifest(
    review_mode: str,
    local_payloads: list[dict[str, Any]],
    long_payloads: list[dict[str, Any]],
    local_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def turn_types(payload: dict[str, Any]) -> dict[str, list[int]]:
        result: dict[str, list[int]] = defaultdict(list)
        for turn in payload.get("conversation") or []:
            if not isinstance(turn, dict):
                continue
            index = turn.get("turn_index")
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            role = turn.get("role")
            if role == "assistant":
                call_count = len(turn.get("tool_calls") or [])
                kind = (
                    "assistant_multi"
                    if call_count > 1
                    else "assistant_single"
                    if call_count == 1
                    else "assistant_non_tool"
                )
            else:
                kind = str(role or "unknown")
            result[kind].append(index)
        return {
            kind: sorted(indices) for kind, indices in sorted(result.items())
        }

    windows_by_id = {str(item["_dialog_id"]): item for item in local_windows}
    if len(windows_by_id) != len(local_windows):
        raise ValueError(f"{review_mode}: duplicate local-window dialog IDs")
    rows = []
    for payload in local_payloads:
        source_hash = payload["payload_sha256"]
        rows.append(
            {
                "dialog_id": str(payload["_dialog_id"]),
                "blind_dialog_id": payload["dialog_id"],
                "review_mode": review_mode,
                "protocol_profile": payload["protocol_profile"],
                "prompt_version": payload["prompt_version"],
                "route": "full/full",
                "source_full_payload_sha256": source_hash,
                "local_input_payload_sha256": source_hash,
                "fallback_input_payload_sha256": source_hash,
                "full_judge_input_tokens": payload["judge_input_tokens"],
                "local_judge_input_tokens": payload["judge_input_tokens"],
                "local_evidence_scope": {"kind": "full_source"},
                "full_source_turn_types": turn_types(payload),
            }
        )
    for payload in long_payloads:
        dialog_id = str(payload["_dialog_id"])
        window = windows_by_id.get(dialog_id)
        if window is None:
            raise ValueError(f"{review_mode}: no local window for {dialog_id}")
        source_hash = payload["payload_sha256"]
        if window.get("source_full_payload_sha256") != source_hash:
            raise ValueError(f"{review_mode}: full/window hash mismatch for {dialog_id}")
        rows.append(
            {
                "dialog_id": dialog_id,
                "blind_dialog_id": payload["dialog_id"],
                "review_mode": review_mode,
                "protocol_profile": payload["protocol_profile"],
                "prompt_version": payload["prompt_version"],
                "route": "window/full",
                "source_full_payload_sha256": source_hash,
                "local_input_payload_sha256": window["payload_sha256"],
                "fallback_input_payload_sha256": source_hash,
                "full_judge_input_tokens": payload["judge_input_tokens"],
                "local_judge_input_tokens": window["judge_input_tokens"],
                "local_evidence_scope": window["evidence_scope"],
                "full_source_turn_types": turn_types(payload),
            }
        )
    counts = Counter(row["dialog_id"] for row in rows)
    duplicates = [dialog_id for dialog_id, count in counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"{review_mode}: duplicate payload manifest IDs: {duplicates[:5]}")
    return sorted(rows, key=lambda row: row["dialog_id"])


def build_calibration_selection(
    features: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    quality_manifest: list[dict[str, Any]],
    limit: int = QUALITY_CALIBRATION_LIMIT,
    seed: int = QUALITY_CALIBRATION_PILOT_SEED,
    exclude_dialog_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude_dialog_ids = exclude_dialog_ids or set()
    feature_by_id = {str(record["dialog_id"]): record for record in features}
    manifest_by_id = {str(row["dialog_id"]): row for row in quality_manifest}
    groups: dict[tuple[str, bool, str, str, bool], list[str]] = defaultdict(list)
    for item in selection:
        dialog_id = str(item["dialog_id"])
        if dialog_id in exclude_dialog_ids:
            continue
        feature = feature_by_id[dialog_id]
        manifest = manifest_by_id[dialog_id]
        key = (
            str(feature["domain"]),
            has_multi(feature),
            label_group(feature),
            str(manifest["route"]),
            bool(full_static(feature).get("mechanical_hard_issues")),
        )
        groups[key].append(dialog_id)
    queues = {
        key: sorted(
            dialog_ids,
            key=lambda dialog_id: stable_json_hash([seed, key, dialog_id]),
        )
        for key, dialog_ids in groups.items()
    }
    selected_ids = []
    target = min(
        limit,
        sum(str(item["dialog_id"]) not in exclude_dialog_ids for item in selection),
    )
    while len(selected_ids) < target:
        progressed = False
        for key in sorted(queues, key=str):
            if not queues[key]:
                continue
            selected_ids.append(queues[key].pop())
            progressed = True
            if len(selected_ids) == target:
                break
        if not progressed:
            break
    result = []
    for dialog_id in selected_ids:
        feature = feature_by_id[dialog_id]
        manifest = manifest_by_id[dialog_id]
        result.append(
            {
                "dialog_id": dialog_id,
                "domain": feature["domain"],
                "call_mode": "multi" if has_multi(feature) else "non-multi",
                "label_group": label_group(feature),
                "route": manifest["route"],
                "mechanical_hard": bool(
                    full_static(feature).get("mechanical_hard_issues")
                ),
                "selection_hash": stable_json_hash([seed, dialog_id]),
            }
        )
    return sorted(result, key=lambda row: row["dialog_id"])


def build_quality_v1_selection(
    features: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    quality_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild the frozen v1 cohort disjoint from the diagnostic pilot."""

    pilot = build_calibration_selection(
        features,
        selection,
        quality_manifest,
        limit=QUALITY_CALIBRATION_LIMIT,
        seed=QUALITY_CALIBRATION_PILOT_SEED,
    )
    return build_calibration_selection(
        features,
        selection,
        quality_manifest,
        limit=QUALITY_CALIBRATION_LIMIT,
        seed=QUALITY_CALIBRATION_V1_SEED,
        exclude_dialog_ids={str(row["dialog_id"]) for row in pilot},
    )


def build_quality_validation_selection(
    features: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    quality_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select v2 validation disjoint from both frozen diagnostic cohorts."""

    pilot = build_calibration_selection(
        features,
        selection,
        quality_manifest,
        limit=QUALITY_CALIBRATION_LIMIT,
        seed=QUALITY_CALIBRATION_PILOT_SEED,
    )
    validation_v1 = build_quality_v1_selection(
        features, selection, quality_manifest
    )
    excluded = {
        str(row["dialog_id"]) for row in pilot + validation_v1
    }
    return build_calibration_selection(
        features,
        selection,
        quality_manifest,
        limit=QUALITY_CALIBRATION_LIMIT,
        seed=QUALITY_CALIBRATION_VALIDATION_SEED,
        exclude_dialog_ids=excluded,
    )


def _build_failure_calibration_selection(
    features: list[dict[str, Any]],
    failure_manifest: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    version: str,
    exclude_dialog_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a sealed failure-attribution stress set before reading reviews."""

    exclude_dialog_ids = exclude_dialog_ids or set()
    feature_by_id = {str(record["dialog_id"]): record for record in features}
    failure_ids = {
        dialog_id
        for dialog_id, feature in feature_by_id.items()
        if label_group(feature) != "consensus_success"
    }
    manifest_by_id = {str(row["dialog_id"]): row for row in failure_manifest}
    if len(manifest_by_id) != len(failure_manifest) or set(manifest_by_id) != failure_ids:
        raise ValueError(
            "failure calibration requires exactly the non-consensus failure manifest"
        )

    groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for dialog_id in failure_ids - exclude_dialog_ids:
        feature = feature_by_id[dialog_id]
        multi_exposure = (
            "multi_exposed" if has_multi(feature) else "non_multi_exposed"
        )
        key = (
            str(feature["domain"]),
            label_group(feature),
            multi_exposure,
            str(manifest_by_id[dialog_id]["route"]),
        )
        groups[key].append(dialog_id)

    queues = {
        key: sorted(
            dialog_ids,
            key=lambda dialog_id: stable_json_hash(
                [
                    seed,
                    version,
                    key,
                    dialog_id,
                ]
            ),
        )
        for key, dialog_ids in groups.items()
    }
    target = min(limit, len(failure_ids - exclude_dialog_ids))
    selected: list[tuple[str, tuple[str, str, str, str]]] = []
    while len(selected) < target:
        progressed = False
        for key in sorted(queues, key=str):
            if not queues[key]:
                continue
            selected.append((queues[key].pop(), key))
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            break

    rows = [
        {
            "dialog_id": dialog_id,
            "domain": key[0],
            "label_group": key[1],
            "multi_exposure": key[2],
            "route": key[3],
            "selection_hash": stable_json_hash(
                [
                    seed,
                    version,
                    key,
                    dialog_id,
                ]
            ),
        }
        for dialog_id, key in selected
    ]
    return sorted(rows, key=lambda row: row["dialog_id"])


def build_failure_calibration_pilot_selection(
    features: list[dict[str, Any]],
    failure_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild the frozen exact-consensus diagnostic pilot."""

    return _build_failure_calibration_selection(
        features,
        failure_manifest,
        limit=FAILURE_CALIBRATION_LIMIT,
        seed=FAILURE_CALIBRATION_PILOT_SEED,
        version=FAILURE_CALIBRATION_PILOT_VERSION,
    )


def build_failure_calibration_selection(
    features: list[dict[str, Any]],
    failure_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select Luna-primary validation disjoint from the diagnostic pilot."""

    pilot = build_failure_calibration_pilot_selection(
        features, failure_manifest
    )
    return _build_failure_calibration_selection(
        features,
        failure_manifest,
        limit=FAILURE_CALIBRATION_LIMIT,
        seed=FAILURE_CALIBRATION_SEED,
        version=FAILURE_CALIBRATION_VERSION,
        exclude_dialog_ids={str(row["dialog_id"]) for row in pilot},
    )


def _counterfactual_payload(
    feature: dict[str, Any], payload: dict[str, Any], *, review_mode: str
) -> dict[str, Any]:
    original_policy = str(
        payload.get("original_domain_policy")
        or payload.get("domain_policy")
        or payload.get("policy")
        or ""
    )
    candidate_policy = str(payload.get("domain_policy") or "")
    clause_replaced = bool(payload.get("counterfactual_policy_applied"))
    if not candidate_policy:
        candidate_policy, clause_replaced = amend_policy(original_policy)
    item = dict(payload)
    item.pop("payload_sha256", None)
    item.pop("original_domain_policy", None)
    item.pop("training_visible_view", None)
    item.pop("view_alignment", None)
    # Quality comparison is blinded; failure attribution receives only the
    # explicitly declared outcome block below.
    for key in ("label_group", "source_correct", "source_reward", "correct", "reward"):
        item.pop(key, None)
    item.update(
        {
            "review_mode": review_mode,
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "protocol_profile": PROTOCOL_DEPENDENCY_SAFE_MULTI,
            "candidate_protocol": CANDIDATE_POLICY_RULE,
            "domain_policy": candidate_policy,
            "original_policy_sha256": stable_json_hash(original_policy),
            "candidate_policy_sha256": stable_json_hash(candidate_policy),
            "single_call_clause_replaced": clause_replaced,
            "mechanical_audit_facts": full_static(feature),
        }
    )
    if review_mode == "failure":
        failed_signals = [
            signal
            for signal, value in (
                ("reward", feature.get("source_reward")),
                ("correct", feature.get("source_correct")),
            )
            if not _is_one(value)
        ]
        item["known_outcome"] = {
            "reward": feature.get("source_reward"),
            "correct": feature.get("source_correct"),
            "label_group": label_group(feature),
            "failed_signals": failed_signals,
        }
    else:
        item.pop("known_outcome", None)
    item["payload_sha256"] = payload_hash(item)
    return item


def _event_turns(payload: dict[str, Any]) -> set[int]:
    facts = payload.get("mechanical_audit_facts") or {}
    selected: set[int] = set()
    for message in payload.get("conversation") or []:
        turn_index = message.get("turn_index")
        if not isinstance(turn_index, int) or isinstance(turn_index, bool):
            continue
        if message.get("role") == "tool" or message.get("tool_calls"):
            selected.add(turn_index)
    for value in facts.get("multitool_turns") or []:
        if isinstance(value, int):
            selected.add(value)
        elif isinstance(value, dict) and isinstance(value.get("turn_index"), int):
            selected.add(value["turn_index"])
    for key in ("tool_error_turns", "multiwrite_turns", "mixed_content_tool_turns"):
        selected.update(value for value in facts.get(key) or [] if isinstance(value, int))
    for error in facts.get("tool_argument_schema_errors") or []:
        if isinstance(error, dict) and isinstance(error.get("turn_index"), int):
            selected.add(error["turn_index"])
    for detail in facts.get("multitool_details") or []:
        if not isinstance(detail, dict):
            continue
        for association in detail.get("result_associations") or []:
            if not isinstance(association, dict):
                continue
            selected.update(
                turn
                for turn in association.get("result_turn_indices") or []
                if isinstance(turn, int) and not isinstance(turn, bool)
            )
    return selected


def _compact_strings(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        half = max(1, limit // 2)
        return value[:half] + "\n...[window excerpt omitted]...\n" + value[-half:]
    if isinstance(value, list):
        return [_compact_strings(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: _compact_strings(item, limit) for key, item in value.items()}
    return value


def _evenly_spaced(values: list[int], limit: int, required: set[int]) -> set[int]:
    if len(values) <= limit:
        return set(values)
    chosen = set(value for value in values if value in required)
    if len(chosen) >= limit:
        return set(sorted(chosen)[:limit])
    candidates = [value for value in values if value not in chosen]
    remaining = limit - len(chosen)
    if remaining == 1:
        chosen.add(candidates[len(candidates) // 2])
        return chosen
    positions = {
        round(index * (len(candidates) - 1) / (remaining - 1))
        for index in range(remaining)
    }
    chosen.update(candidates[position] for position in sorted(positions))
    return chosen


def _project_mechanical_facts(
    facts: dict[str, Any], selected_event_turns: set[int], content_limit: int
) -> tuple[dict[str, Any], list[int]]:
    """Keep every scalar fact but compact the verbose per-batch evidence graph."""

    projected = {
        key: _compact_strings(value, content_limit)
        for key, value in facts.items()
        if key != "multitool_details"
    }
    compact_details = []
    omitted = []
    for detail in facts.get("multitool_details") or []:
        if not isinstance(detail, dict):
            continue
        turn_index = detail.get("turn_index")
        if (
            selected_event_turns
            and isinstance(turn_index, int)
            and turn_index not in selected_event_turns
        ):
            omitted.append(turn_index)
            continue
        detail_projection = {
            key: _compact_strings(detail.get(key), content_limit)
            for key in (
                "turn_index",
                "batch_size",
                "read_call_count",
                "write_call_count",
                "duplicate_call_indices",
                "result_associations",
                "unassociated_result_turns",
            )
        }
        detail_projection["calls"] = [
            {
                key: _compact_strings(call.get(key), content_limit)
                for key in (
                    "call_index",
                    "id",
                    "name",
                    "arguments",
                    "arguments_parse_error",
                    "requestor",
                    "is_write",
                )
            }
            for call in detail.get("calls") or []
            if isinstance(call, dict)
        ]
        compact_details.append(detail_projection)
    projected["multitool_details"] = compact_details
    return projected, sorted(set(omitted))


def build_local_window(
    payload: dict[str, Any],
    *,
    radius: int = 2,
    leading: int = 6,
    trailing: int = 8,
    content_limit: int = 1200,
    max_event_turns: int | None = None,
    compact_reference_text: bool = False,
) -> dict[str, Any]:
    """Build deterministic event windows for a second review of over-context dialogs."""

    conversation = list(payload.get("conversation") or [])
    if not conversation:
        return dict(payload)
    positions = {
        message.get("turn_index"): index
        for index, message in enumerate(conversation)
        if isinstance(message.get("turn_index"), int)
    }
    all_event_turns = sorted(_event_turns(payload))
    selected_event_turns = set(all_event_turns)
    if max_event_turns is not None and len(all_event_turns) > max_event_turns:
        facts = payload.get("mechanical_audit_facts") or {}
        required = {
            turn
            for key in (
                "mixed_content_tool_turns",
                "tool_error_turns",
                "namespace_confusion_turns",
            )
            for turn in facts.get(key) or []
            if isinstance(turn, int) and not isinstance(turn, bool)
        }
        required.update(
            error["turn_index"]
            for error in facts.get("tool_argument_schema_errors") or []
            if isinstance(error, dict)
            and isinstance(error.get("turn_index"), int)
            and not isinstance(error.get("turn_index"), bool)
        )
        selected_event_turns = _evenly_spaced(
            all_event_turns, max_event_turns, required
        )
    keep = set(range(min(leading, len(conversation))))
    keep.update(range(max(0, len(conversation) - trailing), len(conversation)))
    for turn in selected_event_turns:
        position = positions.get(turn)
        if position is None:
            continue
        keep.update(
            range(
                max(0, position - radius),
                min(len(conversation), position + radius + 1),
            )
        )
    compact = []
    for index in sorted(keep):
        message = _compact_strings(dict(conversation[index]), content_limit)
        compact.append(message)
    full_payload_sha256 = payload["payload_sha256"]
    window = dict(payload)
    window.pop("payload_sha256", None)
    window["conversation"] = compact
    projected_facts, omitted_detail_turns = _project_mechanical_facts(
        window.get("mechanical_audit_facts") or {},
        selected_event_turns,
        content_limit,
    )
    window["mechanical_audit_facts"] = projected_facts
    if compact_reference_text:
        for key in ("agent_tools", "user_tools"):
            window[key] = _compact_strings(window.get(key) or [], content_limit)
    window["evidence_scope"] = {
        "kind": "deterministic_event_windows",
        "full_message_count": len(conversation),
        "included_message_count": len(compact),
        "included_turn_indices": [
            message.get("turn_index")
            for message in compact
            if message.get("turn_index") is not None
        ],
        "all_event_turn_indices": all_event_turns,
        "selected_event_turn_indices": sorted(selected_event_turns),
        "omitted_event_turn_indices": sorted(
            set(all_event_turns) - selected_event_turns
        ),
        "omitted_multitool_detail_turn_indices": omitted_detail_turns,
        "mechanical_projection": (
            "all scalar facts; compact calls and result associations; "
            "result message hashes omitted"
        ),
        "content_excerpt_character_limit": content_limit,
    }
    window["source_full_payload_sha256"] = full_payload_sha256
    window["payload_sha256"] = payload_hash(window)
    return window


def _token_sequence_length(encoded: Any) -> int:
    """Normalize legacy list and modern BatchEncoding chat-template outputs."""

    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer BatchEncoding has no input_ids")
        encoded = encoded["input_ids"]
    elif getattr(encoded, "input_ids", None) is not None:
        encoded = encoded.input_ids
    shape = getattr(encoded, "shape", None)
    if shape is not None and len(shape):
        return int(shape[-1])
    if not isinstance(encoded, (list, tuple)):
        raise TypeError(f"unsupported tokenizer output: {type(encoded).__name__}")
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError("expected a single rendered chat sequence")
        encoded = encoded[0]
    return len(encoded)


def _judge_input_tokens(payload: dict[str, Any], tokenizer: Any, review_mode: str) -> int:
    messages = request_messages(
        payload,
        review_mode=review_mode,
        protocol_profile=PROTOCOL_DEPENDENCY_SAFE_MULTI,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    count = _token_sequence_length(encoded)
    if count < 16:
        raise ValueError(
            f"{payload.get('_dialog_id')}: implausible judge input token count {count}"
        )
    return count


def _seal_routed_payload(
    payload: dict[str, Any], tokenizer: Any, review_mode: str
) -> dict[str, Any]:
    """Add the self-describing token count, then seal the routed payload hash."""

    item = dict(payload)
    for _ in range(3):
        count = _judge_input_tokens(item, tokenizer, review_mode)
        if item.get("judge_input_tokens") == count:
            break
        item["judge_input_tokens"] = count
    else:
        raise ValueError(f"{item.get('_dialog_id')}: judge token count did not stabilize")
    item.pop("payload_sha256", None)
    item["payload_sha256"] = payload_hash(item)
    return item


def _split_payloads(
    payloads: list[dict[str, Any]],
    *,
    tokenizer: Any,
    review_mode: str,
    local_input_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    local = []
    long = []
    windows = []
    for payload in payloads:
        item = _seal_routed_payload(payload, tokenizer, review_mode)
        if item["judge_input_tokens"] <= local_input_limit:
            local.append(item)
            continue
        long.append(item)
        window = None
        # Each retry is an explicitly described structured evidence window. The
        # corresponding fallback input above remains the unmodified full dialog.
        window_profiles = (
            {},
            {"radius": 1, "leading": 4, "trailing": 6, "content_limit": 600},
            {"radius": 0, "leading": 3, "trailing": 4, "content_limit": 300},
            {
                "radius": 0,
                "leading": 2,
                "trailing": 3,
                "content_limit": 180,
                "max_event_turns": 64,
                "compact_reference_text": True,
            },
            {
                "radius": 0,
                "leading": 2,
                "trailing": 2,
                "content_limit": 100,
                "max_event_turns": 32,
                "compact_reference_text": True,
            },
            {
                "radius": 0,
                "leading": 1,
                "trailing": 2,
                "content_limit": 72,
                "max_event_turns": 16,
                "compact_reference_text": True,
            },
        )
        for profile_index, profile in enumerate(window_profiles):
            candidate = build_local_window(item, **profile)
            candidate["evidence_scope"]["compression_profile"] = profile_index
            candidate.pop("payload_sha256", None)
            candidate["payload_sha256"] = payload_hash(candidate)
            candidate = _seal_routed_payload(candidate, tokenizer, review_mode)
            if candidate["judge_input_tokens"] <= local_input_limit:
                window = candidate
                break
        if window is None:
            raise ValueError(
                f"{item.get('_dialog_id')}: structured local window still exceeds "
                f"{local_input_limit} tokens after deterministic compression"
            )
        windows.append(window)
    return local, long, windows


def _project_feature(record: dict[str, Any], view: str) -> dict[str, Any]:
    excluded = {"full_source_static", "training_visible_static", "static"}
    projected = {key: value for key, value in record.items() if key not in excluded}
    projected["view"] = view
    projected["static"] = full_static(record) if view == "full_source" else training_static(record)
    return projected


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (math.nan, math.nan)
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((probability * (1 - probability) + z * z / (4 * total)) / total)
        / denominator
    )
    return center - margin, center + margin


def _percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2%}" if denominator else "n/a"


def _is_one(value: Any) -> bool:
    try:
        return math.isclose(float(value), 1.0)
    except (TypeError, ValueError):
        return False


def failure_flags(record: dict[str, Any]) -> dict[str, bool]:
    return {
        "reward": not _is_one(record.get("source_reward")),
        "correct": not _is_one(record.get("source_correct")),
        "consensus": label_group(record) == "consensus_failure",
        "non_consensus": label_group(record) != "consensus_success",
    }


def _risk_difference(
    multi_failures: int,
    multi_total: int,
    single_failures: int,
    single_total: int,
) -> tuple[float, float, float]:
    if not multi_total or not single_total:
        return (math.nan, math.nan, math.nan)
    multi_rate = multi_failures / multi_total
    single_rate = single_failures / single_total
    difference = multi_rate - single_rate
    standard_error = math.sqrt(
        multi_rate * (1 - multi_rate) / multi_total
        + single_rate * (1 - single_rate) / single_total
    )
    return difference, difference - 1.96 * standard_error, difference + 1.96 * standard_error


def comparison_report(features: list[dict[str, Any]], selection: list[dict[str, Any]]) -> str:
    label_counts = Counter(label_group(record) for record in features)
    full_multi = sum(has_multi(record) for record in features)
    visible_multi = sum(has_multi(record, view="training") for record in features)
    hidden_multi = sum(
        has_multi(record) and not has_multi(record, view="training") for record in features
    )
    lines = [
        "# tau2 SFT 单/多工具协议中立对照",
        "",
        "实验名：`tau2-sft-multitool-quality-audit`。本报告把 multi 当作行为事实，",
        "不再把调用数量直接等同于 policy 违规或任务失败。",
        "",
        "## 数据视图",
        "",
        f"- 训练可见 dialog：{len(features):,}",
        f"- full-source multi：{full_multi:,}/{len(features):,} ({_percent(full_multi, len(features))})",
        f"- longest-training-prefix multi：{visible_multi:,}/{len(features):,} "
        f"({_percent(visible_multi, len(features))})",
        f"- prefix 看似 single、full source 实为 multi：{hidden_multi:,}",
        "",
        "## 标签口径",
        "",
        "| group | dialogs |",
        "|---|---:|",
    ]
    for group in ("consensus_success", "consensus_failure", "reward_only", "correct_only", "unknown"):
        if label_counts[group]:
            lines.append(f"| `{group}` | {label_counts[group]:,} |")
    lines.extend(
        [
            "",
            "## Full-source 描述统计",
            "",
            "| domain | call mode | dialogs | reward fail | correct fail | consensus fail | consensus success |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for domain in ("airline", "retail", "telecom"):
        domain_records = [record for record in features if record["domain"] == domain]
        for multi in (False, True):
            cohort = [record for record in domain_records if has_multi(record) is multi]
            reward_failures = sum(failure_flags(record)["reward"] for record in cohort)
            correct_failures = sum(failure_flags(record)["correct"] for record in cohort)
            consensus_failures = sum(
                failure_flags(record)["consensus"] for record in cohort
            )
            successes = sum(label_group(record) == "consensus_success" for record in cohort)
            lines.append(
                f"| {domain} | {'multi' if multi else 'non-multi'} | {len(cohort):,} | "
                f"{_percent(reward_failures, len(cohort))} | "
                f"{_percent(correct_failures, len(cohort))} | "
                f"{_percent(consensus_failures, len(cohort))} | "
                f"{_percent(successes, len(cohort))} |"
            )
    lines.extend(
        [
            "",
            "## Multi minus non-multi failure risk",
            "",
            "下表是观察性 risk difference，不是 multi 的因果效果。CI 为未调整正态近似；",
            "single 样本少于 30 的 domain 标为样本量不足；`n>=30` 也不代表 task-level",
            "common support 已成立。",
            "",
            "| domain | label | multi n | single n | risk difference (95% CI) | size check |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for domain in ("all", "airline", "retail", "telecom"):
        domain_records = (
            features if domain == "all" else [r for r in features if r["domain"] == domain]
        )
        multi_records = [record for record in domain_records if has_multi(record)]
        single_records = [record for record in domain_records if not has_multi(record)]
        support = "n>=30" if min(len(multi_records), len(single_records)) >= 30 else "insufficient"
        for outcome in ("reward", "correct", "consensus"):
            difference, low, high = _risk_difference(
                sum(failure_flags(record)[outcome] for record in multi_records),
                len(multi_records),
                sum(failure_flags(record)[outcome] for record in single_records),
                len(single_records),
            )
            estimate = (
                "n/a"
                if math.isnan(difference)
                else f"{difference:+.2%} ({low:+.2%}–{high:+.2%})"
            )
            lines.append(
                f"| {domain} | {outcome} | {len(multi_records):,} | "
                f"{len(single_records):,} | {estimate} | {support} |"
            )
    selected_roles = Counter(role for item in selection for role in item["selection_roles"])
    lines.extend(
        [
            "",
            "## 语义复审集合",
            "",
            f"- 唯一 dialogs：{len(selection):,}",
            f"- 非一致成功：{selected_roles['non_consensus']:,}",
            f"- full-source single 成功 anchors：{selected_roles['single_success_anchor']:,}",
            f"- 去重后的 multi 成功 controls：{selected_roles['multi_success_control']:,}",
            "",
            "整体比例只用于描述；airline/retail 缺乏 non-multi common support 时不输出",
            "multi 的因果效果。失败归因由独立 schema 输出，unknown 始终保留在分母中。",
            "",
        ]
    )
    return "\n".join(lines)


def filter_spec_draft() -> str:
    return f"""# dependency-safe multi 筛选草案

实验名：`tau2-sft-multitool-quality-audit`。本文件是复审标准，不是训练 JSONL。

## 候选协议

{CANDIDATE_POLICY_RULE}

## 决策规则

- `keep-candidate`：标签为 `consensus_success`，双 judge 对非调用数量质量结论一致，
  且 `included_targets[].conversation_turn_index` 对应的每个 multi target 在两份
  `multi_turn_assessments` 中逐字段一致并为三类 `safe_independent_*`；参数必须 grounded，
  写入必须已授权且 order-independent，调用必须必要。不得有无效工具/schema、同批
  observation 依赖、未确认写入或无依据重复/fanout。
- `review`：标签冲突、judge 分歧、`batch_safety=uncertain`、长上下文窗口与完整审查
  不一致，或最终 evaluator/DB 证据不可见。
- `drop-candidate`：存在确定性工具错误或双 judge 一致确认的关键流程/参数/依赖错误。
  multi 的存在本身不是 drop reason。

## 后续训练约束

本轮不直接写训练数据。标准确认后从 raw source 重新生成 prompt 一致的数据，使用
`step_loss_mask` 只监督获准 target turn；single-only 与 dependency-safe-multi 对照必须
匹配训练 token、optimizer update、seed 和 official eval 配置。
"""


def audit_design() -> str:
    return f"""# tau2 SFT multi-tool 审计设计

实验名：`tau2-sft-multitool-quality-audit`。本实验保留现有 single-policy 结果作为
baseline，不编辑 tau2 policy，也不直接生成训练 JSONL。

## 现有筛选的问题

- legacy deterministic gate 把任意 multi turn 直接记为
  `multi_tool_policy_violation`，并在 judge 前进入 drop；因此不能用该结果检验 multi
  是否安全或更好。
- legacy judge system prompt 同样把 multi 定义为违规，`SUCCESS_CASES.md` 又把
  “每轮最多一个 tool call”写成成功共性；三者共享同一前提，不是独立证据。
- source `correct` 与 `reward` 曾合并为单一正/负结论；本实验保留
  consensus-success、consensus-failure、reward-only、correct-only 四组。
- raw full dialog、最长 retained raw prefix 和实际 converted training row 是不同对象。
  本实验分别保存行为事实与 converted row/system hash，并逐行验证转换。

## 待回答的问题

- 全量样本分别估计 reward/correct/consensus failure 在 multi/non-multi 下的观察比例；
  domain 缺少 common support 时不作因果判断。
- 对所有非 consensus-success 轨迹做失败归因，区分“轨迹出现 multi”与“决定性错误
  发生在 multi turn”。
- 对所有 single consensus-success 和领域/任务匹配的 multi controls 做双 judge
  示范质量比较；success-conditioned 结果不用于估计失败概率。

## 候选协议

{CANDIDATE_POLICY_RULE}

## 有效性边界

完整轨迹只交给可容纳其长度的 fallback；本地 judge 对超长轨迹读取显式记录省略项的
structured window。full/full 与 window/full 分开报告。payload、prompt version、模型配置、
输入数据和 converted artifact 均带 hash；陈旧、缺失或同 provider 的复审不能进入汇总。
"""


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    catalog = load_tool_catalog(args.tau2_root)
    features, payloads, inventory = audit_source_dialogs(
        args.source,
        args.training,
        catalog,
        protocol_profile=args.protocol_profile,
    )
    if inventory.get("conversion_mismatch_rows"):
        raise ValueError(
            "actual converted training rows do not match their raw source targets; "
            f"mismatches={inventory['conversion_mismatch_rows']}"
        )
    payload_by_id = {str(payload["_dialog_id"]): payload for payload in payloads}
    selection = build_comparison_selection(features)
    pairs = build_comparison_pairs(features, selection)
    selected_ids = {item["dialog_id"] for item in selection}
    quality_payloads = [
        _counterfactual_payload(feature, payload_by_id[str(feature["dialog_id"])], review_mode="quality")
        for feature in features
        if str(feature["dialog_id"]) in selected_ids
    ]
    failure_payloads = [
        _counterfactual_payload(feature, payload_by_id[str(feature["dialog_id"])], review_mode="failure")
        for feature in features
        if label_group(feature) != "consensus_success"
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    local_input_limit = args.local_context_limit - args.judge_max_tokens
    quality_local, quality_long, quality_windows = _split_payloads(
        quality_payloads,
        tokenizer=tokenizer,
        review_mode="quality",
        local_input_limit=local_input_limit,
    )
    failure_local, failure_long, failure_windows = _split_payloads(
        failure_payloads,
        tokenizer=tokenizer,
        review_mode="failure",
        local_input_limit=local_input_limit,
    )
    quality_payload_manifest = build_payload_manifest(
        "quality", quality_local, quality_long, quality_windows
    )
    failure_payload_manifest = build_payload_manifest(
        "failure", failure_local, failure_long, failure_windows
    )
    payload_manifest = quality_payload_manifest + failure_payload_manifest
    calibration_pilot_selection = build_calibration_selection(
        features, selection, quality_payload_manifest
    )
    calibration_v1_selection = build_quality_v1_selection(
        features, selection, quality_payload_manifest
    )
    calibration_selection = build_quality_validation_selection(
        features, selection, quality_payload_manifest
    )
    pilot_ids = {
        str(row["dialog_id"]) for row in calibration_pilot_selection
    }
    calibration_v1_ids = {
        str(row["dialog_id"]) for row in calibration_v1_selection
    }
    calibration_ids = {str(row["dialog_id"]) for row in calibration_selection}
    if (
        pilot_ids & calibration_v1_ids
        or pilot_ids & calibration_ids
        or calibration_v1_ids & calibration_ids
    ):
        raise ValueError(
            "quality pilot, validation v1, and validation v2 must be disjoint"
        )
    calibration_quality_local = [
        payload
        for payload in quality_local
        if str(payload["_dialog_id"]) in calibration_ids
    ]
    calibration_quality_long = [
        payload
        for payload in quality_long
        if str(payload["_dialog_id"]) in calibration_ids
    ]
    calibration_quality_windows = [
        payload
        for payload in quality_windows
        if str(payload["_dialog_id"]) in calibration_ids
    ]
    calibration_local_ids = {
        str(payload["_dialog_id"]) for payload in calibration_quality_local
    }
    calibration_long_ids = {
        str(payload["_dialog_id"]) for payload in calibration_quality_long
    }
    calibration_window_ids = {
        str(payload["_dialog_id"]) for payload in calibration_quality_windows
    }
    if (
        calibration_local_ids | calibration_long_ids != calibration_ids
        or calibration_local_ids & calibration_long_ids
        or calibration_long_ids != calibration_window_ids
    ):
        raise ValueError("calibration payload partitions do not match selection routes")

    failure_calibration_pilot_selection = (
        build_failure_calibration_pilot_selection(
            features, failure_payload_manifest
        )
    )
    failure_calibration_selection = build_failure_calibration_selection(
        features, failure_payload_manifest
    )
    failure_calibration_pilot_ids = {
        str(row["dialog_id"]) for row in failure_calibration_pilot_selection
    }
    failure_calibration_ids = {
        str(row["dialog_id"]) for row in failure_calibration_selection
    }
    if failure_calibration_pilot_ids & failure_calibration_ids:
        raise ValueError(
            "failure diagnostic pilot and validation must be disjoint"
        )
    calibration_failure_local = [
        payload
        for payload in failure_local
        if str(payload["_dialog_id"]) in failure_calibration_ids
    ]
    calibration_failure_long = [
        payload
        for payload in failure_long
        if str(payload["_dialog_id"]) in failure_calibration_ids
    ]
    calibration_failure_windows = [
        payload
        for payload in failure_windows
        if str(payload["_dialog_id"]) in failure_calibration_ids
    ]
    failure_calibration_local_ids = {
        str(payload["_dialog_id"]) for payload in calibration_failure_local
    }
    failure_calibration_long_ids = {
        str(payload["_dialog_id"]) for payload in calibration_failure_long
    }
    failure_calibration_window_ids = {
        str(payload["_dialog_id"]) for payload in calibration_failure_windows
    }
    if (
        failure_calibration_local_ids | failure_calibration_long_ids
        != failure_calibration_ids
        or failure_calibration_local_ids & failure_calibration_long_ids
        or failure_calibration_long_ids != failure_calibration_window_ids
    ):
        raise ValueError(
            "failure calibration payload partitions do not match selection routes"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    prepared_jsonl = {
        "features.jsonl": features,
        "full_source_features.jsonl": [
            _project_feature(record, "full_source") for record in features
        ],
        "training_visible_features.jsonl": [
            _project_feature(record, "training_visible") for record in features
        ],
        "comparison_selection.jsonl": selection,
        "comparison_pairs.jsonl": pairs,
        "calibration_pilot_selection.jsonl": calibration_pilot_selection,
        "calibration_validation_v1_selection.jsonl": calibration_v1_selection,
        "calibration_selection.jsonl": calibration_selection,
        "calibration_quality_judge_payloads_local.jsonl": calibration_quality_local,
        "calibration_quality_judge_payloads_long.jsonl": calibration_quality_long,
        "calibration_quality_local_windows.jsonl": calibration_quality_windows,
        "failure_calibration_pilot_selection.jsonl": (
            failure_calibration_pilot_selection
        ),
        "failure_calibration_selection.jsonl": failure_calibration_selection,
        "calibration_failure_judge_payloads_local.jsonl": calibration_failure_local,
        "calibration_failure_judge_payloads_long.jsonl": calibration_failure_long,
        "calibration_failure_local_windows.jsonl": calibration_failure_windows,
        "quality_judge_payloads_local.jsonl": quality_local,
        "quality_judge_payloads_long.jsonl": quality_long,
        "quality_local_windows.jsonl": quality_windows,
        "failure_judge_payloads_local.jsonl": failure_local,
        "failure_judge_payloads_long.jsonl": failure_long,
        "failure_local_windows.jsonl": failure_windows,
        "payload_manifest.jsonl": payload_manifest,
    }
    for filename, rows in prepared_jsonl.items():
        write_jsonl(args.out / filename, rows)
    tokenizer_files = {}
    if args.tokenizer.is_dir():
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
        ):
            candidate = args.tokenizer / filename
            if candidate.is_file():
                tokenizer_files[filename] = file_sha256(candidate)
    compression_profiles = Counter(
        window["evidence_scope"]["compression_profile"]
        for window in quality_windows + failure_windows
    )
    inventory = {
        **inventory,
        "protocol_profile": args.protocol_profile,
        "candidate_policy": CANDIDATE_POLICY_RULE,
        "label_groups": dict(sorted(Counter(label_group(record) for record in features).items())),
        "full_source_multitool_dialogs": sum(has_multi(record) for record in features),
        "training_visible_multitool_dialogs": sum(
            has_multi(record, view="training") for record in features
        ),
        "comparison_review_dialogs": len(selection),
        "comparison_matched_pairs": sum(
            pair["match_status"] == "matched" for pair in pairs
        ),
        "comparison_unmatched_anchors": sum(
            pair["match_status"] == "unmatched" for pair in pairs
        ),
        "failure_review_dialogs": len(failure_payloads),
        "calibration_pilot_dialogs": len(calibration_pilot_selection),
        "calibration_validation_v1_dialogs": len(calibration_v1_selection),
        "calibration_dialogs": len(calibration_selection),
        "calibration_quality_local_dialogs": len(calibration_quality_local),
        "calibration_quality_long_dialogs": len(calibration_quality_long),
        "quality_calibration_pilot_seed": QUALITY_CALIBRATION_PILOT_SEED,
        "quality_calibration_v1_seed": QUALITY_CALIBRATION_V1_SEED,
        "quality_calibration_validation_seed": QUALITY_CALIBRATION_VALIDATION_SEED,
        "quality_calibration_version": QUALITY_CALIBRATION_VERSION,
        "quality_calibration_pilot_selection_sha256": stable_json_hash(
            calibration_pilot_selection
        ),
        "quality_calibration_v1_selection_sha256": stable_json_hash(
            calibration_v1_selection
        ),
        "quality_calibration_selection_sha256": stable_json_hash(
            calibration_selection
        ),
        "failure_calibration_pilot_dialogs": len(
            failure_calibration_pilot_selection
        ),
        "failure_calibration_dialogs": len(failure_calibration_selection),
        "failure_calibration_local_dialogs": len(calibration_failure_local),
        "failure_calibration_long_dialogs": len(calibration_failure_long),
        "failure_calibration_pilot_seed": FAILURE_CALIBRATION_PILOT_SEED,
        "failure_calibration_pilot_version": FAILURE_CALIBRATION_PILOT_VERSION,
        "failure_calibration_pilot_selection_sha256": stable_json_hash(
            failure_calibration_pilot_selection
        ),
        "failure_calibration_seed": FAILURE_CALIBRATION_SEED,
        "failure_calibration_version": FAILURE_CALIBRATION_VERSION,
        "failure_calibration_selection_sha256": stable_json_hash(
            failure_calibration_selection
        ),
        "quality_local_dialogs": len(quality_local),
        "quality_long_dialogs": len(quality_long),
        "failure_local_dialogs": len(failure_local),
        "failure_long_dialogs": len(failure_long),
        "local_context_limit": args.local_context_limit,
        "judge_max_tokens": args.judge_max_tokens,
        "local_input_limit": local_input_limit,
        "window_compression_profiles": dict(sorted(compression_profiles.items())),
        "input_sha256": {
            "source": file_sha256(args.source),
            "training": file_sha256(args.training),
        },
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_file_sha256": tokenizer_files,
        "payload_manifest_sha256": stable_json_hash(payload_manifest),
    }
    artifact_hashes = {
        filename: {
            "rows": len(rows),
            "sha256": file_sha256(args.out / filename),
        }
        for filename, rows in prepared_jsonl.items()
    }
    write_json(args.out / "inventory.json", inventory)
    artifact_hashes["inventory.json"] = {
        "rows": None,
        "sha256": file_sha256(args.out / "inventory.json"),
    }
    analysis_dir = Path(__file__).resolve().parent
    code_paths = {
        "audit_sft_multitool.py": Path(__file__).resolve(),
        "sft_quality.py": analysis_dir / "sft_quality.py",
        "judge_sft_quality.py": analysis_dir / "judge_sft_quality.py",
        "run_multitool_quality_audit.sh": args.runner_script.resolve(),
    }
    write_json(
        args.out / "prepare_manifest.json",
        {
            "protocol_profile": args.protocol_profile,
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "input_sha256": inventory["input_sha256"],
            "tokenizer_path": str(args.tokenizer),
            "tokenizer_file_sha256": tokenizer_files,
            "runner_script_path": str(args.runner_script.resolve()),
            "code_sha256": {
                name: file_sha256(path) for name, path in code_paths.items()
            },
            "artifacts": artifact_hashes,
        },
    )
    (args.out / "COMPARISON_REPORT.md").write_text(
        comparison_report(features, selection), encoding="utf-8"
    )
    (args.out / "FILTER_SPEC_DRAFT.md").write_text(filter_spec_draft(), encoding="utf-8")
    (args.out / "AUDIT_DESIGN.md").write_text(audit_design(), encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True))


def _review_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    records = list(read_jsonl(path))
    counts = Counter(str(record.get("dialog_id")) for record in records)
    duplicates = sorted(dialog_id for dialog_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{path}: duplicate dialog reviews: {duplicates[:5]}")
    return {str(record["dialog_id"]): record for record in records}


def _manifest_map(
    path: Path, review_mode: str
) -> dict[str, dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("review_mode") == review_mode]
    counts = Counter(str(row.get("dialog_id")) for row in rows)
    duplicates = [dialog_id for dialog_id, count in counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"{path}: duplicate {review_mode} manifest IDs: {duplicates[:5]}")
    return {str(row["dialog_id"]): row for row in rows}


def _validated_review_map(
    path: Path,
    expected: dict[str, dict[str, Any]],
    *,
    review_mode: str,
    side: str,
    allow_additional: bool = False,
) -> dict[str, dict[str, Any]]:
    reviews = _review_map(path)
    unexpected = [] if allow_additional else sorted(set(reviews) - set(expected))
    missing = sorted(set(expected) - set(reviews))
    if unexpected or missing:
        raise ValueError(
            f"{path}: {review_mode}/{side} coverage mismatch; "
            f"unexpected={unexpected[:5]} missing={missing[:5]}"
        )
    hash_key = (
        "local_input_payload_sha256"
        if side == "local"
        else "fallback_input_payload_sha256"
    )
    selected_reviews = {
        dialog_id: reviews[dialog_id] for dialog_id in expected if dialog_id in reviews
    }
    for dialog_id, record in selected_reviews.items():
        manifest = expected[dialog_id]
        expected_scope = (
            manifest["local_evidence_scope"]
            if side == "local"
            else {"kind": "full_source"}
        )
        errors = []
        if record.get("review_mode") != review_mode:
            errors.append("review_mode")
        if record.get("protocol_profile") != manifest["protocol_profile"]:
            errors.append("protocol_profile")
        if record.get("prompt_version") != manifest["prompt_version"]:
            errors.append("prompt_version")
        if record.get("input_payload_sha256") != manifest[hash_key]:
            errors.append("input_payload_sha256")
        if (
            record.get("source_full_payload_sha256")
            != manifest["source_full_payload_sha256"]
        ):
            errors.append("source_full_payload_sha256")
        if record.get("evidence_scope") != expected_scope:
            errors.append("evidence_scope")
        judge_config = record.get("judge_config")
        judge_config_sha256 = record.get("judge_config_sha256")
        if not isinstance(judge_config_sha256, str) or not judge_config_sha256:
            errors.append("judge_config_sha256")
        if not isinstance(judge_config, dict):
            errors.append("judge_config")
        elif stable_json_hash(judge_config) != judge_config_sha256:
            errors.append("judge_config_sha256")
        if not isinstance(record.get("selected_review"), dict):
            errors.append("selected_review")
        selected_provider = str(record.get("selected_provider") or "")
        expected_prefix = "local:" if side == "local" else "fallback:"
        if not selected_provider.startswith(expected_prefix):
            errors.append("selected_provider")
        if errors:
            raise ValueError(
                f"{path}: {dialog_id} failed manifest validation: {errors}"
            )
    return selected_reviews


def _selected_review(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    review = record.get("selected_review")
    return review if isinstance(review, dict) else record


def _selected_judge_identity(record: dict[str, Any]) -> tuple[str, str]:
    """Resolve the actual selected endpoint/model, not its cosmetic label."""

    selected_provider = record.get("selected_provider")
    config = record.get("judge_config") or {}
    if selected_provider == record.get("primary_provider"):
        selected_config = config.get("primary")
    elif selected_provider == record.get("fallback_provider"):
        selected_config = config.get("fallback")
    else:
        raise ValueError(f"cannot resolve selected provider {selected_provider!r}")
    if not isinstance(selected_config, dict):
        raise ValueError(f"missing config for selected provider {selected_provider!r}")
    url = str(selected_config.get("url") or "").strip().rstrip("/").lower()
    model = str(selected_config.get("model") or "").strip().lower()
    if not url or not model:
        raise ValueError(f"incomplete selected judge identity {selected_config!r}")
    return url, model


def _validate_independent_reviews(
    review_mode: str,
    local_reviews: dict[str, dict[str, Any]],
    fallback_reviews: dict[str, dict[str, Any]],
) -> None:
    non_independent = []
    for dialog_id in local_reviews:
        local_url, local_model = _selected_judge_identity(local_reviews[dialog_id])
        fallback_url, fallback_model = _selected_judge_identity(
            fallback_reviews[dialog_id]
        )
        if local_url == fallback_url or local_model == fallback_model:
            non_independent.append(dialog_id)
    if non_independent:
        raise ValueError(
            f"{review_mode}: judges must use different actual endpoints and models; "
            f"non-independent IDs={non_independent[:5]}"
        )


def _review_reliability(record: dict[str, Any]) -> dict[str, Any]:
    """Return whether one judge record is safe to use as a determinate vote."""

    reasons = []
    selected = _selected_review(record)
    config = record.get("judge_config") or {}
    threshold = config.get("confidence_threshold", 0.75)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        reasons.append("invalid_confidence_threshold")
        threshold = 0.75
    confidence = selected.get("confidence") if isinstance(selected, dict) else None
    if not isinstance(selected, dict):
        reasons.append("missing_selected_review")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        reasons.append("invalid_confidence")
    elif confidence < threshold:
        reasons.append("low_confidence")
    if record.get("unresolved_escalation"):
        reasons.append("unresolved_escalation")
    if record.get("errors"):
        reasons.append("judge_errors")
    return {
        "reliable": not reasons,
        "reasons": sorted(set(reasons)),
        "confidence": confidence,
        "threshold": threshold,
    }


def _agreement(
    local: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
    fields: tuple[str, ...],
) -> bool:
    first = _selected_review(local)
    second = _selected_review(fallback)
    return bool(first and second and all(first.get(field) == second.get(field) for field in fields))


def _assessment_is_strict_safe(
    assessment: dict[str, Any], write_turns: set[int]
) -> bool:
    return bool(
        assessment.get("batch_safety")
        in {
            "safe_independent_reads",
            "safe_independent_writes",
            "safe_independent_mixed",
        }
        and assessment.get("arguments_grounded") == "yes"
        and (
            assessment.get("writes_authorized") == "yes"
            if assessment.get("turn_index") in write_turns
            else assessment.get("writes_authorized") == "not_applicable"
        )
        and assessment.get("calls_necessary") == "yes"
    )


def _quality_consensus(
    local_record: dict[str, Any],
    fallback_record: dict[str, Any],
    write_turns: set[int] | None = None,
) -> dict[str, Any]:
    local = _selected_review(local_record) or {}
    fallback = _selected_review(fallback_record) or {}
    local_reliability = _review_reliability(local_record)
    fallback_reliability = _review_reliability(fallback_record)
    both_reliable = bool(
        local_reliability["reliable"] and fallback_reliability["reliable"]
    )
    verdict_agrees = bool(
        both_reliable
        and local.get("quality_verdict") == fallback.get("quality_verdict")
    )
    batch_agrees = bool(
        both_reliable and local.get("batch_safety") == fallback.get("batch_safety")
    )
    dimensions = {}
    dimension_within_one = True
    for name in DEPENDENCY_SAFE_QUALITY_DIMENSIONS:
        local_score = local.get("dimensions", {}).get(name, {}).get("score")
        fallback_score = fallback.get("dimensions", {}).get(name, {}).get("score")
        if (
            not both_reliable
            or not isinstance(local_score, int)
            or not isinstance(fallback_score, int)
        ):
            dimensions[name] = None
            dimension_within_one = False
            continue
        if abs(local_score - fallback_score) > 1:
            dimensions[name] = None
            dimension_within_one = False
            continue
        dimensions[name] = (local_score + fallback_score) / 2

    local_turns = {
        item.get("turn_index"): item
        for item in local.get("multi_turn_assessments") or []
        if isinstance(item, dict)
    }
    fallback_turns = {
        item.get("turn_index"): item
        for item in fallback.get("multi_turn_assessments") or []
        if isinstance(item, dict)
    }
    per_turn_fields = (
        "batch_safety",
        "arguments_grounded",
        "writes_authorized",
        "calls_necessary",
    )
    per_turn_consensus = []
    per_turn_agrees = both_reliable and set(local_turns) == set(fallback_turns)
    for turn in sorted(set(local_turns) | set(fallback_turns), key=lambda value: (value is None, value)):
        first = local_turns.get(turn) or {}
        second = fallback_turns.get(turn) or {}
        result = {"turn_index": turn}
        for field in per_turn_fields:
            agrees = bool(
                both_reliable
                and first.get(field) == second.get(field)
                and first.get(field) is not None
            )
            result[field] = first.get(field) if agrees else "uncertain"
            per_turn_agrees &= agrees
        per_turn_consensus.append(result)
    write_turns = write_turns or set()
    if not per_turn_consensus:
        strict_batch_safe = batch_agrees and local.get("batch_safety") == "not_multi"
    else:
        strict_batch_safe = per_turn_agrees and all(
            _assessment_is_strict_safe(item, write_turns)
            for item in per_turn_consensus
        )
    quality_values = [value for value in dimensions.values() if value is not None]
    quality_verdict = (
        local.get("quality_verdict")
        if verdict_agrees
        else "review"
        if not both_reliable
        else "disagreement"
    )
    batch_safety = (
        local.get("batch_safety")
        if batch_agrees
        else "uncertain"
        if not both_reliable
        else "disagreement"
    )
    return {
        "quality_verdict": quality_verdict,
        "batch_safety": batch_safety,
        "both_reliable": both_reliable,
        "local_reliability": local_reliability,
        "fallback_reliability": fallback_reliability,
        "verdict_agrees": verdict_agrees,
        "batch_agrees": batch_agrees,
        "dimension_within_one": dimension_within_one,
        "per_turn_agrees": per_turn_agrees,
        "strict_batch_safe": strict_batch_safe,
        "keep_candidate": quality_verdict == "keep" and strict_batch_safe,
        "dimensions": dimensions,
        "mean_quality": (
            statistics.mean(quality_values)
            if dimension_within_one
            and len(quality_values) == len(DEPENDENCY_SAFE_QUALITY_DIMENSIONS)
            else None
        ),
        "multi_turn_assessments": per_turn_consensus,
    }


def _feature_write_turns(feature: dict[str, Any]) -> set[int]:
    return {
        turn
        for turn in full_static(feature).get("multiwrite_turns") or []
        if isinstance(turn, int) and not isinstance(turn, bool)
    }


def _quality_consensus_map(
    features: list[dict[str, Any]],
    local_reviews: dict[str, dict[str, Any]],
    fallback_reviews: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    feature_by_id = {str(record["dialog_id"]): record for record in features}
    return {
        dialog_id: _quality_consensus(
            local_reviews[dialog_id],
            fallback_reviews[dialog_id],
            _feature_write_turns(feature_by_id[dialog_id]),
        )
        for dialog_id in manifest
    }


def filter_decision_rows(
    features: list[dict[str, Any]],
    local_reviews: dict[str, dict[str, Any]],
    fallback_reviews: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a provenance-only decision row for every converted target.

    The rows intentionally contain hashes, indices, labels, judge identities, and
    decisions only. They never contain prompts, conversations, or target text and
    therefore cannot be consumed as training JSONL.
    """

    consensus = _quality_consensus_map(
        features, local_reviews, fallback_reviews, manifest
    )
    rows = []
    for feature in sorted(features, key=lambda item: str(item["dialog_id"])):
        dialog_id = str(feature["dialog_id"])
        manifest_row = manifest.get(dialog_id)
        result = consensus.get(dialog_id)
        mechanical_issues = sorted(
            str(issue)
            for issue in full_static(feature).get("mechanical_hard_issues") or []
        )
        reasons = []
        group = label_group(feature)
        if group != "consensus_success":
            reasons.append(f"label_group:{group}")
        if result is None:
            reasons.append("quality_review:not_selected")
        else:
            if not result["both_reliable"]:
                reasons.append("quality_review:judge_pair_unreliable")
            if result["quality_verdict"] != "keep":
                reasons.append(
                    f"quality_verdict:{result['quality_verdict']}"
                )
            if not result["strict_batch_safe"]:
                reasons.append("batch_safety:not_strict_batch_safe")
        reasons.extend(f"mechanical_hard_issue:{issue}" for issue in mechanical_issues)
        dialog_keep_candidate = bool(
            group == "consensus_success"
            and result is not None
            and result["keep_candidate"]
            and not mechanical_issues
        )
        assessments = {
            item["turn_index"]: item
            for item in (result or {}).get("multi_turn_assessments") or []
            if isinstance(item, dict) and isinstance(item.get("turn_index"), int)
        }
        write_turns = _feature_write_turns(feature)
        conversions_by_source_row = {
            check.get("source_row"): check
            for check in feature.get("conversion_checks") or []
            if isinstance(check, dict)
        }
        targets = list(feature.get("included_targets") or [])
        if not targets:
            targets = [None]
        for target in sorted(
            targets,
            key=lambda item: (
                item is None,
                (item or {}).get("source_row", -1),
                (item or {}).get("conversation_turn_index", -1),
            ),
        ):
            target = target or {}
            target_reasons = list(reasons)
            target_turn = target.get("conversation_turn_index")
            assessment = assessments.get(target_turn) if target.get("is_multitool") else None
            if target.get("is_multitool"):
                if assessment is None:
                    target_reasons.append("target_multi_assessment:missing")
                elif not _assessment_is_strict_safe(assessment, write_turns):
                    target_reasons.append("target_multi_assessment:not_strict_safe")
            target_keep_candidate = dialog_keep_candidate and not target_reasons
            conversion = conversions_by_source_row.get(target.get("source_row")) or {}
            judge_provenance = None
            review_scope = None
            if manifest_row is not None:
                local_record = local_reviews[dialog_id]
                fallback_record = fallback_reviews[dialog_id]
                judge_provenance = {
                    "local_judge_config_sha256": local_record.get(
                        "judge_config_sha256"
                    ),
                    "local_selected_provider": local_record.get("selected_provider"),
                    "local_selected_identity": list(
                        _selected_judge_identity(local_record)
                    ),
                    "fallback_judge_config_sha256": fallback_record.get(
                        "judge_config_sha256"
                    ),
                    "fallback_selected_provider": fallback_record.get(
                        "selected_provider"
                    ),
                    "fallback_selected_identity": list(
                        _selected_judge_identity(fallback_record)
                    ),
                }
                review_scope = {
                    "route": manifest_row["route"],
                    "source_full_payload_sha256": manifest_row[
                        "source_full_payload_sha256"
                    ],
                    "local_input_payload_sha256": manifest_row[
                        "local_input_payload_sha256"
                    ],
                    "fallback_input_payload_sha256": manifest_row[
                        "fallback_input_payload_sha256"
                    ],
                }
            row = {
                "artifact_kind": "provenance_only_filter_decision",
                "contains_training_messages": False,
                "dialog_id": dialog_id,
                "domain": feature.get("domain"),
                "label_group": group,
                "source_correct": feature.get("source_correct"),
                "source_reward": feature.get("source_reward"),
                "protocol_profile": (
                    manifest_row.get("protocol_profile")
                    if manifest_row is not None
                    else feature.get("protocol_profile")
                ),
                "prompt_version": (
                    manifest_row.get("prompt_version")
                    if manifest_row is not None
                    else None
                ),
                "dialog_reviewed": result is not None,
                "dialog_quality_verdict": (
                    result.get("quality_verdict") if result is not None else None
                ),
                "dialog_strict_batch_safe": (
                    result.get("strict_batch_safe") if result is not None else False
                ),
                "dialog_judge_keep_candidate": (
                    result.get("keep_candidate") if result is not None else False
                ),
                "dialog_keep_candidate": dialog_keep_candidate,
                "mechanical_hard_issues": mechanical_issues,
                "review_scope": review_scope,
                "judge_provenance": judge_provenance,
                "target": {
                    "source_row": target.get("source_row"),
                    "training_turn_index": target.get("turn_index"),
                    "conversation_turn_index": target_turn,
                    "fingerprint": target.get("fingerprint"),
                    "is_multitool": bool(target.get("is_multitool")),
                    "num_tool_calls": target.get("num_tool_calls"),
                    "num_write_calls": target.get("num_write_calls"),
                    "tool_names": target.get("tool_names") or [],
                    "multi_turn_assessment": assessment,
                },
                "conversion_provenance": {
                    key: conversion.get(key)
                    for key in (
                        "source_non_system_hash",
                        "converted_non_system_hash",
                        "source_target_fingerprint",
                        "expected_converted_target_hash",
                        "converted_target_hash",
                        "converted_messages_hash",
                        "converted_system_hash",
                        "non_system_messages_match",
                        "source_system_preserved",
                        "target_fingerprint_match",
                    )
                },
                "decision": (
                    "keep-candidate"
                    if target_keep_candidate
                    else "not-keep-candidate"
                ),
                "keep_candidate": target_keep_candidate,
                "decision_reasons": sorted(set(target_reasons)),
            }
            row["decision_sha256"] = stable_json_hash(row)
            rows.append(row)
    return rows


def _bootstrap_mean_ci(values: list[float], seed: int = 20260802) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], values[0])
    generator = random.Random(seed)
    means = sorted(
        statistics.mean(generator.choice(values) for _ in values)
        for _ in range(5000)
    )
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def quality_report(
    features: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    local_reviews: dict[str, dict[str, Any]],
    fallback_reviews: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    feature_by_id = {str(record["dialog_id"]): record for record in features}
    consensus = _quality_consensus_map(
        features, local_reviews, fallback_reviews, manifest
    )
    roles_by_id = {
        item["dialog_id"]: set(item["selection_roles"]) for item in selection
    }
    cohort_ids = {
        "single success anchors": sorted(
            dialog_id
            for dialog_id, roles in roles_by_id.items()
            if "single_success_anchor" in roles
        ),
        "multi success controls": sorted(
            dialog_id
            for dialog_id, roles in roles_by_id.items()
            if "multi_success_control" in roles
        ),
    }
    lines = [
        "## 双 judge 成功轨迹质量",
        "",
        "这里比较的是已成功轨迹的示范质量；失败倾向使用上面的全量标签风险，不能从",
        "success-conditioned 样本推断。`strict batch-safe` 要求两位 judge 对每个 multi turn 的",
        "安全、参数依据、写入授权和必要性完全一致；任一低置信、未解决升级或 judge",
        "错误均按 indeterminate 保留在分母中，不能成为 keep candidate。",
        "",
        "| cohort | dialogs | both reliable | score determinate | both keep | strict batch-safe | keep candidate | verdict disagree | batch disagree | mean quality /4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cohort_summary = {}
    for cohort, dialog_ids in cohort_ids.items():
        results = [consensus[dialog_id] for dialog_id in dialog_ids]
        quality_values = [
            result["mean_quality"]
            for result in results
            if result["mean_quality"] is not None
        ]
        values = {
            "dialogs": len(dialog_ids),
            "both_reliable": sum(result["both_reliable"] for result in results),
            "score_determinate": sum(
                result["dimension_within_one"] for result in results
            ),
            "both_keep": sum(result["quality_verdict"] == "keep" for result in results),
            "strict_batch_safe": sum(
                result["strict_batch_safe"] for result in results
            ),
            "keep_candidate": sum(result["keep_candidate"] for result in results),
            "verdict_disagreement": sum(
                result["both_reliable"] and not result["verdict_agrees"]
                for result in results
            ),
            "batch_disagreement": sum(
                result["both_reliable"] and not result["batch_agrees"]
                for result in results
            ),
            "mean_quality": statistics.mean(quality_values) if quality_values else None,
        }
        cohort_summary[cohort] = values
        lines.append(
            f"| {cohort} | {values['dialogs']:,} | "
            f"{_percent(values['both_reliable'], values['dialogs'])} | "
            f"{_percent(values['score_determinate'], values['dialogs'])} | "
            f"{_percent(values['both_keep'], values['dialogs'])} | "
            f"{_percent(values['strict_batch_safe'], values['dialogs'])} | "
            f"{_percent(values['keep_candidate'], values['dialogs'])} | "
            f"{_percent(values['verdict_disagreement'], values['dialogs'])} | "
            f"{_percent(values['batch_disagreement'], values['dialogs'])} | "
            f"{values['mean_quality']:.3f} |"
            if values["mean_quality"] is not None
            else f"| {cohort} | {values['dialogs']:,} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
        )

    lines.extend(
        [
            "",
            "## Matched success differences",
            "",
            "每个 single anchor 先对其最多三个全局不重复的 multi controls 求平均，再以",
            "anchor 为 bootstrap 单位。差值为 multi − single。",
            "",
            "| domain | metric | matched anchors | support | mean difference | 95% cluster bootstrap CI |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    matched = [pair for pair in pairs if pair.get("match_status") == "matched"]
    metrics = ("mean_quality",) + tuple(DEPENDENCY_SAFE_QUALITY_DIMENSIONS)
    matched_summary = {}
    for domain in ("all", "airline", "retail", "telecom"):
        domain_pairs = [
            pair for pair in matched if domain == "all" or pair["domain"] == domain
        ]
        controls_by_anchor: dict[str, list[str]] = defaultdict(list)
        for pair in domain_pairs:
            controls_by_anchor[pair["single_dialog_id"]].append(
                pair["multi_dialog_id"]
            )
        matched_summary[domain] = {}
        for metric in metrics:
            deltas = []
            for anchor_id, control_ids in controls_by_anchor.items():
                anchor_result = consensus[anchor_id]
                anchor_value = (
                    anchor_result["mean_quality"]
                    if metric == "mean_quality"
                    else anchor_result["dimensions"].get(metric)
                )
                control_values = [
                    consensus[control_id]["mean_quality"]
                    if metric == "mean_quality"
                    else consensus[control_id]["dimensions"].get(metric)
                    for control_id in control_ids
                ]
                control_values = [
                    value for value in control_values if value is not None
                ]
                if anchor_value is None or not control_values:
                    continue
                deltas.append(statistics.mean(control_values) - anchor_value)
            low, high = _bootstrap_mean_ci(deltas)
            mean_delta = statistics.mean(deltas) if deltas else math.nan
            support = "adequate" if len(deltas) >= 5 else "insufficient (<5)"
            matched_summary[domain][metric] = {
                "matched_anchors": len(deltas),
                "support": support,
                "mean_difference": None if math.isnan(mean_delta) else mean_delta,
                "bootstrap_ci": None if math.isnan(low) else [low, high],
            }
            estimate = "n/a" if math.isnan(mean_delta) else f"{mean_delta:+.3f}"
            interval = "n/a" if math.isnan(low) else f"{low:+.3f}–{high:+.3f}"
            lines.append(
                f"| {domain} | `{metric}` | {len(deltas):,} | {support} | "
                f"{estimate} | {interval} |"
            )

    full_scope = sum(row["route"] == "full/full" for row in manifest.values())
    verdict_agreement = sum(result["verdict_agrees"] for result in consensus.values())
    batch_agreement = sum(result["batch_agrees"] for result in consensus.values())
    reliable_pairs = sum(result["both_reliable"] for result in consensus.values())
    determinate_scores = sum(
        result["dimension_within_one"] for result in consensus.values()
    )
    multi_review_ids = [
        dialog_id
        for dialog_id in manifest
        if full_static(feature_by_id[dialog_id]).get("multitool_turns")
    ]
    per_turn_agreement = sum(
        consensus[dialog_id]["per_turn_agrees"] for dialog_id in multi_review_ids
    )
    strict_batch_safe = sum(
        result["strict_batch_safe"] for result in consensus.values()
    )
    keep_candidates = sum(result["keep_candidate"] for result in consensus.values())
    route_summary = {}
    for route in ("full/full", "window/full"):
        route_ids = [
            dialog_id for dialog_id, row in manifest.items() if row["route"] == route
        ]
        route_multi_ids = [
            dialog_id
            for dialog_id in route_ids
            if full_static(feature_by_id[dialog_id]).get("multitool_turns")
        ]
        route_summary[route] = {
            "dialogs": len(route_ids),
            "both_reliable": sum(
                consensus[dialog_id]["both_reliable"] for dialog_id in route_ids
            ),
            "verdict_agreement": sum(
                consensus[dialog_id]["verdict_agrees"] for dialog_id in route_ids
            ),
            "batch_agreement": sum(
                consensus[dialog_id]["batch_agrees"] for dialog_id in route_ids
            ),
            "per_turn_exact_agreement": sum(
                consensus[dialog_id]["per_turn_agrees"]
                for dialog_id in route_multi_ids
            ),
            "multi_dialogs": len(route_multi_ids),
        }
    lines.extend(
        [
            "",
            "## 复审覆盖与一致性",
            "",
            f"- full/full：{full_scope:,}；window/full：{len(manifest) - full_scope:,}",
            f"- both reliable：{reliable_pairs:,}/{len(manifest):,}",
            f"- all-dimension score concordance (每维差值 ≤1)：{determinate_scores:,}/{len(manifest):,}",
            f"- verdict agreement：{verdict_agreement:,}/{len(manifest):,}",
            f"- aggregate batch agreement：{batch_agreement:,}/{len(manifest):,}",
            f"- all-multi-turn exact agreement：{per_turn_agreement:,}/{len(multi_review_ids):,} multi dialogs",
            f"- strict batch-safe：{strict_batch_safe:,}/{len(manifest):,}",
            f"- judge keep-candidate intersection：{keep_candidates:,}/{len(manifest):,}",
            "",
            "| route | dialogs | both reliable | verdict agree | batch agree | all-multi-turn exact |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for route, values in route_summary.items():
        lines.append(
            f"| {route} | {values['dialogs']:,} | "
            f"{_percent(values['both_reliable'], values['dialogs'])} | "
            f"{_percent(values['verdict_agreement'], values['dialogs'])} | "
            f"{_percent(values['batch_agreement'], values['dialogs'])} | "
            f"{_percent(values['per_turn_exact_agreement'], values['multi_dialogs'])} |"
        )
    lines.append("")
    return "\n".join(lines), {
        "review_dialogs": len(manifest),
        "full_scope_dialogs": full_scope,
        "window_scope_dialogs": len(manifest) - full_scope,
        "verdict_agreement": verdict_agreement,
        "batch_agreement": batch_agreement,
        "both_reliable": reliable_pairs,
        "all_dimension_score_concordance": determinate_scores,
        "per_turn_exact_agreement": per_turn_agreement,
        "per_turn_agreement_multi_dialogs": len(multi_review_ids),
        "strict_batch_safe": strict_batch_safe,
        "keep_candidates": keep_candidates,
        "agreement_by_route": route_summary,
        "cohorts": cohort_summary,
        "matched_differences": matched_summary,
    }


def calibration_report(
    calibration: list[dict[str, Any]],
    features: list[dict[str, Any]],
    local_reviews: dict[str, dict[str, Any]],
    fallback_reviews: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    feature_by_id = {str(record["dialog_id"]): record for record in features}
    results = {
        row["dialog_id"]: _quality_consensus(
            local_reviews[row["dialog_id"]],
            fallback_reviews[row["dialog_id"]],
            {
                turn
                for turn in full_static(feature_by_id[row["dialog_id"]]).get(
                    "multiwrite_turns"
                )
                or []
                if isinstance(turn, int) and not isinstance(turn, bool)
            },
        )
        for row in calibration
    }
    dimension_differences = []
    confidences = []
    hard_ids = []
    hard_both_keep = 0
    for row in calibration:
        dialog_id = row["dialog_id"]
        local = _selected_review(local_reviews[dialog_id]) or {}
        fallback = _selected_review(fallback_reviews[dialog_id]) or {}
        confidences.extend(
            confidence
            for confidence in (local.get("confidence"), fallback.get("confidence"))
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        )
        for name in DEPENDENCY_SAFE_QUALITY_DIMENSIONS:
            first = local.get("dimensions", {}).get(name, {}).get("score")
            second = fallback.get("dimensions", {}).get(name, {}).get("score")
            if (
                results[dialog_id]["both_reliable"]
                and isinstance(first, int)
                and isinstance(second, int)
            ):
                dimension_differences.append(abs(first - second))
        mechanical_issues = (
            full_static(feature_by_id[dialog_id]).get("mechanical_hard_issues") or []
        )
        if mechanical_issues:
            hard_ids.append(dialog_id)
            hard_both_keep += int(
                local.get("quality_verdict") == "keep"
                and fallback.get("quality_verdict") == "keep"
            )
    total = len(calibration)
    multi_ids = [
        row["dialog_id"]
        for row in calibration
        if full_static(feature_by_id[row["dialog_id"]]).get("multitool_turns")
    ]
    reliable_ids = [
        row["dialog_id"]
        for row in calibration
        if results[row["dialog_id"]]["both_reliable"]
    ]
    fallback_keep_ids = [
        dialog_id
        for dialog_id in reliable_ids
        if (_selected_review(fallback_reviews[dialog_id]) or {}).get(
            "quality_verdict"
        )
        == "keep"
    ]
    fallback_keep_supported = sum(
        (_selected_review(local_reviews[dialog_id]) or {}).get("quality_verdict")
        == "keep"
        for dialog_id in fallback_keep_ids
    )
    extreme_verdict_disagreements = sum(
        {
            (_selected_review(local_reviews[dialog_id]) or {}).get(
                "quality_verdict"
            ),
            (_selected_review(fallback_reviews[dialog_id]) or {}).get(
                "quality_verdict"
            ),
        }
        == {"keep", "drop"}
        for dialog_id in reliable_ids
    )
    local_keep_fallback_drop = sum(
        (_selected_review(local_reviews[dialog_id]) or {}).get(
            "quality_verdict"
        )
        == "keep"
        and (_selected_review(fallback_reviews[dialog_id]) or {}).get(
            "quality_verdict"
        )
        == "drop"
        for dialog_id in reliable_ids
    )
    fallback_keep_local_not_keep = (
        len(fallback_keep_ids) - fallback_keep_supported
    )
    metrics = {
        "dialogs": total,
        "both_reliable": sum(result["both_reliable"] for result in results.values()),
        "verdict_agreement": sum(result["verdict_agrees"] for result in results.values()),
        "batch_agreement": sum(result["batch_agrees"] for result in results.values()),
        "per_turn_exact_agreement": sum(
            results[dialog_id]["per_turn_agrees"] for dialog_id in multi_ids
        ),
        "per_turn_agreement_multi_dialogs": len(multi_ids),
        "mean_absolute_dimension_difference": (
            statistics.mean(dimension_differences) if dimension_differences else None
        ),
        "median_confidence": statistics.median(confidences) if confidences else None,
        "mechanical_hard_dialogs": len(hard_ids),
        "mechanical_hard_both_keep": hard_both_keep,
        "fallback_keep_dialogs": len(fallback_keep_ids),
        "fallback_keep_supported_by_local": fallback_keep_supported,
        "extreme_verdict_disagreements": extreme_verdict_disagreements,
        "local_keep_fallback_drop": local_keep_fallback_drop,
        "fallback_keep_local_not_keep": fallback_keep_local_not_keep,
    }
    false_keep_rate = hard_both_keep / len(hard_ids) if hard_ids else 0.0
    gates = {
        "declared_quality_calibration_size_72": (
            total == QUALITY_CALIBRATION_LIMIT
        ),
        "fallback_keep_coverage_nonzero": bool(fallback_keep_ids),
        "fallback_keep_supported_by_local_at_least_0.90": (
            fallback_keep_supported / len(fallback_keep_ids) >= 0.90
            if fallback_keep_ids
            else False
        ),
        "batch_agreement_at_least_0.80": (
            metrics["batch_agreement"] / total >= 0.80 if total else False
        ),
        "both_reliable_at_least_0.90": (
            metrics["both_reliable"] / total >= 0.90 if total else False
        ),
        "per_turn_agreement_at_least_0.70": (
            metrics["per_turn_exact_agreement"] / len(multi_ids) >= 0.70
            if multi_ids
            else False
        ),
        "mean_dimension_difference_at_most_0.75": (
            metrics["mean_absolute_dimension_difference"] is not None
            and metrics["mean_absolute_dimension_difference"] <= 0.75
        ),
        "median_confidence_at_least_0.70": (
            metrics["median_confidence"] is not None
            and metrics["median_confidence"] >= 0.70
        ),
        "mechanical_hard_coverage_nonzero": bool(hard_ids),
        "mechanical_false_keep_at_most_0.05": (
            bool(hard_ids) and false_keep_rate <= 0.05
        ),
    }
    status = "pass" if all(gates.values()) else "fail"
    route_counts = Counter(row["route"] for row in calibration)
    route_metrics = {}
    route_by_id = {row["dialog_id"]: row["route"] for row in calibration}
    for route in ("full/full", "window/full"):
        route_ids = [
            dialog_id for dialog_id, value in route_by_id.items() if value == route
        ]
        route_multi_ids = [
            dialog_id
            for dialog_id in route_ids
            if full_static(feature_by_id[dialog_id]).get("multitool_turns")
        ]
        route_metrics[route] = {
            "dialogs": len(route_ids),
            "both_reliable": sum(
                results[dialog_id]["both_reliable"] for dialog_id in route_ids
            ),
            "verdict_agreement": sum(
                results[dialog_id]["verdict_agrees"] for dialog_id in route_ids
            ),
            "batch_agreement": sum(
                results[dialog_id]["batch_agrees"] for dialog_id in route_ids
            ),
            "per_turn_exact_agreement": sum(
                results[dialog_id]["per_turn_agrees"]
                for dialog_id in route_multi_ids
            ),
            "multi_dialogs": len(route_multi_ids),
        }
    metrics["by_route"] = route_metrics
    lines = [
        "# tau2 SFT multi-tool judge calibration",
        "",
        "这是与两个 frozen diagnostic cohort 都不重叠、由 prepare 前置封存的",
        "72-dialog validation-v2 set。",
        "最终筛选只保留 local 与 fallback 的 keep 交集，因此 verdict gate 检验更严格",
        "fallback keep 是否得到 local 支持；local keep / fallback drop 会被交集安全否决，",
        "只作为方向性诊断报告。三分类 exact agreement 仍报告但不把 judge 的系统性",
        "尺度偏移误当成不安全 keep。",
        "一致性不等于 ground-truth accuracy；gate 失败时不得发布自动 keep 数据。",
        "",
        "## Gate",
        "",
        f"Status：`{status}`。",
        "",
        "| gate | pass |",
        "|---|---|",
    ]
    for gate, passed in gates.items():
        lines.append(f"| `{gate}` | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- dialogs：{total:,} (full/full {route_counts['full/full']:,}; window/full {route_counts['window/full']:,})",
            f"- verdict agreement：{metrics['verdict_agreement']:,}/{total:,}",
            f"- fallback keep supported by local：{fallback_keep_supported:,}/{len(fallback_keep_ids):,}",
            f"- extreme keep/drop disagreements：{extreme_verdict_disagreements:,}/{len(reliable_ids):,}",
            f"- local keep / fallback drop (conservative veto)：{local_keep_fallback_drop:,}/{len(reliable_ids):,}",
            f"- fallback keep / local non-keep：{fallback_keep_local_not_keep:,}/{len(fallback_keep_ids):,}",
            f"- batch agreement：{metrics['batch_agreement']:,}/{total:,}",
            f"- both reliable：{metrics['both_reliable']:,}/{total:,}",
            f"- all-multi-turn exact agreement：{metrics['per_turn_exact_agreement']:,}/{len(multi_ids):,} multi dialogs",
            f"- mean absolute dimension difference：{metrics['mean_absolute_dimension_difference']}",
            f"- median confidence：{metrics['median_confidence']}",
            f"- mechanical hard both-keep：{hard_both_keep:,}/{len(hard_ids):,}",
            "",
            "| route | dialogs | both reliable | verdict agree | batch agree | all-multi-turn exact |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for route, values in route_metrics.items():
        lines.append(
            f"| {route} | {values['dialogs']:,} | "
            f"{_percent(values['both_reliable'], values['dialogs'])} | "
            f"{_percent(values['verdict_agreement'], values['dialogs'])} | "
            f"{_percent(values['batch_agreement'], values['dialogs'])} | "
            f"{_percent(values['per_turn_exact_agreement'], values['multi_dialogs'])} |"
        )
    lines.append("")
    return "\n".join(lines), {"status": status, "gates": gates, "metrics": metrics}


def _failure_field_consensus(
    local_record: dict[str, Any],
    fallback_record: dict[str, Any],
    field: str,
    *,
    both_reliable: bool | None = None,
) -> tuple[str, str]:
    """Resolve one failure field without treating agreed unknown as attribution."""

    if field not in FAILURE_UNKNOWN_VALUES:
        raise ValueError(f"unknown failure-attribution field: {field}")
    if both_reliable is None:
        both_reliable = bool(
            _review_reliability(local_record)["reliable"]
            and _review_reliability(fallback_record)["reliable"]
        )
    unknown = FAILURE_UNKNOWN_VALUES[field]
    if not both_reliable:
        return unknown, "unreliable"
    first = (_selected_review(local_record) or {}).get(field)
    second = (_selected_review(fallback_record) or {}).get(field)
    if first is None or first != second:
        return unknown, "disagreed"
    value = str(first)
    if value == unknown:
        return value, "agreed_unknown"
    return value, "agreed_resolved"


def _failure_mode_evidence_consistent(
    review: dict[str, Any], manifest_row: dict[str, Any]
) -> bool:
    """Check Luna's causal mode against sealed full-source turn types."""

    turn_types = manifest_row.get("full_source_turn_types") or {}
    decisive = {
        turn
        for turn in review.get("decisive_turns") or []
        if isinstance(turn, int) and not isinstance(turn, bool)
    }
    mode = review.get("causal_call_mode")
    expected_kind = {
        "multi_turn": "assistant_multi",
        "single_call_turn": "assistant_single",
        "non_tool": "assistant_non_tool",
    }.get(mode)
    if expected_kind is not None:
        return bool(decisive & set(turn_types.get(expected_kind) or []))
    if mode == "environment_or_user":
        return bool(
            decisive
            & (
                set(turn_types.get("user") or [])
                | set(turn_types.get("tool") or [])
            )
        )
    return mode == "unknown" and not decisive


def failure_calibration_report(
    calibration: list[dict[str, Any]],
    features: list[dict[str, Any]],
    local_reviews: dict[str, dict[str, Any]],
    fallback_reviews: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Gate Luna-primary attribution on evidence consistency, not judge identity."""

    feature_by_id = {str(record["dialog_id"]): record for record in features}
    manifest = manifest or {
        str(row["dialog_id"]): {
            "route": row.get("route", "full/full"),
            "full_source_turn_types": {},
        }
        for row in calibration
    }
    records = []
    for row in calibration:
        dialog_id = str(row["dialog_id"])
        feature = feature_by_id[dialog_id]
        local_record = local_reviews[dialog_id]
        fallback_record = fallback_reviews[dialog_id]
        local = _selected_review(local_record) or {}
        fallback = _selected_review(fallback_record) or {}
        local_reliable = _review_reliability(local_record)["reliable"]
        fallback_reliable = _review_reliability(fallback_record)["reliable"]
        records.append(
            {
                "dialog_id": dialog_id,
                "route": str(row["route"]),
                "domain": str(feature["domain"]),
                "label_group": label_group(feature),
                "multi_exposure": (
                    "multi_exposed" if has_multi(feature) else "non_multi_exposed"
                ),
                "local_reliable": local_reliable,
                "fallback_reliable": fallback_reliable,
                "mode_evidence_consistent": bool(
                    fallback_reliable
                    and _failure_mode_evidence_consistent(
                        fallback, manifest[dialog_id]
                    )
                ),
                "fallback_cause": fallback.get("primary_failure_cause"),
                "fallback_mode": fallback.get("causal_call_mode"),
                "fallback_role": fallback.get("multi_causal_role"),
                "exact_agreement": {
                    field: bool(
                        local_reliable
                        and fallback_reliable
                        and local.get(field) == fallback.get(field)
                    )
                    for field in FAILURE_UNKNOWN_VALUES
                },
            }
        )

    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        fallback_reliable = [
            record for record in subset if record["fallback_reliable"]
        ]
        return {
            "dialogs": len(subset),
            "fallback_reliable": len(fallback_reliable),
            "local_reliable": sum(record["local_reliable"] for record in subset),
            "mode_evidence_consistent": sum(
                record["mode_evidence_consistent"] for record in fallback_reliable
            ),
            "fallback_primary_cause_unknown": sum(
                record["fallback_cause"] == "unknown"
                for record in fallback_reliable
            ),
            "fallback_causal_mode_unknown": sum(
                record["fallback_mode"] == "unknown"
                for record in fallback_reliable
            ),
            "fallback_primary_causes": dict(
                sorted(Counter(record["fallback_cause"] for record in fallback_reliable).items())
            ),
            "fallback_causal_modes": dict(
                sorted(Counter(record["fallback_mode"] for record in fallback_reliable).items())
            ),
            "fallback_multi_roles": dict(
                sorted(Counter(record["fallback_role"] for record in fallback_reliable).items())
            ),
            "local_fallback_exact_agreement": {
                field: sum(record["exact_agreement"][field] for record in subset)
                for field in FAILURE_UNKNOWN_VALUES
            },
        }

    overall = aggregate(records)
    reliable = overall["fallback_reliable"]
    gates = {
        "declared_failure_calibration_size_72": (
            overall["dialogs"] == FAILURE_CALIBRATION_LIMIT
        ),
        "fallback_reliable_at_least_0.90": (
            reliable / overall["dialogs"] >= 0.90 if overall["dialogs"] else False
        ),
        "causal_mode_evidence_consistency_at_least_0.95": (
            overall["mode_evidence_consistent"] / reliable >= 0.95
            if reliable
            else False
        ),
        "fallback_primary_cause_unknown_at_most_0.30": (
            overall["fallback_primary_cause_unknown"] / reliable <= 0.30
            if reliable
            else False
        ),
        "fallback_causal_mode_unknown_at_most_0.30": (
            overall["fallback_causal_mode_unknown"] / reliable <= 0.30
            if reliable
            else False
        ),
    }

    margins = {}
    margin_gates = {}
    for dimension in ("route", "domain", "label_group", "multi_exposure"):
        margins[dimension] = {}
        for value in sorted({str(record[dimension]) for record in records}):
            subset = [
                record for record in records if str(record[dimension]) == value
            ]
            values = aggregate(subset)
            margins[dimension][value] = values
            gate_key = f"{dimension}={value}"
            if values["dialogs"] < 12:
                margin_gates[gate_key] = {
                    "gated": False,
                    "status": "not_gated_n_lt_12",
                    "pass": None,
                    "checks": {},
                }
                continue
            margin_reliable = values["fallback_reliable"]
            checks = {
                "fallback_reliable_at_least_0.85": (
                    margin_reliable / values["dialogs"] >= 0.85
                ),
                "mode_evidence_consistency_at_least_0.90": (
                    values["mode_evidence_consistent"] / margin_reliable >= 0.90
                    if margin_reliable
                    else False
                ),
            }
            margin_gates[gate_key] = {
                "gated": True,
                "status": "pass" if all(checks.values()) else "fail",
                "pass": all(checks.values()),
                "checks": checks,
            }
    gated_margins = [
        values["pass"] for values in margin_gates.values() if values["gated"]
    ]
    gates["all_n_ge_12_one_dimensional_margins_pass"] = bool(
        gated_margins
    ) and all(gated_margins)
    status = "pass" if all(gates.values()) else "fail"

    lines = [
        "# tau2 SFT failure-attribution calibration",
        "",
        "这是与 exact-consensus diagnostic pilot 不重叠的 72-dialog validation set。",
        "用户指定的 gpt-5.6-luna 是主归因器；Qwen 只用于 sensitivity/contradiction",
        "审计。gate 校验 Luna reliability、unknown 覆盖及 causal mode 与 sealed turn-type",
        "证据的一致性，不把模型归因误称为 ground truth。",
        "",
        "## Gate",
        "",
        f"Status：`{status}`。",
        "",
        "| gate | pass |",
        "|---|---|",
    ]
    for gate, passed in gates.items():
        lines.append(f"| `{gate}` | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- dialogs：{overall['dialogs']:,}",
            f"- Luna reliable：{reliable:,}/{overall['dialogs']:,}",
            f"- Luna causal-mode / decisive-turn consistency：{overall['mode_evidence_consistent']:,}/{reliable:,}",
            f"- Luna primary cause unknown：{overall['fallback_primary_cause_unknown']:,}/{reliable:,}",
            f"- Luna causal mode unknown：{overall['fallback_causal_mode_unknown']:,}/{reliable:,}",
            "- local/Luna exact agreement is diagnostic only："
            + ", ".join(
                f"{field} {count}/{overall['dialogs']}"
                for field, count in overall["local_fallback_exact_agreement"].items()
            ),
            "",
            "## One-dimensional margins",
            "",
            "只有 n >= 12 的 margin 进入 reliability/evidence-consistency gate。",
            "",
            "| margin | dialogs | Luna reliable | mode evidence-consistent | gate |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for dimension, by_value in margins.items():
        for value, values in by_value.items():
            margin_gate = margin_gates[f"{dimension}={value}"]
            lines.append(
                f"| `{dimension}={value}` | {values['dialogs']:,} | "
                f"{values['fallback_reliable']:,} | "
                f"{values['mode_evidence_consistent']:,} | "
                f"{margin_gate['status']} |"
            )
    lines.append("")
    return "\n".join(lines), {
        "status": status,
        "attribution_semantics": "luna_primary_qwen_sensitivity",
        "gates": gates,
        "margin_gates": margin_gates,
        "metrics": {"overall": overall, "by_margin": margins},
    }


def failure_report(
    features: list[dict[str, Any]],
    local_reviews: dict[str, dict[str, Any]],
    fallback_reviews: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    failures = [record for record in features if label_group(record) != "consensus_success"]
    if manifest is None:
        manifest = {
            str(record["dialog_id"]): {"route": "full/full"} for record in failures
        }
    causes_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    call_modes_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    call_modes_by_exposure: dict[str, Counter[str]] = defaultdict(Counter)
    causes_by_domain_exposure: dict[tuple[str, str], Counter[str]] = defaultdict(
        Counter
    )
    call_modes_by_domain_exposure: dict[
        tuple[str, str], Counter[str]
    ] = defaultdict(Counter)
    multi_roles = Counter()
    qwen_causes = Counter()
    qwen_call_modes = Counter()
    qwen_multi_roles = Counter()
    field_exact_denominators = Counter()
    field_exact_agreement = Counter()
    field_exact_resolved = Counter()
    field_exact_unknown = Counter()
    field_non_exact_or_unreliable = Counter()
    field_crosstabs: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    route_stats: dict[str, Counter[str]] = defaultdict(Counter)
    luna_reliable_reviews = 0
    qwen_reliable_reviews = 0
    both_reliable_reviews = 0
    luna_evidence_checked = 0
    qwen_evidence_checked = 0
    luna_evidence_consistent = 0
    qwen_evidence_consistent = 0
    luna_usable_reviews = 0
    qwen_usable_reviews = 0

    def attributed_value(
        review: dict[str, Any],
        field: str,
        *,
        usable: bool,
        multi_exposed: bool,
    ) -> str:
        if field == "multi_causal_role":
            if not multi_exposed:
                return "not_applicable"
            value = str(review.get(field) or "uncertain") if usable else "uncertain"
            return "uncertain" if value == "not_applicable" else value
        if not usable:
            return FAILURE_UNKNOWN_VALUES[field]
        return str(review.get(field) or FAILURE_UNKNOWN_VALUES[field])

    for feature in failures:
        dialog_id = str(feature["dialog_id"])
        local_record = local_reviews[dialog_id]
        fallback_record = fallback_reviews[dialog_id]
        local = _selected_review(local_record) or {}
        fallback = _selected_review(fallback_record) or {}
        local_reliable = bool(_review_reliability(local_record)["reliable"])
        fallback_reliable = bool(
            _review_reliability(fallback_record)["reliable"]
        )
        both_reliable = local_reliable and fallback_reliable
        manifest_row = manifest[dialog_id]
        route = str(manifest_row.get("route", "full/full"))
        evidence_available = isinstance(
            manifest_row.get("full_source_turn_types"), dict
        )
        local_consistent = bool(
            local_reliable
            and (
                not evidence_available
                or _failure_mode_evidence_consistent(local, manifest_row)
            )
        )
        fallback_consistent = bool(
            fallback_reliable
            and (
                not evidence_available
                or _failure_mode_evidence_consistent(fallback, manifest_row)
            )
        )
        local_usable = local_reliable and local_consistent
        fallback_usable = fallback_reliable and fallback_consistent
        luna_reliable_reviews += int(fallback_reliable)
        qwen_reliable_reviews += int(local_reliable)
        both_reliable_reviews += int(both_reliable)
        luna_evidence_checked += int(fallback_reliable and evidence_available)
        qwen_evidence_checked += int(local_reliable and evidence_available)
        luna_evidence_consistent += int(
            fallback_reliable and evidence_available and fallback_consistent
        )
        qwen_evidence_consistent += int(
            local_reliable and evidence_available and local_consistent
        )
        luna_usable_reviews += int(fallback_usable)
        qwen_usable_reviews += int(local_usable)
        route_stats[route]["dialogs"] += 1
        route_stats[route]["luna_reliable"] += int(fallback_reliable)
        route_stats[route]["qwen_reliable"] += int(local_reliable)
        route_stats[route]["luna_usable"] += int(fallback_usable)
        route_stats[route]["qwen_usable"] += int(local_usable)

        group = label_group(feature)
        exposure = "multi_exposed" if has_multi(feature) else "non_multi_exposed"
        multi_exposed = exposure == "multi_exposed"
        cause = attributed_value(
            fallback,
            "primary_failure_cause",
            usable=fallback_usable,
            multi_exposed=multi_exposed,
        )
        mode = attributed_value(
            fallback,
            "causal_call_mode",
            usable=fallback_usable,
            multi_exposed=multi_exposed,
        )
        role = attributed_value(
            fallback,
            "multi_causal_role",
            usable=fallback_usable,
            multi_exposed=multi_exposed,
        )
        qwen_cause = attributed_value(
            local,
            "primary_failure_cause",
            usable=local_usable,
            multi_exposed=multi_exposed,
        )
        qwen_mode = attributed_value(
            local,
            "causal_call_mode",
            usable=local_usable,
            multi_exposed=multi_exposed,
        )
        qwen_role = attributed_value(
            local,
            "multi_causal_role",
            usable=local_usable,
            multi_exposed=multi_exposed,
        )
        causes_by_label["all_non_consensus"][cause] += 1
        causes_by_label[group][cause] += 1
        call_modes_by_label["all_non_consensus"][mode] += 1
        call_modes_by_label[group][mode] += 1
        call_modes_by_exposure[exposure][mode] += 1
        domain_exposure = (str(feature["domain"]), exposure)
        causes_by_domain_exposure[domain_exposure][cause] += 1
        call_modes_by_domain_exposure[domain_exposure][mode] += 1
        qwen_causes[qwen_cause] += 1
        qwen_call_modes[qwen_mode] += 1
        if multi_exposed:
            multi_roles[role] += 1
            qwen_multi_roles[qwen_role] += 1

        for field in FAILURE_UNKNOWN_VALUES:
            if field == "multi_causal_role" and not multi_exposed:
                continue
            if both_reliable:
                field_exact_denominators[field] += 1
                local_value = str(local.get(field))
                fallback_value = str(fallback.get(field))
                field_crosstabs[field][(local_value, fallback_value)] += 1
                if local_value == fallback_value:
                    field_exact_agreement[field] += 1
                    if local_value == FAILURE_UNKNOWN_VALUES[field]:
                        field_exact_unknown[field] += 1
                    else:
                        field_exact_resolved[field] += 1
                    route_stats[route][f"exact_{field}"] += 1
            eligible = multi_exposed or field != "multi_causal_role"
            if eligible and not (
                both_reliable and local.get(field) == fallback.get(field)
            ):
                field_non_exact_or_unreliable[field] += 1

    label_counts = Counter(label_group(record) for record in failures)
    reward_failures = sum(failure_flags(record)["reward"] for record in features)
    correct_failures = sum(failure_flags(record)["correct"] for record in features)
    consensus_failures = label_counts["consensus_failure"]
    lines = [
        "# tau2 SFT 失败归因",
        "",
        "本报告以 `label_group != consensus_success` 的 full-source dialog 为归因分母；",
        "reward、correct 与 consensus failure 另行报告。用户指定的 gpt-5.6-luna",
        "给出主归因，Qwen 仅用于独立敏感性检查。Luna 低置信度或 causal mode",
        "与 sealed turn type 不一致时计入 unknown，不从分母移除；模型归因不是",
        "ground truth。",
        "",
        "## 标签口径与覆盖",
        "",
        f"- 归因池（任一标签非成功）：{len(failures):,}",
        f"- reward failure：{reward_failures:,}",
        f"- correct failure：{correct_failures:,}",
        f"- consensus failure：{consensus_failures:,}",
        f"- discordant：reward-only {label_counts['reward_only']:,}；correct-only {label_counts['correct_only']:,}",
        f"- Luna reliable：{luna_reliable_reviews:,}/{len(failures):,}；evidence-usable：{luna_usable_reviews:,}/{len(failures):,}",
        f"- Qwen reliable：{qwen_reliable_reviews:,}/{len(failures):,}；evidence-usable：{qwen_usable_reviews:,}/{len(failures):,}",
        "",
        "## Qwen sensitivity audit",
        "",
        "Exact agreement 只作敏感性诊断，不决定 Luna 主归因。",
        "",
        "| field | both reliable denominator | exact | exact rate |",
        "|---|---:|---:|---:|",
    ]
    for field in ("primary_failure_cause", "causal_call_mode", "multi_causal_role"):
        lines.append(
            f"| `{field}` | {field_exact_denominators[field]:,} | "
            f"{field_exact_agreement[field]:,} | "
            f"{_percent(field_exact_agreement[field], field_exact_denominators[field])} |"
        )
    lines.extend(
        [
            "",
            "| route | dialogs | Luna usable | Qwen usable | exact cause | exact mode |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for route, values in sorted(route_stats.items()):
        lines.append(
            f"| {route} | {values['dialogs']:,} | {values['luna_usable']:,} | "
            f"{values['qwen_usable']:,} | "
            f"{values['exact_primary_failure_cause']:,} | "
            f"{values['exact_causal_call_mode']:,} |"
        )
    lines.extend(
        [
        "",
        "## Primary cause by label stratum",
        "",
        "| cause | all non-consensus | consensus failure | reward-only | correct-only |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    all_causes = sorted(
        set().union(*(counter.keys() for counter in causes_by_label.values())),
        key=lambda cause: (-causes_by_label["all_non_consensus"][cause], cause),
    )
    for cause in all_causes:
        lines.append(
            f"| `{cause}` | {causes_by_label['all_non_consensus'][cause]:,} | "
            f"{causes_by_label['consensus_failure'][cause]:,} | "
            f"{causes_by_label['reward_only'][cause]:,} | "
            f"{causes_by_label['correct_only'][cause]:,} |"
        )
    lines.extend(
        [
            "",
            "## Decisive call mode by trajectory exposure",
            "",
            "这里区分“轨迹含 multi”与“决定性错误发生在 multi turn”；两者不能互换。",
            "",
            "| exposure | failures | multi-turn cause | single-call cause | non-tool | env/user | unknown |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for exposure in ("multi_exposed", "non_multi_exposed"):
        counter = call_modes_by_exposure[exposure]
        total = sum(counter.values())
        lines.append(
            f"| {exposure} | {total:,} | {_percent(counter['multi_turn'], total)} | "
            f"{_percent(counter['single_call_turn'], total)} | "
            f"{_percent(counter['non_tool'], total)} | "
            f"{_percent(counter['environment_or_user'], total)} | "
            f"{_percent(counter['unknown'], total)} |"
        )
    lines.extend(
        [
            "",
            "| domain | exposure | failures | top primary cause | top-cause share | multi-turn cause | single-call cause | non-tool | env/user | unknown |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for domain in ("airline", "retail", "telecom"):
        for exposure in ("multi_exposed", "non_multi_exposed"):
            key = (domain, exposure)
            counter = call_modes_by_domain_exposure[key]
            causes = causes_by_domain_exposure[key]
            total = sum(counter.values())
            top_cause, top_count = (
                causes.most_common(1)[0] if causes else ("n/a", 0)
            )
            lines.append(
                f"| {domain} | {exposure} | {total:,} | `{top_cause}` | "
                f"{_percent(top_count, total)} | "
                f"{_percent(counter['multi_turn'], total)} | "
                f"{_percent(counter['single_call_turn'], total)} | "
                f"{_percent(counter['non_tool'], total)} | "
                f"{_percent(counter['environment_or_user'], total)} | "
                f"{_percent(counter['unknown'], total)} |"
            )
    multi_failures = sum(has_multi(record) for record in failures)
    harmful_multi = multi_roles["caused"] + multi_roles["contributed"]
    lines.extend(
        [
            "",
            "## Multi causal role",
            "",
            f"Full-source multi non-consensus trajectories：{multi_failures:,}/{len(failures):,}。",
            f"其中 Luna 归因为 caused/contributed：{harmful_multi:,}/{multi_failures:,}。",
            "这回答的是失败轨迹内部的归因构成，不等于 multi 相对 single 的失败风险；",
            "失败风险应看确定性、按 domain/label 分层或匹配后的比较。",
            "",
            "| role | dialogs | rate over multi-exposed failures |",
            "|---|---:|---:|",
        ]
    )
    for role, count in sorted(multi_roles.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{role}` | {count:,} | {_percent(count, multi_failures)} |")
    lines.append("")
    crosstabs = {
        field: {
            f"{local_value} -> {fallback_value}": count
            for (local_value, fallback_value), count in sorted(counter.items())
        }
        for field, counter in sorted(field_crosstabs.items())
    }
    route_summary = {
        route: dict(sorted(values.items()))
        for route, values in sorted(route_stats.items())
    }
    summary = {
        "attribution_semantics": "luna_primary_qwen_sensitivity",
        "attribution_pool": len(failures),
        "reward_failures": reward_failures,
        "correct_failures": correct_failures,
        "consensus_failures": consensus_failures,
        "label_groups": dict(sorted(label_counts.items())),
        "luna_primary": {
            "reliable_reviews": luna_reliable_reviews,
            "mode_evidence_checked": luna_evidence_checked,
            "mode_evidence_consistent": luna_evidence_consistent,
            "usable_reviews": luna_usable_reviews,
        },
        "qwen_sensitivity": {
            "reliable_reviews": qwen_reliable_reviews,
            "mode_evidence_checked": qwen_evidence_checked,
            "mode_evidence_consistent": qwen_evidence_consistent,
            "usable_reviews": qwen_usable_reviews,
            "primary_failure_causes": dict(sorted(qwen_causes.items())),
            "causal_call_modes": dict(sorted(qwen_call_modes.items())),
            "multi_causal_roles": dict(sorted(qwen_multi_roles.items())),
            "field_crosstabs_qwen_to_luna": crosstabs,
        },
        "both_reliable_reviews": both_reliable_reviews,
        # Compatibility keys now describe the Qwen/Luna sensitivity audit;
        # Luna's attributed distributions below are the primary result.
        "field_agreement": dict(sorted(field_exact_agreement.items())),
        "field_exact_agreement": dict(sorted(field_exact_agreement.items())),
        "field_exact_agreement_denominators": dict(
            sorted(field_exact_denominators.items())
        ),
        "field_agreed_resolved": dict(sorted(field_exact_resolved.items())),
        "field_agreed_unknown": dict(sorted(field_exact_unknown.items())),
        "field_disagreement": dict(
            sorted(field_non_exact_or_unreliable.items())
        ),
        "unreliable_judge_pairs": len(failures) - both_reliable_reviews,
        "judge_coverage_by_route": route_summary,
        "primary_failure_causes_by_label": {
            label: dict(sorted(counter.items()))
            for label, counter in sorted(causes_by_label.items())
        },
        "causal_call_modes_by_label": {
            label: dict(sorted(counter.items()))
            for label, counter in sorted(call_modes_by_label.items())
        },
        "causal_call_modes_by_exposure": {
            exposure: dict(sorted(counter.items()))
            for exposure, counter in sorted(call_modes_by_exposure.items())
        },
        "primary_failure_causes_by_domain_exposure": {
            f"{domain}/{exposure}": dict(sorted(counter.items()))
            for (domain, exposure), counter in sorted(
                causes_by_domain_exposure.items()
            )
        },
        "causal_call_modes_by_domain_exposure": {
            f"{domain}/{exposure}": dict(sorted(counter.items()))
            for (domain, exposure), counter in sorted(
                call_modes_by_domain_exposure.items()
            )
        },
        "multi_causal_roles": dict(sorted(multi_roles.items())),
        "multi_causal_role_denominator": multi_failures,
        "multi_harmful_role": {
            "definition": "caused_or_contributed",
            "dialogs": harmful_multi,
            "denominator": multi_failures,
        },
    }
    return "\n".join(lines), summary


def calibrate(args: argparse.Namespace) -> None:
    """Validate and score the declared calibration subset before full review."""

    features = list(read_jsonl(args.features))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    artifact_root = args.features.parent
    prepare_manifest_path = (
        args.prepare_manifest or artifact_root / "prepare_manifest.json"
    )
    manifest_path = args.payload_manifest or artifact_root / "payload_manifest.jsonl"
    selection_path = args.selection or artifact_root / "comparison_selection.jsonl"
    calibration_path = (
        args.calibration or artifact_root / "calibration_selection.jsonl"
    )
    calibration_pilot_path = artifact_root / "calibration_pilot_selection.jsonl"
    calibration_v1_path = (
        artifact_root / "calibration_validation_v1_selection.jsonl"
    )
    prepare_manifest = validate_prepare_manifest(prepare_manifest_path, artifact_root)
    validate_selected_artifacts(
        prepare_manifest,
        {
            "features.jsonl": args.features,
            "inventory.json": args.inventory,
            "comparison_selection.jsonl": selection_path,
            "calibration_pilot_selection.jsonl": calibration_pilot_path,
            "calibration_validation_v1_selection.jsonl": calibration_v1_path,
            "calibration_selection.jsonl": calibration_path,
            "payload_manifest.jsonl": manifest_path,
        },
    )
    manifest_rows = list(read_jsonl(manifest_path))
    if stable_json_hash(manifest_rows) != inventory.get("payload_manifest_sha256"):
        raise ValueError(f"{manifest_path}: manifest hash does not match inventory")
    quality_manifest = _manifest_map(manifest_path, "quality")
    selection = list(read_jsonl(selection_path))
    calibration_pilot = list(read_jsonl(calibration_pilot_path))
    calibration_v1 = list(read_jsonl(calibration_v1_path))
    calibration = list(read_jsonl(calibration_path))
    expected_pilot = build_calibration_selection(
        features, selection, list(quality_manifest.values())
    )
    expected_calibration = build_quality_validation_selection(
        features, selection, list(quality_manifest.values())
    )
    expected_v1 = build_quality_v1_selection(
        features, selection, list(quality_manifest.values())
    )
    if calibration_pilot != expected_pilot:
        raise ValueError("calibration pilot is stale or was not deterministically prepared")
    if calibration_v1 != expected_v1:
        raise ValueError("calibration v1 is stale or was not deterministically prepared")
    if calibration != expected_calibration:
        raise ValueError("calibration selection is stale or was not deterministically prepared")
    pilot_ids = {str(row["dialog_id"]) for row in calibration_pilot}
    calibration_v1_ids = {str(row["dialog_id"]) for row in calibration_v1}
    calibration_ids = {str(row["dialog_id"]) for row in calibration}
    if (
        len(calibration_ids) != len(calibration)
        or calibration_ids - set(quality_manifest)
        or calibration_v1_ids - set(quality_manifest)
        or pilot_ids & calibration_v1_ids
        or pilot_ids & calibration_ids
        or calibration_v1_ids & calibration_ids
        or len(calibration) != inventory.get("calibration_dialogs")
        or len(calibration_pilot) != inventory.get("calibration_pilot_dialogs")
        or len(calibration_v1)
        != inventory.get("calibration_validation_v1_dialogs")
        or len(calibration) != QUALITY_CALIBRATION_LIMIT
        or len(calibration_pilot) != QUALITY_CALIBRATION_LIMIT
        or len(calibration_v1) != QUALITY_CALIBRATION_LIMIT
        or inventory.get("quality_calibration_pilot_seed")
        != QUALITY_CALIBRATION_PILOT_SEED
        or inventory.get("quality_calibration_v1_seed")
        != QUALITY_CALIBRATION_V1_SEED
        or inventory.get("quality_calibration_validation_seed")
        != QUALITY_CALIBRATION_VALIDATION_SEED
        or inventory.get("quality_calibration_version")
        != QUALITY_CALIBRATION_VERSION
        or stable_json_hash(calibration_pilot)
        != inventory.get("quality_calibration_pilot_selection_sha256")
        or stable_json_hash(calibration_v1)
        != inventory.get("quality_calibration_v1_selection_sha256")
        or stable_json_hash(calibration)
        != inventory.get("quality_calibration_selection_sha256")
    ):
        raise ValueError("calibration selection is duplicate, stale, or incomplete")
    calibration_manifest = {
        dialog_id: quality_manifest[dialog_id] for dialog_id in calibration_ids
    }
    local_reviews = _validated_review_map(
        args.quality_local_reviews,
        calibration_manifest,
        review_mode="quality",
        side="local",
        allow_additional=True,
    )
    fallback_reviews = _validated_review_map(
        args.quality_fallback_reviews,
        calibration_manifest,
        review_mode="quality",
        side="fallback",
        allow_additional=True,
    )
    _validate_independent_reviews("quality calibration", local_reviews, fallback_reviews)
    markdown, summary = calibration_report(
        calibration, features, local_reviews, fallback_reviews
    )
    summary.update(
        {
            "protocol_profile": inventory.get("protocol_profile"),
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "payload_manifest_sha256": inventory.get("payload_manifest_sha256"),
            "calibration_selection_sha256": stable_json_hash(calibration),
            "selection_seed": QUALITY_CALIBRATION_VALIDATION_SEED,
            "selection_version": QUALITY_CALIBRATION_VERSION,
            "pilot_selection_sha256": stable_json_hash(calibration_pilot),
            "validation_v1_selection_sha256": stable_json_hash(calibration_v1),
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "CALIBRATION_REPORT.md").write_text(markdown, encoding="utf-8")
    write_json(args.out / "CALIBRATION_REPORT.json", summary)
    write_json(args.out / "calibration_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_pass and summary["status"] != "pass":
        raise SystemExit("calibration gate failed; full review is blocked")


def calibrate_failure(args: argparse.Namespace) -> None:
    """Validate the predeclared failure-attribution calibration subset."""

    features = list(read_jsonl(args.features))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    artifact_root = args.features.parent
    prepare_manifest_path = (
        args.prepare_manifest or artifact_root / "prepare_manifest.json"
    )
    manifest_path = args.payload_manifest or artifact_root / "payload_manifest.jsonl"
    calibration_path = (
        args.failure_calibration
        or artifact_root / "failure_calibration_selection.jsonl"
    )
    calibration_pilot_path = (
        artifact_root / "failure_calibration_pilot_selection.jsonl"
    )
    prepare_manifest = validate_prepare_manifest(prepare_manifest_path, artifact_root)
    validate_selected_artifacts(
        prepare_manifest,
        {
            "features.jsonl": args.features,
            "inventory.json": args.inventory,
            "failure_calibration_pilot_selection.jsonl": (
                calibration_pilot_path
            ),
            "failure_calibration_selection.jsonl": calibration_path,
            "payload_manifest.jsonl": manifest_path,
        },
    )
    manifest_rows = list(read_jsonl(manifest_path))
    if stable_json_hash(manifest_rows) != inventory.get("payload_manifest_sha256"):
        raise ValueError(f"{manifest_path}: manifest hash does not match inventory")
    failure_manifest = _manifest_map(manifest_path, "failure")
    calibration_pilot = list(read_jsonl(calibration_pilot_path))
    calibration = list(read_jsonl(calibration_path))
    expected_pilot = build_failure_calibration_pilot_selection(
        features, list(failure_manifest.values())
    )
    expected_calibration = build_failure_calibration_selection(
        features, list(failure_manifest.values())
    )
    if calibration_pilot != expected_pilot:
        raise ValueError(
            "failure calibration pilot is stale or was not deterministically prepared"
        )
    if calibration != expected_calibration:
        raise ValueError(
            "failure calibration selection is stale or was not deterministically prepared"
        )
    pilot_ids = {str(row["dialog_id"]) for row in calibration_pilot}
    calibration_ids = {str(row["dialog_id"]) for row in calibration}
    if (
        len(calibration_ids) != len(calibration)
        or calibration_ids - set(failure_manifest)
        or pilot_ids - set(failure_manifest)
        or pilot_ids & calibration_ids
        or len(calibration_pilot)
        != inventory.get("failure_calibration_pilot_dialogs")
        or len(calibration) != inventory.get("failure_calibration_dialogs")
        or len(calibration_pilot) != FAILURE_CALIBRATION_LIMIT
        or len(calibration) != FAILURE_CALIBRATION_LIMIT
        or inventory.get("failure_calibration_pilot_seed")
        != FAILURE_CALIBRATION_PILOT_SEED
        or inventory.get("failure_calibration_pilot_version")
        != FAILURE_CALIBRATION_PILOT_VERSION
        or inventory.get("failure_calibration_seed")
        != FAILURE_CALIBRATION_SEED
        or inventory.get("failure_calibration_version")
        != FAILURE_CALIBRATION_VERSION
        or stable_json_hash(calibration)
        != inventory.get("failure_calibration_selection_sha256")
        or stable_json_hash(calibration_pilot)
        != inventory.get("failure_calibration_pilot_selection_sha256")
    ):
        raise ValueError(
            "failure calibration selection is duplicate, stale, or incomplete"
        )
    calibration_manifest = {
        dialog_id: failure_manifest[dialog_id] for dialog_id in calibration_ids
    }
    local_reviews = _validated_review_map(
        args.failure_local_reviews,
        calibration_manifest,
        review_mode="failure",
        side="local",
        allow_additional=True,
    )
    fallback_reviews = _validated_review_map(
        args.failure_fallback_reviews,
        calibration_manifest,
        review_mode="failure",
        side="fallback",
        allow_additional=True,
    )
    _validate_independent_reviews(
        "failure calibration", local_reviews, fallback_reviews
    )
    markdown, summary = failure_calibration_report(
        calibration,
        features,
        local_reviews,
        fallback_reviews,
        calibration_manifest,
    )
    summary.update(
        {
            "protocol_profile": inventory.get("protocol_profile"),
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "payload_manifest_sha256": inventory.get("payload_manifest_sha256"),
            "selection_seed": FAILURE_CALIBRATION_SEED,
            "selection_version": FAILURE_CALIBRATION_VERSION,
            "pilot_selection_sha256": stable_json_hash(calibration_pilot),
            "failure_calibration_selection_sha256": stable_json_hash(calibration),
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "FAILURE_CALIBRATION_REPORT.md").write_text(
        markdown, encoding="utf-8"
    )
    write_json(args.out / "FAILURE_CALIBRATION_REPORT.json", summary)
    write_json(args.out / "failure_calibration_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_pass and summary["status"] != "pass":
        raise SystemExit(
            "failure calibration gate failed; full failure review is blocked"
        )


def summarize(args: argparse.Namespace) -> None:
    features = list(read_jsonl(args.features))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    selection_path = args.selection or args.features.parent / "comparison_selection.jsonl"
    pairs_path = args.pairs or args.features.parent / "comparison_pairs.jsonl"
    manifest_path = args.payload_manifest or args.features.parent / "payload_manifest.jsonl"
    calibration_path = (
        args.calibration or args.features.parent / "calibration_selection.jsonl"
    )
    calibration_pilot_path = (
        args.features.parent / "calibration_pilot_selection.jsonl"
    )
    calibration_v1_path = (
        args.features.parent / "calibration_validation_v1_selection.jsonl"
    )
    failure_calibration_path = (
        getattr(args, "failure_calibration", None)
        or args.features.parent / "failure_calibration_selection.jsonl"
    )
    failure_calibration_pilot_path = (
        args.features.parent / "failure_calibration_pilot_selection.jsonl"
    )
    selection = list(read_jsonl(selection_path))
    pairs = list(read_jsonl(pairs_path))
    calibration_pilot = list(read_jsonl(calibration_pilot_path))
    calibration_v1 = list(read_jsonl(calibration_v1_path))
    calibration = list(read_jsonl(calibration_path))
    manifest_rows = list(read_jsonl(manifest_path))
    if stable_json_hash(manifest_rows) != inventory.get("payload_manifest_sha256"):
        raise ValueError(f"{manifest_path}: manifest hash does not match inventory")
    failure_calibration_pilot = list(
        read_jsonl(failure_calibration_pilot_path)
    )
    failure_calibration = list(read_jsonl(failure_calibration_path))
    prepare_manifest_path = (
        getattr(args, "prepare_manifest", None)
        or args.features.parent / "prepare_manifest.json"
    )
    prepare_manifest = validate_prepare_manifest(
        prepare_manifest_path, args.features.parent
    )
    validate_selected_artifacts(
        prepare_manifest,
        {
            "features.jsonl": args.features,
            "inventory.json": args.inventory,
            "comparison_selection.jsonl": selection_path,
            "comparison_pairs.jsonl": pairs_path,
            "calibration_pilot_selection.jsonl": calibration_pilot_path,
            "calibration_validation_v1_selection.jsonl": calibration_v1_path,
            "calibration_selection.jsonl": calibration_path,
            "failure_calibration_pilot_selection.jsonl": (
                failure_calibration_pilot_path
            ),
            "failure_calibration_selection.jsonl": failure_calibration_path,
            "payload_manifest.jsonl": manifest_path,
        },
    )
    quality_manifest = _manifest_map(manifest_path, "quality")
    failure_manifest = _manifest_map(manifest_path, "failure")
    selected_ids = {str(item["dialog_id"]) for item in selection}
    expected_failure_ids = {
        str(record["dialog_id"])
        for record in features
        if label_group(record) != "consensus_success"
    }
    if set(quality_manifest) != selected_ids:
        raise ValueError("quality payload manifest does not match comparison selection")
    if set(failure_manifest) != expected_failure_ids:
        raise ValueError("failure payload manifest does not match non-consensus label pool")
    expected_pilot = build_calibration_selection(
        features, selection, list(quality_manifest.values())
    )
    expected_calibration = build_quality_validation_selection(
        features, selection, list(quality_manifest.values())
    )
    expected_v1 = build_quality_v1_selection(
        features, selection, list(quality_manifest.values())
    )
    if calibration_pilot != expected_pilot:
        raise ValueError(
            "calibration pilot was not deterministically prepared"
        )
    if calibration_v1 != expected_v1:
        raise ValueError(
            "calibration validation v1 was not deterministically prepared"
        )
    if calibration != expected_calibration:
        raise ValueError("calibration selection was not deterministically prepared")
    calibration_pilot_ids = [
        str(row["dialog_id"]) for row in calibration_pilot
    ]
    calibration_v1_ids = [
        str(row["dialog_id"]) for row in calibration_v1
    ]
    calibration_ids = [str(row["dialog_id"]) for row in calibration]
    if (
        len(calibration_pilot_ids) != len(set(calibration_pilot_ids))
        or len(calibration_v1_ids) != len(set(calibration_v1_ids))
        or len(calibration_ids) != len(set(calibration_ids))
        or set(calibration_pilot_ids) & set(calibration_v1_ids)
        or set(calibration_pilot_ids) & set(calibration_ids)
        or set(calibration_v1_ids) & set(calibration_ids)
        or not set(calibration_pilot_ids) <= set(quality_manifest)
        or not set(calibration_v1_ids) <= set(quality_manifest)
        or not set(calibration_ids) <= set(quality_manifest)
        or len(calibration_pilot_ids)
        != inventory.get("calibration_pilot_dialogs")
        or len(calibration_ids) != inventory.get("calibration_dialogs")
        or len(calibration_v1_ids)
        != inventory.get("calibration_validation_v1_dialogs")
        or len(calibration_pilot_ids) != QUALITY_CALIBRATION_LIMIT
        or len(calibration_v1_ids) != QUALITY_CALIBRATION_LIMIT
        or len(calibration_ids) != QUALITY_CALIBRATION_LIMIT
        or inventory.get("quality_calibration_pilot_seed")
        != QUALITY_CALIBRATION_PILOT_SEED
        or inventory.get("quality_calibration_validation_seed")
        != QUALITY_CALIBRATION_VALIDATION_SEED
        or inventory.get("quality_calibration_v1_seed")
        != QUALITY_CALIBRATION_V1_SEED
        or inventory.get("quality_calibration_version")
        != QUALITY_CALIBRATION_VERSION
        or stable_json_hash(calibration_pilot)
        != inventory.get("quality_calibration_pilot_selection_sha256")
        or stable_json_hash(calibration_v1)
        != inventory.get("quality_calibration_v1_selection_sha256")
        or stable_json_hash(calibration)
        != inventory.get("quality_calibration_selection_sha256")
    ):
        raise ValueError("calibration selection is duplicate, stale, or incomplete")
    expected_failure_calibration_pilot = (
        build_failure_calibration_pilot_selection(
            features, list(failure_manifest.values())
        )
    )
    expected_failure_calibration = build_failure_calibration_selection(
        features, list(failure_manifest.values())
    )
    if failure_calibration_pilot != expected_failure_calibration_pilot:
        raise ValueError(
            "failure calibration pilot was not deterministically prepared"
        )
    if failure_calibration != expected_failure_calibration:
        raise ValueError(
            "failure calibration selection was not deterministically prepared"
        )
    failure_calibration_pilot_ids = [
        str(row["dialog_id"]) for row in failure_calibration_pilot
    ]
    failure_calibration_ids = [
        str(row["dialog_id"]) for row in failure_calibration
    ]
    if (
        len(failure_calibration_pilot_ids)
        != len(set(failure_calibration_pilot_ids))
        or len(failure_calibration_ids) != len(set(failure_calibration_ids))
        or set(failure_calibration_pilot_ids) & set(failure_calibration_ids)
        or not set(failure_calibration_pilot_ids) <= set(failure_manifest)
        or not set(failure_calibration_ids) <= set(failure_manifest)
        or len(failure_calibration_pilot_ids)
        != inventory.get("failure_calibration_pilot_dialogs")
        or len(failure_calibration_ids)
        != inventory.get("failure_calibration_dialogs")
        or len(failure_calibration_pilot_ids) != FAILURE_CALIBRATION_LIMIT
        or len(failure_calibration_ids) != FAILURE_CALIBRATION_LIMIT
        or inventory.get("failure_calibration_pilot_seed")
        != FAILURE_CALIBRATION_PILOT_SEED
        or inventory.get("failure_calibration_pilot_version")
        != FAILURE_CALIBRATION_PILOT_VERSION
        or inventory.get("failure_calibration_seed")
        != FAILURE_CALIBRATION_SEED
        or inventory.get("failure_calibration_version")
        != FAILURE_CALIBRATION_VERSION
        or stable_json_hash(failure_calibration)
        != inventory.get("failure_calibration_selection_sha256")
        or stable_json_hash(failure_calibration_pilot)
        != inventory.get("failure_calibration_pilot_selection_sha256")
    ):
        raise ValueError(
            "failure calibration selection is duplicate, stale, or incomplete"
        )
    quality_local = _validated_review_map(
        args.quality_local_reviews,
        quality_manifest,
        review_mode="quality",
        side="local",
    )
    quality_fallback = _validated_review_map(
        args.quality_fallback_reviews,
        quality_manifest,
        review_mode="quality",
        side="fallback",
    )
    _validate_independent_reviews("quality", quality_local, quality_fallback)
    calibration_markdown, calibration_summary = calibration_report(
        calibration, features, quality_local, quality_fallback
    )
    calibration_summary.update(
        {
            "protocol_profile": inventory.get("protocol_profile"),
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "payload_manifest_sha256": inventory.get("payload_manifest_sha256"),
            "calibration_selection_sha256": stable_json_hash(calibration),
            "selection_seed": QUALITY_CALIBRATION_VALIDATION_SEED,
            "selection_version": QUALITY_CALIBRATION_VERSION,
            "pilot_selection_sha256": stable_json_hash(calibration_pilot),
            "validation_v1_selection_sha256": stable_json_hash(calibration_v1),
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "CALIBRATION_REPORT.md").write_text(
        calibration_markdown, encoding="utf-8"
    )
    write_json(args.out / "CALIBRATION_REPORT.json", calibration_summary)
    write_json(args.out / "calibration_report.json", calibration_summary)
    if calibration_summary["status"] != "pass":
        raise SystemExit("calibration gate failed; summary generation is blocked")

    failure_local = _validated_review_map(
        args.failure_local_reviews,
        failure_manifest,
        review_mode="failure",
        side="local",
    )
    failure_fallback = _validated_review_map(
        args.failure_fallback_reviews,
        failure_manifest,
        review_mode="failure",
        side="fallback",
    )
    _validate_independent_reviews("failure", failure_local, failure_fallback)

    failure_calibration_markdown, failure_calibration_summary = (
        failure_calibration_report(
            failure_calibration,
            features,
            failure_local,
            failure_fallback,
            {
                dialog_id: failure_manifest[dialog_id]
                for dialog_id in failure_calibration_ids
            },
        )
    )
    failure_calibration_summary.update(
        {
            "protocol_profile": inventory.get("protocol_profile"),
            "prompt_version": MULTITOOL_PROMPT_VERSION,
            "payload_manifest_sha256": inventory.get("payload_manifest_sha256"),
            "selection_seed": FAILURE_CALIBRATION_SEED,
            "selection_version": FAILURE_CALIBRATION_VERSION,
            "pilot_selection_sha256": stable_json_hash(
                failure_calibration_pilot
            ),
            "failure_calibration_selection_sha256": stable_json_hash(
                failure_calibration
            ),
        }
    )
    (args.out / "FAILURE_CALIBRATION_REPORT.md").write_text(
        failure_calibration_markdown, encoding="utf-8"
    )
    write_json(
        args.out / "FAILURE_CALIBRATION_REPORT.json",
        failure_calibration_summary,
    )
    write_json(
        args.out / "failure_calibration_report.json",
        failure_calibration_summary,
    )
    if failure_calibration_summary["status"] != "pass":
        raise SystemExit(
            "failure calibration gate failed; summary generation is blocked"
        )

    failure_markdown, failure_summary = failure_report(
        features, failure_local, failure_fallback, failure_manifest
    )
    quality_markdown, quality_summary = quality_report(
        features,
        selection,
        pairs,
        quality_local,
        quality_fallback,
        quality_manifest,
    )
    decision_rows = filter_decision_rows(
        features, quality_local, quality_fallback, quality_manifest
    )
    decision_path = args.out / "filter_decisions_draft.jsonl"
    write_jsonl(decision_path, decision_rows)
    keep_candidate_dialogs = {
        row["dialog_id"]
        for row in decision_rows
        if row["dialog_keep_candidate"]
    }
    decision_summary = {
        "artifact": decision_path.name,
        "artifact_kind": "provenance_only_filter_decision",
        "contains_training_messages": False,
        "sha256": file_sha256(decision_path),
        "dialogs": len(features),
        "reviewed_dialogs": len(quality_manifest),
        "keep_candidate_dialogs": len(keep_candidate_dialogs),
        "target_rows": len(decision_rows),
        "keep_candidate_targets": sum(
            row["keep_candidate"] for row in decision_rows
        ),
        "unreviewed_targets": sum(
            not row["dialog_reviewed"] for row in decision_rows
        ),
    }
    (args.out / "FAILURE_ATTRIBUTION.md").write_text(failure_markdown, encoding="utf-8")
    (args.out / "COMPARISON_REPORT.md").write_text(
        comparison_report(features, selection) + "\n" + quality_markdown,
        encoding="utf-8",
    )
    summary = {
        "protocol_profile": inventory.get("protocol_profile"),
        "prompt_version": MULTITOOL_PROMPT_VERSION,
        "payload_manifest_sha256": inventory.get("payload_manifest_sha256"),
        "quality_comparison": quality_summary,
        "calibration": calibration_summary,
        "failure_calibration": failure_calibration_summary,
        "failure_attribution": failure_summary,
        "filter_decisions_draft": decision_summary,
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build protocol-neutral comparison inputs")
    prepare_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare_parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    prepare_parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    prepare_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    prepare_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    prepare_parser.add_argument(
        "--protocol-profile",
        choices=(PROTOCOL_DEPENDENCY_SAFE_MULTI,),
        default=PROTOCOL_DEPENDENCY_SAFE_MULTI,
    )
    prepare_parser.add_argument("--local-context-limit", type=int, default=32768)
    prepare_parser.add_argument("--judge-max-tokens", type=int, default=2048)
    prepare_parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path(__file__).resolve().parent / "run_multitool_quality_audit.sh",
        help="actual submitted runner copy used to generate this prepare manifest",
    )
    prepare_parser.set_defaults(func=prepare)

    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Validate the declared dual-judge calibration subset"
    )
    calibrate_parser.add_argument(
        "--features", type=Path, default=DEFAULT_OUT / "features.jsonl"
    )
    calibrate_parser.add_argument(
        "--inventory", type=Path, default=DEFAULT_OUT / "inventory.json"
    )
    calibrate_parser.add_argument("--prepare-manifest", type=Path)
    calibrate_parser.add_argument("--selection", type=Path)
    calibrate_parser.add_argument("--payload-manifest", type=Path)
    calibrate_parser.add_argument("--calibration", type=Path)
    calibrate_parser.add_argument("--quality-local-reviews", type=Path, required=True)
    calibrate_parser.add_argument(
        "--quality-fallback-reviews", type=Path, required=True
    )
    calibrate_parser.add_argument("--require-pass", action="store_true")
    calibrate_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    calibrate_parser.set_defaults(func=calibrate)

    failure_calibrate_parser = subparsers.add_parser(
        "calibrate-failure",
        help="Validate the declared failure-attribution calibration subset",
    )
    failure_calibrate_parser.add_argument(
        "--features", type=Path, default=DEFAULT_OUT / "features.jsonl"
    )
    failure_calibrate_parser.add_argument(
        "--inventory", type=Path, default=DEFAULT_OUT / "inventory.json"
    )
    failure_calibrate_parser.add_argument("--prepare-manifest", type=Path)
    failure_calibrate_parser.add_argument("--payload-manifest", type=Path)
    failure_calibrate_parser.add_argument("--failure-calibration", type=Path)
    failure_calibrate_parser.add_argument(
        "--failure-local-reviews", type=Path, required=True
    )
    failure_calibrate_parser.add_argument(
        "--failure-fallback-reviews", type=Path, required=True
    )
    failure_calibrate_parser.add_argument("--require-pass", action="store_true")
    failure_calibrate_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    failure_calibrate_parser.set_defaults(func=calibrate_failure)

    summarize_parser = subparsers.add_parser("summarize", help="Merge independent reviews")
    summarize_parser.add_argument("--features", type=Path, default=DEFAULT_OUT / "features.jsonl")
    summarize_parser.add_argument("--inventory", type=Path, default=DEFAULT_OUT / "inventory.json")
    summarize_parser.add_argument("--prepare-manifest", type=Path)
    summarize_parser.add_argument("--selection", type=Path)
    summarize_parser.add_argument("--pairs", type=Path)
    summarize_parser.add_argument("--payload-manifest", type=Path)
    summarize_parser.add_argument("--calibration", type=Path)
    summarize_parser.add_argument("--failure-calibration", type=Path)
    summarize_parser.add_argument("--quality-local-reviews", type=Path, required=True)
    summarize_parser.add_argument("--quality-fallback-reviews", type=Path, required=True)
    summarize_parser.add_argument("--failure-local-reviews", type=Path, required=True)
    summarize_parser.add_argument("--failure-fallback-reviews", type=Path, required=True)
    summarize_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    summarize_parser.set_defaults(func=summarize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
