#!/usr/bin/env python3
"""Summarize strict-single tau2 credit-assignment training artifacts.

Trajectory JSONL files contain every generated candidate, including groups
later replaced by the dynamic filter.  Run-log metrics describe the groups
that actually reached an update.  The output keeps those two views separate.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(
    r"\b(?P<kind>perf|rollout|step) (?P<step>\d+): "
    r"(?P<payload>\{.*\})\s*$"
)
RULE_ERROR_KEYS = (
    "wrong_namespace_tool",
    "nonexistent_tool",
    "malformed_json",
    "wrong_argument_fields",
    "tool_execution_error",
    "repetition",
    "max_steps",
)
PARTIAL_COMPONENT_KEYS = (
    "tool_name",
    "argument",
    "action",
    "db",
    "env_assertion",
    "communicate",
)
SPARSE_PERF_PREFIXES = (
    "rollout/zero_std/count_",
    "rollout/dynamic_filter/drop_",
)


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _rate(numerator: int | float, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _token_weighted_percentile(
    values: list[tuple[float, int]], percentile: float
) -> float | None:
    """Linear percentile over per-turn values repeated by their token counts."""

    ordered = sorted((value, count) for value, count in values if count > 0)
    total = sum(count for _, count in ordered)
    if total == 0:
        return None

    position = (total - 1) * percentile
    lower_rank = math.floor(position)
    upper_rank = math.ceil(position)

    def value_at(rank: int) -> float:
        seen = 0
        for value, count in ordered:
            seen += count
            if rank < seen:
                return value
        raise AssertionError("token percentile rank exceeds the reconstructed values")

    lower = value_at(lower_rank)
    upper = value_at(upper_rank)
    return lower + (upper - lower) * (position - lower_rank)


def _official_reward(metadata: dict[str, Any]) -> float | None:
    value = metadata.get("raw_reward")
    if not _is_number(value):
        reward_info = metadata.get("tau2_reward_info") or metadata.get("reward_info") or {}
        value = reward_info.get("reward") if isinstance(reward_info, dict) else None
    return float(value) if _is_number(value) else None


def _reference_action_tier(action: dict[str, Any]) -> str:
    if not action.get("tool_name_match"):
        return "unmatched"
    arguments = action.get("all_arguments") or action.get("arguments") or {}
    values = list(arguments.values()) if isinstance(arguments, dict) else []
    if not values or all(bool(value) for value in values):
        return "exact"
    if any(bool(value) for value in values):
        return "partial_arguments"
    return "tool_name_only"


def _reconstruct_v2_modifiers(
    turn_details: list[dict[str, Any]],
) -> list[float] | None:
    if not turn_details or any("critique" not in detail for detail in turn_details):
        return None
    positive = [
        index for index, detail in enumerate(turn_details) if detail.get("critique") == 1
    ]
    negative = [
        index for index, detail in enumerate(turn_details) if detail.get("critique") == -1
    ]
    modifiers = [0.0] * len(turn_details)
    if not positive and negative:
        positive = [
            index
            for index, detail in enumerate(turn_details[:-1])
            if detail.get("critique") == 0
        ]
    if not positive or not negative:
        return modifiers
    for index in positive:
        modifiers[index] = 0.5 / len(positive)
    for index in negative:
        modifiers[index] = -0.5 / len(negative)
    return modifiers


def _resolved_files(paths: Iterable[Path], *, pattern: str) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(candidate for candidate in path.rglob(pattern) if candidate.is_file()))
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(path)
    return list(dict.fromkeys(path.resolve() for path in resolved))


def discover_inputs(
    *,
    arm_dir: Path | None,
    trajectory_inputs: list[Path],
    log_inputs: list[Path],
) -> tuple[list[Path], list[Path]]:
    trajectory_paths = _resolved_files(trajectory_inputs, pattern="*.jsonl")
    log_paths = _resolved_files(log_inputs, pattern="run_*.log")
    if arm_dir is not None:
        if not arm_dir.is_dir():
            raise NotADirectoryError(arm_dir)
        trajectory_paths.extend(
            path.resolve()
            for path in arm_dir.rglob("*.jsonl")
            if "trajectories" in path.parts
        )
        log_paths.extend(path.resolve() for path in arm_dir.rglob("run_*.log"))
    trajectory_paths = list(dict.fromkeys(sorted(trajectory_paths)))
    log_paths = list(dict.fromkeys(sorted(log_paths)))
    if not trajectory_paths:
        raise ValueError("no trajectory JSONL files found")
    return trajectory_paths, log_paths


def summarize_trajectories(
    paths: list[Path], *, reallocation_weight: float | None = None
) -> dict[str, Any]:
    rows_by_file: dict[str, int] = {}
    groups: dict[
        tuple[str, int | None, int, int],
        list[tuple[float | None, bool, bool]],
    ] = defaultdict(list)
    latest_rollout_occurrence: dict[tuple[str, int | None], int] = {}
    missing_group_index = 0
    total_rows = 0
    removed_rows = 0
    domains: Counter[str] = Counter()
    rewards: list[float] = []
    not_removed_rewards: list[float] = []
    domain_rewards: dict[str, list[float]] = defaultdict(list)
    not_removed_domain_rewards: dict[str, list[float]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    rule_events: Counter[str] = Counter()
    rule_affected_rows: Counter[str] = Counter()
    reference_tiers: Counter[str] = Counter()
    partial_components: dict[str, list[float]] = defaultdict(list)
    turn_credit_versions: Counter[str] = Counter()
    v2_reconstruction_available = 0
    raw_directional_active = 0
    raw_positive_turns = 0
    raw_negative_turns = 0
    modifier_active = 0
    modifier_positive_turns = 0
    modifier_negative_turns = 0
    complementary_neutral_turns = 0
    stored_modifier_comparisons = 0
    stored_modifier_mismatches = 0
    weighted_local_turn_values: list[tuple[float, int]] = []
    weighted_local_reconstructed_trajectories = 0
    replacement_rows = 0
    replacement_groups: set[tuple[str, int | None, int, int]] = set()
    dropped_too_long = 0
    permanently_too_long = 0
    retried_rows = 0
    extra_rollout_attempts = 0

    for path in paths:
        path_key = str(path.resolve())
        file_rows = 0
        last_rollout_id: int | None | object = object()
        rollout_occurrences: Counter[int | None] = Counter()
        rollout_occurrence = 0
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
                total_rows += 1
                file_rows += 1
                metadata = row.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                removed = bool(row.get("remove_sample"))
                removed_rows += int(removed)

                domain = str(metadata.get("tau2_domain") or metadata.get("domain") or "<missing>")
                domains[domain] += 1
                reward = _official_reward(metadata)
                if reward is not None:
                    rewards.append(reward)
                    domain_rewards[domain].append(reward)
                    if not removed:
                        not_removed_rewards.append(reward)
                        not_removed_domain_rewards[domain].append(reward)

                rollout_id = metadata.get("tau2_domain_quota_rollout_id")
                if not isinstance(rollout_id, int) or isinstance(rollout_id, bool):
                    rollout_id = None
                if rollout_id != last_rollout_id:
                    rollout_occurrences[rollout_id] += 1
                    rollout_occurrence = rollout_occurrences[rollout_id]
                    last_rollout_id = rollout_id
                latest_rollout_occurrence[(path_key, rollout_id)] = rollout_occurrence

                group_key = None
                group_index = row.get("group_index")
                if isinstance(group_index, int) and not isinstance(group_index, bool):
                    group_key = (
                        path_key,
                        rollout_id,
                        rollout_occurrence,
                        group_index,
                    )
                    if metadata.get("tau2_domain_quota_replacement"):
                        replacement_groups.add(group_key)
                else:
                    missing_group_index += 1

                statuses[str(row.get("status") or "<missing>")] += 1
                termination = metadata.get("tau2_termination_reason")
                if termination is None:
                    simulation = row.get("simulation") or {}
                    termination = simulation.get("termination_reason") if isinstance(simulation, dict) else None
                terminations[str(termination or "<missing>")] += 1

                signals = metadata.get("tau2_field_reward_signals") or {}
                if not isinstance(signals, dict):
                    signals = {}
                counts = signals.get("counts") or {}
                if not isinstance(counts, dict):
                    counts = {}
                for key in RULE_ERROR_KEYS:
                    value = counts.get(key, 0)
                    if _is_number(value):
                        rule_events[key] += int(value)
                        rule_affected_rows[key] += int(float(value) > 0)

                action_details = signals.get("action_details") or []
                for action in action_details:
                    if isinstance(action, dict):
                        reference_tiers[_reference_action_tier(action)] += 1

                components = metadata.get("partial_components") or signals.get("components") or {}
                if isinstance(components, dict):
                    for key, value in components.items():
                        if _is_number(value):
                            partial_components[str(key)].append(float(value))

                version = str(metadata.get("tau2_turn_credit_version") or "<missing>")
                turn_credit_versions[version] += 1
                details = metadata.get("tau2_turn_credits")
                sample_has_local_signal = False
                if isinstance(details, list) and all(isinstance(detail, dict) for detail in details):
                    modifiers = _reconstruct_v2_modifiers(details)
                    if modifiers is not None:
                        sample_has_local_signal = bool(
                            reallocation_weight is not None
                            and reallocation_weight > 0.0
                            and any(value != 0.0 for value in modifiers)
                        )
                        v2_reconstruction_available += 1
                        positive_count = sum(detail.get("critique") == 1 for detail in details)
                        negative_count = sum(detail.get("critique") == -1 for detail in details)
                        raw_positive_turns += positive_count
                        raw_negative_turns += negative_count
                        raw_directional_active += int(positive_count + negative_count > 0)
                        modifier_positive_turns += sum(value > 0 for value in modifiers)
                        modifier_negative_turns += sum(value < 0 for value in modifiers)
                        modifier_active += int(any(value != 0.0 for value in modifiers))
                        complementary_neutral_turns += sum(
                            detail.get("critique") == 0 and modifier > 0
                            for detail, modifier in zip(details, modifiers)
                        )
                        for detail, modifier in zip(details, modifiers):
                            stored = detail.get("reallocation_modifier")
                            if _is_number(stored):
                                stored_modifier_comparisons += 1
                                stored_modifier_mismatches += int(
                                    abs(float(stored) - modifier) > 1e-9
                                )

                        if reallocation_weight is not None and not removed:
                            loss_mask = row.get("loss_mask")
                            if isinstance(loss_mask, list):
                                trainable_token_count = sum(bool(value) for value in loss_mask)
                                reconstructed_turns: list[tuple[float, int]] = []
                                for detail, modifier in zip(details, modifiers):
                                    span = detail.get("response_span")
                                    if (
                                        not isinstance(span, list)
                                        or len(span) != 2
                                        or any(
                                            isinstance(value, bool) or not isinstance(value, int)
                                            for value in span
                                        )
                                    ):
                                        break
                                    span_start, span_end = span
                                    if not 0 <= span_start < span_end <= len(loss_mask):
                                        break
                                    turn_token_count = span_end - span_start
                                    reconstructed_turns.append(
                                        (
                                            reallocation_weight
                                            * trainable_token_count
                                            * modifier
                                            / turn_token_count,
                                            turn_token_count,
                                        )
                                    )
                                else:
                                    if trainable_token_count > 0:
                                        weighted_local_reconstructed_trajectories += 1
                                        weighted_local_turn_values.extend(reconstructed_turns)

                if group_key is not None:
                    groups[group_key].append(
                        (reward, removed, sample_has_local_signal)
                    )

                replacement_rows += int(bool(metadata.get("tau2_domain_quota_replacement")))
                dropped_too_long += int(bool(metadata.get("tau2_dropped_too_long")))
                permanently_too_long += int(bool(metadata.get("tau2_permanently_too_long")))
                attempts = metadata.get("tau2_rollout_attempts")
                if _is_number(attempts):
                    attempts = int(attempts)
                    retried_rows += int(attempts > 1)
                    extra_rollout_attempts += max(0, attempts - 1)
        rows_by_file[path_key] = file_rows

    zero_variance_groups = 0
    zero_success_groups = 0
    zero_failure_groups = 0
    groups_with_missing_reward = 0
    groups_with_valid_outcomes = 0
    trainable_candidate_groups = 0
    outcome_signal_groups = 0
    local_signal_groups = 0
    rescued_zero_variance_groups = 0
    zero_signal_groups = 0
    valid_group_sizes: Counter[int] = Counter()
    dumped_group_sizes: Counter[int] = Counter()
    for group_key, group_rows in groups.items():
        dumped_group_sizes[len(group_rows)] += 1
        valid_values = [
            reward
            for reward, removed, _ in group_rows
            if not removed and reward is not None
        ]
        valid_group_sizes[len(valid_values)] += 1
        groups_with_missing_reward += int(
            any(
                reward is None and not removed
                for reward, removed, _ in group_rows
            )
        )
        if valid_values and max(valid_values) == min(valid_values):
            zero_variance_groups += 1
            zero_failure_groups += int(valid_values[0] == 0.0)
            zero_success_groups += int(valid_values[0] == 1.0)
        groups_with_valid_outcomes += int(bool(valid_values))

        path_key, rollout_id, rollout_occurrence, _ = group_key
        if rollout_occurrence != latest_rollout_occurrence[(path_key, rollout_id)]:
            continue
        if any(removed or reward is None for reward, removed, _ in group_rows):
            continue
        trainable_candidate_groups += 1
        has_outcome_signal = max(valid_values) != min(valid_values)
        has_local_signal = any(
            local_signal for _, _, local_signal in group_rows
        )
        outcome_signal_groups += int(has_outcome_signal)
        local_signal_groups += int(has_local_signal)
        rescued_zero_variance_groups += int(
            not has_outcome_signal and has_local_signal
        )
        zero_signal_groups += int(
            not has_outcome_signal and not has_local_signal
        )

    reference_total = sum(reference_tiers.values())
    component_summary = {}
    for key in list(PARTIAL_COMPONENT_KEYS) + sorted(
        set(partial_components) - set(PARTIAL_COMPONENT_KEYS)
    ):
        values = partial_components.get(key, [])
        component_summary[key] = {
            "available_trajectories": len(values),
            "mean": _mean(values),
        }

    rule_summary = {
        key: {
            "events": int(rule_events[key]),
            "affected_trajectories": int(rule_affected_rows[key]),
            "affected_rate": _rate(rule_affected_rows[key], total_rows),
        }
        for key in RULE_ERROR_KEYS
    }
    domain_summary = {
        domain: {
            "dumped_trajectories": domains[domain],
            "official_reward_available": len(domain_rewards[domain]),
            "official_reward_mean": _mean(domain_rewards[domain]),
            "official_successes": sum(value == 1.0 for value in domain_rewards[domain]),
            "not_removed_official_reward_available": len(
                not_removed_domain_rewards[domain]
            ),
            "not_removed_official_reward_mean": _mean(
                not_removed_domain_rewards[domain]
            ),
        }
        for domain in sorted(domains)
    }

    weighted_local_active = (
        None
        if reallocation_weight is None
        else modifier_active * int(reallocation_weight > 0.0)
    )
    weighted_local_activation_rate = (
        None
        if weighted_local_active is None
        else _rate(weighted_local_active, v2_reconstruction_available)
    )
    weighted_l1_budget_total = (
        None
        if reallocation_weight is None
        else reallocation_weight * modifier_active
    )
    weighted_l1_budget_per_modifier_active = (
        None
        if reallocation_weight is None or modifier_active == 0
        else reallocation_weight
    )
    weighted_local_per_token_advantage = None
    if reallocation_weight is not None:
        absolute_values = [
            (abs(value), token_count)
            for value, token_count in weighted_local_turn_values
        ]
        token_count = sum(count for _, count in weighted_local_turn_values)
        weighted_local_per_token_advantage = {
            "view": (
                "reconstructed from eligible trajectory-dump rows before dynamic "
                "filtering; groups later rejected or replaced may be included"
            ),
            "formula": (
                "reallocation_weight * trainable_token_count * "
                "reallocation_modifier / turn_token_count"
            ),
            "reconstructed_trajectories": weighted_local_reconstructed_trajectories,
            "turn_count": len(weighted_local_turn_values),
            "token_count": token_count,
            "signed_min": (
                min(value for value, _ in weighted_local_turn_values)
                if weighted_local_turn_values
                else None
            ),
            "signed_max": (
                max(value for value, _ in weighted_local_turn_values)
                if weighted_local_turn_values
                else None
            ),
            "abs_p90": _token_weighted_percentile(absolute_values, 0.90),
            "abs_p99": _token_weighted_percentile(absolute_values, 0.99),
            "abs_max": (
                max(value for value, _ in absolute_values)
                if absolute_values
                else None
            ),
        }

    return {
        "sources": [str(path.resolve()) for path in paths],
        "rows_by_file": rows_by_file,
        "trajectory_counts": {
            "dumped": total_rows,
            "removed": removed_rows,
            "not_marked_removed": total_rows - removed_rows,
            "by_domain": dict(sorted(domains.items())),
            "status": dict(sorted(statuses.items())),
        },
        "group_counts": {
            "dumped": len(groups),
            "missing_group_index_trajectories": missing_group_index,
            "dumped_size_histogram": {str(key): value for key, value in sorted(dumped_group_sizes.items())},
            "valid_size_histogram": {str(key): value for key, value in sorted(valid_group_sizes.items())},
            "groups_with_missing_official_reward": groups_with_missing_reward,
        },
        "official_reward": {
            "view": "all dumped candidates; run-log rollout/raw_reward is the accepted-update view",
            "available_trajectories": len(rewards),
            "successes": sum(value == 1.0 for value in rewards),
            "mean": _mean(rewards),
            "not_removed_available_trajectories": len(not_removed_rewards),
            "not_removed_successes": sum(
                value == 1.0 for value in not_removed_rewards
            ),
            "not_removed_mean": _mean(not_removed_rewards),
            "by_domain": domain_summary,
        },
        "binary_outcome_groups": {
            "view": "valid rows in dumped candidate groups",
            "zero_variance": zero_variance_groups,
            "zero_failure": zero_failure_groups,
            "zero_success": zero_success_groups,
            "groups_with_valid_outcomes": groups_with_valid_outcomes,
            "zero_variance_rate": _rate(
                zero_variance_groups, groups_with_valid_outcomes
            ),
        },
        "policy_signal_groups": {
            "view": (
                "latest attempt of each rollout, with no removed trajectory and "
                "complete official rewards; local signal uses the configured "
                "reallocation weight"
            ),
            "trainable_candidates": trainable_candidate_groups,
            "outcome_signal": outcome_signal_groups,
            "local_signal": local_signal_groups,
            "rescued_zero_variance": rescued_zero_variance_groups,
            "zero_signal": zero_signal_groups,
            "zero_signal_rate": _rate(
                zero_signal_groups, trainable_candidate_groups
            ),
        },
        "turn_credit": {
            "versions": dict(sorted(turn_credit_versions.items())),
            "reallocation_weight": reallocation_weight,
            "v2_reconstruction_available_trajectories": v2_reconstruction_available,
            "raw_directional_active_trajectories": raw_directional_active,
            "raw_directional_activation_rate": _rate(
                raw_directional_active, v2_reconstruction_available
            ),
            "raw_positive_turns": raw_positive_turns,
            "raw_negative_turns": raw_negative_turns,
            "modifier_active_trajectories": modifier_active,
            "modifier_activation_rate": _rate(
                modifier_active, v2_reconstruction_available
            ),
            "modifier_positive_turns": modifier_positive_turns,
            "modifier_negative_turns": modifier_negative_turns,
            "complementary_neutral_turns": complementary_neutral_turns,
            "weighted_local_channel_active_trajectories": weighted_local_active,
            "weighted_local_channel_activation_rate": weighted_local_activation_rate,
            "weighted_local_l1_budget_per_modifier_active_trajectory": (
                weighted_l1_budget_per_modifier_active
            ),
            "weighted_local_l1_budget_across_dumped_modifier_active_trajectories": (
                weighted_l1_budget_total
            ),
            "weighted_local_per_token_advantage": weighted_local_per_token_advantage,
            "stored_modifier_comparisons": stored_modifier_comparisons,
            "stored_modifier_mismatches": stored_modifier_mismatches,
        },
        "reference_actions": {
            "golden_actions": reference_total,
            "tiers": dict(sorted(reference_tiers.items())),
            "rates": {
                key: _rate(reference_tiers[key], reference_total)
                for key in ("exact", "partial_arguments", "tool_name_only", "unmatched")
            },
        },
        "rule_errors": rule_summary,
        "partial_components": component_summary,
        "termination": {
            "reasons": dict(sorted(terminations.items())),
            "max_steps": int(terminations["max_steps"]),
            "max_steps_rate": _rate(terminations["max_steps"], total_rows),
            "truncated_status": int(statuses["truncated"]),
            "truncated_status_rate": _rate(statuses["truncated"], total_rows),
        },
        "sampling": {
            "replacement_draw_trajectories": replacement_rows,
            "replacement_draw_groups": len(replacement_groups),
            "removed_trajectories": removed_rows,
            "dropped_too_long": dropped_too_long,
            "permanently_too_long": permanently_too_long,
            "retried_trajectories": retried_rows,
            "extra_rollout_attempts": extra_rollout_attempts,
        },
    }


def _metric_payload(raw: str, *, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid metric payload") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: metric payload is not a dictionary")
    return value


def _parse_log(path: Path) -> tuple[dict[int, dict[str, Any]], list[int]]:
    records: dict[int, dict[str, Any]] = {}
    parse_error_lines: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line_number, raw_line in enumerate(file, start=1):
            match = METRIC_RE.search(ANSI_RE.sub("", raw_line))
            if not match:
                continue
            step = int(match.group("step"))
            try:
                payload = _metric_payload(
                    match.group("payload"), path=path, line_number=line_number
                )
            except ValueError:
                parse_error_lines.append(line_number)
                continue
            record = records.setdefault(step, {"metrics": {}, "event_kinds": set()})
            record["event_kinds"].add(match.group("kind"))
            record["metrics"].update(
                {
                    str(key): float(value)
                    for key, value in payload.items()
                    if _is_number(value)
                }
            )
    return records, parse_error_lines


def _metric_stats(values: list[tuple[int, float]]) -> dict[str, Any]:
    final_step, final_value = max(values, key=lambda item: item[0])
    numeric = [value for _, value in values]
    return {
        "updates": len(values),
        "min": min(numeric),
        "mean": statistics.fmean(numeric),
        "final": final_value,
        "final_step": final_step,
    }


def summarize_logs(paths: list[Path]) -> dict[str, Any]:
    per_file: dict[str, dict[int, dict[str, Any]]] = {}
    source_latest: dict[str, int | None] = {}
    parse_error_lines: dict[str, list[int]] = {}
    for path in paths:
        path_key = str(path.resolve())
        records, errors = _parse_log(path)
        per_file[path_key] = records
        source_latest[path_key] = max(records) if records else None
        if errors:
            parse_error_lines[path_key] = errors

    # Stage logs normally own disjoint update ranges.  When a smoke log and a
    # real stage both contain step 0, prefer the source whose run reaches the
    # later step, so arm-directory discovery does not double-count smoke.
    selected: dict[int, tuple[int, str, dict[str, Any]]] = {}
    duplicate_step_sources: dict[int, list[str]] = defaultdict(list)
    for path_key, records in per_file.items():
        latest = source_latest[path_key]
        priority = latest if latest is not None else -1
        for step, record in records.items():
            duplicate_step_sources[step].append(path_key)
            current = selected.get(step)
            if current is None or priority > current[0]:
                selected[step] = (priority, path_key, record)

    perf_steps = {
        step
        for step, (_, _, record) in selected.items()
        if "perf" in record["event_kinds"]
    }
    all_metric_keys = {
        key
        for _, _, record in selected.values()
        for key in record["metrics"]
    }
    metric_values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for key in sorted(all_metric_keys):
        if key.startswith(SPARSE_PERF_PREFIXES):
            metric_values[key] = [
                (step, float(selected[step][2]["metrics"].get(key, 0.0)))
                for step in sorted(perf_steps)
            ]
        else:
            metric_values[key] = [
                (step, float(record["metrics"][key]))
                for step, (_, _, record) in sorted(selected.items())
                if key in record["metrics"]
            ]

    wanted = {
        key: _metric_stats(values)
        for key, values in metric_values.items()
        if values
        and (
            key in {
                "rollout/raw_reward",
                "rollout/truncated",
                "rollout/truncated_ratio",
                "rollout/kl",
                "train/loss",
                "train/pg_loss",
                "train/kl_loss",
                "train/ppo_kl",
                "train/grad_norm",
            }
            or key.startswith(SPARSE_PERF_PREFIXES)
        )
    }
    count_totals = {
        key: sum(value for _, value in values)
        for key, values in metric_values.items()
        if key.startswith(SPARSE_PERF_PREFIXES)
    }
    dynamic_filter_drop_total = sum(
        value
        for key, value in count_totals.items()
        if key.startswith("rollout/dynamic_filter/drop_")
    )
    zero_std_count_totals = {
        key: value
        for key, value in count_totals.items()
        if key.startswith("rollout/zero_std/count_")
    }
    return {
        "sources": [str(path.resolve()) for path in paths],
        "source_latest_step": source_latest,
        "observed_updates": len(selected),
        "observed_steps": sorted(selected),
        "perf_updates": len(perf_steps),
        "duplicate_step_sources": {
            str(step): sources
            for step, sources in sorted(duplicate_step_sources.items())
            if len(sources) > 1
        },
        "metric_parse_error_lines": parse_error_lines,
        "metrics": wanted,
        "sparse_count_totals": count_totals,
        "dynamic_filter_drop_total": dynamic_filter_drop_total,
        "zero_std_count_totals": zero_std_count_totals,
    }


def build_summary(
    *,
    arm: str,
    trajectory_paths: list[Path],
    log_paths: list[Path],
    reallocation_weight: float | None = None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "reallocation_weight": reallocation_weight,
        "trajectory_summary": summarize_trajectories(
            trajectory_paths, reallocation_weight=reallocation_weight
        ),
        "run_log_summary": summarize_logs(log_paths),
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a float") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative float")
    return value


def render_markdown(summary: dict[str, Any]) -> str:
    trajectory = summary["trajectory_summary"]
    logs = summary["run_log_summary"]
    reward = trajectory["official_reward"]
    groups = trajectory["binary_outcome_groups"]
    policy_groups = trajectory["policy_signal_groups"]
    turn_credit = trajectory["turn_credit"]
    weighted_local_advantage = turn_credit["weighted_local_per_token_advantage"]
    reference = trajectory["reference_actions"]

    lines = [
        f"# Training diagnostics: {summary['arm']}",
        "",
        "## Dumped trajectories",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Trajectories | {trajectory['trajectory_counts']['dumped']} |",
        f"| Groups | {trajectory['group_counts']['dumped']} |",
        f"| Removed | {trajectory['trajectory_counts']['removed']} |",
        f"| Official reward | {_format_value(reward['mean'])} |",
        f"| Official reward, not removed | {_format_value(reward['not_removed_mean'])} |",
        f"| Binary zero-variance groups | {groups['zero_variance']} ({_format_value(groups['zero_variance_rate'])}) |",
        f"| Trainable candidate groups | {policy_groups['trainable_candidates']} |",
        f"| Outcome-signal groups | {policy_groups['outcome_signal']} |",
        f"| Zero-variance groups rescued by local credit | {policy_groups['rescued_zero_variance']} |",
        f"| Zero-signal groups | {policy_groups['zero_signal']} ({_format_value(policy_groups['zero_signal_rate'])}) |",
        "",
        "The binary trajectory view includes candidates later replaced by the dynamic "
        "filter. Policy-signal counts exclude any group containing a removed trajectory.",
        "",
        "## Reward by domain",
        "",
        "| Domain | Trajectories | Successes | Reward mean | Not-removed mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain, values in reward["by_domain"].items():
        lines.append(
            f"| {domain} | {values['dumped_trajectories']} | "
            f"{values['official_successes']} | {_format_value(values['official_reward_mean'])} | "
            f"{_format_value(values['not_removed_official_reward_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Turn credit",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Versions | {json.dumps(turn_credit['versions'], sort_keys=True)} |",
            f"| Reallocation weight | {_format_value(turn_credit['reallocation_weight'])} |",
            f"| V2 reconstruction available | {turn_credit['v2_reconstruction_available_trajectories']} |",
            f"| Raw directional activation | {turn_credit['raw_directional_active_trajectories']} ({_format_value(turn_credit['raw_directional_activation_rate'])}) |",
            f"| Modifier allocation activation | {turn_credit['modifier_active_trajectories']} ({_format_value(turn_credit['modifier_activation_rate'])}) |",
            f"| Weighted local-channel activation | {_format_value(turn_credit['weighted_local_channel_active_trajectories'])} ({_format_value(turn_credit['weighted_local_channel_activation_rate'])}) |",
            f"| Weighted local L1 budget per modifier-active trajectory | {_format_value(turn_credit['weighted_local_l1_budget_per_modifier_active_trajectory'])} |",
            f"| Weighted local L1 budget across dumped modifier-active trajectories | {_format_value(turn_credit['weighted_local_l1_budget_across_dumped_modifier_active_trajectories'])} |",
            f"| Complementary-neutral turns | {turn_credit['complementary_neutral_turns']} |",
            f"| Stored/reconstructed mismatches | {turn_credit['stored_modifier_mismatches']} / {turn_credit['stored_modifier_comparisons']} |",
        ]
    )
    if weighted_local_advantage is not None:
        lines.extend(
            [
                f"| Weighted local per-token view | {weighted_local_advantage['view']} |",
                f"| Weighted local per-token count | {weighted_local_advantage['token_count']} |",
                f"| Weighted local per-token signed min / max | {_format_value(weighted_local_advantage['signed_min'])} / {_format_value(weighted_local_advantage['signed_max'])} |",
                f"| Weighted local per-token abs p90 / p99 / max | {_format_value(weighted_local_advantage['abs_p90'])} / {_format_value(weighted_local_advantage['abs_p99'])} / {_format_value(weighted_local_advantage['abs_max'])} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Reference actions",
            "",
            "| Tier | Count | Rate |",
            "|---|---:|---:|",
        ]
    )
    for tier in ("exact", "partial_arguments", "tool_name_only", "unmatched"):
        lines.append(
            f"| {tier} | {reference['tiers'].get(tier, 0)} | "
            f"{_format_value(reference['rates'][tier])} |"
        )

    lines.extend(
        [
            "",
            "## Rule errors",
            "",
            "| Error | Events | Affected trajectories | Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for error, values in trajectory["rule_errors"].items():
        lines.append(
            f"| {error} | {values['events']} | {values['affected_trajectories']} | "
            f"{_format_value(values['affected_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Partial components",
            "",
            "| Component | Available trajectories | Mean |",
            "|---|---:|---:|",
        ]
    )
    for component, values in trajectory["partial_components"].items():
        lines.append(
            f"| {component} | {values['available_trajectories']} | "
            f"{_format_value(values['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Termination and sampling",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Termination reasons | {json.dumps(trajectory['termination']['reasons'], sort_keys=True)} |",
            f"| max_steps | {trajectory['termination']['max_steps']} ({_format_value(trajectory['termination']['max_steps_rate'])}) |",
            f"| Truncated status | {trajectory['termination']['truncated_status']} ({_format_value(trajectory['termination']['truncated_status_rate'])}) |",
        ]
    )
    for key, value in trajectory["sampling"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Accepted-update log metrics",
            "",
            f"Parsed {logs['observed_updates']} update steps from {len(logs['sources'])} log files.",
            f"Dynamic-filter drops recorded: {_format_value(logs['dynamic_filter_drop_total'])}.",
            "",
            "| Metric | Updates | Min | Mean | Final | Final step |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric, values in logs["metrics"].items():
        lines.append(
            f"| {metric} | {values['updates']} | {_format_value(values['min'])} | "
            f"{_format_value(values['mean'])} | {_format_value(values['final'])} | "
            f"{values['final_step']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", help="experiment arm label")
    parser.add_argument(
        "--trajectory",
        action="append",
        default=[],
        type=Path,
        help="trajectory JSONL file or directory; repeat as needed",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        type=Path,
        help="run log file or job directory; repeat as needed",
    )
    parser.add_argument(
        "--arm-dir",
        type=Path,
        help="discover files below an arm directory",
    )
    parser.add_argument(
        "--reallocation-weight",
        type=_nonnegative_float,
        help="nonnegative v2 local-credit weight; required for weighted activation",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, help="write output instead of stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trajectory_paths, log_paths = discover_inputs(
        arm_dir=args.arm_dir,
        trajectory_inputs=args.trajectory,
        log_inputs=args.log,
    )
    summary = build_summary(
        arm=args.arm,
        trajectory_paths=trajectory_paths,
        log_paths=log_paths,
        reallocation_weight=args.reallocation_weight,
    )
    rendered = (
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(summary)
    )
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
