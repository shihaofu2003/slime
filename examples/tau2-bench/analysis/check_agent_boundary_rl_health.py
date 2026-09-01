#!/usr/bin/env python3
"""Validate boundary-v2 RL health without gating on behavior or eval."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from check_agent_boundary_rl_pilot import (
    PROFILE,
    PROTOCOL_SIGNATURE,
    TURN_CREDIT_VERSION,
    _turn_credit_alignment_error,
)
from check_tau2_rl_promotion import (
    ANSI_RE,
    FATAL_TRAINING_PATTERNS,
    ROLLOUT_METRIC_RE,
    TRAIN_METRIC_RE,
)


GATE_VERSION = "turn-aware-rl-health-v1"
LONG200_GATE_VERSION = "turn-aware-rl-health-v2"
LONG200_KL_WAIVER_GATE_VERSION = "turn-aware-rl-health-v2-kl-waiver"
KL_WAIVER_POLICY = "kl-waiver-v1"
KL_CHECK_NAMES = ("last_10_k2_kl", "no_consecutive_high_kl")
LONG100_STAGES: dict[str, dict[str, Any]] = {
    "iter9": {
        "runner_stage": "long100-a",
        "start": 0,
        "end": 9,
        "quota": {"telecom": 3, "airline": 2, "retail": 1},
        "prerequisite": None,
    },
    "iter19": {
        "runner_stage": "long100-b",
        "start": 10,
        "end": 19,
        "quota": {"telecom": 3, "airline": 1, "retail": 2},
        "prerequisite": "iter9",
    },
    "iter99": {
        "runner_stage": "long100-final",
        "start": 20,
        "end": 99,
        "quota": {"telecom": 2, "airline": 2, "retail": 2},
        "prerequisite": "iter19",
    },
}
LONG200_STAGES: dict[str, dict[str, Any]] = {
    "iter199": {
        "runner_stage": "long200-final",
        "start": 100,
        "end": 199,
        "quota": {"telecom": 2, "airline": 2, "retail": 2},
        "prerequisite": None,
        "source_iteration": 99,
    }
}
HEALTH_STAGES = {**LONG100_STAGES, **LONG200_STAGES}
EXPECTED_SAMPLES_PER_GROUP = 8
REQUIRED_STEP_METRICS = (
    "train/loss",
    "train/pg_loss",
    "train/kl_loss",
    "train/grad_norm",
    "train/lr-pg_0",
    "rollout/rewards",
)
SUCCESSFUL_LOAD_RE = re.compile(
    r"successfully loaded checkpoint from (?P<path>.+?)\s+"
    r"\[\s*t .*?\]\s+at iteration\s+(?P<iteration>\d+)"
)
LOADING_CHECKPOINT_RE = re.compile(
    r"loading distributed checkpoint from (?P<path>.+?)\s+"
    r"at iteration\s+(?P<iteration>\d+)"
)
RNG_IGNORED_RE = re.compile(r"RNG state will be ignored", re.IGNORECASE)
SUCCESSFUL_SAVE_RE = re.compile(
    r"successfully saved checkpoint from iteration\s+(?P<iteration>\d+)\s+"
    r"to\s+(?P<path>.+?)\s+\[\s*t "
)
EXPLICIT_OOM_RE = re.compile(
    r"cuda out of memory|outofmemoryerror",
    re.IGNORECASE,
)


class _NonfiniteNameTransformer(ast.NodeTransformer):
    """Allow only the non-finite names emitted by Python metric reprs."""

    _VALUES = {
        "nan": math.nan,
        "inf": math.inf,
        "infinity": math.inf,
    }

    def visit_Name(self, node: ast.Name) -> ast.AST:
        value = self._VALUES.get(node.id.lower())
        if value is None:
            return node
        return ast.copy_location(ast.Constant(value=value), node)


def _metric_payload(raw: str) -> dict[str, Any]:
    try:
        tree = ast.parse(raw, mode="eval")
        tree = _NonfiniteNameTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        payload = ast.literal_eval(tree)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid training metric payload: {raw[:200]}") from exc
    if not isinstance(payload, dict):
        raise ValueError("training metric payload is not a dictionary")
    return payload


def _resolved(value: str | Path) -> str:
    return str(Path(value).resolve())


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Inf" if value > 0 else "-Inf"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def parse_training_logs(
    paths: list[Path],
    *,
    start: int,
    end: int,
    checkpoint_root: Path,
    initial_checkpoint_root: Path,
    initial_iteration: int,
    mode: str = "final",
    checkpoint_scoped: bool = False,
) -> dict[str, Any]:
    """Merge ordered retry/resume logs and validate each effective load point."""

    if not paths:
        raise ValueError("at least one --training-log is required")
    fatal_matches = {label: 0 for label in FATAL_TRAINING_PATTERNS}
    metrics: dict[int, dict[str, Any]] = {}
    step_sources: dict[int, list[str]] = {}
    metric_occurrences: list[dict[str, Any]] = []
    all_save_events: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    log_reports: list[dict[str, Any]] = []

    if mode not in {"partial", "final"}:
        raise ValueError(f"unsupported health mode: {mode}")
    expected_checkpoint = _resolved(checkpoint_root)
    expected_initial = _resolved(initial_checkpoint_root)
    for path in paths:
        per_log_metrics: dict[int, dict[str, Any]] = {}
        load_events: list[dict[str, Any]] = []
        rng_ignored_loads: list[dict[str, Any]] = []
        pending_rng_ignored_lines: list[int] = []
        save_events: list[dict[str, Any]] = []
        scope_closed = False
        with path.open(encoding="utf-8", errors="replace") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = ANSI_RE.sub("", raw_line)
                structured_matches = [
                    match for pattern in (TRAIN_METRIC_RE, ROLLOUT_METRIC_RE)
                    if (match := pattern.search(line))
                ]
                if (
                    checkpoint_scoped
                    and structured_matches
                    and int(structured_matches[0].group("step")) > end
                ):
                    scope_closed = True
                if scope_closed:
                    continue
                for label, pattern in FATAL_TRAINING_PATTERNS.items():
                    # RolloutManager logs include verbatim Agent/User text.  Words
                    # such as "infinity" in a telecom conversation are not
                    # numerical failures.  Non-finite metric values are parsed
                    # structurally below; keep OOM scanning global because a real
                    # worker failure can still be relayed through this actor.
                    if label in {"nan", "inf"} and "RolloutManager pid=" in line:
                        continue
                    if label == "oom" and "RolloutManager pid=" in line:
                        # A telecom response can literally contain phrases such
                        # as "No OOM value".  Keep explicit CUDA/Python OOM
                        # exceptions fatal while ignoring only the ambiguous
                        # bare abbreviation in verbatim conversation text.
                        fatal_matches[label] += len(EXPLICIT_OOM_RE.findall(line))
                        continue
                    fatal_matches[label] += len(pattern.findall(line))
                if RNG_IGNORED_RE.search(line):
                    pending_rng_ignored_lines.append(line_number)
                loading_match = LOADING_CHECKPOINT_RE.search(line)
                if loading_match and pending_rng_ignored_lines:
                    rng_ignored_loads.append(
                        {
                            "path": _resolved(loading_match.group("path").strip()),
                            "iteration": int(loading_match.group("iteration")),
                            "warning_lines": pending_rng_ignored_lines,
                        }
                    )
                    pending_rng_ignored_lines = []
                load_match = SUCCESSFUL_LOAD_RE.search(line)
                if load_match:
                    load_events.append(
                        {
                            "path": _resolved(load_match.group("path").strip()),
                            "iteration": int(load_match.group("iteration")),
                        }
                    )
                save_match = SUCCESSFUL_SAVE_RE.search(line)
                if save_match:
                    event = {
                        "path": _resolved(save_match.group("path").strip()),
                        "iteration": int(save_match.group("iteration")),
                    }
                    save_events.append(event)
                    all_save_events.append(event)
                for match in structured_matches:
                    step = int(match.group("step"))
                    try:
                        payload = _metric_payload(match.group("payload"))
                    except ValueError as exc:
                        parse_errors.append(f"{path.resolve()}:{line_number}: {exc}")
                        break
                    per_log_metrics.setdefault(step, {}).update(payload)
                    break

        source = str(path.resolve())
        for step, payload in per_log_metrics.items():
            step_sources.setdefault(step, []).append(source)
            # Logs are ordered by attempt. A later retry owns the complete
            # effective payload for a duplicate update; retaining keys from an
            # abandoned attempt could hide a missing metric or bad quota.
            metrics[step] = payload
            metric_occurrences.append(
                {"path": source, "step": step, "metrics": payload}
            )

        observed_steps = sorted(per_log_metrics)
        recovery_requirement: dict[str, Any] | None = None
        recovery_passed = True
        if observed_steps:
            first_step = observed_steps[0]
            if first_step == start:
                recovery_requirement = {
                    "mode": (
                        "fresh_sft"
                        if start == 0
                        else (
                            "cross_root_resume"
                            if expected_initial != expected_checkpoint
                            else "resume_rl"
                        )
                    ),
                    "path": expected_initial,
                    "iteration": initial_iteration,
                }
            else:
                recovery_requirement = {
                    "mode": "resume_rl",
                    "path": expected_checkpoint,
                    "iteration": first_step - 1,
                }
            recovery_passed = any(
                event == {
                    "path": recovery_requirement["path"],
                    "iteration": recovery_requirement["iteration"],
                }
                for event in load_events
            )
        recovery_integrity_passed = bool(
            not pending_rng_ignored_lines
            and (
                recovery_requirement is None
                or not any(
                    event["path"] == recovery_requirement["path"]
                    and event["iteration"] == recovery_requirement["iteration"]
                    for event in rng_ignored_loads
                )
            )
        )
        log_reports.append(
            {
                "path": source,
                "observed_steps": observed_steps,
                "load_events": load_events,
                "rng_ignored_loads": rng_ignored_loads,
                "unassigned_rng_ignored_lines": pending_rng_ignored_lines,
                "save_events": save_events,
                "recovery_requirement": recovery_requirement,
                "recovery_passed": recovery_passed,
                "recovery_integrity_passed": recovery_integrity_passed,
            }
        )

    stage_steps = sorted(step for step in metrics if start <= step <= end)
    if mode == "partial":
        expected_steps = list(range(start, stage_steps[-1] + 1)) if stage_steps else []
    else:
        expected_steps = list(range(start, end + 1))
    unexpected_steps = sorted(
        step
        for step in metrics
        if step < start or (step > end and not checkpoint_scoped)
    )
    missing_steps = [step for step in expected_steps if step not in metrics]
    missing_kl_steps = [
        step
        for step in expected_steps
        if "train/kl_loss" not in metrics.get(step, {})
    ]
    missing_truncation_steps = [
        step
        for step in expected_steps
        if "rollout/truncated" not in metrics.get(step, {})
        and "rollout/truncated_ratio" not in metrics.get(step, {})
    ]
    missing_required_step_metrics = {
        str(step): [
            key for key in REQUIRED_STEP_METRICS if key not in metrics.get(step, {})
        ]
        for step in expected_steps
        if any(key not in metrics.get(step, {}) for key in REQUIRED_STEP_METRICS)
    }
    nonfinite_metrics = []
    for step in expected_steps:
        for key, value in metrics.get(step, {}).items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isfinite(float(value))
            ):
                nonfinite_metrics.append(f"{step}:{key}")

    kl_by_step = {
        step: float(metrics[step]["train/kl_loss"])
        for step in expected_steps
        if "train/kl_loss" in metrics.get(step, {})
        and isinstance(metrics[step]["train/kl_loss"], (int, float))
        and not isinstance(metrics[step]["train/kl_loss"], bool)
    }
    last_10_steps = expected_steps[-10:]
    last_10_values = [
        kl_by_step[step]
        for step in last_10_steps
        if step in kl_by_step and math.isfinite(kl_by_step[step])
    ]
    consecutive_high_kl = [
        [left, right]
        for left, right in zip(expected_steps, expected_steps[1:])
        if left in kl_by_step
        and right in kl_by_step
        and kl_by_step[left] >= 0.20
        and kl_by_step[right] >= 0.20
    ]
    return {
        "paths": [str(path.resolve()) for path in paths],
        "expected_steps": expected_steps,
        "observed_steps": sorted(metrics),
        "unexpected_steps": unexpected_steps,
        "missing_steps": missing_steps,
        "missing_kl_steps": missing_kl_steps,
        "missing_truncation_steps": missing_truncation_steps,
        "missing_required_step_metrics": missing_required_step_metrics,
        "fatal_matches": fatal_matches,
        "parse_errors": parse_errors,
        "nonfinite_metrics": nonfinite_metrics,
        "last_10_k2_kl_mean": (
            sum(last_10_values) / len(last_10_values)
            if last_10_steps and len(last_10_values) == len(last_10_steps)
            else None
        ),
        "consecutive_high_kl_steps": consecutive_high_kl,
        "duplicate_step_sources": {
            str(step): sources
            for step, sources in step_sources.items()
            if len(sources) > 1
        },
        "logs": log_reports,
        "save_events": all_save_events,
        "expected_step_metrics": {
            str(step): _json_safe(metrics.get(step, {})) for step in expected_steps
        },
        "_raw_metrics": metrics,
        "_metric_occurrences": [
            next(
                occurrence
                for occurrence in reversed(metric_occurrences)
                if occurrence["step"] == step
            )
            for step in expected_steps
            if step in metrics
        ],
    }


def _quota_observations(
    training: dict[str, Any],
    *,
    quota: dict[str, int],
) -> tuple[bool, dict[str, Any]]:
    observations: dict[str, Any] = {}
    passed = True
    expected_steps = set(training["expected_steps"])
    for occurrence_index, occurrence in enumerate(training["_metric_occurrences"]):
        step = occurrence["step"]
        if step not in expected_steps:
            continue
        metrics = occurrence["metrics"]
        step_observation: dict[str, Any] = {}
        for domain, expected in quota.items():
            prefix = f"rollout/domain_quota/{domain}"
            accepted = metrics.get(f"{prefix}/accepted")
            rejected = metrics.get(f"{prefix}/rejected")
            replacements = metrics.get(f"{prefix}/replacement_draws")
            step_observation[domain] = {
                "accepted": _json_safe(accepted),
                "rejected": _json_safe(rejected),
                "replacement_draws": _json_safe(replacements),
            }
            rejected_is_finite = (
                isinstance(rejected, (int, float))
                and not isinstance(rejected, bool)
                and math.isfinite(float(rejected))
                and rejected >= 0
            )
            passed = passed and accepted == expected
            passed = passed and rejected_is_finite
            passed = passed and replacements == rejected
        observations[f"{step}@{occurrence_index}"] = {
            "path": occurrence["path"],
            "domains": step_observation,
        }
    return passed, observations


def _empty_trajectory_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "domains": Counter(),
        "wrong_profiles": Counter(),
        "wrong_protocol_signatures": Counter(),
        "contract_signatures": {},
        "missing_contract_signatures": 0,
        "turn_credit_errors": Counter(),
        "termination_reasons": Counter(),
        "truncated_breakdown": Counter(),
        "behavior": Counter(),
    }


def _update_trajectory_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["rows"] += 1
    metadata = row.get("metadata") or {}
    domain = str(
        metadata.get("tau2_domain") or metadata.get("domain") or "<missing>"
    )
    stats["domains"][domain] += 1
    profile = str(metadata.get("tau2_agent_protocol_profile") or "<missing>")
    signature = str(metadata.get("tau2_agent_protocol_signature") or "<missing>")
    if profile != PROFILE:
        stats["wrong_profiles"][profile] += 1
    if signature != PROTOCOL_SIGNATURE:
        stats["wrong_protocol_signatures"][signature] += 1
    contract_signature = metadata.get("agent_contract_signature")
    if not isinstance(contract_signature, str) or not contract_signature.startswith(
        "sha256:"
    ):
        stats["missing_contract_signatures"] += 1
    else:
        stats["contract_signatures"].setdefault(domain, set()).add(
            contract_signature
        )

    alignment_error = _turn_credit_alignment_error(row)
    if alignment_error:
        stats["turn_credit_errors"][alignment_error] += 1

    counts = ((metadata.get("tau2_field_reward_signals") or {}).get("counts") or {})
    stats["behavior"]["malformed_json"] += int(
        int(counts.get("malformed_json", 0)) > 0
    )
    stats["behavior"]["nonexistent_tool"] += int(
        int(counts.get("nonexistent_tool", 0)) > 0
    )
    stats["behavior"]["wrong_argument_field"] += int(
        int(counts.get("wrong_argument_fields", 0)) > 0
    )
    stats["behavior"]["tool_execution_error"] += int(
        int(counts.get("tool_execution_error", 0)) > 0
    )
    stats["behavior"]["repetition"] += int(int(counts.get("repetition", 0)) > 0)

    reason = str(
        metadata.get("tau2_termination_reason")
        or (row.get("simulation") or {}).get("termination_reason")
        or "unknown"
    )
    stats["termination_reasons"][reason] += 1
    if row.get("status") != "truncated":
        return
    over_cap = bool(
        metadata.get("tau2_dropped_too_long")
        or metadata.get("tau2_permanently_too_long")
        or metadata.get("tau2_turn_credit_invalid") == "over_train_token_cap"
    )
    if over_cap:
        stats["truncated_breakdown"]["over_token_cap"] += 1
    elif reason == "max_steps":
        stats["truncated_breakdown"]["max_steps"] += 1
    elif reason == "timeout":
        stats["truncated_breakdown"]["timeout"] += 1
    elif reason == "context_window_exceeded":
        stats["truncated_breakdown"]["context_window"] += 1
    else:
        stats["truncated_breakdown"]["other"] += 1


def _merge_trajectory_stats(
    destination: dict[str, Any], source: dict[str, Any]
) -> None:
    destination["rows"] += source["rows"]
    for key in (
        "domains",
        "wrong_profiles",
        "wrong_protocol_signatures",
        "turn_credit_errors",
        "termination_reasons",
        "truncated_breakdown",
        "behavior",
    ):
        destination[key].update(source[key])
    destination["missing_contract_signatures"] += source[
        "missing_contract_signatures"
    ]
    for domain, signatures in source["contract_signatures"].items():
        destination["contract_signatures"].setdefault(domain, set()).update(
            signatures
        )


def summarize_health_trajectories(
    paths: list[Path],
    *,
    effective_steps: set[int] | None = None,
    allow_incomplete_tail: bool = False,
) -> dict[str, Any]:
    """Select the last artifact for each update, then summarize its rows."""

    if not paths:
        raise ValueError("at least one --trajectories artifact is required")
    parsed_artifacts: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    raw_total_rows = 0
    missing_rollout_ids = 0
    ignored_incomplete_tails = 0
    for path in paths:
        digest = hashlib.sha256()
        stats_by_rollout: dict[int, dict[str, Any]] = {}
        legacy_stats = _empty_trajectory_stats()
        path_rows = 0
        with path.open("rb") as file:
            for line_number, raw_line in enumerate(file, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if allow_incomplete_tail and not raw_line.endswith(b"\n"):
                        ignored_incomplete_tails += 1
                        continue
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                path_rows += 1
                metadata = row.get("metadata") or {}
                rollout_id = metadata.get("tau2_domain_quota_rollout_id")
                if isinstance(rollout_id, str) and rollout_id.isdigit():
                    rollout_id = int(rollout_id)
                if isinstance(rollout_id, int) and not isinstance(rollout_id, bool):
                    stats = stats_by_rollout.setdefault(
                        rollout_id, _empty_trajectory_stats()
                    )
                else:
                    missing_rollout_ids += 1
                    stats = legacy_stats
                _update_trajectory_stats(stats, row)

        resolved = str(path.resolve())
        raw_total_rows += path_rows
        parsed_artifacts.append(
            {
                "path": resolved,
                "stats_by_rollout": stats_by_rollout,
                "legacy_stats": legacy_stats,
            }
        )
        artifacts[resolved] = {
            "sha256": digest.hexdigest(),
            "rows": path_rows,
            "bytes": path.stat().st_size,
        }

    effective_owner: dict[int, int] = {}
    duplicate_rollouts: set[int] = set()
    for artifact_index, artifact in enumerate(parsed_artifacts):
        for rollout_id in artifact["stats_by_rollout"]:
            if effective_steps is not None and rollout_id not in effective_steps:
                continue
            if rollout_id in effective_owner:
                duplicate_rollouts.add(rollout_id)
            effective_owner[rollout_id] = artifact_index

    selected = _empty_trajectory_stats()
    rows_by_path = {artifact["path"]: 0 for artifact in parsed_artifacts}
    for artifact_index, artifact in enumerate(parsed_artifacts):
        legacy_stats = artifact["legacy_stats"]
        _merge_trajectory_stats(selected, legacy_stats)
        rows_by_path[artifact["path"]] += legacy_stats["rows"]
        for rollout_id, stats in artifact["stats_by_rollout"].items():
            if effective_owner.get(rollout_id) != artifact_index:
                continue
            _merge_trajectory_stats(selected, stats)
            rows_by_path[artifact["path"]] += stats["rows"]

    denominator = selected["rows"]
    framework_truncated = sum(selected["truncated_breakdown"].values())
    expected_domains = set().union(
        *(stage["quota"] for stage in HEALTH_STAGES.values())
    )
    rendered_contracts = {
        domain: sorted(signatures)
        for domain, signatures in sorted(selected["contract_signatures"].items())
    }
    return {
        "paths": [str(path.resolve()) for path in paths],
        "artifacts": artifacts,
        "rows_by_path": rows_by_path,
        "raw_total_rows": raw_total_rows,
        "total_rows": denominator,
        "ignored_rows": raw_total_rows - denominator,
        "ignored_incomplete_tails": ignored_incomplete_tails,
        "missing_rollout_ids": missing_rollout_ids,
        "effective_rollout_ids": sorted(effective_owner),
        "duplicate_rollout_ids": sorted(duplicate_rollouts),
        "domains": dict(sorted(selected["domains"].items())),
        "wrong_protocol_profiles": dict(selected["wrong_profiles"]),
        "wrong_protocol_signatures": dict(
            selected["wrong_protocol_signatures"]
        ),
        "missing_contract_signatures": selected["missing_contract_signatures"],
        "agent_contract_signatures": rendered_contracts,
        "contract_signatures_aligned": (
            expected_domains.issubset(rendered_contracts)
            and all(len(rendered_contracts[domain]) == 1 for domain in expected_domains)
        ),
        "turn_credit_alignment_errors": dict(selected["turn_credit_errors"]),
        "termination_reasons": dict(sorted(selected["termination_reasons"].items())),
        "framework_truncated": {
            "count": framework_truncated,
            "fraction": framework_truncated / denominator if denominator else None,
            "breakdown": {
                name: selected["truncated_breakdown"].get(name, 0)
                for name in (
                    "max_steps",
                    "timeout",
                    "context_window",
                    "over_token_cap",
                    "other",
                )
            },
        },
        "behavior_diagnostics": {
            key: value
            for name, count in sorted(selected["behavior"].items())
            for key, value in (
                (f"{name}_count", count),
                (
                    f"{name}_fraction",
                    count / denominator if denominator else None,
                ),
            )
        },
    }


def _read_latest(checkpoint_root: Path) -> int | None:
    marker = checkpoint_root / "latest_checkpointed_iteration.txt"
    raw = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    return int(raw) if raw.isdigit() else None


def _prerequisite_observation(
    payload: dict[str, Any] | None,
    *,
    expected_stage: str | None,
    checkpoint_root: Path,
) -> tuple[bool, dict[str, Any] | None]:
    if expected_stage is None:
        return payload is None, payload
    config = LONG100_STAGES[expected_stage]
    passed = bool(
        payload
        and payload.get("status") == "pass"
        and payload.get("gate_version") == GATE_VERSION
        and payload.get("stage") == expected_stage
        and payload.get("expected_latest") == config["end"]
        and payload.get("checkpoint_root") == _resolved(checkpoint_root)
        and payload.get("turn_credit_version") == TURN_CREDIT_VERSION
        and payload.get("agent_protocol_signature") == PROTOCOL_SIGNATURE
    )
    observation = None if payload is None else {
        "status": payload.get("status"),
        "gate_version": payload.get("gate_version"),
        "stage": payload.get("stage"),
        "expected_latest": payload.get("expected_latest"),
        "checkpoint_root": payload.get("checkpoint_root"),
        "agent_contract_signatures": (
            (payload.get("trajectories") or {}).get("agent_contract_signatures")
        ),
    }
    return passed, observation


def _checkpoint_state(root: Path, iteration: int, *, require_shards: bool) -> dict[str, Any]:
    iteration_dir = root / f"iter_{iteration:07d}"
    shards = list(iteration_dir.glob("__*.distcp")) if iteration_dir.is_dir() else []
    return {
        "iteration_dir": iteration_dir.is_dir(),
        "metadata": (iteration_dir / ".metadata").is_file(),
        "common": (iteration_dir / "common.pt").is_file(),
        "rollout_state": (
            root / "rollout" / f"global_dataset_state_dict_{iteration}.pt"
        ).is_file(),
        "nonempty_shards": (
            sum(path.stat().st_size > 0 for path in shards) if require_shards else None
        ),
    }


def _checkpoint_state_complete(state: dict[str, Any], *, require_shards: bool) -> bool:
    core = all(
        state[key] for key in ("iteration_dir", "metadata", "common", "rollout_state")
    )
    return core and (not require_shards or bool(state["nonempty_shards"]))


def _source_health_observation(
    payload: dict[str, Any] | None,
    *,
    source_checkpoint_root: Path | None,
    source_health_sha256: str | None,
) -> tuple[bool, dict[str, Any]]:
    source_root = (
        _resolved(source_checkpoint_root) if source_checkpoint_root is not None else None
    )
    source_state = (
        _checkpoint_state(source_checkpoint_root, 99, require_shards=True)
        if source_checkpoint_root is not None
        else None
    )
    latest = (
        _read_latest(source_checkpoint_root)
        if source_checkpoint_root is not None
        else None
    )
    passed = bool(
        payload
        and source_root
        and source_health_sha256
        and payload.get("status") == "pass"
        and payload.get("gate_version") == GATE_VERSION
        and payload.get("stage") == "iter99"
        and payload.get("checkpoint_root") == source_root
        and payload.get("expected_checkpoint_root") == source_root
        and payload.get("expected_latest") == 99
        and payload.get("turn_credit_version") == TURN_CREDIT_VERSION
        and payload.get("agent_protocol_signature") == PROTOCOL_SIGNATURE
        and latest == 99
        and source_state is not None
        and _checkpoint_state_complete(source_state, require_shards=True)
    )
    return passed, {
        "status": (payload or {}).get("status"),
        "gate_version": (payload or {}).get("gate_version"),
        "stage": (payload or {}).get("stage"),
        "checkpoint_root": (payload or {}).get("checkpoint_root"),
        "source_checkpoint_root": source_root,
        "source_latest": latest,
        "source_checkpoint_state": source_state,
        "source_health_gate_sha256": source_health_sha256,
    }


def _lineage_observation(
    payload: dict[str, Any] | None,
    *,
    source_checkpoint_root: Path | None,
    destination_checkpoint_root: Path,
    source_health_artifact: Path | None,
    source_health_sha256: str | None,
) -> tuple[bool, dict[str, Any] | None]:
    if payload is None:
        return False, None
    expected = {
        "lineage_version": "boundary-v2-long200-v1",
        "source_checkpoint_root": (
            _resolved(source_checkpoint_root)
            if source_checkpoint_root is not None
            else None
        ),
        "destination_checkpoint_root": _resolved(destination_checkpoint_root),
        "source_iteration": 99,
        "target_iteration": 199,
        "source_health_gate": (
            str(source_health_artifact.resolve())
            if source_health_artifact is not None
            else None
        ),
        "source_health_gate_sha256": source_health_sha256,
        "scheduler_horizon_override": True,
    }
    return payload == expected, {"expected": expected, "observed": payload}


def evaluate_health(
    *,
    stage: str,
    checkpoint_root: Path,
    expected_checkpoint_root: Path,
    sft_checkpoint_root: Path,
    training_logs: list[Path],
    trajectory_paths: list[Path],
    prerequisite_health: dict[str, Any] | None = None,
    source_checkpoint_root: Path | None = None,
    source_health: dict[str, Any] | None = None,
    source_health_sha256: str | None = None,
    source_health_artifact: Path | None = None,
    lineage: dict[str, Any] | None = None,
    mode: str = "final",
    waive_kl: bool = False,
) -> dict[str, Any]:
    config = HEALTH_STAGES[stage]
    is_long200 = stage in LONG200_STAGES
    if is_long200:
        initial_checkpoint_root = source_checkpoint_root or checkpoint_root
        initial_iteration = config["source_iteration"]
    elif config["start"] == 0:
        initial_checkpoint_root = sft_checkpoint_root
        initial_iteration = 0
    else:
        initial_checkpoint_root = checkpoint_root
        initial_iteration = config["start"] - 1
    training = parse_training_logs(
        training_logs,
        start=config["start"],
        end=config["end"],
        checkpoint_root=checkpoint_root,
        initial_checkpoint_root=initial_checkpoint_root,
        initial_iteration=initial_iteration,
        mode=mode,
    )
    trajectories = summarize_health_trajectories(
        trajectory_paths,
        effective_steps=set(training["expected_steps"]),
        allow_incomplete_tail=mode == "partial",
    )
    quota_passed, quota_observed = _quota_observations(training, quota=config["quota"])
    prerequisite_passed, prerequisite_observed = _prerequisite_observation(
        prerequisite_health,
        expected_stage=config["prerequisite"],
        checkpoint_root=checkpoint_root,
    )
    source_health_passed, source_health_observed = _source_health_observation(
        source_health,
        source_checkpoint_root=source_checkpoint_root,
        source_health_sha256=source_health_sha256,
    )
    lineage_passed, lineage_observed = _lineage_observation(
        lineage,
        source_checkpoint_root=source_checkpoint_root,
        destination_checkpoint_root=checkpoint_root,
        source_health_artifact=source_health_artifact,
        source_health_sha256=source_health_sha256,
    )

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, requirement: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "requirement": requirement,
            }
        )

    expected_root = _resolved(expected_checkpoint_root)
    actual_root = _resolved(checkpoint_root)
    add("checkpoint_root", actual_root == expected_root, actual_root, f"== {expected_root}")
    if is_long200:
        add(
            "source_health",
            source_health_passed,
            source_health_observed,
            "passing immutable long100 iter99 health gate and complete exact source checkpoint",
        )
        add(
            "cross_root_lineage",
            lineage_passed,
            lineage_observed,
            "lineage marker matches source/destination roots, source Gate hash, and scheduler override",
        )
    latest = _read_latest(checkpoint_root)
    exact_stage_save = {
        "path": actual_root,
        "iteration": config["end"],
    } in training["save_events"]
    observed_end = (
        training["expected_steps"][-1] if training["expected_steps"] else None
    )
    if mode == "partial":
        checkpoint_iteration_passed = bool(
            latest is None
            and is_long200
            and (observed_end is None or observed_end < config["start"] + 9)
        )
        if latest is not None:
            checkpoint_iteration_passed = bool(
                observed_end is not None
                and config["start"] + 9 <= latest <= observed_end
                and (latest - (config["start"] + 9)) % 10 == 0
                and {"path": actual_root, "iteration": latest}
                in training["save_events"]
            )
        checkpoint_iteration_requirement = (
            "no destination save before iter109, otherwise the latest complete "
            "ten-update save does not exceed observed progress"
        )
    else:
        checkpoint_iteration_passed = bool(
            (
                latest == config["end"]
                and (not is_long200 or exact_stage_save)
            )
            or (
                latest is not None
                and latest > config["end"]
                and exact_stage_save
            )
        )
        checkpoint_iteration_requirement = (
            f"current latest is {config['end']}, or an advanced exact root has "
            f"an explicit successful save for historical iteration {config['end']}"
        )
    add(
        "checkpoint_latest",
        checkpoint_iteration_passed,
        {
            "current_latest": latest,
            "stage_iteration": config["end"],
            "exact_stage_save_event": exact_stage_save,
        },
        checkpoint_iteration_requirement,
    )
    checkpoint_iteration = config["end"] if mode == "final" else latest
    checkpoint_files = (
        _checkpoint_state(
            checkpoint_root,
            checkpoint_iteration,
            require_shards=is_long200,
        )
        if checkpoint_iteration is not None
        else None
    )
    checkpoint_state_passed = bool(
        checkpoint_files is not None
        and _checkpoint_state_complete(checkpoint_files, require_shards=is_long200)
    )
    if mode == "partial" and checkpoint_iteration is None and is_long200:
        checkpoint_state_passed = True
    add(
        "checkpoint_state_complete",
        checkpoint_state_passed,
        checkpoint_files,
        "iteration, optimizer/common metadata, rollout sampler state, and long200 shards exist when a destination save is required",
    )
    if not is_long200:
        add(
            "prerequisite_health",
            prerequisite_passed,
            prerequisite_observed,
            (
                "no prerequisite for a fresh SFT start"
                if config["prerequisite"] is None
                else f"passing {config['prerequisite']} health artifact for this exact root"
            ),
        )
    add(
        "recovery_state",
        all(log["recovery_passed"] for log in training["logs"]),
        training["logs"],
        "each effective log starts from the exact SFT or preceding RL iteration",
    )
    add(
        "recovery_integrity",
        all(log["recovery_integrity_passed"] for log in training["logs"]),
        training["logs"],
        "the effective training checkpoint load never ignores RNG state; warnings bound to the separate reference load are diagnostic only",
    )
    add(
        "training_metrics_complete",
        not training["missing_steps"]
        and not training["missing_kl_steps"]
        and not training["missing_truncation_steps"]
        and not training["missing_required_step_metrics"]
        and not training["unexpected_steps"],
        {
            "missing_steps": training["missing_steps"],
            "missing_kl_steps": training["missing_kl_steps"],
            "missing_truncation_steps": training["missing_truncation_steps"],
            "missing_required_step_metrics": training[
                "missing_required_step_metrics"
            ],
            "unexpected_steps": training["unexpected_steps"],
        },
        (
            f"required updates from {config['start']} through "
            f"{observed_end if mode == 'partial' else config['end']} contain every "
            f"core metric {REQUIRED_STEP_METRICS} plus truncation"
        ),
    )
    add(
        "training_finite",
        not any(training["fatal_matches"].values())
        and not training["parse_errors"]
        and not training["nonfinite_metrics"],
        {
            "fatal_matches": training["fatal_matches"],
            "parse_errors": training["parse_errors"],
            "nonfinite_metrics": training["nonfinite_metrics"],
        },
        "no OOM, NaN, Inf, parse failure, or non-finite metric",
    )
    kl_mean = training["last_10_k2_kl_mean"]
    has_progress = bool(training["expected_steps"])
    add(
        "last_10_k2_kl",
        (not has_progress and mode == "partial")
        or (kl_mean is not None and kl_mean < 0.10),
        kl_mean,
        "< 0.10 for the last min(10, completed updates)",
    )
    add(
        "no_consecutive_high_kl",
        not training["consecutive_high_kl_steps"],
        training["consecutive_high_kl_steps"],
        "no adjacent updates both have train/kl_loss >= 0.20",
    )
    add(
        "domain_quota",
        quota_passed,
        quota_observed,
        f"every update accepts {config['quota']} and replacements equal same-domain rejects",
    )
    expected_minimum_rows = (
        len(training["expected_steps"])
        * sum(config["quota"].values())
        * EXPECTED_SAMPLES_PER_GROUP
    )
    raw_artifacts_nonempty = all(
        artifact["rows"] > 0 for artifact in trajectories["artifacts"].values()
    )
    trajectory_data_passed = bool(
        trajectories["total_rows"] >= expected_minimum_rows
        and raw_artifacts_nonempty
        and (not is_long200 or trajectories["missing_rollout_ids"] == 0)
    )
    if mode == "partial" and not has_progress:
        trajectory_data_passed = True
    add(
        "trajectory_data_complete",
        trajectory_data_passed,
        {
            "rows": trajectories["total_rows"],
            "minimum": expected_minimum_rows,
            "rows_by_path": trajectories["rows_by_path"],
            "raw_artifacts_nonempty": raw_artifacts_nonempty,
            "missing_rollout_ids": trajectories["missing_rollout_ids"],
            "duplicate_rollout_ids": trajectories["duplicate_rollout_ids"],
            "ignored_rows": trajectories["ignored_rows"],
        },
        "at least K=8 effective dumped samples per accepted group, ordered retry deduplication, and no empty artifact",
    )
    trajectory_contract_passed = bool(
        not trajectories["wrong_protocol_profiles"]
        and not trajectories["wrong_protocol_signatures"]
        and trajectories["missing_contract_signatures"] == 0
        and trajectories["contract_signatures_aligned"]
    )
    if mode == "partial" and not has_progress:
        trajectory_contract_passed = True
    add(
        "trajectory_contract",
        trajectory_contract_passed,
        {
            "wrong_profiles": trajectories["wrong_protocol_profiles"],
            "wrong_protocol_signatures": trajectories["wrong_protocol_signatures"],
            "missing_contract_signatures": trajectories["missing_contract_signatures"],
            "agent_contract_signatures": trajectories["agent_contract_signatures"],
        },
        "all rows use the exact protocol and one signed Agent contract per domain",
    )
    prior_health = source_health if is_long200 else prerequisite_health
    prior_contracts = ((prior_health or {}).get("trajectories") or {}).get(
        "agent_contract_signatures"
    )
    continuity_passed = bool(
        prior_contracts is None
        or prior_contracts == trajectories["agent_contract_signatures"]
    )
    if mode == "partial" and not has_progress:
        continuity_passed = True
    add(
        "trajectory_contract_continuity",
        continuity_passed,
        {
            "prerequisite": prior_contracts,
            "current": trajectories["agent_contract_signatures"],
        },
        "contract signatures remain exact across resumed stages",
    )
    add(
        "trajectory_turn_credit_alignment",
        not trajectories["turn_credit_alignment_errors"],
        trajectories["turn_credit_alignment_errors"],
        "turn-credit-v1 response spans, loss masks, and penalty vectors all align",
    )

    training_output = {
        key: value
        for key, value in training.items()
        if key not in {"_raw_metrics", "_metric_occurrences"}
    }
    if waive_kl and not is_long200:
        raise ValueError("KL waiver is supported only for long200 iter199 health")
    waived_checks = set(KL_CHECK_NAMES) if waive_kl else set()
    for check in checks:
        if check["name"] in waived_checks:
            check["enforced"] = False
    all_checks_passed = all(
        check["passed"] or check["name"] in waived_checks for check in checks
    )
    if not all_checks_passed:
        status = "fail"
    elif mode == "partial" and not has_progress:
        status = "pending"
    else:
        status = "pass"
    report = {
        "status": status,
        "gate_version": (
            LONG200_KL_WAIVER_GATE_VERSION
            if waive_kl
            else LONG200_GATE_VERSION
            if is_long200
            else GATE_VERSION
        ),
        "health_mode": mode,
        "stage": stage,
        "runner_stage": config["runner_stage"],
        "checkpoint_root": actual_root,
        "expected_checkpoint_root": expected_root,
        "destination_checkpoint_root": actual_root,
        "source_checkpoint_root": (
            _resolved(source_checkpoint_root)
            if source_checkpoint_root is not None
            else None
        ),
        "source_health_gate": (
            str(source_health_artifact.resolve())
            if source_health_artifact is not None
            else None
        ),
        "source_health_gate_sha256": source_health_sha256,
        "sft_checkpoint_root": _resolved(sft_checkpoint_root),
        "expected_latest": config["end"],
        "expected_steps": [config["start"], config["end"]],
        "completed_steps": training["expected_steps"],
        "domain_quota": config["quota"],
        "protocol_profile": PROFILE,
        "agent_protocol_signature": PROTOCOL_SIGNATURE,
        "turn_credit_version": TURN_CREDIT_VERSION,
        "checks": checks,
        "training": training_output,
        "trajectories": trajectories,
        "diagnostics_only": {
            "framework_truncated": trajectories["framework_truncated"],
            "termination_reasons": trajectories["termination_reasons"],
            "behavior": trajectories["behavior_diagnostics"],
        },
    }
    if waive_kl:
        report["health_policy"] = KL_WAIVER_POLICY
        report["waived_checks"] = list(KL_CHECK_NAMES)
    return report


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(HEALTH_STAGES), required=True)
    parser.add_argument("--mode", choices=("partial", "final"), default="final")
    parser.add_argument("--waive-kl", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-root", type=Path, required=True)
    parser.add_argument("--sft-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path)
    parser.add_argument("--source-health", type=Path)
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--training-log", type=Path, action="append", required=True)
    parser.add_argument("--trajectories", type=Path, action="append", required=True)
    parser.add_argument("--prerequisite-health", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prerequisite = (
        _read_json(args.prerequisite_health) if args.prerequisite_health else None
    )
    source_health = _read_json(args.source_health) if args.source_health else None
    source_health_sha256 = (
        hashlib.sha256(args.source_health.read_bytes()).hexdigest()
        if args.source_health
        else None
    )
    lineage = _read_json(args.lineage) if args.lineage else None
    if args.stage in LONG200_STAGES and not all(
        (args.source_checkpoint_root, args.source_health, args.lineage)
    ):
        parser.error(
            "iter199 requires --source-checkpoint-root, --source-health, and --lineage"
        )
    if args.waive_kl and args.stage not in LONG200_STAGES:
        parser.error("--waive-kl is supported only for long200 iter199 health")
    report = evaluate_health(
        stage=args.stage,
        checkpoint_root=args.checkpoint_root,
        expected_checkpoint_root=args.expected_checkpoint_root,
        sft_checkpoint_root=args.sft_checkpoint_root,
        training_logs=args.training_log,
        trajectory_paths=args.trajectories,
        prerequisite_health=prerequisite,
        source_checkpoint_root=args.source_checkpoint_root,
        source_health=source_health,
        source_health_sha256=source_health_sha256,
        source_health_artifact=args.source_health,
        lineage=lineage,
        mode=args.mode,
        waive_kl=args.waive_kl,
    )
    if args.prerequisite_health:
        report["prerequisite_health_artifact"] = str(
            args.prerequisite_health.resolve()
        )
    write_json(args.output, report)
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
