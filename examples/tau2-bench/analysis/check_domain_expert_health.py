#!/usr/bin/env python3
"""Create checkpoint-scoped health Gates for domain experts and mixed controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from check_agent_boundary_rl_health import (
    KL_CHECK_NAMES,
    REQUIRED_STEP_METRICS,
    _checkpoint_state,
    _checkpoint_state_complete,
    _lineage_observation,
    _quota_observations,
    _read_latest,
    _resolved,
    _source_health_observation,
    parse_training_logs,
    summarize_health_trajectories,
)
from check_agent_boundary_rl_pilot import (
    PROFILE,
    PROTOCOL_SIGNATURE,
    TURN_CREDIT_VERSION,
)
from domain_expert_checkpoint import (
    DOMAINS,
    EXPERT_ITERATIONS,
    LINEAGE_VERSION,
    SOURCE_ITERATION,
    expected_lineage,
    quota_for_domain,
    read_json,
    read_sampler_state,
)


GATE_VERSION = "boundary-v2-domain-expert-prefix-health-v1"
HEALTH_POLICY = "kl-diagnostic-v1"
MIXED_QUOTA = {"airline": 2, "retail": 2, "telecom": 2}
EXPECTED_SAMPLES_PER_GROUP = 8


def _sampler_cursor_advanced(
    source: dict[str, Any], target: dict[str, Any], domain: str
) -> bool:
    source_epoch = source["domain_epochs"][domain]
    target_epoch = target["domain_epochs"][domain]
    return bool(
        target_epoch > source_epoch
        or (
            target_epoch == source_epoch
            and target["domain_offsets"][domain] > source["domain_offsets"][domain]
        )
    )


def evaluate_prefix_health(
    *,
    variant: str,
    iteration: int,
    domain: str | None,
    checkpoint_root: Path,
    source_checkpoint_root: Path,
    source_health: dict[str, Any],
    source_health_path: Path,
    lineage: dict[str, Any],
    training_logs: list[Path],
    trajectory_paths: list[Path],
) -> dict[str, Any]:
    if variant not in {"domain-expert", "mixed-control"}:
        raise ValueError(f"unsupported prefix-health variant: {variant}")
    if iteration not in EXPERT_ITERATIONS:
        raise ValueError(f"iteration must be one of {EXPERT_ITERATIONS}")
    if variant == "domain-expert" and domain not in DOMAINS:
        raise ValueError("domain-expert health requires an explicit expert domain")
    if variant == "mixed-control" and domain is not None:
        raise ValueError("mixed-control health must not specify an expert domain")

    checkpoint_root = checkpoint_root.resolve()
    source_checkpoint_root = source_checkpoint_root.resolve()
    source_sha = hashlib.sha256(source_health_path.read_bytes()).hexdigest()
    quota = quota_for_domain(domain) if domain is not None else MIXED_QUOTA
    training = parse_training_logs(
        training_logs,
        start=100,
        end=iteration,
        checkpoint_root=checkpoint_root,
        initial_checkpoint_root=source_checkpoint_root,
        initial_iteration=SOURCE_ITERATION,
        mode="final",
        checkpoint_scoped=True,
    )
    trajectories = summarize_health_trajectories(
        trajectory_paths,
        effective_steps=set(training["expected_steps"]),
    )
    quota_passed, quota_observed = _quota_observations(training, quota=quota)
    source_passed, source_observed = _source_health_observation(
        source_health,
        source_checkpoint_root=source_checkpoint_root,
        source_health_sha256=source_sha,
    )

    source_sampler = read_sampler_state(source_checkpoint_root, SOURCE_ITERATION)
    target_sampler = read_sampler_state(checkpoint_root, iteration)
    if variant == "domain-expert":
        expected = expected_lineage(
            domain=domain,
            source_root=source_checkpoint_root,
            destination_root=checkpoint_root,
            source_gate_path=source_health_path,
            source_sampler_state=source_sampler,
        )
        lineage_passed = lineage == expected
        lineage_observed: dict[str, Any] | None = {
            "expected": expected,
            "observed": lineage,
        }
    else:
        lineage_passed, lineage_observed = _lineage_observation(
            lineage,
            source_checkpoint_root=source_checkpoint_root,
            destination_checkpoint_root=checkpoint_root,
            source_health_artifact=source_health_path,
            source_health_sha256=source_sha,
        )

    inactive_unchanged = True
    active_advanced = True
    if domain is not None:
        inactive_unchanged = all(
            target_sampler["domain_offsets"][name]
            == source_sampler["domain_offsets"][name]
            and target_sampler["domain_epochs"][name]
            == source_sampler["domain_epochs"][name]
            for name in DOMAINS
            if name != domain
        )
        active_advanced = _sampler_cursor_advanced(source_sampler, target_sampler, domain)
    sampler_passed = bool(
        target_sampler["dataset_fingerprint"]
        == source_sampler["dataset_fingerprint"]
        and inactive_unchanged
        and active_advanced
        and target_sampler["sample_group_index"]
        > source_sampler["sample_group_index"]
        and target_sampler["sample_index"] > source_sampler["sample_index"]
    )

    checks: list[dict[str, Any]] = []

    def add(
        name: str,
        passed: bool,
        observed: Any,
        requirement: str,
        *,
        enforced: bool = True,
    ) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "enforced": enforced,
                "observed": observed,
                "requirement": requirement,
            }
        )

    actual_root = _resolved(checkpoint_root)
    add("checkpoint_root", True, actual_root, f"== {actual_root}")
    add(
        "source_health",
        source_passed,
        source_observed,
        "passing immutable long100 iter99 Gate and complete source checkpoint",
    )
    add(
        "cross_root_lineage",
        lineage_passed,
        lineage_observed,
        "exact source/destination roots, source Gate SHA, domain, quota, fingerprint, cursors, and horizon",
    )
    latest = _read_latest(checkpoint_root)
    save_event = {"path": actual_root, "iteration": iteration}
    exact_save = save_event in training["save_events"]
    add(
        "checkpoint_latest",
        latest is not None and latest >= iteration and exact_save,
        {
            "root_current_latest": latest,
            "target_iteration": iteration,
            "exact_target_save_event": exact_save,
        },
        "root latest is at least the target and the scoped logs contain its successful save",
    )
    checkpoint_state = _checkpoint_state(checkpoint_root, iteration, require_shards=True)
    add(
        "checkpoint_state_complete",
        _checkpoint_state_complete(checkpoint_state, require_shards=True),
        checkpoint_state,
        "metadata, optimizer/common state, rollout state, and nonempty distcp shards exist",
    )
    add(
        "sampler_state",
        sampler_passed,
        {"restored": source_sampler, "target": target_sampler},
        "dataset fingerprint is unchanged, source cursors are restored, and only the expert cursor advances",
    )
    add(
        "recovery_state",
        all(log["recovery_passed"] for log in training["logs"]),
        training["logs"],
        "the first attempt loads iter99 and retries load the latest complete expert checkpoint",
    )
    add(
        "recovery_integrity",
        all(log["recovery_integrity_passed"] for log in training["logs"]),
        training["logs"],
        "model, optimizer, and RNG recovery never ignores RNG state",
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
            "missing_required_step_metrics": training["missing_required_step_metrics"],
            "unexpected_steps": training["unexpected_steps"],
        },
        f"updates 100-{iteration} contain {REQUIRED_STEP_METRICS} plus truncation",
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
        "no OOM, NaN, Inf, parse failure, or non-finite metric in the checkpoint prefix",
    )
    add(
        "last_10_k2_kl",
        training["last_10_k2_kl_mean"] is not None
        and training["last_10_k2_kl_mean"] < 0.10,
        training["last_10_k2_kl_mean"],
        "diagnostic only: < 0.10 over the checkpoint's last 10 updates",
        enforced=False,
    )
    add(
        "no_consecutive_high_kl",
        not training["consecutive_high_kl_steps"],
        training["consecutive_high_kl_steps"],
        "diagnostic only: no adjacent updates both have K2 KL >= 0.20",
        enforced=False,
    )
    add(
        "domain_quota",
        quota_passed,
        quota_observed,
        f"every scoped update accepts {quota} with same-domain replacements",
    )
    expected_rows = (iteration - 99) * sum(quota.values()) * EXPECTED_SAMPLES_PER_GROUP
    add(
        "trajectory_data_complete",
        trajectories["total_rows"] >= expected_rows
        and all(item["rows"] > 0 for item in trajectories["artifacts"].values())
        and trajectories["missing_rollout_ids"] == 0,
        {
            "rows": trajectories["total_rows"],
            "minimum": expected_rows,
            "missing_rollout_ids": trajectories["missing_rollout_ids"],
            "duplicate_rollout_ids": trajectories["duplicate_rollout_ids"],
            "ignored_rows": trajectories["ignored_rows"],
        },
        "K=8 effective samples per accepted group with ordered retry deduplication",
    )
    target_contracts = trajectories["agent_contract_signatures"]
    contract_passed = bool(
        not trajectories["wrong_protocol_profiles"]
        and not trajectories["wrong_protocol_signatures"]
        and trajectories["missing_contract_signatures"] == 0
        and set(target_contracts) == {domain}
        and len(target_contracts[domain]) == 1
    )
    add(
        "trajectory_contract",
        contract_passed,
        {
            "wrong_profiles": trajectories["wrong_protocol_profiles"],
            "wrong_signatures": trajectories["wrong_protocol_signatures"],
            "missing_signatures": trajectories["missing_contract_signatures"],
            "agent_contract_signatures": target_contracts,
        },
        "all rows use the signed Agent-owned Contract",
    )
    source_contracts = (source_health.get("trajectories") or {}).get(
        "agent_contract_signatures"
    )
    expected_source_contracts = (
        {domain: source_contracts.get(domain, [])}
        if isinstance(source_contracts, dict)
        else None
    )
    add(
        "trajectory_contract_continuity",
        expected_source_contracts == target_contracts,
        {
            "source": expected_source_contracts,
            "target": target_contracts,
        },
        "the expert-domain Contract signature equals the immutable iter99 source",
    )
    add(
        "trajectory_turn_credit_alignment",
        not trajectories["turn_credit_alignment_errors"],
        trajectories["turn_credit_alignment_errors"],
        "turn-credit-v1 spans, loss masks, and penalty vectors align",
    )

    status = "pass" if all(check["passed"] for check in checks if check["enforced"]) else "fail"
    training_output = {
        key: value
        for key, value in training.items()
        if key not in {"_raw_metrics", "_metric_occurrences"}
    }
    return {
        "status": status,
        "gate_version": GATE_VERSION,
        "health_policy": HEALTH_POLICY,
        "variant": variant,
        "stage": f"{domain + '-' if domain else 'mixed-'}iter{iteration}",
        "expert_domain": domain,
        "domain_quota": quota,
        "checkpoint_root": actual_root,
        "expected_checkpoint_root": actual_root,
        "source_checkpoint_root": _resolved(source_checkpoint_root),
        "destination_checkpoint_root": actual_root,
        "source_health_gate": str(source_health_path.resolve()),
        "source_health_gate_sha256": source_sha,
        "dataset_fingerprint": source_sampler["dataset_fingerprint"],
        "restored_domain_cursors": {
            "domain_offsets": source_sampler["domain_offsets"],
            "domain_epochs": source_sampler["domain_epochs"],
        },
        "target_checkpoint": str(
            (checkpoint_root / f"iter_{iteration:07d}").resolve()
        ),
        "root_current_latest": latest,
        "expected_latest": iteration,
        "expected_steps": [100, iteration],
        "protocol_profile": PROFILE,
        "agent_protocol_signature": PROTOCOL_SIGNATURE,
        "turn_credit_version": TURN_CREDIT_VERSION,
        "lineage_version": (
            LINEAGE_VERSION if variant == "domain-expert" else lineage.get("lineage_version")
        ),
        "kl_enforced": False,
        "checks": checks,
        "training": training_output,
        "trajectories": trajectories,
        "diagnostics_only": {
            "kl_checks": list(KL_CHECK_NAMES),
            "framework_truncated": trajectories["framework_truncated"],
            "termination_reasons": trajectories["termination_reasons"],
            "behavior": trajectories["behavior_diagnostics"],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("domain-expert", "mixed-control"), required=True)
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--iteration", type=int, choices=EXPERT_ITERATIONS, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-health", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, action="append", required=True)
    parser.add_argument("--trajectories", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_prefix_health(
            variant=args.variant,
            iteration=args.iteration,
            domain=args.domain,
            checkpoint_root=args.checkpoint_root,
            source_checkpoint_root=args.source_checkpoint_root,
            source_health=read_json(args.source_health),
            source_health_path=args.source_health,
            lineage=read_json(args.lineage),
            training_logs=args.training_log,
            trajectory_paths=args.trajectories,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
