#!/usr/bin/env python3
"""Apply the turn-aware-RL-v3 gate and select the tau2 boundary SFT arm."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PROFILE = "agent-owned-dependency-safe-multi"
GATE_VERSION = "turn-aware-rl-v3"
EXPECTED_TELECOM_TRAJECTORIES = 160
EXPECTED_SEEDS = {300, 301}
DEFAULT_CAPABILITY_BASELINES = ("raw_instruct", "old_sft", "current_sft")
THRESHOLDS = {
    "temperature_0": {"namespace_attempts_max": 0, "cases": 60},
    "temperature_0_6": {"namespace_attempts_max": 2, "cases": 240},
    "per_seed_namespace_affected_max": 60,
    "per_seed_namespace_attempts_max": 350,
    "per_seed_namespace_termination_max": 10,
    "per_domain_pass_at_1_gap_max": 0.08,
    "pass_power_4_paired_ci95_lower_min": -0.02,
}
PREVIOUS_STRICT_GATE = {
    "status": "fail",
    "thresholds": {
        "per_seed_namespace_affected_max": 8,
        "per_seed_namespace_attempts_max": 16,
        "per_seed_namespace_termination_max": 0,
        "per_domain_pass_at_1_gap_max": 0.05,
    },
    "reason_for_v3_relaxation": (
        "The prior strict gate rejected both completed SFT arms. The v3 limits "
        "admit only Contract + boundary as the measured RL initialization while "
        "retaining the direct namespace probe, two-seed signed protocol, contract, "
        "task-set, and capability checks; turn-aware RL must then reduce the "
        "remaining namespace errors under its own non-regression gates."
    ),
}


def _specs(values: list[str], *, option: str) -> dict[str, list[Path]]:
    parsed: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must be LABEL=PATH: {value!r}")
        label, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError(f"invalid label in {option}: {label!r}")
        parsed[label].append(Path(raw_path).resolve())
    return dict(parsed)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _paired_row(report: dict[str, Any], candidate: str, baseline: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in report.get("paired_deltas") or []
            if row.get("candidate") == candidate
            and row.get("reference") == baseline
            and row.get("metric") == "pass_power_4"
        ),
        None,
    )


def capability_baseline_values(
    models: dict[str, Any],
    *,
    domain: str,
    labels: tuple[str, ...],
) -> list[float]:
    """Return only the fixed pre-v2 baselines used by the capability gate."""

    values: list[float] = []
    for label in labels:
        model = models.get(label)
        metrics = (model or {}).get("by_domain", {}).get(domain)
        value = (metrics or {}).get("pass_at_1")
        if not isinstance(value, (int, float)):
            raise ValueError(f"missing {domain} pass_at_1 for capability baseline {label}")
        values.append(float(value))
    return values


def evaluate_candidate(
    *,
    label: str,
    checkpoint: Path,
    probe_path: Path,
    eval_paths: list[Path],
    comparison_paths: list[Path],
    baseline: str,
    capability_baselines: tuple[str, ...],
) -> dict[str, Any]:
    reasons: list[str] = []
    probe = _read(probe_path)
    if probe.get("status") != "pass":
        reasons.append("namespace_probe_failed")
    summaries = probe.get("summaries") or {}
    if (summaries.get("temperature_0") or {}).get("namespace_attempts") != THRESHOLDS[
        "temperature_0"
    ]["namespace_attempts_max"]:
        reasons.append("temperature_0_not_0_of_60")
    if (summaries.get("temperature_0") or {}).get("cases") != THRESHOLDS[
        "temperature_0"
    ]["cases"]:
        reasons.append("temperature_0_case_count")
    if (summaries.get("temperature_0_6") or {}).get(
        "namespace_attempts", 10**9
    ) > THRESHOLDS["temperature_0_6"]["namespace_attempts_max"]:
        reasons.append("temperature_0_6_above_2_of_240")
    if (summaries.get("temperature_0_6") or {}).get("cases") != THRESHOLDS[
        "temperature_0_6"
    ]["cases"]:
        reasons.append("temperature_0_6_case_count")
    if probe.get("user_tool_count") != 30:
        reasons.append("probe_user_tool_count")
    probe_checkpoint = Path(str(probe.get("checkpoint") or "")).resolve()
    if checkpoint.resolve() not in (probe_checkpoint, *probe_checkpoint.parents):
        reasons.append("probe_checkpoint_mismatch")

    seeds = set()
    affected_total = 0
    attempts_total = 0
    contract_signatures = {probe.get("agent_contract_signature")}
    for path in eval_paths:
        summary = _read(path)
        if summary.get("agent_protocol_profile") != PROFILE:
            reasons.append(f"unsigned_eval:{path.name}")
            continue
        seed = summary.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            reasons.append(f"missing_eval_seed:{path.name}")
        else:
            seeds.add(seed)
        telecom = (summary.get("domains") or {}).get("telecom") or {}
        namespace = telecom.get("namespace") or {}
        contract = telecom.get("agent_contract") or {}
        contract_signatures.add(contract.get("agent_contract_signature"))
        trajectories = namespace.get("trajectory_count")
        affected = namespace.get("affected_trajectory_count")
        attempts = namespace.get("raw_attempt_count")
        terminations = namespace.get("namespace_attributed_termination_count")
        if trajectories != EXPECTED_TELECOM_TRAJECTORIES:
            reasons.append(f"telecom_trajectory_count:{path.name}:{trajectories}")
        if not isinstance(affected, int) or isinstance(affected, bool):
            reasons.append(f"telecom_affected_invalid:{path.name}:{affected}")
        else:
            affected_total += affected
            if affected > THRESHOLDS["per_seed_namespace_affected_max"]:
                reasons.append(f"telecom_affected_above_60:{path.name}:{affected}")
        if not isinstance(attempts, int) or isinstance(attempts, bool):
            reasons.append(f"telecom_attempts_invalid:{path.name}:{attempts}")
        else:
            attempts_total += attempts
            if attempts > THRESHOLDS["per_seed_namespace_attempts_max"]:
                reasons.append(f"telecom_attempts_above_350:{path.name}:{attempts}")
        if (
            not isinstance(terminations, int)
            or isinstance(terminations, bool)
            or terminations > THRESHOLDS["per_seed_namespace_termination_max"]
        ):
            reasons.append(
                f"namespace_termination_above_10:{path.name}:{terminations}"
            )
    if seeds != EXPECTED_SEEDS:
        reasons.append(f"fixed_eval_seeds_mismatch:{sorted(seeds)}")
    if None in contract_signatures or len(contract_signatures) != 1:
        reasons.append("contract_signature_mismatch")

    pass_power_values = []
    pass_at_1_values = []
    comparison_seeds = set()
    for path in comparison_paths:
        report = _read(path)
        if (
            report.get("status") != "pass"
            or report.get("comparison_mode") != "strict-signed"
            or report.get("protocol_profile") != PROFILE
            or report.get("candidate") != label
        ):
            reasons.append(f"invalid_comparison:{path.name}")
            continue
        comparison_seed = (report.get("protocol_signature") or {}).get("seed")
        if not isinstance(comparison_seed, int) or isinstance(comparison_seed, bool):
            reasons.append(f"missing_comparison_seed:{path.name}")
        else:
            comparison_seeds.add(comparison_seed)
        paired = _paired_row(report, label, baseline)
        if paired is None or (paired.get("ci95") or [-1])[0] < THRESHOLDS[
            "pass_power_4_paired_ci95_lower_min"
        ]:
            reasons.append(f"pass_power_4_ci_below_minus_2pp:{path.name}")
        models = report.get("models") or {}
        candidate_model = models.get(label) or {}
        if not candidate_model:
            reasons.append(f"candidate_missing_from_comparison:{path.name}")
            continue
        for domain, candidate_metrics in (candidate_model.get("by_domain") or {}).items():
            try:
                baseline_values = capability_baseline_values(
                    models,
                    domain=domain,
                    labels=capability_baselines,
                )
            except ValueError as exc:
                reasons.append(f"invalid_capability_baseline:{path.name}:{exc}")
                continue
            if baseline_values:
                capability_gap = max(baseline_values) - float(
                    candidate_metrics["pass_at_1"]
                )
                if capability_gap > THRESHOLDS["per_domain_pass_at_1_gap_max"] + 1e-12:
                    reasons.append(
                        f"{domain}_pass_at_1_below_best_by_8pp:{path.name}"
                    )
        overall = candidate_model.get("overall") or {}
        if isinstance(overall.get("pass_power_4"), (int, float)):
            pass_power_values.append(float(overall["pass_power_4"]))
        if isinstance(overall.get("pass_at_1"), (int, float)):
            pass_at_1_values.append(float(overall["pass_at_1"]))
        report_signature = (report.get("agent_contract_signatures") or {}).get("telecom")
        if report_signature and {report_signature} != contract_signatures:
            reasons.append(f"comparison_contract_signature_mismatch:{path.name}")

    if comparison_seeds != EXPECTED_SEEDS:
        reasons.append(f"fixed_comparison_seeds_mismatch:{sorted(comparison_seeds)}")
    if comparison_seeds != seeds:
        reasons.append("comparison_eval_seed_mismatch")
    return {
        "label": label,
        "status": "pass" if not reasons else "fail",
        "reasons": sorted(set(reasons)),
        "checkpoint_root": str(checkpoint.resolve()),
        "probe": str(probe_path),
        "eval_summaries": [str(path) for path in eval_paths],
        "comparisons": [str(path) for path in comparison_paths],
        "eval_seeds": sorted(seeds),
        "comparison_seeds": sorted(comparison_seeds),
        "namespace_affected_trajectories": affected_total,
        "namespace_attempts": attempts_total,
        "mean_pass_power_4": (
            sum(pass_power_values) / len(pass_power_values) if pass_power_values else 0.0
        ),
        "mean_pass_at_1": (
            sum(pass_at_1_values) / len(pass_at_1_values) if pass_at_1_values else 0.0
        ),
        "agent_contract_signature": next(iter(contract_signatures), None),
    }


def rank_eligible(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return passing arms in the deterministic promotion order."""

    eligible = [decision for decision in decisions if decision["status"] == "pass"]
    eligible.sort(
        key=lambda decision: (
            decision["namespace_affected_trajectories"],
            decision["namespace_attempts"],
            -decision["mean_pass_power_4"],
            -decision["mean_pass_at_1"],
            0 if decision["label"] == "contract-only" else 1,
        )
    )
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--probe", action="append", required=True)
    parser.add_argument("--eval-summary", action="append", default=[])
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument("--baseline", default="old_sft")
    parser.add_argument("--capability-baseline", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = args.candidate
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate labels must be unique")
    checkpoints = _specs(args.checkpoint, option="--checkpoint")
    probes = _specs(args.probe, option="--probe")
    evals = _specs(args.eval_summary, option="--eval-summary")
    comparisons = _specs(args.comparison, option="--comparison")
    capability_baselines = tuple(
        args.capability_baseline or DEFAULT_CAPABILITY_BASELINES
    )
    if len(capability_baselines) != len(set(capability_baselines)):
        raise ValueError("capability baseline labels must be unique")
    decisions = []
    for label in candidates:
        if len(checkpoints.get(label, [])) != 1 or len(probes.get(label, [])) != 1:
            raise ValueError(f"{label}: require exactly one checkpoint and probe")
        decisions.append(
            evaluate_candidate(
                label=label,
                checkpoint=checkpoints[label][0],
                probe_path=probes[label][0],
                eval_paths=evals.get(label, []),
                comparison_paths=comparisons.get(label, []),
                baseline=args.baseline,
                capability_baselines=capability_baselines,
            )
        )

    eligible = rank_eligible(decisions)
    selected = eligible[0] if eligible else None
    output = {
        "status": "pass" if selected else "fail",
        "gate_version": GATE_VERSION,
        "thresholds": THRESHOLDS,
        "previous_strict_gate": PREVIOUS_STRICT_GATE,
        "protocol_profile": PROFILE,
        "baseline": args.baseline,
        "capability_baselines": list(capability_baselines),
        "decisions": decisions,
        "ranking": [decision["label"] for decision in eligible],
        "selected": selected["label"] if selected else None,
        "selected_checkpoint_root": selected["checkpoint_root"] if selected else None,
        "agent_contract_signature": selected["agent_contract_signature"] if selected else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
