#!/usr/bin/env python3
"""Evaluate tau2 RL promotion gates and select the final eligible checkpoint."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from protocol_profiles import protocol_signature  # noqa: E402


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TRAIN_METRIC_RE = re.compile(r"\bstep (?P<step>\d+): (?P<payload>\{.*\})\s*$")
ROLLOUT_METRIC_RE = re.compile(
    r"\b(?:perf |rollout )(?P<step>\d+): (?P<payload>\{.*\})\s*$"
)
FATAL_TRAINING_PATTERNS = {
    # Ray logs contain repr-escaped prompt text.  Without excluding a leading
    # backslash, ``\nAn ownership ...`` is read case-insensitively as ``nan``.
    "nan": re.compile(r"(?<![A-Za-z_\\])nan(?![A-Za-z_])", re.IGNORECASE),
    "inf": re.compile(
        r"(?<![A-Za-z_\\])(?:[-+]?inf|infinity)(?![A-Za-z_])",
        re.IGNORECASE,
    ),
    "oom": re.compile(
        r"cuda out of memory|outofmemoryerror|\boom\b",
        re.IGNORECASE,
    ),
}
PROTOCOL_PROFILE = "dependency-safe-multi"
PROTOCOL_SIGNATURE = protocol_signature(PROTOCOL_PROFILE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_payload(raw: str) -> dict[str, Any]:
    try:
        payload = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid training metric payload: {raw[:200]}") from exc
    if not isinstance(payload, dict):
        raise ValueError("training metric payload is not a dictionary")
    return payload


def parse_training_log(path: Path, *, expected_latest: int) -> dict[str, Any]:
    fatal_matches = {label: 0 for label in FATAL_TRAINING_PATTERNS}
    metrics: dict[int, dict[str, Any]] = {}
    # Training logs can be hundreds of MiB because they include Ray worker
    # output.  Scan one line at a time so promotion checks have bounded memory.
    with path.open(encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = ANSI_RE.sub("", raw_line)
            for label, pattern in FATAL_TRAINING_PATTERNS.items():
                fatal_matches[label] += len(pattern.findall(line))
            for pattern in (TRAIN_METRIC_RE, ROLLOUT_METRIC_RE):
                match = pattern.search(line)
                if not match:
                    continue
                step = int(match.group("step"))
                metrics.setdefault(step, {}).update(
                    _metric_payload(match.group("payload"))
                )
                break

    expected_steps = list(range(max(0, expected_latest - 9), expected_latest + 1))
    missing_steps = [step for step in expected_steps if step not in metrics]
    missing_kl = [step for step in expected_steps if "train/kl_loss" not in metrics.get(step, {})]
    missing_truncation = [
        step
        for step in expected_steps
        if "rollout/truncated" not in metrics.get(step, {})
        and "rollout/truncated_ratio" not in metrics.get(step, {})
    ]

    kl_values = [
        float(metrics[step]["train/kl_loss"])
        for step in expected_steps
        if "train/kl_loss" in metrics.get(step, {})
    ]
    truncation_values = [
        float(
            metrics[step].get(
                "rollout/truncated",
                metrics[step].get("rollout/truncated_ratio"),
            )
        )
        for step in expected_steps
        if "rollout/truncated" in metrics.get(step, {})
        or "rollout/truncated_ratio" in metrics.get(step, {})
    ]
    nonfinite_metrics = []
    for step in expected_steps:
        for key, value in metrics.get(step, {}).items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                nonfinite_metrics.append(f"{step}:{key}")

    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "expected_steps": expected_steps,
        "observed_latest_step": max(metrics) if metrics else None,
        "missing_steps": missing_steps,
        "missing_kl_steps": missing_kl,
        "missing_truncation_steps": missing_truncation,
        "fatal_matches": fatal_matches,
        "nonfinite_metrics": nonfinite_metrics,
        "last_10_k2_kl_mean": sum(kl_values) / len(kl_values) if kl_values else None,
        "last_10_truncation_mean": (
            sum(truncation_values) / len(truncation_values)
            if truncation_values
            else None
        ),
        # Keep the bounded final-ten-step payloads so specialized gates can
        # validate pipeline metrics such as exact domain quotas without
        # materializing or rescanning a potentially huge Ray log.
        "expected_step_metrics": {
            str(step): metrics.get(step, {}) for step in expected_steps
        },
    }


def summarize_recent_trajectories(path: Path, *, window: int) -> dict[str, Any]:
    rows: deque[dict[str, Any]] = deque(maxlen=window)
    total_rows = 0
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
            rows.append(row)
            total_rows += 1

    selected = list(rows)
    malformed = nonexistent = repetition = max_steps = 0
    wrong_profiles: dict[str, int] = {}
    wrong_signatures: dict[str, int] = {}
    for row in selected:
        metadata = row.get("metadata") or {}
        profile = str(metadata.get("tau2_agent_protocol_profile") or "<missing>")
        if profile != PROTOCOL_PROFILE:
            wrong_profiles[profile] = wrong_profiles.get(profile, 0) + 1
        signature = str(
            metadata.get("tau2_agent_protocol_signature") or "<missing>"
        )
        if signature != PROTOCOL_SIGNATURE:
            wrong_signatures[signature] = wrong_signatures.get(signature, 0) + 1
        signals = metadata.get("tau2_field_reward_signals") or {}
        counts = signals.get("counts") or {}
        malformed += int(int(counts.get("malformed_json", 0)) > 0)
        nonexistent += int(int(counts.get("nonexistent_tool", 0)) > 0)
        repetition += int(counts.get("repetition", 0))
        max_steps += int(
            metadata.get("tau2_termination_reason") == "max_steps"
            or int(counts.get("max_steps", 0)) > 0
        )

    denominator = len(selected)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "total_rows": total_rows,
        "window": window,
        "selected_rows": denominator,
        "wrong_protocol_profiles": wrong_profiles,
        "wrong_protocol_signatures": wrong_signatures,
        "malformed_json_fraction": malformed / denominator if denominator else None,
        "nonexistent_tool_fraction": nonexistent / denominator if denominator else None,
        "repeated_identical_calls_per_trajectory": (
            repetition / denominator if denominator else None
        ),
        "max_steps_fraction": max_steps / denominator if denominator else None,
    }


def _termination_count(model: dict[str, Any], reason: str) -> int:
    reasons = model["behavior"]["overall"]["termination_reasons"]
    return int(reasons.get(reason, 0))


def evaluate_gate(
    *,
    comparison: dict[str, Any],
    candidate: str,
    baseline: str,
    prior: str | None,
    gate_mode: str,
    training: dict[str, Any],
    trajectories: dict[str, Any],
    checkpoint_root: Path,
    expected_latest: int,
) -> dict[str, Any]:
    models = comparison.get("models") or {}
    required_labels = [candidate, baseline] + ([prior] if prior else [])
    missing_labels = [label for label in required_labels if label not in models]
    if missing_labels:
        raise ValueError(f"comparison is missing model labels: {missing_labels}")
    if comparison.get("status") != "pass":
        raise ValueError("comparison artifact did not pass protocol validation")
    if comparison.get("protocol_profile") != PROTOCOL_PROFILE:
        raise ValueError("comparison artifact uses the wrong protocol profile")
    if comparison.get("agent_protocol_signature") != PROTOCOL_SIGNATURE:
        raise ValueError("comparison artifact uses the wrong protocol content signature")
    if gate_mode == "promote100" and not prior:
        raise ValueError("promote100 requires --prior")

    latest_file = checkpoint_root / "latest_checkpointed_iteration.txt"
    latest_raw = latest_file.read_text(encoding="utf-8").strip() if latest_file.is_file() else None
    latest = int(latest_raw) if latest_raw and latest_raw.isdigit() else None
    candidate_model = models[candidate]
    baseline_model = models[baseline]

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

    if gate_mode == "promote100":
        for metric in ("pass_at_1", "pass_at_4_any"):
            observed = candidate_model["overall"][metric]
            references = {
                baseline: baseline_model["overall"][metric],
                prior: models[prior]["overall"][metric],
            }
            add(
                f"strict_{metric}_improvement",
                all(observed > value for value in references.values()),
                {candidate: observed, **references},
                f"{candidate} strictly exceeds {baseline} and {prior}",
            )

    pass_power_delta = (
        candidate_model["overall"]["pass_power_4"]
        - baseline_model["overall"]["pass_power_4"]
    )
    add(
        "pass_power_4_noninferiority",
        pass_power_delta >= -0.02,
        pass_power_delta,
        f"{candidate}-{baseline} >= -0.02",
    )
    for domain, candidate_metrics in candidate_model["by_domain"].items():
        delta = (
            candidate_metrics["pass_at_1"]
            - baseline_model["by_domain"][domain]["pass_at_1"]
        )
        add(
            f"{domain}_pass_at_1_noninferiority",
            delta >= -0.05,
            delta,
            f"{candidate}-{baseline} >= -0.05",
        )

    too_many_errors = _termination_count(candidate_model, "too_many_errors")
    max_steps = _termination_count(candidate_model, "max_steps")
    add("eval_too_many_errors", too_many_errors <= 21, too_many_errors, "<= 21")
    add("eval_max_steps", max_steps <= 12, max_steps, "<= 12")

    add("checkpoint_latest", latest == expected_latest, latest, f"== {expected_latest}")
    add(
        "training_last_10_complete",
        not training["missing_steps"]
        and not training["missing_kl_steps"]
        and not training["missing_truncation_steps"],
        {
            "missing_steps": training["missing_steps"],
            "missing_kl_steps": training["missing_kl_steps"],
            "missing_truncation_steps": training["missing_truncation_steps"],
        },
        "all final 10 train and rollout metrics are present",
    )
    add(
        "training_no_nan_inf_oom",
        not any(training["fatal_matches"].values())
        and not training["nonfinite_metrics"],
        {
            "fatal_matches": training["fatal_matches"],
            "nonfinite_metrics": training["nonfinite_metrics"],
        },
        "zero NaN/Inf/OOM matches and finite final metrics",
    )
    kl_mean = training["last_10_k2_kl_mean"]
    truncation_mean = training["last_10_truncation_mean"]
    add("training_k2_kl", kl_mean is not None and kl_mean < 0.10, kl_mean, "< 0.10")
    add(
        "training_truncation",
        truncation_mean is not None and truncation_mean < 0.01,
        truncation_mean,
        "< 0.01",
    )

    add(
        "trajectory_window_complete",
        trajectories["selected_rows"] == trajectories["window"],
        trajectories["selected_rows"],
        f"== {trajectories['window']}",
    )
    add(
        "trajectory_protocol_profile",
        not trajectories["wrong_protocol_profiles"],
        trajectories["wrong_protocol_profiles"],
        f"all rows use {PROTOCOL_PROFILE}",
    )
    add(
        "trajectory_protocol_signature",
        not trajectories["wrong_protocol_signatures"],
        trajectories["wrong_protocol_signatures"],
        f"all rows use {PROTOCOL_SIGNATURE}",
    )
    trajectory_thresholds = {
        "malformed_json_fraction": (0.10, "< 0.10"),
        "nonexistent_tool_fraction": (0.05, "< 0.05"),
        "repeated_identical_calls_per_trajectory": (0.5, "< 0.5"),
        "max_steps_fraction": (0.02, "< 0.02"),
    }
    for name, (threshold, rendered) in trajectory_thresholds.items():
        value = trajectories[name]
        add(name, value is not None and value < threshold, value, rendered)

    status = "pass" if all(check["passed"] for check in checks) else "fail"
    return {
        "status": status,
        "gate_mode": gate_mode,
        "candidate": candidate,
        "baseline": baseline,
        "prior": prior,
        "checkpoint_root": str(checkpoint_root.resolve()),
        "expected_latest": expected_latest,
        "protocol_profile": comparison.get("protocol_profile"),
        "agent_protocol_signature": comparison.get("agent_protocol_signature"),
        "comparison_mode": comparison.get("comparison_mode"),
        "legacy_models": comparison.get("legacy_models", {}),
        "comparability_warning": comparison.get("comparability_warning"),
        "checks": checks,
        "training": training,
        "trajectories": trajectories,
    }


def select_checkpoint(
    comparison: dict[str, Any],
    eligibility: dict[str, dict[str, Any]],
    iterations: dict[str, int],
) -> dict[str, Any]:
    models = comparison.get("models") or {}
    if set(eligibility) != set(iterations):
        raise ValueError("eligibility and iteration labels differ")
    candidates = []
    rejected = {}
    for label, gate in eligibility.items():
        if label not in models:
            raise ValueError(f"comparison is missing checkpoint label {label!r}")
        if gate.get("candidate") != label:
            raise ValueError(f"eligibility artifact candidate mismatch for {label}")
        if gate.get("status") != "pass":
            rejected[label] = "eligibility_gate_failed"
            continue
        metrics = models[label]["overall"]
        candidates.append(
            {
                "label": label,
                "iteration": iterations[label],
                "checkpoint_root": gate.get("checkpoint_root"),
                "metrics": metrics,
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["metrics"]["pass_at_1"],
            -row["metrics"]["pass_at_4_any"],
            -row["metrics"]["pass_power_4"],
            row["iteration"],
        )
    )
    return {
        "status": "pass" if candidates else "fail",
        "selection_order": [
            "pass_at_1",
            "pass_at_4_any",
            "pass_power_4",
            "earlier_iteration_on_exact_tie",
        ],
        "selected": candidates[0] if candidates else None,
        "eligible_ranked": candidates,
        "rejected": rejected,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_label_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"expected LABEL=PATH: {spec}")
    label, raw_path = spec.split("=", 1)
    if not label or not raw_path:
        raise ValueError(f"expected LABEL=PATH: {spec}")
    return label, Path(raw_path)


def parse_label_iteration(spec: str) -> tuple[str, int]:
    label, raw_iteration = parse_label_path(spec)
    return label, int(str(raw_iteration))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--comparison", type=Path, required=True)
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--prior", default=None)
    gate.add_argument(
        "--gate-mode",
        choices=("promote100", "eligibility"),
        default="promote100",
    )
    gate.add_argument("--training-log", type=Path, required=True)
    gate.add_argument("--trajectory-dump", type=Path, required=True)
    gate.add_argument("--trajectory-window", type=int, default=384)
    gate.add_argument("--checkpoint-root", type=Path, required=True)
    gate.add_argument("--expected-latest", type=int, required=True)
    gate.add_argument("--output", type=Path, required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--comparison", type=Path, required=True)
    select.add_argument("--eligibility", action="append", required=True)
    select.add_argument("--iteration", action="append", required=True)
    select.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    if args.command == "gate":
        if args.trajectory_window < 1:
            raise SystemExit("--trajectory-window must be positive")
        training = parse_training_log(
            args.training_log,
            expected_latest=args.expected_latest,
        )
        trajectories = summarize_recent_trajectories(
            args.trajectory_dump,
            window=args.trajectory_window,
        )
        report = evaluate_gate(
            comparison=comparison,
            candidate=args.candidate,
            baseline=args.baseline,
            prior=args.prior,
            gate_mode=args.gate_mode,
            training=training,
            trajectories=trajectories,
            checkpoint_root=args.checkpoint_root,
            expected_latest=args.expected_latest,
        )
    else:
        eligibility = {
            label: json.loads(path.read_text(encoding="utf-8"))
            for label, path in map(parse_label_path, args.eligibility)
        }
        iterations = dict(map(parse_label_iteration, args.iteration))
        report = select_checkpoint(comparison, eligibility, iterations)

    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
