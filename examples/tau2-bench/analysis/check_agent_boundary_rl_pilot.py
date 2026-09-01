#!/usr/bin/env python3
"""Gate tau2 turn-aware RL iter9 and iter19 before the next training stage."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from check_tau2_rl_promotion import parse_training_log  # noqa: E402
from protocol_profiles import protocol_signature  # noqa: E402


PROFILE = "agent-owned-dependency-safe-multi"
PROTOCOL_SIGNATURE = protocol_signature(PROFILE)
TURN_CREDIT_VERSION = "turn-credit-v1"
GATE_VERSION = "turn-aware-rl-v1"
EXPECTED_SEEDS = {300, 301}
PILOT_QUOTAS = {
    "pilot_a": {"telecom": 3, "airline": 2, "retail": 1},
    "pilot_b": {"telecom": 3, "airline": 1, "retail": 2},
}


def _turn_credit_alignment_error(row: dict[str, Any]) -> str | None:
    """Return a reason when dumped training credit is not response aligned."""

    metadata = row.get("metadata") or {}
    if metadata.get("tau2_turn_credit_version") != TURN_CREDIT_VERSION:
        return "wrong_turn_credit_version"
    if not isinstance(metadata.get("tau2_global_score"), (int, float)) or not math.isfinite(
        float(metadata.get("tau2_global_score", math.nan))
    ):
        return "missing_or_nonfinite_global_score"
    response_length = row.get("response_length")
    loss_mask = row.get("loss_mask")
    train_metadata = row.get("train_metadata")
    penalties = (
        train_metadata.get("token_penalties")
        if isinstance(train_metadata, dict)
        else None
    )
    if (
        not isinstance(response_length, int)
        or isinstance(response_length, bool)
        or response_length < 1
        or not isinstance(loss_mask, list)
        or not isinstance(penalties, list)
        or len(loss_mask) != response_length
        or len(penalties) != response_length
    ):
        return "response_vector_length_mismatch"
    # Length equality is checked above; avoid ``zip(strict=True)`` so the
    # offline gate also runs on the Python 3.8 control node.
    for mask, penalty in zip(loss_mask, penalties):
        if not isinstance(penalty, (int, float)) or not math.isfinite(float(penalty)):
            return "nonfinite_token_penalty"
        if float(penalty) < 0 or (not mask and float(penalty) != 0.0):
            return "invalid_or_masked_token_penalty"

    if row.get("remove_sample"):
        if (
            not metadata.get("tau2_turn_credit_invalid")
            or any(loss_mask)
            or any(float(penalty) for penalty in penalties)
        ):
            return "invalid_removed_sample_credit"
        return None

    details = metadata.get("tau2_turn_credits")
    if not isinstance(details, list):
        return "missing_turn_details"
    covered = [False] * response_length
    for detail in details:
        if not isinstance(detail, dict):
            return "invalid_turn_detail"
        response_span = detail.get("response_span")
        penalty = detail.get("penalty")
        if (
            not isinstance(response_span, list)
            or len(response_span) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in response_span)
            or not isinstance(penalty, (int, float))
        ):
            return "invalid_turn_span"
        start, end = response_span
        if not 0 <= start < end <= response_length:
            return "turn_span_outside_response"
        for token_index in range(start, end):
            if not loss_mask[token_index] or covered[token_index]:
                return "turn_span_mask_or_overlap"
            if abs(float(penalties[token_index]) - float(penalty)) > 1e-9:
                return "turn_penalty_span_mismatch"
            covered[token_index] = True
    if any(mask and not covered[index] for index, mask in enumerate(loss_mask)):
        return "unowned_trainable_token"
    return None


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def summarize_trajectories(paths: list[Path], *, window: int) -> dict[str, Any]:
    rows: deque[dict[str, Any]] = deque(maxlen=window)
    total = 0
    totals_by_path: dict[str, int] = {}
    for path in paths:
        path_total = 0
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
                total += 1
                path_total += 1
        totals_by_path[str(path.resolve())] = path_total

    selected = list(rows)
    wrong_profiles = Counter()
    wrong_protocol_signatures = Counter()
    missing_contract_signatures = 0
    namespace_rows = namespace_calls = 0
    malformed = nonexistent = repetition = max_steps = 0
    matched_argument_fields = wrong_argument_fields = 0
    tool_execution_error_rows = 0
    turn_credit_alignment_errors = Counter()
    penalized_turns = clean_turns = removed_rows = 0
    minimum_positive_turn_penalty: float | None = None
    domains = Counter()
    for row in selected:
        metadata = row.get("metadata") or {}
        domains[str(metadata.get("tau2_domain") or metadata.get("domain"))] += 1
        profile = str(metadata.get("tau2_agent_protocol_profile") or "<missing>")
        signature = str(metadata.get("tau2_agent_protocol_signature") or "<missing>")
        if profile != PROFILE:
            wrong_profiles[profile] += 1
        if signature != PROTOCOL_SIGNATURE:
            wrong_protocol_signatures[signature] += 1
        contract_signature = metadata.get("agent_contract_signature")
        missing_contract_signatures += int(
            not isinstance(contract_signature, str)
            or not contract_signature.startswith("sha256:")
        )
        counts = ((metadata.get("tau2_field_reward_signals") or {}).get("counts") or {})
        namespace_count = int(counts.get("wrong_namespace_tool", 0))
        namespace_rows += int(namespace_count > 0)
        namespace_calls += namespace_count
        malformed += int(int(counts.get("malformed_json", 0)) > 0)
        nonexistent += int(int(counts.get("nonexistent_tool", 0)) > 0)
        matched_argument_fields += int(counts.get("matched_argument_fields", 0))
        wrong_argument_fields += int(counts.get("wrong_argument_fields", 0))
        tool_execution_error_rows += int(
            int(counts.get("tool_execution_error", 0)) > 0
        )
        repetition += int(counts.get("repetition", 0))
        max_steps += int(
            metadata.get("tau2_termination_reason") == "max_steps"
            or int(counts.get("max_steps", 0)) > 0
        )
        alignment_error = _turn_credit_alignment_error(row)
        if alignment_error:
            turn_credit_alignment_errors[alignment_error] += 1
        removed_rows += int(bool(row.get("remove_sample")))
        for detail in metadata.get("tau2_turn_credits") or []:
            if isinstance(detail, dict) and float(detail.get("penalty") or 0.0) > 0:
                penalized_turns += 1
                penalty = float(detail["penalty"])
                minimum_positive_turn_penalty = (
                    penalty
                    if minimum_positive_turn_penalty is None
                    else min(minimum_positive_turn_penalty, penalty)
                )
            elif isinstance(detail, dict):
                clean_turns += 1
    denominator = len(selected)
    return {
        "paths": [str(path.resolve()) for path in paths],
        "rows_by_path": totals_by_path,
        "total_rows": total,
        "window": window,
        "selected_rows": denominator,
        "domains": dict(sorted(domains.items())),
        "wrong_protocol_profiles": dict(wrong_profiles),
        "wrong_protocol_signatures": dict(wrong_protocol_signatures),
        "missing_contract_signatures": missing_contract_signatures,
        "turn_credit_alignment_errors": dict(turn_credit_alignment_errors),
        "penalized_turns": penalized_turns,
        "clean_turns": clean_turns,
        "removed_rows": removed_rows,
        "minimum_positive_turn_penalty": minimum_positive_turn_penalty,
        "namespace_affected_fraction": namespace_rows / denominator if denominator else None,
        "namespace_calls_per_trajectory": namespace_calls / denominator if denominator else None,
        "malformed_json_fraction": malformed / denominator if denominator else None,
        "nonexistent_tool_fraction": nonexistent / denominator if denominator else None,
        "wrong_argument_field_rate": (
            wrong_argument_fields / matched_argument_fields
            if matched_argument_fields
            else 0.0
        ),
        "tool_execution_error_fraction": (
            tool_execution_error_rows / denominator if denominator else None
        ),
        "repeated_identical_calls_per_trajectory": repetition / denominator if denominator else None,
        "max_steps_fraction": max_steps / denominator if denominator else None,
    }


def _telecom_namespace(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("agent_protocol_profile") != PROFILE:
        raise ValueError("eval summary does not use the signed Agent-owned protocol")
    namespace = (((summary.get("domains") or {}).get("telecom") or {}).get("namespace"))
    if not isinstance(namespace, dict):
        raise ValueError("eval summary has no telecom namespace analysis")
    return namespace


def _summaries_by_seed(paths: list[Path], *, label: str) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    for path in paths:
        summary = _read(path)
        seed = summary.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"{label} summary has no integer seed: {path}")
        if seed in summaries:
            raise ValueError(f"duplicate {label} summary seed {seed}")
        summaries[seed] = summary
    if set(summaries) != EXPECTED_SEEDS:
        raise ValueError(
            f"{label} requires exactly fixed seeds {sorted(EXPECTED_SEEDS)}; "
            f"got {sorted(summaries)}"
        )
    return summaries


def _comparisons_by_seed(
    paths: list[Path],
    *,
    candidate: str,
    baseline: str,
) -> dict[int, dict[str, Any]]:
    comparisons: dict[int, dict[str, Any]] = {}
    for path in paths:
        comparison = _read(path)
        models = comparison.get("models") or {}
        if candidate not in models or baseline not in models:
            raise ValueError(f"comparison is missing candidate or baseline: {path}")
        if (
            comparison.get("status") != "pass"
            or comparison.get("comparison_mode") != "strict-signed"
            or comparison.get("protocol_profile") != PROFILE
        ):
            raise ValueError(f"comparison is not strict signed boundary-v2: {path}")
        seed = (comparison.get("protocol_signature") or {}).get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"comparison has no integer seed: {path}")
        if seed in comparisons:
            raise ValueError(f"duplicate comparison seed {seed}")
        comparisons[seed] = comparison
    return comparisons


def _namespace_probe_observation(
    path: Path,
    *,
    checkpoint_root: Path,
    iteration: int,
    summaries: dict[int, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Validate probe provenance/completeness without adding a numeric RL gate."""

    probe = _read(path)
    summaries_payload = probe.get("summaries") or {}
    temperature_0 = summaries_payload.get("temperature_0") or {}
    temperature_0_6 = summaries_payload.get("temperature_0_6") or {}
    expected_checkpoint = (
        checkpoint_root / f"iter_{iteration:07d}_hf"
    ).resolve()
    probe_checkpoint = Path(str(probe.get("checkpoint") or "")).resolve()
    summary_signatures = {
        (((summary.get("domains") or {}).get("telecom") or {}).get("agent_contract") or {}).get(
            "agent_contract_signature"
        )
        for summary in summaries.values()
    }
    complete = (
        temperature_0.get("cases") == 60
        and temperature_0_6.get("cases") == 240
        and probe.get("user_tool_count") == 30
        and probe_checkpoint == expected_checkpoint
        and None not in summary_signatures
        and len(summary_signatures) == 1
        and probe.get("agent_contract_signature") in summary_signatures
    )
    return complete, {
        "path": str(path.resolve()),
        "reported_status": probe.get("status"),
        "checkpoint": str(probe_checkpoint),
        "expected_checkpoint": str(expected_checkpoint),
        "agent_contract_signature": probe.get("agent_contract_signature"),
        "temperature_0": temperature_0,
        "temperature_0_6": temperature_0_6,
        "user_tool_count": probe.get("user_tool_count"),
    }


def _quota_observations(
    training: dict[str, Any],
    *,
    quota: dict[str, int],
) -> tuple[bool, dict[str, Any]]:
    observations: dict[str, Any] = {}
    passed = True
    for step in training["expected_steps"]:
        metrics = (training.get("expected_step_metrics") or {}).get(str(step), {})
        step_observation: dict[str, Any] = {}
        for domain, expected in quota.items():
            prefix = f"rollout/domain_quota/{domain}"
            accepted = metrics.get(f"{prefix}/accepted")
            rejected = metrics.get(f"{prefix}/rejected")
            replacements = metrics.get(f"{prefix}/replacement_draws")
            step_observation[domain] = {
                "accepted": accepted,
                "rejected": rejected,
                "replacement_draws": replacements,
            }
            passed = passed and accepted == expected
            passed = passed and isinstance(rejected, (int, float))
            passed = passed and replacements == rejected
        observations[str(step)] = step_observation
    return passed, observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("iter9", "iter19"), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--pilot-a-training-log", type=Path, required=True)
    parser.add_argument("--pilot-b-training-log", type=Path)
    parser.add_argument("--pilot-a-trajectories", type=Path, required=True)
    parser.add_argument("--pilot-b-trajectories", type=Path)
    parser.add_argument("--sft-summary", type=Path, action="append", required=True)
    parser.add_argument("--pilot-summary", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--namespace-probe", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="selected_sft")
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.window < 1:
        raise ValueError("--window must be positive")
    expected_latest = 9 if args.stage == "iter9" else 19
    training = {
        "pilot_a": parse_training_log(args.pilot_a_training_log, expected_latest=9)
    }
    trajectory_paths = [args.pilot_a_trajectories]
    if args.stage == "iter19":
        if args.pilot_b_training_log is None or args.pilot_b_trajectories is None:
            raise ValueError("iter19 requires Pilot B training and trajectory artifacts")
        training["pilot_b"] = parse_training_log(
            args.pilot_b_training_log,
            expected_latest=19,
        )
        trajectory_paths.append(args.pilot_b_trajectories)
    trajectories = summarize_trajectories(trajectory_paths, window=args.window)
    sft_summaries = _summaries_by_seed(args.sft_summary, label="selected SFT")
    pilot_summaries = _summaries_by_seed(
        args.pilot_summary,
        label=f"RL {args.stage}",
    )
    comparisons = _comparisons_by_seed(
        args.comparison,
        candidate=args.candidate,
        baseline=args.baseline,
    )
    if not (set(sft_summaries) == set(pilot_summaries) == set(comparisons)):
        raise ValueError(
            f"SFT, {args.stage}, and comparison seed sets differ: "
            f"{sorted(sft_summaries)}, {sorted(pilot_summaries)}, "
            f"{sorted(comparisons)}"
        )

    checks = []

    def add(name: str, passed: bool, observed: Any, requirement: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "requirement": requirement,
            }
        )

    latest_file = args.checkpoint_root / "latest_checkpointed_iteration.txt"
    latest_raw = latest_file.read_text(encoding="utf-8").strip() if latest_file.is_file() else ""
    latest = int(latest_raw) if latest_raw.isdigit() else None
    add(
        "checkpoint_latest",
        latest == expected_latest,
        latest,
        f"== {expected_latest}",
    )
    for stage, stage_training in training.items():
        add(
            f"{stage}_training_complete",
            not stage_training["missing_steps"]
            and not stage_training["missing_kl_steps"]
            and not stage_training["missing_truncation_steps"],
            {
                "missing_steps": stage_training["missing_steps"],
                "missing_kl_steps": stage_training["missing_kl_steps"],
                "missing_truncation_steps": stage_training["missing_truncation_steps"],
            },
            "all ten stage metrics are present",
        )
        add(
            f"{stage}_no_nan_inf_oom",
            not any(stage_training["fatal_matches"].values())
            and not stage_training["nonfinite_metrics"],
            {
                "fatal_matches": stage_training["fatal_matches"],
                "nonfinite_metrics": stage_training["nonfinite_metrics"],
            },
            "no NaN/Inf/OOM and all metrics finite",
        )
        kl_mean = stage_training["last_10_k2_kl_mean"]
        truncation_mean = stage_training["last_10_truncation_mean"]
        add(f"{stage}_k2_kl", kl_mean is not None and kl_mean < 0.10, kl_mean, "< 0.10")
        add(
            f"{stage}_truncation",
            truncation_mean is not None and truncation_mean < 0.01,
            truncation_mean,
            "< 0.01",
        )
        quota_passed, quota_observed = _quota_observations(
            stage_training,
            quota=PILOT_QUOTAS[stage],
        )
        add(
            f"{stage}_domain_quota",
            quota_passed,
            quota_observed,
            f"every step accepts {PILOT_QUOTAS[stage]} and same-domain replacements equal rejects",
        )

    add(
        "trajectory_window_complete",
        trajectories["selected_rows"] == args.window,
        trajectories["selected_rows"],
        f"== {args.window}",
    )
    add(
        "trajectory_protocol",
        not trajectories["wrong_protocol_profiles"]
        and not trajectories["wrong_protocol_signatures"]
        and trajectories["missing_contract_signatures"] == 0,
        {
            "profiles": trajectories["wrong_protocol_profiles"],
            "protocol_signatures": trajectories["wrong_protocol_signatures"],
            "missing_contract_signatures": trajectories["missing_contract_signatures"],
        },
        "every trajectory carries the signed Agent contract",
    )
    add(
        "trajectory_turn_credit_alignment",
        not trajectories["turn_credit_alignment_errors"],
        trajectories["turn_credit_alignment_errors"],
        "every live trajectory has turn-credit-v1 response-aligned penalties",
    )
    for name, threshold in (
        ("malformed_json_fraction", 0.10),
        ("nonexistent_tool_fraction", 0.05),
        ("wrong_argument_field_rate", 0.50),
        ("tool_execution_error_fraction", 0.10),
        ("max_steps_fraction", 0.02),
    ):
        value = trajectories[name]
        add(name, value is not None and value < threshold, value, f"< {threshold}")

    namespace_by_seed: dict[str, Any] = {}
    namespace_non_regression = True
    namespace_strict_reduction = True
    namespace_totals = {
        "affected_trajectory_count": 0,
        "raw_attempt_count": 0,
        "namespace_attributed_termination_count": 0,
    }
    for seed in sorted(sft_summaries):
        sft_namespace = _telecom_namespace(sft_summaries[seed])
        pilot_namespace = _telecom_namespace(pilot_summaries[seed])
        seed_namespace_passed = all(
            pilot_namespace.get(key, math.inf) <= sft_namespace.get(key, -math.inf)
            for key in (
                "affected_trajectory_count",
                "raw_attempt_count",
                "namespace_attributed_termination_count",
            )
        )
        namespace_non_regression = namespace_non_regression and seed_namespace_passed
        seed_strict_reduction = (
            pilot_namespace.get("affected_trajectory_count", math.inf)
            < sft_namespace.get("affected_trajectory_count", -math.inf)
            or pilot_namespace.get("raw_attempt_count", math.inf)
            < sft_namespace.get("raw_attempt_count", -math.inf)
        )
        namespace_strict_reduction = (
            namespace_strict_reduction and seed_strict_reduction
        )
        for key in namespace_totals:
            namespace_totals[key] += int(pilot_namespace.get(key, 0))
        namespace_by_seed[str(seed)] = {
            "selected_sft": sft_namespace,
            args.candidate: pilot_namespace,
        }
        add(
            f"seed{seed}_namespace_non_regression",
            seed_namespace_passed,
            namespace_by_seed[str(seed)],
            f"{args.stage} does not increase affected trajectories, attempts, or attributed terminations",
        )
        add(
            f"seed{seed}_namespace_strict_reduction",
            seed_strict_reduction,
            namespace_by_seed[str(seed)],
            f"{args.stage} strictly reduces affected trajectories or attempts",
        )
        add(
            f"seed{seed}_telecom_trajectory_count",
            pilot_namespace.get("trajectory_count") == 160,
            pilot_namespace.get("trajectory_count"),
            "== 160",
        )

        models = comparisons[seed]["models"]
        candidate = models[args.candidate]
        baseline = models[args.baseline]
        pass_power_delta = (
            candidate["overall"]["pass_power_4"]
            - baseline["overall"]["pass_power_4"]
        )
        add(
            f"seed{seed}_pass_power_4_noninferiority",
            pass_power_delta >= -0.02,
            pass_power_delta,
            ">= -0.02",
        )
        for domain, metrics in candidate["by_domain"].items():
            delta = metrics["pass_at_1"] - baseline["by_domain"][domain]["pass_at_1"]
            add(
                f"seed{seed}_{domain}_pass_at_1_noninferiority",
                delta >= -0.05,
                delta,
                ">= -0.05",
            )
    if args.stage == "iter19":
        for key, limit in (
            ("affected_trajectory_count", 87),
            ("raw_attempt_count", 512),
            ("namespace_attributed_termination_count", 13),
        ):
            add(
                f"combined_{key}",
                namespace_totals[key] <= limit,
                namespace_totals[key],
                f"<= {limit}",
            )

    probe_complete, probe_observation = _namespace_probe_observation(
        args.namespace_probe,
        checkpoint_root=args.checkpoint_root,
        iteration=expected_latest,
        summaries=pilot_summaries,
    )
    add(
        "namespace_probe_complete",
        probe_complete,
        probe_observation,
        "60+240 cases, signed contract, 30 User tools, and exact checkpoint",
    )

    status = "pass" if all(check["passed"] for check in checks) else "fail"
    output = {
        "status": status,
        "gate_version": GATE_VERSION,
        "stage": args.stage,
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "expected_latest": expected_latest,
        "candidate": args.candidate,
        "baseline": args.baseline,
        "protocol_profile": PROFILE,
        "turn_credit_version": TURN_CREDIT_VERSION,
        "agent_protocol_signature": PROTOCOL_SIGNATURE,
        "namespace_non_regression": namespace_non_regression,
        "namespace_strict_reduction": namespace_strict_reduction,
        "namespace_totals": namespace_totals,
        "namespace_probe": probe_observation,
        "eval_seeds": sorted(sft_summaries),
        "namespace_by_seed": namespace_by_seed,
        "checks": checks,
        "training": training,
        "trajectories": trajectories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
