#!/usr/bin/env python3
"""Gate the isolated two-update tau2 turn-credit engineering smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from check_agent_boundary_rl_pilot import (
    PILOT_QUOTAS,
    _quota_observations,
    summarize_trajectories,
)
from check_tau2_rl_promotion import parse_training_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = parse_training_log(args.training_log, expected_latest=1)
    trajectories = summarize_trajectories([args.trajectories], window=args.window)
    latest_file = args.checkpoint_root / "latest_checkpointed_iteration.txt"
    latest_raw = latest_file.read_text(encoding="utf-8").strip() if latest_file.is_file() else ""
    latest = int(latest_raw) if latest_raw.isdigit() else None
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

    add("checkpoint_latest", latest == 1, latest, "== 1 in the isolated smoke root")
    add(
        "training_complete",
        not training["missing_steps"]
        and not training["missing_kl_steps"]
        and not training["missing_truncation_steps"],
        {
            "missing_steps": training["missing_steps"],
            "missing_kl_steps": training["missing_kl_steps"],
            "missing_truncation_steps": training["missing_truncation_steps"],
        },
        "updates 0 and 1 have train/KL/truncation metrics",
    )
    add(
        "no_nan_inf_oom",
        not any(training["fatal_matches"].values())
        and not training["nonfinite_metrics"],
        {
            "fatal_matches": training["fatal_matches"],
            "nonfinite_metrics": training["nonfinite_metrics"],
        },
        "no NaN, Inf, OOM, or non-finite metric",
    )
    quota_passed, quota_observed = _quota_observations(
        training,
        quota=PILOT_QUOTAS["pilot_a"],
    )
    add(
        "domain_quota",
        quota_passed,
        quota_observed,
        "each update accepts telecom/airline/retail=3/2/1",
    )
    add(
        "trajectory_window_complete",
        trajectories["selected_rows"] == args.window,
        trajectories["selected_rows"],
        f"== {args.window}",
    )
    add(
        "signed_turn_credit_alignment",
        not trajectories["wrong_protocol_profiles"]
        and not trajectories["wrong_protocol_signatures"]
        and trajectories["missing_contract_signatures"] == 0
        and not trajectories["turn_credit_alignment_errors"],
        {
            "profiles": trajectories["wrong_protocol_profiles"],
            "protocol_signatures": trajectories["wrong_protocol_signatures"],
            "missing_contract_signatures": trajectories["missing_contract_signatures"],
            "alignment_errors": trajectories["turn_credit_alignment_errors"],
        },
        "every live response has signed, exact span/token alignment",
    )
    minimum_penalty = trajectories["minimum_positive_turn_penalty"]
    advantage_delta = -minimum_penalty if minimum_penalty is not None else None
    add(
        "penalized_turn_advantage_below_clean_turn",
        trajectories["penalized_turns"] > 0
        and trajectories["clean_turns"] > 0
        and advantage_delta is not None
        and advantage_delta < 0,
        {
            "penalized_turns": trajectories["penalized_turns"],
            "clean_turns": trajectories["clean_turns"],
            "minimum_penalty": minimum_penalty,
            "penalized_minus_clean_advantage": advantage_delta,
        },
        "a live penalized turn has global_advantage-penalty below a clean turn",
    )

    status = "pass" if all(check["passed"] for check in checks) else "fail"
    output = {
        "status": status,
        "gate_version": "turn-aware-rl-smoke-v1",
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "expected_latest": 1,
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
