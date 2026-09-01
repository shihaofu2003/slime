#!/usr/bin/env python3
"""Gate boundary-v2 RL iter99 after the final balanced 80 updates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from check_agent_boundary_rl_pilot import (
    PROFILE,
    PROTOCOL_SIGNATURE,
    _comparisons_by_seed,
    _namespace_probe_observation,
    _quota_observations,
    _summaries_by_seed,
    _telecom_namespace,
    summarize_trajectories,
)
from check_tau2_rl_promotion import parse_training_log


FINAL_QUOTA = {"telecom": 2, "airline": 2, "retail": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--sft-summary", type=Path, action="append", required=True)
    parser.add_argument("--iter19-summary", type=Path, action="append", required=True)
    parser.add_argument("--iter99-summary", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--namespace-probe", type=Path, required=True)
    parser.add_argument("--candidate", default="rl_iter99")
    parser.add_argument("--baseline", default="selected_sft")
    parser.add_argument("--prior", default="rl_iter19")
    parser.add_argument("--window", type=int, default=3840)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.window < 1:
        raise ValueError("--window must be positive")
    training = parse_training_log(args.training_log, expected_latest=99)
    trajectories = summarize_trajectories([args.trajectories], window=args.window)
    sft_summaries = _summaries_by_seed(args.sft_summary, label="selected SFT")
    iter19_summaries = _summaries_by_seed(args.iter19_summary, label="RL iter19")
    iter99_summaries = _summaries_by_seed(args.iter99_summary, label="RL iter99")
    comparisons = _comparisons_by_seed(
        args.comparison,
        candidate=args.candidate,
        baseline=args.baseline,
    )
    seed_sets = (
        set(sft_summaries),
        set(iter19_summaries),
        set(iter99_summaries),
        set(comparisons),
    )
    if len({tuple(sorted(seeds)) for seeds in seed_sets}) != 1:
        raise ValueError(
            "SFT, iter19, iter99, and comparison seed sets differ: "
            + ", ".join(str(sorted(seeds)) for seeds in seed_sets)
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

    latest_file = args.checkpoint_root / "latest_checkpointed_iteration.txt"
    latest_raw = latest_file.read_text(encoding="utf-8").strip() if latest_file.is_file() else ""
    latest = int(latest_raw) if latest_raw.isdigit() else None
    add("checkpoint_latest", latest == 99, latest, "== 99")
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
        "all final ten train and rollout metrics are present",
    )
    add(
        "training_no_nan_inf_oom",
        not any(training["fatal_matches"].values())
        and not training["nonfinite_metrics"],
        {
            "fatal_matches": training["fatal_matches"],
            "nonfinite_metrics": training["nonfinite_metrics"],
        },
        "no NaN/Inf/OOM and all final metrics finite",
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
    quota_passed, quota_observed = _quota_observations(
        training,
        quota=FINAL_QUOTA,
    )
    add(
        "final_domain_quota",
        quota_passed,
        quota_observed,
        f"every final step accepts {FINAL_QUOTA} and same-domain replacements equal rejects",
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

    namespace_by_seed: dict[str, Any] = {}
    namespace_non_regression = True
    for seed in sorted(sft_summaries):
        sft_namespace = _telecom_namespace(sft_summaries[seed])
        iter19_namespace = _telecom_namespace(iter19_summaries[seed])
        iter99_namespace = _telecom_namespace(iter99_summaries[seed])
        namespace_passed = all(
            iter99_namespace.get(key, math.inf)
            <= min(
                sft_namespace.get(key, -math.inf),
                iter19_namespace.get(key, -math.inf),
            )
            for key in (
                "affected_trajectory_count",
                "raw_attempt_count",
                "namespace_attributed_termination_count",
            )
        )
        namespace_non_regression = namespace_non_regression and namespace_passed
        namespace_by_seed[str(seed)] = {
            "selected_sft": sft_namespace,
            "rl_iter19": iter19_namespace,
            "rl_iter99": iter99_namespace,
        }
        add(
            f"seed{seed}_namespace_non_regression",
            namespace_passed,
            namespace_by_seed[str(seed)],
            "iter99 does not increase affected trajectories, attempts, or "
            "attributed terminations over selected SFT or iter19",
        )
        add(
            f"seed{seed}_telecom_trajectory_count",
            iter99_namespace.get("trajectory_count") == 160,
            iter99_namespace.get("trajectory_count"),
            "== 160",
        )

        models = comparisons[seed]["models"]
        missing_models = {
            label for label in (args.candidate, args.baseline, args.prior) if label not in models
        }
        if missing_models:
            raise ValueError(
                f"seed {seed} comparison is missing models: {sorted(missing_models)}"
            )
        candidate = models[args.candidate]
        baseline = models[args.baseline]
        prior = models[args.prior]
        for metric in ("pass_at_1", "pass_at_4_any"):
            observed = candidate["overall"][metric]
            references = {
                args.baseline: baseline["overall"][metric],
                args.prior: prior["overall"][metric],
            }
            add(
                f"seed{seed}_strict_{metric}_improvement",
                all(observed > value for value in references.values()),
                {args.candidate: observed, **references},
                f"{args.candidate} strictly exceeds selected SFT and iter19",
            )
        pass_power_delta = (
            candidate["overall"]["pass_power_4"]
            - baseline["overall"]["pass_power_4"]
        )
        add(
            f"seed{seed}_pass_power_4_noninferiority",
            pass_power_delta >= -0.02,
            pass_power_delta,
            ">= -0.02 versus selected SFT",
        )
        for domain, metrics in candidate["by_domain"].items():
            delta = metrics["pass_at_1"] - baseline["by_domain"][domain]["pass_at_1"]
            add(
                f"seed{seed}_{domain}_pass_at_1_noninferiority",
                delta >= -0.05,
                delta,
                ">= -0.05 versus selected SFT",
            )

    probe_complete, probe_observation = _namespace_probe_observation(
        args.namespace_probe,
        checkpoint_root=args.checkpoint_root,
        iteration=99,
        summaries=iter99_summaries,
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
        "gate_version": "turn-aware-rl-final-v1",
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "expected_latest": 99,
        "protocol_profile": PROFILE,
        "turn_credit_version": "turn-credit-v1",
        "agent_protocol_signature": PROTOCOL_SIGNATURE,
        "namespace_non_regression": namespace_non_regression,
        "eval_seeds": sorted(sft_summaries),
        "namespace_by_seed": namespace_by_seed,
        "namespace_probe": probe_observation,
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
