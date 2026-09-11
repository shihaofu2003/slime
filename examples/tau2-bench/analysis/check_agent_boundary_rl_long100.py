#!/usr/bin/env python3
"""Classify boundary-v2 long100 iter99 as win, usable, or reject."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from check_agent_boundary_rl_health import (
    GATE_VERSION as HEALTH_GATE_VERSION,
    LONG100_STAGES,
)
from check_agent_boundary_rl_pilot import (
    PROFILE,
    PROTOCOL_SIGNATURE,
    TURN_CREDIT_VERSION,
    _comparisons_by_seed,
    _namespace_probe_observation,
    _summaries_by_seed,
    _telecom_namespace,
)


DECISION_VERSION = "turn-aware-rl-long100-v1"
NAMESPACE_KEYS = (
    "affected_trajectory_count",
    "raw_attempt_count",
    "namespace_attributed_termination_count",
)


def _health_gate_passed(
    gate: dict[str, Any],
    *,
    stage: str,
    checkpoint_root: Path,
) -> bool:
    config = LONG100_STAGES[stage]
    root = str(checkpoint_root.resolve())
    return bool(
        gate.get("status") == "pass"
        and gate.get("gate_version") == HEALTH_GATE_VERSION
        and gate.get("stage") == stage
        and gate.get("checkpoint_root") == root
        and gate.get("expected_checkpoint_root") == root
        and gate.get("expected_latest") == config["end"]
        and gate.get("protocol_profile") == PROFILE
        and gate.get("agent_protocol_signature") == PROTOCOL_SIGNATURE
        and gate.get("turn_credit_version") == TURN_CREDIT_VERSION
    )


def evaluate_long100_decision(
    *,
    checkpoint_root: Path,
    health_gates: dict[str, dict[str, Any]],
    sft_summaries: dict[int, dict[str, Any]],
    iter9_summaries: dict[int, dict[str, Any]],
    iter99_summaries: dict[int, dict[str, Any]],
    comparisons: dict[int, dict[str, Any]],
    probe_complete: bool,
    probe_observation: dict[str, Any],
    candidate: str,
    baseline: str,
    prior: str,
) -> dict[str, Any]:
    seed_sets = (
        set(sft_summaries),
        set(iter9_summaries),
        set(iter99_summaries),
        set(comparisons),
    )
    if len({tuple(sorted(seeds)) for seeds in seed_sets}) != 1:
        raise ValueError(
            "SFT, historical iter9, long100 iter99, and comparison seed sets differ: "
            + ", ".join(str(sorted(seeds)) for seeds in seed_sets)
        )

    checks: list[dict[str, Any]] = []

    def add(
        name: str,
        passed: bool,
        observed: Any,
        requirement: str,
        *,
        category: str = "hard",
    ) -> None:
        checks.append(
            {
                "name": name,
                "category": category,
                "passed": bool(passed),
                "observed": observed,
                "requirement": requirement,
            }
        )

    for stage in LONG100_STAGES:
        gate = health_gates.get(stage) or {}
        add(
            f"{stage}_health",
            _health_gate_passed(gate, stage=stage, checkpoint_root=checkpoint_root),
            {
                key: gate.get(key)
                for key in (
                    "status",
                    "gate_version",
                    "stage",
                    "checkpoint_root",
                    "expected_latest",
                )
            },
            "passing long100 health artifact for the exact checkpoint root",
        )
    contract_maps = [
        ((health_gates.get(stage) or {}).get("trajectories") or {}).get(
            "agent_contract_signatures"
        )
        for stage in LONG100_STAGES
    ]
    add(
        "health_contract_continuity",
        None not in contract_maps
        and len({json.dumps(value, sort_keys=True) for value in contract_maps}) == 1,
        dict(zip(LONG100_STAGES, contract_maps)),
        "all three training stages carry identical per-domain Contract signatures",
    )
    add(
        "namespace_probe_complete",
        probe_complete,
        probe_observation,
        "seed-300 iter99 probe contains 60+240 cases, 30 User tools, and exact checkpoint",
    )

    namespace_by_seed: dict[str, Any] = {}
    metrics_by_seed: dict[str, Any] = {}
    paired_deltas_by_seed: dict[str, Any] = {}
    for seed in sorted(sft_summaries):
        sft_namespace = _telecom_namespace(sft_summaries[seed])
        iter9_namespace = _telecom_namespace(iter9_summaries[seed])
        iter99_namespace = _telecom_namespace(iter99_summaries[seed])
        namespace_observation = {
            baseline: sft_namespace,
            prior: iter9_namespace,
            candidate: iter99_namespace,
        }
        namespace_by_seed[str(seed)] = namespace_observation
        namespace_passed = all(
            iter99_namespace.get(key, math.inf)
            <= min(
                sft_namespace.get(key, -math.inf),
                iter9_namespace.get(key, -math.inf),
            )
            for key in NAMESPACE_KEYS
        )
        add(
            f"seed{seed}_namespace_non_regression",
            namespace_passed,
            namespace_observation,
            "iter99 affected trajectories, attempts, and attributed terminations do not exceed SFT or historical iter9",
        )
        add(
            f"seed{seed}_telecom_trajectory_count",
            iter99_namespace.get("trajectory_count") == 160,
            iter99_namespace.get("trajectory_count"),
            "== 160",
        )

        models = comparisons[seed].get("models") or {}
        missing_models = {
            label for label in (candidate, baseline, prior) if label not in models
        }
        if missing_models:
            raise ValueError(
                f"seed {seed} comparison is missing models: {sorted(missing_models)}"
            )
        candidate_model = models[candidate]
        baseline_model = models[baseline]
        prior_model = models[prior]
        metrics_by_seed[str(seed)] = {
            baseline: {
                "overall": baseline_model["overall"],
                "by_domain": baseline_model["by_domain"],
            },
            prior: {
                "overall": prior_model["overall"],
                "by_domain": prior_model["by_domain"],
            },
            candidate: {
                "overall": candidate_model["overall"],
                "by_domain": candidate_model["by_domain"],
            },
        }
        paired_deltas_by_seed[str(seed)] = comparisons[seed].get("paired_deltas") or []
        add(
            f"seed{seed}_official_eval_complete",
            candidate_model.get("tasks") == 100
            and candidate_model.get("simulations") == 400,
            {
                "tasks": candidate_model.get("tasks"),
                "simulations": candidate_model.get("simulations"),
            },
            "100 tasks x 4 trials",
        )

        pass_power_delta = (
            candidate_model["overall"]["pass_power_4"]
            - baseline_model["overall"]["pass_power_4"]
        )
        add(
            f"seed{seed}_pass_power_4_noninferiority",
            pass_power_delta >= -0.02 - 1e-12,
            pass_power_delta,
            ">= -0.02 versus selected SFT",
        )
        for domain, candidate_metrics in candidate_model["by_domain"].items():
            delta = (
                candidate_metrics["pass_at_1"]
                - baseline_model["by_domain"][domain]["pass_at_1"]
            )
            add(
                f"seed{seed}_{domain}_pass_at_1_noninferiority",
                delta >= -0.05 - 1e-12,
                delta,
                ">= -0.05 versus selected SFT",
            )

        for metric in ("pass_at_1", "pass_at_4_any"):
            observed = candidate_model["overall"][metric]
            references = {
                baseline: baseline_model["overall"][metric],
                prior: prior_model["overall"][metric],
            }
            add(
                f"seed{seed}_strict_{metric}_improvement",
                all(observed > value for value in references.values()),
                {candidate: observed, **references},
                f"{candidate} strictly exceeds both controls",
                category="strict_win",
            )

    hard_passed = all(
        check["passed"] for check in checks if check["category"] == "hard"
    )
    strict_passed = all(
        check["passed"] for check in checks if check["category"] == "strict_win"
    )
    if not hard_passed:
        decision = "reject"
    elif strict_passed:
        decision = "win"
    else:
        decision = "usable"

    return {
        "status": decision,
        "decision": decision,
        "decision_version": DECISION_VERSION,
        "checkpoint_root": str(checkpoint_root.resolve()),
        "expected_latest": 99,
        "candidate": candidate,
        "baseline": baseline,
        "prior": prior,
        "protocol_profile": PROFILE,
        "agent_protocol_signature": PROTOCOL_SIGNATURE,
        "turn_credit_version": TURN_CREDIT_VERSION,
        "eval_seeds": sorted(sft_summaries),
        "hard_conditions_passed": hard_passed,
        "strict_improvements_passed": strict_passed,
        "namespace_non_regression": all(
            check["passed"]
            for check in checks
            if check["name"].endswith("namespace_non_regression")
        ),
        "capability_noninferiority": all(
            check["passed"]
            for check in checks
            if "noninferiority" in check["name"]
        ),
        "checks": checks,
        "namespace_by_seed": namespace_by_seed,
        "namespace_probe": probe_observation,
        "metrics_by_seed": metrics_by_seed,
        "paired_deltas_by_seed": paired_deltas_by_seed,
        "behavior_diagnostics_only": {
            "training": {
                stage: (health_gates[stage].get("diagnostics_only") or {})
                for stage in LONG100_STAGES
                if stage in health_gates
            },
            "evaluation": {
                str(seed): {
                    label: model.get("behavior") or {}
                    for label, model in comparisons[seed]["models"].items()
                }
                for seed in sorted(comparisons)
            },
        },
    }


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
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
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--iter9-health", type=Path, required=True)
    parser.add_argument("--iter19-health", type=Path, required=True)
    parser.add_argument("--iter99-health", type=Path, required=True)
    parser.add_argument("--sft-summary", type=Path, action="append", required=True)
    parser.add_argument("--iter9-summary", type=Path, action="append", required=True)
    parser.add_argument("--iter99-summary", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--namespace-probe", type=Path, required=True)
    parser.add_argument("--candidate", default="long100_iter99")
    parser.add_argument("--baseline", default="selected_sft")
    parser.add_argument("--prior", default="historical_rl_iter9")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    health_gates = {
        "iter9": _read(args.iter9_health),
        "iter19": _read(args.iter19_health),
        "iter99": _read(args.iter99_health),
    }
    sft_summaries = _summaries_by_seed(args.sft_summary, label="selected SFT")
    iter9_summaries = _summaries_by_seed(
        args.iter9_summary,
        label="historical RL iter9",
    )
    iter99_summaries = _summaries_by_seed(
        args.iter99_summary,
        label="long100 RL iter99",
    )
    comparisons = _comparisons_by_seed(
        args.comparison,
        candidate=args.candidate,
        baseline=args.baseline,
    )
    probe_complete, probe_observation = _namespace_probe_observation(
        args.namespace_probe,
        checkpoint_root=args.checkpoint_root,
        iteration=99,
        summaries=iter99_summaries,
    )
    report = evaluate_long100_decision(
        checkpoint_root=args.checkpoint_root,
        health_gates=health_gates,
        sft_summaries=sft_summaries,
        iter9_summaries=iter9_summaries,
        iter99_summaries=iter99_summaries,
        comparisons=comparisons,
        probe_complete=probe_complete,
        probe_observation=probe_observation,
        candidate=args.candidate,
        baseline=args.baseline,
        prior=args.prior,
    )
    report["artifacts"] = {
        "health": {
            "iter9": str(args.iter9_health.resolve()),
            "iter19": str(args.iter19_health.resolve()),
            "iter99": str(args.iter99_health.resolve()),
        },
        "comparisons": [str(path.resolve()) for path in args.comparison],
    }
    _write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] == "reject":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
