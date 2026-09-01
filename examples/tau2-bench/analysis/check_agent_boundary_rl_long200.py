#!/usr/bin/env python3
"""Classify boundary-v2 long200 iter199 as win, usable, or reject."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from check_agent_boundary_rl_health import (
    KL_CHECK_NAMES,
    KL_WAIVER_POLICY,
    LONG200_GATE_VERSION,
    LONG200_KL_WAIVER_GATE_VERSION,
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


DECISION_VERSION = "turn-aware-rl-long200-v1"
KL_WAIVER_RESULT_VERSION = "turn-aware-rl-long200-kl-waiver-v1"
NAMESPACE_KEYS = (
    "affected_trajectory_count",
    "raw_attempt_count",
    "namespace_attributed_termination_count",
)


def _health_gate_passed(
    gate: dict[str, Any],
    *,
    checkpoint_root: Path,
    source_checkpoint_root: Path,
) -> bool:
    root = str(checkpoint_root.resolve())
    source = str(source_checkpoint_root.resolve())
    checks = gate.get("checks") or []
    return bool(
        gate.get("status") == "pass"
        and gate.get("gate_version") == LONG200_GATE_VERSION
        and gate.get("health_mode") == "final"
        and gate.get("stage") == "iter199"
        and gate.get("checkpoint_root") == root
        and gate.get("expected_checkpoint_root") == root
        and gate.get("destination_checkpoint_root") == root
        and gate.get("source_checkpoint_root") == source
        and gate.get("expected_latest") == 199
        and gate.get("protocol_profile") == PROFILE
        and gate.get("agent_protocol_signature") == PROTOCOL_SIGNATURE
        and gate.get("turn_credit_version") == TURN_CREDIT_VERSION
        and isinstance(gate.get("source_health_gate_sha256"), str)
        and len(gate["source_health_gate_sha256"]) == 64
        and checks
        and all(check.get("passed") is True for check in checks)
    )


def _early_stop_health_gate_valid(
    gate: dict[str, Any],
    *,
    checkpoint_root: Path,
    source_checkpoint_root: Path,
) -> bool:
    root = str(checkpoint_root.resolve())
    source = str(source_checkpoint_root.resolve())
    checks = gate.get("checks") or []
    return bool(
        gate.get("status") == "fail"
        and gate.get("gate_version") == LONG200_GATE_VERSION
        and gate.get("health_mode") == "partial"
        and gate.get("stage") == "iter199"
        and gate.get("checkpoint_root") == root
        and gate.get("expected_checkpoint_root") == root
        and gate.get("destination_checkpoint_root") == root
        and gate.get("source_checkpoint_root") == source
        and gate.get("expected_latest") == 199
        and gate.get("protocol_profile") == PROFILE
        and gate.get("agent_protocol_signature") == PROTOCOL_SIGNATURE
        and gate.get("turn_credit_version") == TURN_CREDIT_VERSION
        and isinstance(gate.get("source_health_gate_sha256"), str)
        and len(gate["source_health_gate_sha256"]) == 64
        and checks
        and any(check.get("passed") is False for check in checks)
    )


def _kl_waiver_health_gate_valid(
    gate: dict[str, Any],
    *,
    checkpoint_root: Path,
    source_checkpoint_root: Path,
) -> bool:
    root = str(checkpoint_root.resolve())
    source = str(source_checkpoint_root.resolve())
    checks = gate.get("checks") or []
    checks_by_name = {check.get("name"): check for check in checks}
    return bool(
        gate.get("status") == "pass"
        and gate.get("gate_version") == LONG200_KL_WAIVER_GATE_VERSION
        and gate.get("health_mode") == "final"
        and gate.get("health_policy") == KL_WAIVER_POLICY
        and gate.get("waived_checks") == list(KL_CHECK_NAMES)
        and gate.get("stage") == "iter199"
        and gate.get("checkpoint_root") == root
        and gate.get("expected_checkpoint_root") == root
        and gate.get("destination_checkpoint_root") == root
        and gate.get("source_checkpoint_root") == source
        and gate.get("expected_latest") == 199
        and gate.get("protocol_profile") == PROFILE
        and gate.get("agent_protocol_signature") == PROTOCOL_SIGNATURE
        and gate.get("turn_credit_version") == TURN_CREDIT_VERSION
        and isinstance(gate.get("source_health_gate_sha256"), str)
        and len(gate["source_health_gate_sha256"]) == 64
        and all(name in checks_by_name for name in KL_CHECK_NAMES)
        and all(
            checks_by_name[name].get("enforced") is False
            for name in KL_CHECK_NAMES
        )
        and all(
            check.get("passed") is True
            for check in checks
            if check.get("name") not in KL_CHECK_NAMES
        )
    )


def evaluate_long200_early_reject(
    *,
    checkpoint_root: Path,
    source_checkpoint_root: Path,
    health_gate: dict[str, Any],
    candidate: str,
    baseline: str,
    prior: str,
) -> dict[str, Any]:
    if not _early_stop_health_gate_valid(
        health_gate,
        checkpoint_root=checkpoint_root,
        source_checkpoint_root=source_checkpoint_root,
    ):
        raise ValueError(
            "early-stop health gate does not match the long200 lineage or contain "
            "a hard health failure"
        )
    failed_checks = [
        check for check in health_gate["checks"] if check.get("passed") is False
    ]
    completed_steps = health_gate.get("completed_steps") or []
    return {
        "status": "reject",
        "decision": "reject",
        "decision_version": DECISION_VERSION,
        "decision_basis": "early_health_stop",
        "checkpoint_root": str(checkpoint_root.resolve()),
        "source_checkpoint_root": str(source_checkpoint_root.resolve()),
        "expected_latest": 199,
        "completed_latest": max(completed_steps) if completed_steps else None,
        "candidate": candidate,
        "baseline": baseline,
        "prior": prior,
        "selected_model": prior,
        "replacement_authorized": False,
        "continuation_200_299_authorized": False,
        "protocol_profile": PROFILE,
        "agent_protocol_signature": PROTOCOL_SIGNATURE,
        "turn_credit_version": TURN_CREDIT_VERSION,
        "eval_seeds": [],
        "hard_conditions_passed": False,
        "strict_improvements_passed": False,
        "namespace_non_regression": None,
        "capability_noninferiority": None,
        "checks": [
            {
                "name": "long200_health",
                "category": "hard",
                "passed": False,
                "observed": {
                    "status": health_gate.get("status"),
                    "health_mode": health_gate.get("health_mode"),
                    "completed_latest": max(completed_steps) if completed_steps else None,
                    "failed_checks": [check.get("name") for check in failed_checks],
                },
                "requirement": "long200 must reach iter199 with every hard health check passing",
            }
        ],
        "health_failed_checks": failed_checks,
        "namespace_by_seed": {},
        "namespace_probe": None,
        "metrics_by_seed": {},
        "paired_deltas_by_seed": {},
        "evaluation_skipped": {
            "conversion": True,
            "seed300": True,
            "seed301": True,
            "namespace_probe": True,
            "reason": "predeclared hard health failure before iter199",
        },
        "behavior_diagnostics_only": {
            "training": health_gate.get("diagnostics_only") or {},
            "evaluation": {},
        },
    }


def evaluate_long200_kl_waiver_result(
    *,
    checkpoint_root: Path,
    source_checkpoint_root: Path,
    health_gate: dict[str, Any],
    sft_summaries: dict[int, dict[str, Any]],
    iter99_summaries: dict[int, dict[str, Any]],
    iter199_summaries: dict[int, dict[str, Any]],
    comparisons: dict[int, dict[str, Any]],
    probe_complete: bool,
    probe_observation: dict[str, Any],
    candidate: str,
    baseline: str,
    prior: str,
) -> dict[str, Any]:
    seed_sets = (
        set(sft_summaries),
        set(iter99_summaries),
        set(iter199_summaries),
        set(comparisons),
    )
    if len({tuple(sorted(seeds)) for seeds in seed_sets}) != 1:
        raise ValueError(
            "SFT, iter99, iter199, and comparison seed sets differ: "
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

    add(
        "kl_waiver_health",
        _kl_waiver_health_gate_valid(
            health_gate,
            checkpoint_root=checkpoint_root,
            source_checkpoint_root=source_checkpoint_root,
        ),
        {
            key: health_gate.get(key)
            for key in (
                "status",
                "gate_version",
                "health_mode",
                "health_policy",
                "waived_checks",
                "checkpoint_root",
                "source_checkpoint_root",
                "expected_latest",
            )
        },
        "final iter199 health passes every non-KL check under the explicit KL waiver",
    )
    temperature_0 = probe_observation.get("temperature_0") or {}
    temperature_0_6 = probe_observation.get("temperature_0_6") or {}
    probe_safe = bool(
        probe_complete
        and probe_observation.get("reported_status") == "pass"
        and temperature_0.get("pass") is True
        and temperature_0_6.get("pass") is True
        and temperature_0.get("namespace_attempts") == 0
        and temperature_0_6.get("namespace_attempts") == 0
    )
    add(
        "namespace_probe_safe",
        probe_safe,
        probe_observation,
        "complete seed-300 probe passes all 60+240 cases with zero namespace attempts",
    )

    metrics_by_seed: dict[str, Any] = {}
    deltas_by_seed: dict[str, Any] = {}
    namespace_by_seed: dict[str, Any] = {}
    strict_point_checks: list[bool] = []
    noninferiority_checks: list[bool] = []
    for seed in sorted(comparisons):
        models = comparisons[seed].get("models") or {}
        missing = {label for label in (candidate, baseline, prior) if label not in models}
        if missing:
            raise ValueError(f"seed {seed} comparison is missing models: {sorted(missing)}")
        candidate_model = models[candidate]
        baseline_model = models[baseline]
        prior_model = models[prior]
        metrics_by_seed[str(seed)] = {
            label: {
                "overall": model["overall"],
                "by_domain": model["by_domain"],
            }
            for label, model in (
                (baseline, baseline_model),
                (prior, prior_model),
                (candidate, candidate_model),
            )
        }
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
        overall_deltas: dict[str, dict[str, float]] = {}
        for reference, model in ((baseline, baseline_model), (prior, prior_model)):
            overall_deltas[reference] = {
                metric: candidate_model["overall"][metric] - model["overall"][metric]
                for metric in ("pass_at_1", "pass_at_4_any", "pass_power_4")
            }
        domain_deltas = {
            domain: {
                baseline: candidate_metrics["pass_at_1"]
                - baseline_model["by_domain"][domain]["pass_at_1"],
                prior: candidate_metrics["pass_at_1"]
                - prior_model["by_domain"][domain]["pass_at_1"],
            }
            for domain, candidate_metrics in candidate_model["by_domain"].items()
        }
        deltas_by_seed[str(seed)] = {
            "overall": overall_deltas,
            "domain_pass_at_1": domain_deltas,
            "paired_bootstrap": comparisons[seed].get("paired_deltas") or [],
        }
        strict_point_checks.extend(
            candidate_model["overall"][metric]
            > max(
                baseline_model["overall"][metric],
                prior_model["overall"][metric],
            )
            for metric in ("pass_at_1", "pass_at_4_any")
        )
        noninferiority_checks.extend(
            (
                overall_deltas[prior]["pass_at_1"] >= -0.02 - 1e-12,
                overall_deltas[prior]["pass_at_4_any"] >= -0.02 - 1e-12,
                overall_deltas[prior]["pass_power_4"] >= -0.02 - 1e-12,
                overall_deltas[baseline]["pass_power_4"] >= -0.02 - 1e-12,
                *(delta >= -0.05 - 1e-12 for values in domain_deltas.values() for delta in values.values()),
            )
        )
        namespace_by_seed[str(seed)] = {
            baseline: _telecom_namespace(sft_summaries[seed]),
            prior: _telecom_namespace(iter99_summaries[seed]),
            candidate: _telecom_namespace(iter199_summaries[seed]),
        }

    return {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "result_version": KL_WAIVER_RESULT_VERSION,
        "result_type": "diagnostic_kl_waiver",
        "checkpoint_root": str(checkpoint_root.resolve()),
        "source_checkpoint_root": str(source_checkpoint_root.resolve()),
        "candidate": candidate,
        "baseline": baseline,
        "prior": prior,
        "eval_seeds": sorted(comparisons),
        "checks": checks,
        "strict_capability_point_estimates_passed": all(strict_point_checks),
        "capability_noninferiority_passed": all(noninferiority_checks),
        "metrics_by_seed": metrics_by_seed,
        "deltas_by_seed": deltas_by_seed,
        "namespace_by_seed": namespace_by_seed,
        "namespace_probe": probe_observation,
        "waived_health_failures": [
            check
            for check in health_gate.get("checks") or []
            if check.get("name") in KL_CHECK_NAMES and check.get("passed") is not True
        ],
        "formal_decision_unchanged": "reject",
        "selected_model": prior,
        "replacement_authorized": False,
        "continuation_200_299_authorized": False,
        "behavior_diagnostics_only": {
            "training": health_gate.get("diagnostics_only") or {},
            "evaluation": {
                str(seed): {
                    label: model.get("behavior") or {}
                    for label, model in comparisons[seed]["models"].items()
                }
                for seed in sorted(comparisons)
            },
        },
    }


def evaluate_long200_decision(
    *,
    checkpoint_root: Path,
    source_checkpoint_root: Path,
    health_gate: dict[str, Any],
    sft_summaries: dict[int, dict[str, Any]],
    iter99_summaries: dict[int, dict[str, Any]],
    iter199_summaries: dict[int, dict[str, Any]],
    comparisons: dict[int, dict[str, Any]],
    probe_complete: bool,
    probe_observation: dict[str, Any],
    candidate: str,
    baseline: str,
    prior: str,
) -> dict[str, Any]:
    seed_sets = (
        set(sft_summaries),
        set(iter99_summaries),
        set(iter199_summaries),
        set(comparisons),
    )
    if len({tuple(sorted(seeds)) for seeds in seed_sets}) != 1:
        raise ValueError(
            "SFT, iter99, iter199, and comparison seed sets differ: "
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

    add(
        "iter199_health",
        _health_gate_passed(
            health_gate,
            checkpoint_root=checkpoint_root,
            source_checkpoint_root=source_checkpoint_root,
        ),
        {
            key: health_gate.get(key)
            for key in (
                "status",
                "gate_version",
                "health_mode",
                "checkpoint_root",
                "source_checkpoint_root",
                "expected_latest",
                "source_health_gate_sha256",
            )
        },
        "passing final iter199 health artifact for the exact source and destination roots",
    )
    temperature_0 = probe_observation.get("temperature_0") or {}
    temperature_0_6 = probe_observation.get("temperature_0_6") or {}
    probe_safe = bool(
        probe_complete
        and probe_observation.get("reported_status") == "pass"
        and temperature_0.get("pass") is True
        and temperature_0_6.get("pass") is True
        and temperature_0.get("namespace_attempts") == 0
        and temperature_0_6.get("namespace_attempts") == 0
    )
    add(
        "namespace_probe_safe",
        probe_safe,
        probe_observation,
        "complete seed-300 probe passes all 60+240 cases with zero namespace attempts",
    )

    namespace_by_seed: dict[str, Any] = {}
    metrics_by_seed: dict[str, Any] = {}
    paired_deltas_by_seed: dict[str, Any] = {}
    for seed in sorted(sft_summaries):
        sft_namespace = _telecom_namespace(sft_summaries[seed])
        iter99_namespace = _telecom_namespace(iter99_summaries[seed])
        iter199_namespace = _telecom_namespace(iter199_summaries[seed])
        namespace_observation = {
            baseline: sft_namespace,
            prior: iter99_namespace,
            candidate: iter199_namespace,
        }
        namespace_by_seed[str(seed)] = namespace_observation
        add(
            f"seed{seed}_namespace_non_regression",
            all(
                iter199_namespace.get(key, math.inf)
                <= min(
                    sft_namespace.get(key, -math.inf),
                    iter99_namespace.get(key, -math.inf),
                )
                for key in NAMESPACE_KEYS
            ),
            namespace_observation,
            "iter199 namespace affected/attempt/termination counts do not exceed SFT or iter99",
        )
        add(
            f"seed{seed}_telecom_trajectory_count",
            iter199_namespace.get("trajectory_count") == 160,
            iter199_namespace.get("trajectory_count"),
            "== 160",
        )

        models = comparisons[seed].get("models") or {}
        missing = {label for label in (candidate, baseline, prior) if label not in models}
        if missing:
            raise ValueError(f"seed {seed} comparison is missing models: {sorted(missing)}")
        candidate_model = models[candidate]
        baseline_model = models[baseline]
        prior_model = models[prior]
        metrics_by_seed[str(seed)] = {
            label: {
                "overall": model["overall"],
                "by_domain": model["by_domain"],
            }
            for label, model in (
                (baseline, baseline_model),
                (prior, prior_model),
                (candidate, candidate_model),
            )
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

        sft_pass_power_delta = (
            candidate_model["overall"]["pass_power_4"]
            - baseline_model["overall"]["pass_power_4"]
        )
        add(
            f"seed{seed}_sft_pass_power_4_noninferiority",
            sft_pass_power_delta >= -0.02 - 1e-12,
            sft_pass_power_delta,
            ">= -0.02 versus selected SFT",
        )
        for domain, candidate_metrics in candidate_model["by_domain"].items():
            sft_delta = (
                candidate_metrics["pass_at_1"]
                - baseline_model["by_domain"][domain]["pass_at_1"]
            )
            add(
                f"seed{seed}_{domain}_sft_pass_at_1_noninferiority",
                sft_delta >= -0.05 - 1e-12,
                sft_delta,
                ">= -0.05 versus selected SFT",
            )
            prior_delta = (
                candidate_metrics["pass_at_1"]
                - prior_model["by_domain"][domain]["pass_at_1"]
            )
            add(
                f"seed{seed}_{domain}_iter99_pass_at_1_noninferiority",
                prior_delta >= -0.05 - 1e-12,
                prior_delta,
                ">= -0.05 versus iter99",
            )
        for metric in ("pass_at_1", "pass_at_4_any", "pass_power_4"):
            prior_delta = (
                candidate_model["overall"][metric]
                - prior_model["overall"][metric]
            )
            add(
                f"seed{seed}_iter99_{metric}_noninferiority",
                prior_delta >= -0.02 - 1e-12,
                prior_delta,
                ">= -0.02 versus iter99",
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
                f"{candidate} strictly exceeds selected SFT and iter99",
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
        "source_checkpoint_root": str(source_checkpoint_root.resolve()),
        "expected_latest": 199,
        "candidate": candidate,
        "baseline": baseline,
        "prior": prior,
        "selected_model": candidate if decision == "win" else prior,
        "replacement_authorized": decision == "win",
        "continuation_200_299_authorized": False,
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
            check["passed"] for check in checks if "noninferiority" in check["name"]
        ),
        "checks": checks,
        "namespace_by_seed": namespace_by_seed,
        "namespace_probe": probe_observation,
        "metrics_by_seed": metrics_by_seed,
        "paired_deltas_by_seed": paired_deltas_by_seed,
        "behavior_diagnostics_only": {
            "training": health_gate.get("diagnostics_only") or {},
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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--early-stop-health", type=Path)
    parser.add_argument("--kl-waiver-health", type=Path)
    parser.add_argument("--iter199-health", type=Path)
    parser.add_argument("--eval-checkpoint-root", type=Path)
    parser.add_argument("--sft-summary", type=Path, action="append")
    parser.add_argument("--iter99-summary", type=Path, action="append")
    parser.add_argument("--iter199-summary", type=Path, action="append")
    parser.add_argument("--comparison", type=Path, action="append")
    parser.add_argument("--namespace-probe", type=Path)
    parser.add_argument("--candidate", default="long200_iter199")
    parser.add_argument("--baseline", default="selected_sft")
    parser.add_argument("--prior", default="long100_iter99")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.kl_waiver_health is not None:
        if args.early_stop_health is not None or args.iter199_health is not None:
            parser.error(
                "--kl-waiver-health cannot be combined with strict or early-stop health"
            )
        required_waiver_inputs = {
            "--eval-checkpoint-root": args.eval_checkpoint_root,
            "--sft-summary": args.sft_summary,
            "--iter99-summary": args.iter99_summary,
            "--iter199-summary": args.iter199_summary,
            "--comparison": args.comparison,
            "--namespace-probe": args.namespace_probe,
        }
        missing = [
            name for name, value in required_waiver_inputs.items() if value is None
        ]
        if missing:
            parser.error("KL-waiver result is missing: " + ", ".join(missing))
        health_gate = _read(args.kl_waiver_health)
        sft_summaries = _summaries_by_seed(args.sft_summary, label="selected SFT")
        iter99_summaries = _summaries_by_seed(
            args.iter99_summary, label="long100 iter99"
        )
        iter199_summaries = _summaries_by_seed(
            args.iter199_summary, label="long200 iter199 KL waiver"
        )
        comparisons = _comparisons_by_seed(
            args.comparison,
            candidate=args.candidate,
            baseline=args.baseline,
        )
        probe_complete, probe_observation = _namespace_probe_observation(
            args.namespace_probe,
            checkpoint_root=args.eval_checkpoint_root,
            iteration=199,
            summaries=iter199_summaries,
        )
        try:
            report = evaluate_long200_kl_waiver_result(
                checkpoint_root=args.checkpoint_root,
                source_checkpoint_root=args.source_checkpoint_root,
                health_gate=health_gate,
                sft_summaries=sft_summaries,
                iter99_summaries=iter99_summaries,
                iter199_summaries=iter199_summaries,
                comparisons=comparisons,
                probe_complete=probe_complete,
                probe_observation=probe_observation,
                candidate=args.candidate,
                baseline=args.baseline,
                prior=args.prior,
            )
        except ValueError as exc:
            parser.error(str(exc))
        report["artifacts"] = {
            "health": str(args.kl_waiver_health.resolve()),
            "comparisons": [str(path.resolve()) for path in args.comparison],
        }
        _write(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        if report["status"] == "fail":
            raise SystemExit(1)
        return

    if args.early_stop_health is not None:
        full_inputs = (
            args.iter199_health,
            args.sft_summary,
            args.iter99_summary,
            args.iter199_summary,
            args.comparison,
            args.namespace_probe,
        )
        if any(value is not None for value in full_inputs):
            parser.error(
                "--early-stop-health cannot be combined with final evaluation inputs"
            )
        health_gate = _read(args.early_stop_health)
        try:
            report = evaluate_long200_early_reject(
                checkpoint_root=args.checkpoint_root,
                source_checkpoint_root=args.source_checkpoint_root,
                health_gate=health_gate,
                candidate=args.candidate,
                baseline=args.baseline,
                prior=args.prior,
            )
        except ValueError as exc:
            parser.error(str(exc))
        report["artifacts"] = {
            "early_stop_health": str(args.early_stop_health.resolve()),
            "comparisons": [],
        }
        _write(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(1)

    required_final_inputs = {
        "--iter199-health": args.iter199_health,
        "--sft-summary": args.sft_summary,
        "--iter99-summary": args.iter99_summary,
        "--iter199-summary": args.iter199_summary,
        "--comparison": args.comparison,
        "--namespace-probe": args.namespace_probe,
    }
    missing = [name for name, value in required_final_inputs.items() if value is None]
    if missing:
        parser.error("final decision is missing: " + ", ".join(missing))

    health_gate = _read(args.iter199_health)
    sft_summaries = _summaries_by_seed(args.sft_summary, label="selected SFT")
    iter99_summaries = _summaries_by_seed(args.iter99_summary, label="long100 iter99")
    iter199_summaries = _summaries_by_seed(args.iter199_summary, label="long200 iter199")
    comparisons = _comparisons_by_seed(
        args.comparison,
        candidate=args.candidate,
        baseline=args.baseline,
    )
    probe_complete, probe_observation = _namespace_probe_observation(
        args.namespace_probe,
        checkpoint_root=args.checkpoint_root,
        iteration=199,
        summaries=iter199_summaries,
    )
    report = evaluate_long200_decision(
        checkpoint_root=args.checkpoint_root,
        source_checkpoint_root=args.source_checkpoint_root,
        health_gate=health_gate,
        sft_summaries=sft_summaries,
        iter99_summaries=iter99_summaries,
        iter199_summaries=iter199_summaries,
        comparisons=comparisons,
        probe_complete=probe_complete,
        probe_observation=probe_observation,
        candidate=args.candidate,
        baseline=args.baseline,
        prior=args.prior,
    )
    report["artifacts"] = {
        "health": str(args.iter199_health.resolve()),
        "comparisons": [str(path.resolve()) for path in args.comparison],
    }
    _write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] == "reject":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
