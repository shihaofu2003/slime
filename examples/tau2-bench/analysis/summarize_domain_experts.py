#!/usr/bin/env python3
"""Build tau2 domain-expert curves, selections, and OPD eligibility decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from check_agent_boundary_rl_pilot import PROFILE
from compare_official_sft_evals import (
    EXPECTED_TEST_TASKS,
    METRIC_LABELS,
    aggregate_behavior,
    aggregate_metrics,
    load_model_eval,
    paired_bootstrap_ci,
    task_metrics,
)
from domain_expert_checkpoint import DOMAINS, EXPERT_ITERATIONS


CURVE_VERSION = "boundary-v2-domain-expert-curve-v1"
DECISION_VERSION = "boundary-v2-domain-expert-decision-v1"
SUMMARY_VERSION = "boundary-v2-domain-expert-summary-v1"
KEY_METRICS = ("pass_at_1", "pass_at_4_any")
NONINFERIORITY_MARGIN = 0.05


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def select_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    healthy = [
        item
        for item in candidates
        if item.get("health_passed") is True
        and item.get("protocol_passed") is True
    ]
    if not healthy:
        raise ValueError("no healthy protocol-valid checkpoint is selectable")
    return min(
        healthy,
        key=lambda item: (
            -float(item["metrics"]["pass_at_1"]),
            -float(item["metrics"]["pass_at_4_any"]),
            -float(item["metrics"]["pass_power_4"]),
            int(item["iteration"]),
        ),
    )


def classify_expert(
    *,
    expert_by_seed: dict[int, dict[str, float]],
    baseline_by_seed: dict[int, dict[str, float]],
    mixed_by_seed: dict[int, dict[str, float]],
    hard_conditions: dict[str, bool],
    margin: float = NONINFERIORITY_MARGIN,
) -> dict[str, Any]:
    if set(expert_by_seed) != {300, 301}:
        raise ValueError("expert decision requires seeds 300 and 301")
    comparator: dict[int, dict[str, float]] = {}
    deltas: dict[int, dict[str, float]] = {}
    for seed in (300, 301):
        comparator[seed] = {
            metric: max(
                float(baseline_by_seed[seed][metric]),
                float(mixed_by_seed[seed][metric]),
            )
            for metric in KEY_METRICS
        }
        deltas[seed] = {
            metric: float(expert_by_seed[seed][metric]) - comparator[seed][metric]
            for metric in KEY_METRICS
        }
    noninferior = all(
        delta >= -margin - 1e-12
        for seed_deltas in deltas.values()
        for delta in seed_deltas.values()
    )
    mean_delta = {
        metric: sum(deltas[seed][metric] for seed in (300, 301)) / 2
        for metric in KEY_METRICS
    }
    hard_passed = all(hard_conditions.values())
    if not hard_passed or not noninferior:
        classification = "reject"
    elif any(value > 0 for value in mean_delta.values()):
        classification = "advantage"
    else:
        classification = "near-parity"
    return {
        "classification": classification,
        "opd_eligible": classification in {"advantage", "near-parity"},
        "margin": margin,
        "hard_conditions": hard_conditions,
        "hard_conditions_passed": hard_passed,
        "comparator_by_seed": comparator,
        "delta_to_comparator_by_seed": deltas,
        "mean_delta_to_comparator": mean_delta,
    }


def _parse_mapping(spec: str, *, pattern: str, label: str) -> tuple[tuple[Any, ...], Path]:
    if "=" not in spec:
        raise ValueError(f"{label} must end in =SUMMARY.json: {spec}")
    key, raw_path = spec.split("=", 1)
    match = re.fullmatch(pattern, key)
    if not match:
        raise ValueError(f"invalid {label}: {spec}")
    values: list[Any] = []
    for value in match.groups():
        values.append(int(value) if value.isdigit() else value)
    return tuple(values), Path(raw_path).resolve()


def _load_eval(label: str, path: Path, results_root: Path) -> dict[str, Any]:
    return load_model_eval(
        label=label,
        summary_path=path,
        results_root=results_root,
        expected_profile=PROFILE,
    )


def _domain_metrics(item: dict[str, Any], domain: str) -> dict[str, float]:
    rewards = item["task_rewards"]
    tasks = [key for key in rewards if key[0] == domain]
    if len(tasks) != EXPECTED_TEST_TASKS[domain]:
        raise ValueError(
            f"{item['label']}/{domain}: expected {EXPECTED_TEST_TASKS[domain]} tasks, "
            f"found {len(tasks)}"
        )
    return aggregate_metrics(rewards, domain=domain)


def _paired_domain(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    domain: str,
    comparison: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    candidate_rewards = candidate["task_rewards"]
    reference_rewards = reference["task_rewards"]
    tasks = sorted(key for key in candidate_rewards if key[0] == domain)
    if set(tasks) != {key for key in reference_rewards if key[0] == domain}:
        raise ValueError(f"{comparison}: task sets differ")
    rows = []
    for metric in METRIC_LABELS:
        deltas = [
            task_metrics(candidate_rewards[key])[metric]
            - task_metrics(reference_rewards[key])[metric]
            for key in tasks
        ]
        salt = int.from_bytes(
            hashlib.sha256(f"{comparison}:{metric}".encode()).digest()[:4], "big"
        )
        observed, low, high = paired_bootstrap_ci(
            deltas,
            samples=bootstrap_samples,
            seed=bootstrap_seed + salt,
        )
        rows.append(
            {
                "comparison": comparison,
                "domain": domain,
                "metric": metric,
                "tasks": len(tasks),
                "delta": observed,
                "ci95": [low, high],
                "ci_excludes_zero": low > 0 or high < 0,
            }
        )
    return rows


def _training_raw_success(gate: dict[str, Any]) -> dict[str, float | None]:
    values = [
        metrics.get("rollout/raw_reward")
        for metrics in (gate.get("training") or {}).get("expected_step_metrics", {}).values()
    ]
    finite = [float(value) for value in values if isinstance(value, (int, float))]
    return {
        "mean": sum(finite) / len(finite) if finite else None,
        "last_10_mean": sum(finite[-10:]) / len(finite[-10:]) if finite else None,
    }


def build_curve_report(
    *,
    baseline: dict[str, Any],
    mixed: dict[int, dict[str, Any]],
    experts: dict[tuple[str, int], dict[str, Any]],
    gates: dict[tuple[str, int], dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if set(mixed) != set(EXPERT_ITERATIONS):
        raise ValueError("curve requires mixed iter109/119/129")
    expected_experts = {
        (domain, iteration) for domain in DOMAINS for iteration in EXPERT_ITERATIONS
    }
    if set(experts) != expected_experts:
        raise ValueError("curve requires all nine expert summaries")
    expected_gates = expected_experts | {
        ("mixed", iteration) for iteration in EXPERT_ITERATIONS
    }
    if set(gates) != expected_gates:
        raise ValueError("curve requires all nine expert and three mixed prefix Gates")

    models: dict[str, Any] = {
        "long100_iter99": {
            "by_domain": {
                domain: _domain_metrics(baseline, domain) for domain in DOMAINS
            }
        },
        "mixed": {},
        "experts": {},
    }
    paired = []
    selections = {}
    for iteration, item in sorted(mixed.items()):
        models["mixed"][str(iteration)] = {
            "by_domain": {
                domain: _domain_metrics(item, domain) for domain in DOMAINS
            },
            "health_status": gates[("mixed", iteration)].get("status"),
            "training_raw_success": _training_raw_success(gates[("mixed", iteration)]),
        }
    for domain in DOMAINS:
        candidates = []
        models["experts"][domain] = {}
        for iteration in EXPERT_ITERATIONS:
            item = experts[(domain, iteration)]
            metrics = _domain_metrics(item, domain)
            gate = gates[(domain, iteration)]
            protocol_passed = item["protocol_provenance"] == "signed-current"
            candidate = {
                "iteration": iteration,
                "metrics": metrics,
                "health_passed": gate.get("status") == "pass",
                "protocol_passed": protocol_passed,
            }
            candidates.append(candidate)
            models["experts"][domain][str(iteration)] = {
                **candidate,
                "training_raw_success": _training_raw_success(gate),
                "behavior": aggregate_behavior(item["simulation_rows"], domain=domain),
                "official_metrics": item["official_metrics_by_domain"].get(domain),
            }
            paired.extend(
                _paired_domain(
                    item,
                    mixed[iteration],
                    domain=domain,
                    comparison=f"{domain}_expert{iteration}-mixed{iteration}",
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                )
            )
        paired.extend(
            _paired_domain(
                experts[(domain, 109)],
                mixed[129],
                domain=domain,
                comparison=f"{domain}_expert109-mixed129_equal_domain_samples",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )
        selected = select_checkpoint(candidates)
        selections[domain] = {
            "iteration": selected["iteration"],
            "selection_order": [
                "health_and_protocol",
                "max_pass_at_1",
                "max_pass_at_4_any",
                "max_pass_power_4",
                "earliest_iteration",
            ],
            "metrics": selected["metrics"],
        }
    return {
        "status": "pass",
        "curve_version": CURVE_VERSION,
        "seed": 300,
        "protocol_profile": PROFILE,
        "comparisons": {
            "same_updates": [[109, 109], [119, 119], [129, 129]],
            "same_domain_task_groups": {
                "expert_iteration": 109,
                "mixed_iteration": 129,
                "domain_task_groups": 60,
                "trajectories": 480,
            },
        },
        "bootstrap": {
            "unit": "domain/task_id",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "interval": "percentile_95",
        },
        "models": models,
        "paired_deltas": paired,
        "selected_checkpoints": selections,
    }


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_curve(report: dict[str, Any]) -> str:
    lines = [
        "# Domain expert curve",
        "",
        "Seed 300; target-domain official test tasks × 4 trials; temperature 0.6; max_steps 200.",
        "",
        "## Target-domain checkpoints",
        "",
        "| Domain | Model | pass@1 | pass@4(any) | pass^4 | Health |",
        "|---|---|---:|---:|---:|---|",
    ]
    for domain in DOMAINS:
        baseline = report["models"]["long100_iter99"]["by_domain"][domain]
        lines.append(
            f"| {domain} | long100 iter99 | {_percent(baseline['pass_at_1'])} | "
            f"{_percent(baseline['pass_at_4_any'])} | {_percent(baseline['pass_power_4'])} | pass |"
        )
        for iteration in EXPERT_ITERATIONS:
            mixed = report["models"]["mixed"][str(iteration)]["by_domain"][domain]
            expert = report["models"]["experts"][domain][str(iteration)]
            lines.append(
                f"| {domain} | mixed iter{iteration} | {_percent(mixed['pass_at_1'])} | "
                f"{_percent(mixed['pass_at_4_any'])} | {_percent(mixed['pass_power_4'])} | "
                f"{report['models']['mixed'][str(iteration)]['health_status']} |"
            )
            metrics = expert["metrics"]
            lines.append(
                f"| {domain} | expert iter{iteration} | {_percent(metrics['pass_at_1'])} | "
                f"{_percent(metrics['pass_at_4_any'])} | {_percent(metrics['pass_power_4'])} | "
                f"{'pass' if expert['health_passed'] and expert['protocol_passed'] else 'fail'} |"
            )
    lines.extend(["", "## Selected checkpoints", "", "| Domain | Iteration | Rule winner |", "|---|---:|---|"])
    for domain in DOMAINS:
        selected = report["selected_checkpoints"][domain]
        lines.append(f"| {domain} | {selected['iteration']} | pass@1/pass@4(any)/pass^4/earlier |")
    lines.extend(
        [
            "",
            "## Paired intervals",
            "",
            "| Comparison | Metric | Delta | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in report["paired_deltas"]:
        lines.append(
            f"| {row['comparison']} | {METRIC_LABELS[row['metric']]} | "
            f"{100 * row['delta']:+.2f} pp | "
            f"[{100 * row['ci95'][0]:+.2f}, {100 * row['ci95'][1]:+.2f}] pp |"
        )
    return "\n".join(lines) + "\n"


def namespace_probe_passed(payload: dict[str, Any]) -> bool:
    summaries = payload.get("summaries") or {}
    return bool(
        payload.get("status") == "pass"
        and set(summaries) == {"temperature_0", "temperature_0_6"}
        and all(summary.get("pass") is True for summary in summaries.values())
    )


def build_decision_report(
    *,
    domain: str,
    iteration: int,
    expert: dict[int, dict[str, Any]],
    baseline: dict[int, dict[str, Any]],
    mixed: dict[int, dict[str, Any]],
    health_gate: dict[str, Any],
    forgetting: dict[str, Any],
    namespace_probe: dict[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics = {
        "expert": {seed: _domain_metrics(item, domain) for seed, item in expert.items()},
        "long100_iter99": {seed: _domain_metrics(item, domain) for seed, item in baseline.items()},
        "mixed_control": {seed: _domain_metrics(item, domain) for seed, item in mixed.items()},
    }
    hard_conditions = {
        "health": health_gate.get("status") == "pass",
        "protocol": all(
            item["protocol_provenance"] == "signed-current"
            for group in (expert, baseline, mixed)
            for item in group.values()
        ),
        "complete_seed_coverage": set(expert) == set(baseline) == set(mixed) == {300, 301},
        "forgetting_diagnostic_complete": set(forgetting["result_artifacts"]) == (set(DOMAINS) - {domain}),
        "namespace_probe": domain != "telecom" or namespace_probe_passed(namespace_probe),
    }
    classification = classify_expert(
        expert_by_seed=metrics["expert"],
        baseline_by_seed=metrics["long100_iter99"],
        mixed_by_seed=metrics["mixed_control"],
        hard_conditions=hard_conditions,
    )
    paired = []
    for seed in (300, 301):
        for reference_name, reference in (
            ("long100_iter99", baseline[seed]),
            (f"mixed_iter{iteration}", mixed[seed]),
        ):
            paired.extend(
                _paired_domain(
                    expert[seed],
                    reference,
                    domain=domain,
                    comparison=f"{domain}_expert{iteration}-{reference_name}_seed{seed}",
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                )
            )
    return {
        "status": "pass" if classification["classification"] != "reject" else "fail",
        "decision_version": DECISION_VERSION,
        "domain": domain,
        "selected_iteration": iteration,
        **classification,
        "metrics": metrics,
        "paired_deltas": paired,
        "diagnostics_only": {
            "pass_power_4": {
                model: {seed: values["pass_power_4"] for seed, values in by_seed.items()}
                for model, by_seed in metrics.items()
            },
            "forgetting": {
                other: aggregate_metrics(forgetting["task_rewards"], domain=other)
                for other in DOMAINS
                if other != domain
            },
            "forgetting_behavior": aggregate_behavior(forgetting["simulation_rows"]),
            "namespace_probe": namespace_probe,
            "health": health_gate.get("diagnostics_only"),
        },
        "opd_action": "eligibility_only_no_distillation_started",
    }


def render_decision(report: dict[str, Any]) -> str:
    return (
        f"# {report['domain']} expert decision\n\n"
        f"- Selected checkpoint: iter{report['selected_iteration']}\n"
        f"- Classification: `{report['classification']}`\n"
        f"- OPD eligible: `{str(report['opd_eligible']).lower()}`\n"
        f"- Margin: {100 * report['margin']:.0f} pp per seed for pass@1 and pass@4(any).\n"
        "- This artifact does not start distillation.\n"
    )


def build_summary(decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(decisions) != set(DOMAINS):
        raise ValueError("summary requires airline, retail, and telecom decisions")
    return {
        "status": "pass",
        "summary_version": SUMMARY_VERSION,
        "opd_started": False,
        "experts": {
            domain: {
                "selected_iteration": decisions[domain]["selected_iteration"],
                "classification": decisions[domain]["classification"],
                "opd_eligible": decisions[domain]["opd_eligible"],
            }
            for domain in DOMAINS
        },
        "eligible_domains": [
            domain for domain in DOMAINS if decisions[domain]["opd_eligible"]
        ],
    }


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Domain expert summary",
        "",
        "| Domain | Selected | Classification | OPD eligible |",
        "|---|---:|---|---|",
    ]
    for domain in DOMAINS:
        item = report["experts"][domain]
        lines.append(
            f"| {domain} | iter{item['selected_iteration']} | {item['classification']} | "
            f"{str(item['opd_eligible']).lower()} |"
        )
    lines.extend(["", "No OPD or distillation job is started by this experiment."])
    return "\n".join(lines) + "\n"


def _write_report(json_path: Path, markdown_path: Path, report: dict[str, Any], markdown: str) -> None:
    write_atomic(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    write_atomic(markdown_path, markdown)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    curve = subparsers.add_parser("curve")
    curve.add_argument("--baseline-summary", type=Path, required=True)
    curve.add_argument("--mixed-summary", action="append", required=True)
    curve.add_argument("--expert-summary", action="append", required=True)
    curve.add_argument("--health-gate", action="append", required=True)
    curve.add_argument("--results-root", type=Path, required=True)
    curve.add_argument("--bootstrap-samples", type=int, default=100_000)
    curve.add_argument("--bootstrap-seed", type=int, default=20260806)
    curve.add_argument("--json-output", type=Path, required=True)
    curve.add_argument("--markdown-output", type=Path, required=True)

    decision = subparsers.add_parser("decision")
    decision.add_argument("--domain", choices=DOMAINS, required=True)
    decision.add_argument("--iteration", type=int, choices=EXPERT_ITERATIONS, required=True)
    decision.add_argument("--expert-summary", action="append", required=True)
    decision.add_argument("--baseline-summary", action="append", required=True)
    decision.add_argument("--mixed-summary", action="append", required=True)
    decision.add_argument("--forgetting-summary", type=Path, required=True)
    decision.add_argument("--health-gate", type=Path, required=True)
    decision.add_argument("--namespace-probe", type=Path, required=True)
    decision.add_argument("--results-root", type=Path, required=True)
    decision.add_argument("--bootstrap-samples", type=int, default=100_000)
    decision.add_argument("--bootstrap-seed", type=int, default=20260806)
    decision.add_argument("--json-output", type=Path, required=True)
    decision.add_argument("--markdown-output", type=Path, required=True)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--decision", action="append", required=True)
    summary.add_argument("--json-output", type=Path, required=True)
    summary.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "curve":
            mixed_paths = dict(
                _parse_mapping(spec, pattern=r"(109|119|129)", label="mixed summary")
                for spec in args.mixed_summary
            )
            expert_paths = dict(
                _parse_mapping(
                    spec,
                    pattern=r"(airline|retail|telecom):(109|119|129)",
                    label="expert summary",
                )
                for spec in args.expert_summary
            )
            gate_paths = dict(
                _parse_mapping(
                    spec,
                    pattern=r"(airline|retail|telecom|mixed):(109|119|129)",
                    label="health Gate",
                )
                for spec in args.health_gate
            )
            results_root = args.results_root.resolve()
            report = build_curve_report(
                baseline=_load_eval("long100_iter99", args.baseline_summary.resolve(), results_root),
                mixed={
                    key[0]: _load_eval(f"mixed{key[0]}", path, results_root)
                    for key, path in mixed_paths.items()
                },
                experts={
                    key: _load_eval(f"{key[0]}_expert{key[1]}", path, results_root)
                    for key, path in expert_paths.items()
                },
                gates={key: read_json(path) for key, path in gate_paths.items()},
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
            _write_report(args.json_output, args.markdown_output, report, render_curve(report))
        elif args.command == "decision":
            def seed_paths(specs: list[str], label: str) -> dict[int, Path]:
                parsed = dict(
                    _parse_mapping(spec, pattern=r"(300|301)", label=label)
                    for spec in specs
                )
                return {key[0]: path for key, path in parsed.items()}

            results_root = args.results_root.resolve()
            expert_paths = seed_paths(args.expert_summary, "expert summary")
            baseline_paths = seed_paths(args.baseline_summary, "baseline summary")
            mixed_paths = seed_paths(args.mixed_summary, "mixed summary")
            report = build_decision_report(
                domain=args.domain,
                iteration=args.iteration,
                expert={seed: _load_eval(f"expert_seed{seed}", path, results_root) for seed, path in expert_paths.items()},
                baseline={seed: _load_eval(f"baseline_seed{seed}", path, results_root) for seed, path in baseline_paths.items()},
                mixed={seed: _load_eval(f"mixed_seed{seed}", path, results_root) for seed, path in mixed_paths.items()},
                health_gate=read_json(args.health_gate),
                forgetting=_load_eval("forgetting", args.forgetting_summary.resolve(), results_root),
                namespace_probe=read_json(args.namespace_probe),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
            _write_report(args.json_output, args.markdown_output, report, render_decision(report))
        else:
            decisions = {}
            for spec in args.decision:
                (domain,), path = _parse_mapping(
                    spec, pattern=r"(airline|retail|telecom)", label="decision"
                )
                decisions[domain] = read_json(path)
            report = build_summary(decisions)
            _write_report(args.json_output, args.markdown_output, report, render_summary(report))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
