#!/usr/bin/env python3
"""Summarize the signed boundary-v2 seed-300 checkpoint curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
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


BASELINE = "selected_sft"
LINEAGE = (
    "lineage_iter9",
    "lineage_iter39",
    "lineage_iter69",
    "lineage_iter89",
    "lineage_iter99",
)
BACKGROUND = ("historical_iter9", "raw_instruct")
EXPECTED_LABELS = (BASELINE, *LINEAGE, *BACKGROUND)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _model_report(item: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    task_rewards = item["task_rewards"]
    domains = sorted({domain for domain, _ in task_rewards})
    behavior = aggregate_behavior(item["simulation_rows"])
    namespace = (((summary.get("domains") or {}).get("telecom") or {}).get("namespace"))
    if not isinstance(namespace, dict):
        raise ValueError(f"{item['label']}: missing telecom namespace diagnostics")
    return {
        "tasks": len(task_rewards),
        "simulations": sum(len(rewards) for rewards in task_rewards.values()),
        "overall": aggregate_metrics(task_rewards),
        "by_domain": {
            domain: aggregate_metrics(task_rewards, domain=domain)
            for domain in domains
        },
        "behavior": behavior,
        "telecom_namespace": namespace,
    }


def _paired_rows(
    *,
    loaded: dict[str, dict[str, Any]],
    pairs: list[tuple[str, str, str]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    expected_tasks = set(loaded[BASELINE]["task_rewards"])
    domains = sorted({domain for domain, _ in expected_tasks})
    for comparison_type, candidate, reference in pairs:
        candidate_rewards = loaded[candidate]["task_rewards"]
        reference_rewards = loaded[reference]["task_rewards"]
        for scope, selected_domain in [("overall", None), *[(d, d) for d in domains]]:
            selected_tasks = [
                key
                for key in sorted(expected_tasks)
                if selected_domain is None or key[0] == selected_domain
            ]
            for metric in METRIC_LABELS:
                deltas = [
                    task_metrics(candidate_rewards[key])[metric]
                    - task_metrics(reference_rewards[key])[metric]
                    for key in selected_tasks
                ]
                salt = int.from_bytes(
                    hashlib.sha256(
                        f"{comparison_type}:{candidate}:{reference}:{scope}:{metric}".encode()
                    ).digest()[:4],
                    "big",
                )
                observed, low, high = paired_bootstrap_ci(
                    deltas,
                    samples=samples,
                    seed=seed + salt,
                )
                rows.append(
                    {
                        "comparison_type": comparison_type,
                        "comparison": f"{candidate}-{reference}",
                        "candidate": candidate,
                        "reference": reference,
                        "scope": scope,
                        "metric": metric,
                        "tasks": len(deltas),
                        "delta": observed,
                        "ci95": [low, high],
                        "ci_excludes_zero": low > 0 or high < 0,
                    }
                )
    return rows


def build_curve_report(
    *,
    loaded: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if tuple(loaded) != EXPECTED_LABELS or set(summaries) != set(EXPECTED_LABELS):
        raise ValueError(f"curve requires labels in this exact order: {EXPECTED_LABELS}")
    task_sets = {label: set(item["task_rewards"]) for label, item in loaded.items()}
    if len({frozenset(tasks) for tasks in task_sets.values()}) != 1:
        raise ValueError("checkpoint-curve task sets differ")
    domain_counts = Counter(domain for domain, _ in task_sets[BASELINE])
    if dict(sorted(domain_counts.items())) != EXPECTED_TEST_TASKS:
        raise ValueError(f"official task coverage mismatch: {dict(domain_counts)}")
    signatures = {
        json.dumps(item["signature"], sort_keys=True) for item in loaded.values()
    }
    contracts = {
        json.dumps(item["agent_contract_signatures"], sort_keys=True)
        for item in loaded.values()
    }
    if len(signatures) != 1 or len(contracts) != 1:
        raise ValueError("curve models do not share one signed evaluation protocol")

    models = {
        label: {
            **_model_report(loaded[label], summaries[label]),
            "role": (
                "baseline"
                if label == BASELINE
                else "same_lineage" if label in LINEAGE else "background"
            ),
        }
        for label in EXPECTED_LABELS
    }
    pairs = [("relative_to_sft", label, BASELINE) for label in LINEAGE]
    pairs.extend(
        ("adjacent_checkpoint", candidate, reference)
        for reference, candidate in zip(LINEAGE, LINEAGE[1:])
    )
    paired = _paired_rows(
        loaded=loaded,
        pairs=pairs,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    adjacent_trend = []
    for reference, candidate in zip(LINEAGE, LINEAGE[1:]):
        deltas = {
            metric: models[candidate]["overall"][metric]
            - models[reference]["overall"][metric]
            for metric in ("pass_at_1", "pass_at_4_any")
        }
        score = sum(deltas.values())
        adjacent_trend.append(
            {
                "comparison": f"{candidate}-{reference}",
                "direction": "rise" if score > 0 else "decline" if score < 0 else "plateau",
                "deltas": deltas,
            }
        )
    return {
        "status": "pass",
        "curve_version": "boundary-v2-checkpoint-curve-v1",
        "protocol_profile": PROFILE,
        "seed": 300,
        "bootstrap": {
            "unit": "domain/task_id",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "interval": "percentile_95",
        },
        "same_lineage": list(LINEAGE),
        "background_controls": list(BACKGROUND),
        "trend_scope": "same-lineage checkpoints only",
        "selection_policy": "diagnostic_only_no_automatic_model_selection",
        "models": models,
        "adjacent_trend": adjacent_trend,
        "paired_deltas": paired,
        "artifacts": {
            label: {
                "summary_path": loaded[label]["summary_path"],
                "summary_sha256": loaded[label]["summary_sha256"],
                "result_artifacts": loaded[label]["result_artifacts"],
            }
            for label in EXPECTED_LABELS
        },
    }


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# boundary-v2 checkpoint curve",
        "",
        "## Scope",
        "",
        "Seed 300, 100 test tasks × 4 trials, temperature 0.6, max_steps 200. "
        "Trend statements use only the same-lineage iter9/39/69/89/99 checkpoints; "
        "historical iter9 and raw Instruct are background controls. No model is selected automatically.",
        "",
        "## Overall",
        "",
        "| Model | Role | pass@1 | pass@4(any) | pass^4 | too_many_errors | max_steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, model in report["models"].items():
        reasons = model["behavior"]["termination_reasons"]
        metrics = model["overall"]
        lines.append(
            f"| {label} | {model['role']} | {_percent(metrics['pass_at_1'])} | "
            f"{_percent(metrics['pass_at_4_any'])} | {_percent(metrics['pass_power_4'])} | "
            f"{reasons.get('too_many_errors', 0)} | {reasons.get('max_steps', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Domains and telecom namespace",
            "",
            "| Model | Domain | pass@1 | pass@4(any) | pass^4 | namespace affected/attempts/terminations |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, model in report["models"].items():
        namespace = model["telecom_namespace"]
        rendered_namespace = (
            f"{namespace.get('affected_trajectory_count', 0)}/"
            f"{namespace.get('raw_attempt_count', 0)}/"
            f"{namespace.get('namespace_attributed_termination_count', 0)}"
        )
        for domain, metrics in model["by_domain"].items():
            lines.append(
                f"| {label} | {domain} | {_percent(metrics['pass_at_1'])} | "
                f"{_percent(metrics['pass_at_4_any'])} | {_percent(metrics['pass_power_4'])} | "
                f"{rendered_namespace if domain == 'telecom' else '—'} |"
            )
    lines.extend(
        [
            "",
            "## Same-lineage direction",
            "",
            "| Adjacent checkpoints | Direction | Δ pass@1 | Δ pass@4(any) |",
            "|---|---|---:|---:|",
        ]
    )
    for row in report["adjacent_trend"]:
        lines.append(
            f"| {row['comparison']} | {row['direction']} | "
            f"{100 * row['deltas']['pass_at_1']:+.2f} pp | "
            f"{100 * row['deltas']['pass_at_4_any']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap intervals",
            "",
            "| Type | Comparison | Scope | Metric | Delta | 95% CI |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in report["paired_deltas"]:
        lines.append(
            f"| {row['comparison_type']} | {row['comparison']} | {row['scope']} | "
            f"{METRIC_LABELS[row['metric']]} | {100 * row['delta']:+.2f} pp | "
            f"[{100 * row['ci95'][0]:+.2f}, {100 * row['ci95'][1]:+.2f}] pp |"
        )
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    parsed = []
    for spec in args.model:
        if "=" not in spec:
            parser.error(f"--model must be LABEL=SUMMARY.json: {spec}")
        label, raw_path = spec.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            parser.error(f"invalid model label: {label}")
        parsed.append((label, Path(raw_path).resolve()))
    if tuple(label for label, _ in parsed) != EXPECTED_LABELS:
        parser.error(f"models must be provided in this exact order: {EXPECTED_LABELS}")
    loaded = {
        label: load_model_eval(
            label=label,
            summary_path=path,
            results_root=args.results_root.resolve(),
            expected_profile=PROFILE,
        )
        for label, path in parsed
    }
    summaries = {label: _read(path) for label, path in parsed}
    report = build_curve_report(
        loaded=loaded,
        summaries=summaries,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_atomic(
        args.json_output,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    write_atomic(args.markdown_output, render_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
