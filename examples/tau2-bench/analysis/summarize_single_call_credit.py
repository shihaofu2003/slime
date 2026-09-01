#!/usr/bin/env python3
"""Summarize the strict-single-v1 credit-assignment experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from compare_official_sft_evals import paired_bootstrap_ci, task_metrics


LABELS = ("sft", "v1-matched", "v2-l000", "v2-l010")
ARMS = LABELS[1:]
SEEDS = (300, 301)
EXPECTED_TASKS = {"airline": 20, "retail": 40, "telecom": 40}
PASS_METRICS = ("pass_at_1", "pass_at_4_any", "pass_power_4")
PRIMARY_METRICS = PASS_METRICS[:2]
RULE_DIAGNOSTICS = (
    "agent_errors_by_severity",
    "user_errors_by_severity",
    "sims_by_max_agent_severity",
    "sims_by_max_user_severity",
    "sims_by_first_critical_source",
    "agent_error_tags_by_severity",
    "user_error_tags_by_severity",
)
NAMESPACE_COUNTS = (
    "trajectory_count",
    "affected_trajectory_count",
    "raw_attempt_count",
    "parsed_call_count",
    "executed_call_count",
    "namespace_attributed_termination_count",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def parse_summary_specs(specs: list[str]) -> dict[str, dict[int, dict[str, Any]]]:
    records: dict[str, dict[int, dict[str, Any]]] = {
        label: {} for label in LABELS
    }
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--summary must be LABEL=SUMMARY.json: {spec}")
        label, raw_path = spec.split("=", 1)
        if label not in records:
            raise ValueError(f"unknown summary label {label!r}; expected {LABELS}")
        path = Path(raw_path).resolve()
        payload = read_json(path)
        seed = payload.get("seed")
        if seed not in SEEDS:
            raise ValueError(f"{label}: expected summary seed 300 or 301, found {seed!r}")
        if seed in records[label]:
            raise ValueError(f"duplicate summary for {label} seed {seed}")
        records[label][seed] = {"path": path, "summary": payload}

    missing = [
        f"{label}/seed{seed}"
        for label in LABELS
        for seed in SEEDS
        if seed not in records[label]
    ]
    if missing:
        raise ValueError(f"missing required summaries: {', '.join(missing)}")
    return records


def validate_summaries(records: dict[str, dict[int, dict[str, Any]]]) -> None:
    expected_protocol = {
        "task_split_name": "test",
        "num_trials": 4,
        "max_steps": 200,
        "max_errors": 10,
        "agent_protocol_profile": "strict-single-v1",
    }
    for label in LABELS:
        for seed in SEEDS:
            summary = records[label][seed]["summary"]
            if summary.get("seed") != seed:
                raise ValueError(f"{label}/seed{seed}: summary seed does not match its slot")
            actual_protocol = {
                key: summary.get(key) for key in expected_protocol
            }
            if actual_protocol != expected_protocol:
                raise ValueError(
                    f"{label}/seed{seed}: unexpected evaluation protocol: "
                    f"{actual_protocol}"
                )
            domains = summary.get("domains")
            if not isinstance(domains, dict) or set(domains) != set(EXPECTED_TASKS):
                raise ValueError(f"{label}/seed{seed}: expected all three tau2 domains")
            for domain, task_count in EXPECTED_TASKS.items():
                domain_summary = domains[domain]
                pass_metrics = domain_summary.get("pass_metrics") or {}
                if (
                    pass_metrics.get("tasks") != task_count
                    or pass_metrics.get("simulations") != 4 * task_count
                ):
                    raise ValueError(
                        f"{label}/seed{seed}/{domain}: official task coverage mismatch"
                    )
                metrics = domain_summary.get("metrics") or {}
                if metrics.get("infra_error_count") != 0:
                    raise ValueError(
                        f"{label}/seed{seed}/{domain}: evaluation has infrastructure errors"
                    )
                for metric in PASS_METRICS:
                    value = pass_metrics.get(metric)
                    if (
                        not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or not 0 <= value <= 1
                    ):
                        raise ValueError(
                            f"{label}/seed{seed}/{domain}: invalid {metric}={value!r}"
                        )


def _merge_count_maps(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            nested = target.setdefault(key, {})
            if isinstance(nested, dict):
                _merge_count_maps(nested, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value


def _selected_domains(summary: dict[str, Any], domain: str | None) -> list[dict[str, Any]]:
    domains = summary["domains"]
    return [domains[domain]] if domain is not None else list(domains.values())


def _accuracy(
    summaries: list[dict[str, Any]],
    *,
    domain: str | None,
) -> tuple[float | None, float | None, dict[str, int]]:
    action_accuracies = []
    db_accuracies = []
    action_correct = 0
    action_total = 0
    db_correct = 0
    db_total = 0
    for summary in summaries:
        seed_action_correct = 0
        seed_action_total = 0
        seed_db_correct = 0
        seed_db_total = 0
        for domain_summary in _selected_domains(summary, domain):
            metrics = domain_summary["metrics"]
            seed_action_correct += int(metrics.get("correct_read_actions", 0))
            seed_action_correct += int(metrics.get("correct_write_actions", 0))
            seed_action_total += int(metrics.get("total_read_actions", 0))
            seed_action_total += int(metrics.get("total_write_actions", 0))
            seed_db_correct += int(metrics.get("db_match_count", 0))
            seed_db_total += int(metrics.get("db_match_count", 0))
            seed_db_total += int(metrics.get("db_mismatch_count", 0))
        if seed_action_total:
            action_accuracies.append(seed_action_correct / seed_action_total)
        if seed_db_total:
            db_accuracies.append(seed_db_correct / seed_db_total)
        action_correct += seed_action_correct
        action_total += seed_action_total
        db_correct += seed_db_correct
        db_total += seed_db_total
    return (
        sum(action_accuracies) / len(action_accuracies) if action_accuracies else None,
        sum(db_accuracies) / len(db_accuracies) if db_accuracies else None,
        {
            "correct_actions": action_correct,
            "total_actions": action_total,
            "db_matches": db_correct,
            "db_checked": db_total,
        },
    )


def _scope_report(
    summaries: list[dict[str, Any]],
    *,
    domain: str | None,
) -> dict[str, Any]:
    rows = [
        summary["overall"] if domain is None else summary["domains"][domain]["pass_metrics"]
        for summary in summaries
    ]
    pass_metrics = {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in PASS_METRICS
    }
    action_accuracy, db_accuracy, counts = _accuracy(summaries, domain=domain)
    terminations: dict[str, Any] = {}
    rule_diagnostics: dict[str, Any] = {}
    single_call: dict[str, Any] = {}
    namespace = {key: 0 for key in NAMESPACE_COUNTS}
    for summary in summaries:
        for domain_summary in _selected_domains(summary, domain):
            metrics = domain_summary["metrics"]
            _merge_count_maps(
                terminations,
                {
                    key: value
                    for key, value in metrics.items()
                    if key.startswith("termination_")
                },
            )
            _merge_count_maps(
                rule_diagnostics,
                {
                    key: metrics.get(key, {})
                    for key in RULE_DIAGNOSTICS
                    if isinstance(metrics.get(key), dict)
                },
            )
            _merge_count_maps(single_call, domain_summary.get("single_call") or {})
            namespace_diagnostics = domain_summary.get("namespace") or {}
            for key in NAMESPACE_COUNTS:
                value = namespace_diagnostics.get(key, 0)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    namespace[key] += value
    return {
        **pass_metrics,
        "action_accuracy": action_accuracy,
        "db_accuracy": db_accuracy,
        "diagnostic_counts": counts,
        "terminations": terminations,
        "rule_diagnostics": rule_diagnostics,
        "single_call": single_call,
        "namespace": namespace,
    }


def build_models(
    records: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    return {
        label: {
            "seeds": list(SEEDS),
            "overall": _scope_report(
                [records[label][seed]["summary"] for seed in SEEDS],
                domain=None,
            ),
            "by_domain": {
                domain: _scope_report(
                    [records[label][seed]["summary"] for seed in SEEDS],
                    domain=domain,
                )
                for domain in EXPECTED_TASKS
            },
        }
        for label in LABELS
    }


def decide(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = {
        arm: tuple(models[arm]["overall"][metric] for metric in PRIMARY_METRICS)
        for arm in ARMS
    }
    strict_winners = [
        arm
        for arm in ARMS
        if all(
            all(primary[arm][index] > primary[other][index] for index in range(2))
            for other in ARMS
            if other != arm
        )
    ]
    if strict_winners:
        selected = strict_winners[0]
        return {
            "selected_arm": selected,
            "changed_from_v1": selected != "v1-matched",
            "reason": "strictly_higher_pass_at_1_and_pass_at_4_any_than_both_other_arms",
            "primary_metrics": list(PRIMARY_METRICS),
            "pass_power_4_role": "report_only",
        }

    tied_pairs: dict[tuple[float, float], list[str]] = defaultdict(list)
    for arm, values in primary.items():
        tied_pairs[values].append(arm)
    leading_ties = [
        tied
        for values, tied in tied_pairs.items()
        if len(tied) > 1
        and all(
            values[index] >= primary[other][index]
            for other in ARMS
            for index in range(2)
        )
    ]
    if leading_ties:
        tied = leading_ties[0]
        def diagnostic_value(arm: str, metric: str) -> float:
            value = models[arm]["overall"][metric]
            return float(value) if value is not None else -math.inf

        db_best = max(
            diagnostic_value(arm, "db_accuracy") for arm in tied
        )
        db_tied = [
            arm for arm in tied if diagnostic_value(arm, "db_accuracy") == db_best
        ]
        if len(db_tied) == 1:
            selected = db_tied[0]
            reason = "exact_primary_pass_tie_broken_by_db_accuracy"
        else:
            action_best = max(
                diagnostic_value(arm, "action_accuracy") for arm in db_tied
            )
            action_tied = [
                arm
                for arm in db_tied
                if diagnostic_value(arm, "action_accuracy") == action_best
            ]
            if len(action_tied) == 1:
                selected = action_tied[0]
                reason = "exact_primary_and_db_tie_broken_by_action_accuracy"
            else:
                selected = "v1-matched"
                reason = "exact_tie_unresolved_retain_v1"
        return {
            "selected_arm": selected,
            "changed_from_v1": selected != "v1-matched",
            "reason": reason,
            "primary_metrics": list(PRIMARY_METRICS),
            "pass_power_4_role": "report_only",
        }

    return {
        "selected_arm": "v1-matched",
        "changed_from_v1": False,
        "reason": "mixed_or_no_strict_primary_winner_retain_v1",
        "primary_metrics": list(PRIMARY_METRICS),
        "pass_power_4_role": "report_only",
    }


def _resolve_results_path(
    reference: str,
    *,
    summary_path: Path,
    results_root: Path | None,
) -> Path | None:
    path = Path(reference)
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute() and results_root is not None:
        candidates.append(results_root / path)
    if not path.is_absolute():
        candidates.append(summary_path.parent / path)
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _load_raw_task_rewards(
    record: dict[str, Any],
    *,
    results_root: Path | None,
) -> tuple[dict[tuple[str, str], list[float]] | None, list[str]]:
    summary = record["summary"]
    seed = summary["seed"]
    missing = []
    trials_by_task: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for domain in EXPECTED_TASKS:
        reference = summary["domains"][domain].get("results_file")
        if not isinstance(reference, str):
            missing.append(f"seed{seed}/{domain}:missing results_file")
            continue
        result_path = _resolve_results_path(
            reference,
            summary_path=record["path"],
            results_root=results_root,
        )
        if result_path is None:
            missing.append(reference)
            continue
        result = read_json(result_path)
        info = result.get("info") or {}
        if info.get("seed") != seed or info.get("num_trials") != 4:
            raise ValueError(f"raw result protocol mismatch: {result_path}")
        for simulation in result.get("simulations") or []:
            task_id = str(simulation.get("task_id") or "")
            trial = simulation.get("trial")
            reward = (simulation.get("reward_info") or {}).get("reward")
            if not task_id or not isinstance(trial, int) or not isinstance(reward, (int, float)):
                raise ValueError(f"malformed simulation in {result_path}")
            key = (domain, task_id)
            if trial in trials_by_task[key]:
                raise ValueError(f"duplicate {domain}/{task_id}/trial{trial} in {result_path}")
            trials_by_task[key][trial] = float(reward)
    if missing:
        return None, missing
    rewards = {
        key: [trials[index] for index in range(4)]
        for key, trials in trials_by_task.items()
        if set(trials) == set(range(4))
    }
    expected_count = sum(EXPECTED_TASKS.values())
    if len(rewards) != expected_count:
        raise ValueError(
            f"seed{seed}: raw results contain {len(rewards)} complete tasks, "
            f"expected {expected_count}"
        )
    for values in rewards.values():
        task_metrics(values)
    return rewards, []


def build_paired_bootstrap(
    *,
    records: dict[str, dict[int, dict[str, Any]]],
    models: dict[str, dict[str, Any]],
    results_root: Path | None,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    raw: dict[str, dict[int, dict[tuple[str, str], list[float]]]] = {
        label: {} for label in LABELS
    }
    missing = []
    for label in LABELS:
        for eval_seed in SEEDS:
            task_rewards, missing_paths = _load_raw_task_rewards(
                records[label][eval_seed],
                results_root=results_root,
            )
            if task_rewards is not None:
                raw[label][eval_seed] = task_rewards
            missing.extend(f"{label}/seed{eval_seed}:{path}" for path in missing_paths)
    if missing:
        return {
            "status": "unavailable",
            "reason": "raw per-task result files referenced by the summaries are unavailable",
            "missing": missing,
            "used_for_decision": False,
            "rows": [],
        }

    task_sets = {
        (label, eval_seed): set(raw[label][eval_seed])
        for label in LABELS
        for eval_seed in SEEDS
    }
    if len({frozenset(tasks) for tasks in task_sets.values()}) != 1:
        raise ValueError("raw result task sets differ across models or seeds")
    all_tasks = sorted(next(iter(task_sets.values())))
    pairs = [(arm, "sft") for arm in ARMS]
    pairs.extend(
        [
            ("v2-l000", "v1-matched"),
            ("v2-l010", "v1-matched"),
            ("v2-l010", "v2-l000"),
        ]
    )
    rows = []
    scopes = [("overall", None), *((domain, domain) for domain in EXPECTED_TASKS)]
    for pair_index, (candidate, reference) in enumerate(pairs):
        for scope_index, (scope, selected_domain) in enumerate(scopes):
            selected_tasks = [
                key
                for key in all_tasks
                if selected_domain is None or key[0] == selected_domain
            ]
            for metric_index, metric in enumerate(PASS_METRICS):
                deltas = [
                    sum(
                        task_metrics(raw[candidate][eval_seed][key])[metric]
                        - task_metrics(raw[reference][eval_seed][key])[metric]
                        for eval_seed in SEEDS
                    )
                    / len(SEEDS)
                    for key in selected_tasks
                ]
                observed, low, high = paired_bootstrap_ci(
                    deltas,
                    samples=samples,
                    seed=seed + pair_index * 100 + scope_index * 10 + metric_index,
                )
                candidate_metrics = (
                    models[candidate]["overall"]
                    if selected_domain is None
                    else models[candidate]["by_domain"][selected_domain]
                )
                reference_metrics = (
                    models[reference]["overall"]
                    if selected_domain is None
                    else models[reference]["by_domain"][selected_domain]
                )
                expected_value = candidate_metrics[metric] - reference_metrics[metric]
                if not math.isclose(observed, expected_value, abs_tol=1e-12):
                    raise ValueError(
                        f"raw/summary metric mismatch for {candidate}-{reference}/"
                        f"{scope}/{metric}: {observed} != {expected_value}"
                    )
                rows.append(
                    {
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
    return {
        "status": "available",
        "unit": "domain/task_id; each task delta is averaged across seeds 300 and 301",
        "samples": samples,
        "seed": seed,
        "interval": "percentile_95",
        "used_for_decision": False,
        "rows": rows,
    }


def build_report(
    *,
    records: dict[str, dict[int, dict[str, Any]]],
    results_root: Path | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    validate_summaries(records)
    models = build_models(records)
    return {
        "report_version": "strict-single-credit-comparison-v1",
        "protocol": {
            "profile": "strict-single-v1",
            "task_split_name": "test",
            "seeds": list(SEEDS),
            "num_trials": 4,
            "max_steps": 200,
            "max_errors": 10,
        },
        "selection_policy": {
            "arms": list(ARMS),
            "primary_metrics": list(PRIMARY_METRICS),
            "pass_power_4": "report_only",
            "exact_primary_tie_breakers": ["db_accuracy", "action_accuracy"],
            "mixed_or_unresolved_result": "retain_v1-matched",
        },
        "decision": decide(models),
        "models": models,
        "paired_bootstrap": build_paired_bootstrap(
            records=records,
            models=models,
            results_root=results_root,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "artifacts": {
            label: {
                f"seed{seed}": str(records[label][seed]["path"])
                for seed in SEEDS
            }
            for label in LABELS
        },
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _flatten_nonzero(value: dict[str, Any], prefix: str = "") -> list[str]:
    rows = []
    for key, item in sorted(value.items()):
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            rows.extend(_flatten_nonzero(item, name))
        elif item:
            rows.append(f"{name}={item}")
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# strict-single credit assignment comparison",
        "",
        "## Decision",
        "",
        f"Selected arm: `{decision['selected_arm']}`; reason: `{decision['reason']}`. "
        "The decision uses only the two-seed mean pass@1 and pass@4(any); pass^4 "
        "and bootstrap intervals are report-only.",
        "",
        "## Overall",
        "",
        "| Model | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        metrics = report["models"][label]["overall"]
        lines.append(
            f"| {label} | {_percent(metrics['pass_at_1'])} | "
            f"{_percent(metrics['pass_at_4_any'])} | "
            f"{_percent(metrics['pass_power_4'])} | "
            f"{_percent(metrics['action_accuracy'])} | "
            f"{_percent(metrics['db_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Domains",
            "",
            "| Model | Domain | pass@1 | pass@4(any) | pass^4 | Action accuracy | DB accuracy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        for domain in EXPECTED_TASKS:
            metrics = report["models"][label]["by_domain"][domain]
            lines.append(
                f"| {label} | {domain} | {_percent(metrics['pass_at_1'])} | "
                f"{_percent(metrics['pass_at_4_any'])} | "
                f"{_percent(metrics['pass_power_4'])} | "
                f"{_percent(metrics['action_accuracy'])} | "
                f"{_percent(metrics['db_accuracy'])} |"
            )
    lines.extend(
        [
            "",
            "## Termination and rule diagnostics",
            "",
            "Counts pool both evaluation seeds. Empty diagnostics are shown as `none`.",
            "",
            "| Model | Scope | Terminations | Rule diagnostics | Protocol diagnostics |",
            "|---|---|---|---|---|",
        ]
    )
    for label in LABELS:
        scopes = [("overall", report["models"][label]["overall"])]
        scopes.extend(report["models"][label]["by_domain"].items())
        for scope, metrics in scopes:
            terminations = ", ".join(_flatten_nonzero(metrics["terminations"])) or "none"
            rules = ", ".join(_flatten_nonzero(metrics["rule_diagnostics"])) or "none"
            protocol = {
                "single_call": {
                    key: value
                    for key, value in metrics["single_call"].items()
                    if key != "trajectory_count"
                },
                "namespace": {
                    key: value
                    for key, value in metrics["namespace"].items()
                    if key != "trajectory_count"
                },
            }
            rendered_protocol = ", ".join(_flatten_nonzero(protocol)) or "none"
            lines.append(
                f"| {label} | {scope} | {terminations} | {rules} | {rendered_protocol} |"
            )
    bootstrap = report["paired_bootstrap"]
    lines.extend(["", "## Paired task bootstrap", ""])
    if bootstrap["status"] != "available":
        lines.append(
            "Unavailable: raw per-task result files referenced by one or more summaries "
            "could not be resolved. This does not affect point estimates or the decision."
        )
    else:
        lines.extend(
            [
                f"Descriptive only; {bootstrap['samples']:,} resamples over 100 matched tasks. "
                "Each task delta is averaged across seeds 300 and 301 before resampling.",
                "",
                "| Comparison | Metric | Delta | 95% paired bootstrap CI |",
                "|---|---|---:|---:|",
            ]
        )
        for row in bootstrap["rows"]:
            if row["scope"] != "overall":
                continue
            lines.append(
                f"| {row['comparison']} | {row['metric']} | "
                f"{100 * row['delta']:+.2f} pp | "
                f"[{100 * row['ci95'][0]:+.2f}, {100 * row['ci95'][1]:+.2f}] pp |"
            )
    return "\n".join(lines) + "\n"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help=(
            "LABEL=SUMMARY.json; provide seed300 and seed301 for each of "
            "sft, v1-matched, v2-l000, and v2-l010"
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Root used to resolve data/simulations paths for paired bootstrap",
    )
    parser.add_argument("--bootstrap-samples", type=positive_int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = parse_summary_specs(args.summary)
    report = build_report(
        records=records,
        results_root=args.results_root.resolve() if args.results_root else None,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
