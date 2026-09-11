#!/usr/bin/env python3
"""Summarize model results for the expanded Banking curriculum."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACTS = (
    PROJECT_ROOT
    / "output/experiments/tau2-banking-task-curriculum/expanded_contracts.jsonl"
)
RETRY_PATTERN = re.compile(
    r"Task (?P<task_id>\S+) failed \(attempt (?P<attempt>\d+)/\d+\): (?P<reason>.*)"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def grouped_accuracy(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        value: {
            "tasks": len(items),
            "correct": sum(item["correct"] for item in items),
            "accuracy": sum(item["correct"] for item in items) / len(items),
            "mean_reward": sum(item["reward"] for item in items) / len(items),
        }
        for value, items in sorted(grouped.items())
    }


def summarize_results(
    results_paths: list[Path],
    contracts: list[dict[str, Any]],
    *,
    replacement_results_paths: Optional[list[Path]] = None,
    expected_task_count: Optional[int] = None,
    run_logs: Optional[list[Path]] = None,
) -> dict[str, Any]:
    contract_by_id = {item["task_id"]: item for item in contracts}
    if len(contract_by_id) != len(contracts):
        raise ValueError("contract ids must be unique")

    simulations_by_task_id: dict[str, dict[str, Any]] = {}
    models = set()
    replacement_task_ids = set()
    for results_path in results_paths:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        agent_info = ((payload.get("info") or {}).get("agent_info") or {})
        if agent_info.get("llm"):
            models.add(agent_info["llm"])
        for simulation in payload.get("simulations") or []:
            task_id = simulation["task_id"]
            if task_id in simulations_by_task_id:
                raise ValueError(f"duplicate evaluated task: {task_id}")
            simulations_by_task_id[task_id] = simulation

    for results_path in replacement_results_paths or []:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        agent_info = ((payload.get("info") or {}).get("agent_info") or {})
        if agent_info.get("llm"):
            models.add(agent_info["llm"])
        for simulation in payload.get("simulations") or []:
            task_id = simulation["task_id"]
            if task_id not in simulations_by_task_id:
                raise ValueError(f"replacement task was not in base results: {task_id}")
            simulations_by_task_id[task_id] = simulation
            replacement_task_ids.add(task_id)

    rows = []
    atomic_checks: dict[str, list[bool]] = defaultdict(list)
    action_checks_by_requestor: dict[str, list[bool]] = defaultdict(list)
    for task_id, simulation in simulations_by_task_id.items():
        if task_id not in contract_by_id:
            raise ValueError(f"missing curriculum contract: {task_id}")
        contract = contract_by_id[task_id]
        reward_info = simulation.get("reward_info") or {}
        reward = float(reward_info.get("reward") or 0.0)
        for check in reward_info.get("action_checks") or []:
            matched = bool(check.get("action_match"))
            atomic_checks["ACTION"].append(matched)
            requestor = ((check.get("action") or {}).get("requestor") or "unknown")
            action_checks_by_requestor[str(requestor)].append(matched)
        for check in reward_info.get("communicate_checks") or []:
            atomic_checks["COMMUNICATE"].append(bool(check.get("met")))
        db_check = reward_info.get("db_check") or {}
        if db_check.get("db_match") is not None:
            atomic_checks["DB"].append(bool(db_check["db_match"]))
        for check in reward_info.get("nl_assertions") or []:
            atomic_checks["NL_ASSERTION"].append(bool(check.get("met")))
        rows.append(
            {
                "task_id": task_id,
                "source_task_id": contract["source_task_id"],
                "variant": contract["variant"],
                "level": contract["level"],
                "business_category": contract["business_category"],
                "retrieval_variant": contract["retrieval_variant"],
                "reward": reward,
                "correct": int(reward == 1.0),
                "termination_reason": simulation.get("termination_reason"),
                "reward_breakdown": reward_info.get("reward_breakdown") or {},
            }
        )

    if expected_task_count is not None and len(rows) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} evaluated tasks, found {len(rows)}"
        )
    if not rows:
        raise ValueError("no simulations found")
    if len(models) > 1:
        raise ValueError(f"results contain multiple Agent models: {sorted(models)}")

    component_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for component, value in row["reward_breakdown"].items():
            if isinstance(value, (int, float)):
                component_values[component].append(float(value))

    correct = sum(row["correct"] for row in rows)
    summary = {
        "model": next(iter(models), None),
        "results_files": [str(path) for path in results_paths],
        "replacement_results_files": [
            str(path) for path in replacement_results_paths or []
        ],
        "replacement_task_ids": sorted(replacement_task_ids),
        "tasks": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_reward": sum(row["reward"] for row in rows) / len(rows),
        "by_variant": grouped_accuracy(rows, "variant"),
        "by_level": grouped_accuracy(rows, "level"),
        "by_retrieval_variant": grouped_accuracy(rows, "retrieval_variant"),
        "by_business_category": grouped_accuracy(rows, "business_category"),
        "reward_components": {
            component: {
                "checks": len(values),
                "mean": sum(values) / len(values),
            }
            for component, values in sorted(component_values.items())
        },
        "atomic_checks": {
            component: {
                "passed": sum(values),
                "checks": len(values),
                "pass_rate": sum(values) / len(values),
            }
            for component, values in sorted(atomic_checks.items())
        },
        "action_checks_by_requestor": {
            requestor: {
                "passed": sum(values),
                "checks": len(values),
                "pass_rate": sum(values) / len(values),
            }
            for requestor, values in sorted(action_checks_by_requestor.items())
        },
        "termination_reasons": dict(
            sorted(Counter(row["termination_reason"] for row in rows).items())
        ),
        "failed_tasks": [
            {
                key: row[key]
                for key in (
                    "task_id",
                    "source_task_id",
                    "variant",
                    "level",
                    "business_category",
                    "retrieval_variant",
                    "reward",
                    "termination_reason",
                )
            }
            for row in rows
            if not row["correct"]
        ],
    }
    if run_logs:
        retry_events = []
        for path in run_logs:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = RETRY_PATTERN.search(line)
                if match:
                    retry_events.append(
                        {
                            "task_id": match.group("task_id"),
                            "attempt": int(match.group("attempt")),
                            "reason": match.group("reason"),
                        }
                    )
        retried_task_ids = {event["task_id"] for event in retry_events}
        correct_task_ids = {row["task_id"] for row in rows if row["correct"]}
        first_attempt_correct = len(correct_task_ids - retried_task_ids)
        summary["retry_diagnostics"] = {
            "events": len(retry_events),
            "tasks": len(retried_task_ids),
            "task_ids": sorted(retried_task_ids),
            "reasons": dict(sorted(Counter(event["reason"] for event in retry_events).items())),
            "first_attempt_correct": first_attempt_correct,
            "first_attempt_accuracy": first_attempt_correct / len(rows),
        }
    return summary


def metric_table(title: str, values: dict[str, dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| Group | Correct | Tasks | Accuracy |", "|---|---:|---:|---:|"]
    for name, item in values.items():
        lines.append(
            f"| {name} | {item['correct']} | {item['tasks']} | {item['accuracy']:.2%} |"
        )
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Banking Curriculum Evaluation",
        "",
        f"- Model: `{summary['model']}`",
        f"- Correct: {summary['correct']}/{summary['tasks']}",
        f"- Accuracy: {summary['accuracy']:.2%}",
        f"- Mean reward: {summary['mean_reward']:.4f}",
    ]
    if summary["replacement_task_ids"]:
        lines.append(
            f"- Recovered tasks: {len(summary['replacement_task_ids'])} "
            f"(`{', '.join(summary['replacement_task_ids'])}`)"
        )
    lines.append("")
    lines.extend(metric_table("By variant", summary["by_variant"]))
    lines.append("")
    lines.extend(metric_table("By level", summary["by_level"]))
    lines.append("")
    lines.extend(
        metric_table("By retrieval mode", summary["by_retrieval_variant"])
    )
    lines.append("")
    lines.extend(
        metric_table("By business category", summary["by_business_category"])
    )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Reward components: `{json.dumps(summary['reward_components'], ensure_ascii=False, sort_keys=True)}`",
            f"- Atomic checks: `{json.dumps(summary['atomic_checks'], ensure_ascii=False, sort_keys=True)}`",
            f"- Action checks by requestor: `{json.dumps(summary['action_checks_by_requestor'], ensure_ascii=False, sort_keys=True)}`",
            f"- Termination reasons: `{json.dumps(summary['termination_reasons'], ensure_ascii=False, sort_keys=True)}`",
            f"- Failed task count: {len(summary['failed_tasks'])}",
        ]
    )
    retry_diagnostics = summary.get("retry_diagnostics")
    if retry_diagnostics is not None:
        lines.extend(
            [
                f"- Framework retry events: {retry_diagnostics['events']} across {retry_diagnostics['tasks']} tasks",
                f"- Conservative first-attempt exact accuracy: {retry_diagnostics['first_attempt_correct']}/{summary['tasks']} ({retry_diagnostics['first_attempt_accuracy']:.2%})",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--replacement-results", type=Path, nargs="+", default=None)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--run-logs", type=Path, nargs="+", default=None)
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_results(
        args.results,
        read_jsonl(args.contracts),
        replacement_results_paths=args.replacement_results,
        expected_task_count=args.expected_task_count,
        run_logs=args.run_logs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("tasks", "correct", "accuracy")}, sort_keys=True))


if __name__ == "__main__":
    main()
