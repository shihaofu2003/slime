#!/usr/bin/env python3
"""Extract deterministic behavior features from official tau2 trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from trajectory_patterns import (
    duplicate_ids,
    extract_features,
    load_result_cells,
    load_tool_catalog,
)


def parse_model_spec(spec: str) -> tuple[str, str]:
    label, separator, prefix = spec.partition(":")
    if not separator or not label.strip() or not prefix.strip():
        raise argparse.ArgumentTypeError(f"expected LABEL:PREFIX, got {spec!r}")
    return label.strip(), prefix.strip()


def percentile(values: list[int], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def write_deterministic_summary(records: list[dict], outdir: Path) -> None:
    fields = (
        "model",
        "domain",
        "success",
        "workflow_type",
        "n",
        "reward_rate",
        "gold_write_rate",
        "observed_write_rate",
        "exact_action_match_rate",
        "expected_write_name_called_failure_n",
        "multitool_violation_rate",
        "multiwrite_violation_rate",
        "invalid_tool_rate",
        "argument_schema_error_rate",
        "namespace_confusion_rate",
        "repeat_rate",
        "tool_calls_p50",
        "tool_calls_p90",
        "tool_calls_max",
    )
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[
            (
                record["model"],
                record["domain"],
                "success" if record["success"] else "failure",
                record["workflow_type"],
            )
        ].append(record)
    rows = []
    for key, group in sorted(groups.items()):
        n = len(group)
        tools = [record["num_tool_calls"] for record in group]
        rows.append(
            {
                "model": key[0],
                "domain": key[1],
                "success": key[2],
                "workflow_type": key[3],
                "n": n,
                "reward_rate": statistics.mean(record["success"] for record in group),
                "gold_write_rate": statistics.mean(
                    record["gold_requires_agent_write"] for record in group
                ),
                "observed_write_rate": statistics.mean(
                    record["num_observed_writes"] > 0 for record in group
                ),
                "exact_action_match_rate": statistics.mean(
                    record["exact_action_match"] for record in group
                ),
                "expected_write_name_called_failure_n": sum(
                    record["expected_write_name_called_failure"] for record in group
                ),
                "multitool_violation_rate": statistics.mean(
                    record["multitool_policy_violation"] for record in group
                ),
                "multiwrite_violation_rate": statistics.mean(
                    record["multiwrite_turns"] > 0 for record in group
                ),
                "invalid_tool_rate": statistics.mean(
                    record["invalid_tool_calls"] > 0 for record in group
                ),
                "argument_schema_error_rate": statistics.mean(
                    bool(record["tool_argument_schema_errors"]) for record in group
                ),
                "namespace_confusion_rate": statistics.mean(
                    bool(record["namespace_confusion_turns"]) for record in group
                ),
                "repeat_rate": statistics.mean(
                    record["adjacent_repeated_calls"] + record["nonadjacent_repeated_calls"]
                    > 0
                    for record in group
                ),
                "tool_calls_p50": percentile(tools, 0.5),
                "tool_calls_p90": percentile(tools, 0.9),
                "tool_calls_max": max(tools),
            }
        )
    with (outdir / "deterministic_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_task_success(records: list[dict], outdir: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["domain"], record["task_id"])].append(record)
    with (outdir / "task_success_counts.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ("model", "domain", "task_id", "successes", "trials")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, group in sorted(grouped.items()):
            writer.writerow(
                {
                    "model": key[0],
                    "domain": key[1],
                    "task_id": key[2],
                    "successes": sum(record["success"] for record in group),
                    "trials": len(group),
                }
            )


def print_cross_checks(records: list[dict], expected_per_model: int | None) -> None:
    by_model = defaultdict(list)
    for record in records:
        by_model[record["model"]].append(record)
    for model, group in by_model.items():
        rate = statistics.mean(record["success"] for record in group)
        print(f"{model}: n={len(group)} reward_rate={rate:.6f}")
        if expected_per_model is not None and len(group) != expected_per_model:
            raise SystemExit(
                f"{model}: expected {expected_per_model} trajectories, got {len(group)}"
            )
    duplicates = duplicate_ids(records)
    if duplicates:
        raise SystemExit(f"duplicate trajectory IDs: {duplicates[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--model", type=parse_model_spec, action="append", required=True)
    parser.add_argument("--domains", default="airline,retail,telecom")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-per-model", type=int, default=400)
    args = parser.parse_args()

    domains = [domain.strip() for domain in args.domains.split(",") if domain.strip()]
    catalog = load_tool_catalog(args.tau2_root)
    cells = load_result_cells(args.sim_root, args.model, domains)
    records = [
        extract_features(
            cell["simulation"], cell["task"], cell["model"], cell["domain"], catalog
        )
        for cell in cells
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "features.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    write_deterministic_summary(records, args.out)
    write_task_success(records, args.out)
    print_cross_checks(records, args.expected_per_model)

    no_write_success = Counter()
    write_success = Counter()
    expected_write_called_failure = Counter()
    for record in records:
        if record["success"]:
            (write_success if record["gold_requires_agent_write"] else no_write_success)[
                record["model"]
            ] += 1
        if record["expected_write_name_called_failure"]:
            expected_write_called_failure[record["model"]] += 1
    print("read-only successes:", dict(no_write_success))
    print("state-changing successes:", dict(write_success))
    print("expected-write-name-called failures:", dict(expected_write_called_failure))


if __name__ == "__main__":
    main()
