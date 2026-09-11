#!/usr/bin/env python3
"""Assemble accepted train tasks and request fresh scenarios for quota gaps."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import (  # type: ignore
        FRONTIER_FORM_PRIORITY,
        MIN_GRPO_FRONTIER,
        TRAIN_TASK_COUNT,
    )
    from banking_synthetic.documents import read_json, read_jsonl, write_json, write_jsonl  # type: ignore
    from banking_synthetic.generate import assignments  # type: ignore
else:
    from .constants import FRONTIER_FORM_PRIORITY, MIN_GRPO_FRONTIER, TRAIN_TASK_COUNT
    from .documents import read_json, read_jsonl, write_json, write_jsonl
    from .generate import assignments


def assemble(
    *,
    tasks: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    eligible_task_ids: set[str],
    frontier_task_ids: set[str],
    four_trial_task_ids: set[str] | None = None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    task_by_id = {task["id"]: task for task in tasks}
    spec_by_scenario = {spec["scenario_id"]: spec for spec in specs}
    desired = Counter(assignments("train", seed))
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for contract in contracts:
        if contract["task_id"] in eligible_task_ids:
            candidates[(contract["business_category"], contract["task_form"])].append(contract)
    selected_contracts = []
    missing = []
    four_trial_task_ids = four_trial_task_ids or set()
    for pair, count in sorted(desired.items()):
        rows = sorted(
            candidates[pair],
            key=lambda row: (
                row["task_id"] not in frontier_task_ids,
                row["task_id"] not in four_trial_task_ids,
                row["task_id"],
            ),
        )
        selected_contracts.extend(rows[:count])
        missing.extend(
            {
                "business_category": pair[0],
                "task_form": pair[1],
            }
            for _ in range(count - min(count, len(rows)))
        )
    selected_frontier = sum(
        row["task_id"] in frontier_task_ids for row in selected_contracts
    )
    quota_missing = len(missing)
    extra_frontier_candidates = max(
        0, MIN_GRPO_FRONTIER - selected_frontier - quota_missing
    )
    form_priority = {
        form: index for index, form in enumerate(FRONTIER_FORM_PRIORITY)
    }
    replaceable_rows = sorted(
        (
            row
            for row in selected_contracts
            if row["task_id"] not in frontier_task_ids
        ),
        key=lambda row: (
            form_priority.get(row["task_form"], len(form_priority)),
            row["business_category"],
            row["task_form"],
            row["task_id"],
        ),
    )
    if extra_frontier_candidates and not replaceable_rows:
        raise ValueError("frontier shortfall has no replaceable category/form cells")
    for index in range(extra_frontier_candidates):
        row = replaceable_rows[index % len(replaceable_rows)]
        missing.append(
            {
                "business_category": row["business_category"],
                "task_form": row["task_form"],
                "reason": "frontier_shortfall",
            }
        )
    selected_tasks = [task_by_id[row["task_id"]] for row in selected_contracts]
    selected_specs = [spec_by_scenario[row["scenario_id"]] for row in selected_contracts]
    if len({task["id"] for task in selected_tasks}) != len(selected_tasks):
        raise ValueError("selected task ids are not unique")
    if len({spec["scenario_id"] for spec in selected_specs}) != len(selected_specs):
        raise ValueError("selected scenario ids are not unique")
    return selected_tasks, selected_specs, selected_contracts, missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--specs", type=Path, action="append", required=True)
    parser.add_argument("--contracts", type=Path, action="append", required=True)
    parser.add_argument("--pool-manifest", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (len(args.tasks) == len(args.specs) == len(args.contracts)):
        raise ValueError("tasks/specs/contracts inputs must have matching counts")
    tasks = [row for path in args.tasks for row in read_json(path)]
    specs = [row for path in args.specs for row in read_jsonl(path)]
    contracts = [row for path in args.contracts for row in read_jsonl(path)]
    pool_manifests = [read_json(path) for path in args.pool_manifest]
    eligible = {
        task_id
        for manifest in pool_manifests
        for task_id in (manifest.get("pools") or {}).get("sft_candidate", [])
    }
    frontier = {
        task_id
        for manifest in pool_manifests
        for task_id in (manifest.get("pools") or {}).get("grpo_frontier", [])
    }
    four_trial = {
        task_id
        for manifest in pool_manifests
        for task_id in manifest.get("four_trial_task_ids") or []
    }
    selected_tasks, selected_specs, selected_contracts, missing = assemble(
        tasks=tasks,
        specs=specs,
        contracts=contracts,
        eligible_task_ids=eligible,
        frontier_task_ids=frontier,
        four_trial_task_ids=four_trial,
        seed=args.seed,
    )
    write_json(args.output_dir / "train_tasks.json", selected_tasks)
    write_jsonl(args.output_dir / "train_scenario_specs.jsonl", selected_specs)
    write_jsonl(args.output_dir / "train_contracts.jsonl", selected_contracts)
    write_json(args.output_dir / "missing_pairs.json", missing)
    frontier_count = sum(
        row["task_id"] in frontier for row in selected_contracts
    )
    report = {
        "accepted_train_count": len(selected_tasks),
        "distinct_scenario_count": len(selected_specs),
        "missing_count": len(missing),
        "quota_missing_count": max(0, TRAIN_TASK_COUNT - len(selected_tasks)),
        "frontier_count": frontier_count,
        "frontier_shortfall": max(0, MIN_GRPO_FRONTIER - frontier_count),
        "complete": len(selected_tasks) == TRAIN_TASK_COUNT
        and not missing
        and frontier_count >= MIN_GRPO_FRONTIER,
        "by_category": dict(sorted(Counter(row["business_category"] for row in selected_contracts).items())),
        "by_form": dict(sorted(Counter(row["task_form"] for row in selected_contracts).items())),
    }
    write_json(args.output_dir / "assembly_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
