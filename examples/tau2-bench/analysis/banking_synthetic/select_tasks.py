#!/usr/bin/env python3
"""Select balanced synthetic task subsets without consulting benchmark data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import BUSINESS_CATEGORIES, CASE_KINDS  # type: ignore
    from banking_synthetic.documents import read_json, read_jsonl, write_json, write_jsonl  # type: ignore
    from banking_synthetic.generate import assignments  # type: ignore
else:
    from .constants import BUSINESS_CATEGORIES, CASE_KINDS
    from .documents import read_json, read_jsonl, write_json, write_jsonl
    from .generate import assignments


FORM_ORDER = {
    "l1_evidence": 0,
    "l2_retrieval": 1,
    "l3_decision": 2,
    "l3_single_action": 3,
    "l4_two_skill": 4,
    "l5_medium_workflow": 5,
    "l6_full_workflow": 6,
}
ACTION_FORMS = tuple(list(FORM_ORDER)[3:])


def quota_selection(
    tasks: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the declared category/form quotas for every final split."""

    task_by_id = {task["id"]: task for task in tasks}
    spec_by_scenario = {spec["scenario_id"]: spec for spec in specs}
    candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for contract in contracts:
        candidates[
            (
                contract["split"],
                contract["business_category"],
                contract["task_form"],
            )
        ].append(contract)

    selected_contracts = []
    for split in ("train", "dev", "challenge"):
        desired = Counter(assignments(split, seed))
        for pair, count in sorted(desired.items()):
            key = (split, pair[0], pair[1])
            rows = sorted(candidates[key], key=lambda row: row["task_id"])
            if len(rows) < count:
                raise ValueError(
                    f"{split} quota {pair} needs {count} candidates, found {len(rows)}"
                )
            selected_contracts.extend(rows[:count])
    selected_tasks = [task_by_id[row["task_id"]] for row in selected_contracts]
    selected_specs = [
        spec_by_scenario[row["scenario_id"]] for row in selected_contracts
    ]
    return selected_tasks, selected_specs, selected_contracts


def balanced_selection(
    tasks: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    per_category: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task_by_id = {task["id"]: task for task in tasks}
    spec_by_scenario = {spec["scenario_id"]: spec for spec in specs}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for contract in contracts:
        grouped[contract["business_category"]].append(contract)

    selected_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_ids: set[str] = set()
    selected_form_counts: defaultdict[str, int] = defaultdict(int)
    selected_case_counts: defaultdict[str, int] = defaultdict(int)
    uncovered_forms = set(FORM_ORDER)
    uncovered_cases = set(CASE_KINDS)

    def add_best(candidates: list[dict[str, Any]]) -> None:
        available = [
            row
            for row in candidates
            if row["task_id"] not in selected_ids
            and len(selected_by_category[row["business_category"]]) < per_category
        ]
        if not available:
            raise ValueError("balanced smoke selection cannot satisfy feature coverage")
        chosen = min(
            available,
            key=lambda row: (
                -int(row["task_form"] in uncovered_forms)
                - int(row["case_kind"] in uncovered_cases),
                selected_form_counts[row["task_form"]],
                selected_case_counts[row["case_kind"]],
                len(selected_by_category[row["business_category"]]),
                FORM_ORDER[row["task_form"]],
                row["task_id"],
            ),
        )
        selected_by_category[chosen["business_category"]].append(chosen)
        selected_ids.add(chosen["task_id"])
        selected_form_counts[chosen["task_form"]] += 1
        selected_case_counts[chosen["case_kind"]] += 1
        uncovered_forms.discard(chosen["task_form"])
        uncovered_cases.discard(chosen["case_kind"])

    for task_form in ACTION_FORMS:
        add_best(
            [
                row
                for rows in grouped.values()
                for row in rows
                if row["task_form"] == task_form and row["case_kind"] == "positive"
            ]
        )
    for case_kind in reversed(CASE_KINDS):
        if case_kind in uncovered_cases:
            add_best(
                [row for rows in grouped.values() for row in rows if row["case_kind"] == case_kind]
            )
    for task_form in FORM_ORDER:
        if task_form in uncovered_forms:
            add_best(
                [row for rows in grouped.values() for row in rows if row["task_form"] == task_form]
            )
    for category in BUSINESS_CATEGORIES:
        while len(selected_by_category[category]) < per_category:
            add_best(grouped[category])

    selected_contracts = [
        row
        for category in BUSINESS_CATEGORIES
        for row in selected_by_category[category]
    ]

    selected_tasks = [task_by_id[row["task_id"]] for row in selected_contracts]
    selected_specs = [spec_by_scenario[row["scenario_id"]] for row in selected_contracts]
    return selected_tasks, selected_specs, selected_contracts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--train-quota", action="store_true")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_tasks = read_json(args.tasks)
    source_specs = read_jsonl(args.specs)
    source_contracts = read_jsonl(args.contracts)
    if args.train_quota:
        tasks, specs, contracts = quota_selection(
            source_tasks, source_specs, source_contracts, args.seed
        )
    else:
        tasks, specs, contracts = balanced_selection(
            source_tasks, source_specs, source_contracts, args.per_category
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "tasks.json", tasks)
    write_jsonl(args.output_dir / "scenario_specs.jsonl", specs)
    write_jsonl(args.output_dir / "contracts.jsonl", contracts)
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "category_count": len({row["business_category"] for row in contracts}),
                "task_ids": [task["id"] for task in tasks],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
