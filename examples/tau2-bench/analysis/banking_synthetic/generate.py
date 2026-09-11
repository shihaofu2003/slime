#!/usr/bin/env python3
"""Generate independent Banking ScenarioSpec, Task, and TaskContract artifacts."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import (  # type: ignore
        BUSINESS_CATEGORIES,
        LOW_FORM_CATEGORIES,
        PILOT_FORM_QUOTAS,
        TRAIN_CATEGORY_QUOTAS,
        TRAIN_FORM_QUOTAS,
    )
    from banking_synthetic.documents import DocumentCatalog, write_json, write_jsonl  # type: ignore
    from banking_synthetic.models import scenario_to_task  # type: ignore
    from banking_synthetic.templates import build_scenario  # type: ignore
else:
    from .constants import (
        BUSINESS_CATEGORIES,
        LOW_FORM_CATEGORIES,
        PILOT_FORM_QUOTAS,
        TRAIN_CATEGORY_QUOTAS,
        TRAIN_FORM_QUOTAS,
    )
    from .documents import DocumentCatalog, write_json, write_jsonl
    from .models import scenario_to_task
    from .templates import build_scenario


def _expanded(quota: dict[str, int]) -> list[str]:
    return [name for name, count in quota.items() for _ in range(count)]


def assignments(phase: str, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    if phase == "pilot":
        categories = [name for name in BUSINESS_CATEGORIES for _ in range(100)]
        forms = _expanded(PILOT_FORM_QUOTAS)
    elif phase == "train":
        categories = _expanded(TRAIN_CATEGORY_QUOTAS)
        forms = _expanded(TRAIN_FORM_QUOTAS)
    elif phase == "dev":
        rows = [
            (category, form)
            for category in BUSINESS_CATEGORIES
            for form in (
                "l3_decision",
                "l3_single_action",
                "l4_two_skill",
                "l5_medium_workflow",
                "l6_full_workflow",
            )
        ]
        rng.shuffle(rows)
        return rows
    elif phase == "challenge":
        rows = [
            (category, form)
            for category in BUSINESS_CATEGORIES
            for form in ("l5_medium_workflow", "l6_full_workflow")
        ]
        rng.shuffle(rows)
        return rows
    else:
        raise ValueError(f"unknown phase: {phase}")
    if len(categories) != len(forms):
        raise AssertionError("category and form quotas have different totals")
    rng.shuffle(categories)
    low_forms = [form for form in forms if form in {"l1_evidence", "l2_retrieval"}]
    other_forms = [form for form in forms if form not in {"l1_evidence", "l2_retrieval"}]
    eligible_slots = [
        index
        for index, category in enumerate(categories)
        if category in LOW_FORM_CATEGORIES
    ]
    if len(eligible_slots) < len(low_forms):
        raise AssertionError("low-form document categories do not have enough quota")
    rng.shuffle(eligible_slots)
    low_slots = set(eligible_slots[: len(low_forms)])
    rng.shuffle(low_forms)
    rng.shuffle(other_forms)
    low_iter = iter(low_forms)
    other_iter = iter(other_forms)
    rows = [
        (category, next(low_iter) if index in low_slots else next(other_iter))
        for index, category in enumerate(categories)
    ]
    rng.shuffle(rows)
    return rows


def case_matrix_assignments() -> list[tuple[str, str, str]]:
    """One executable template test for every category and supported case kind."""

    form_for_case = {
        "positive": "l4_two_skill",
        "refusal": "l5_medium_workflow",
        "user_tool": "l5_medium_workflow",
        "dynamic_tool": "l4_two_skill",
        "multi_entity": "l6_full_workflow",
    }
    return [
        (category, form_for_case[case_kind], case_kind)
        for category in BUSINESS_CATEGORIES
        for case_kind in form_for_case
    ]


def generate(
    *,
    phase: str,
    seed: int,
    catalog: DocumentCatalog,
    start: int = 0,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if phase == "matrix":
        rows = case_matrix_assignments()
    else:
        rows = [
            (category, form, None)
            for category, form in assignments(phase, seed)
        ]
    if start < 0 or start > len(rows):
        raise ValueError("start is outside the phase assignment range")
    rows = rows[start:]
    if limit is not None:
        rows = rows[:limit]
    specs = []
    tasks = []
    contracts = []
    split = "train" if phase in {"pilot", "train"} else phase
    serial_base = {
        "pilot": 0,
        "train": 100000,
        "dev": 200000,
        "challenge": 300000,
        "matrix": 350000,
    }[phase]
    for index, (category, form, case_kind) in enumerate(rows, start=start):
        spec = build_scenario(
            split=split,
            serial=serial_base + index,
            category=category,
            form=form,
            catalog=catalog,
            case_kind=case_kind,
        )
        task, contract = scenario_to_task(spec)
        specs.append(spec.to_dict())
        tasks.append(task)
        contracts.append(contract.to_dict())
    return specs, tasks, contracts


def summary(specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenario_count": len({spec["scenario_id"] for spec in specs}),
        "task_count": len(specs),
        "by_category": dict(sorted(Counter(spec["business_category"] for spec in specs).items())),
        "by_form": dict(sorted(Counter(spec["task_form"] for spec in specs).items())),
        "by_level": dict(sorted(Counter(spec["difficulty"] for spec in specs).items())),
        "by_case_kind": dict(sorted(Counter(spec["case_kind"] for spec in specs).items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("pilot", "train", "dev", "challenge", "matrix"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog = DocumentCatalog.load(args.documents_dir, args.allowlist)
    specs, tasks, contracts = generate(
        phase=args.phase,
        seed=args.seed,
        catalog=catalog,
        start=args.start,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / f"{args.phase}_scenario_specs.jsonl", specs)
    write_json(args.output_dir / f"{args.phase}_tasks.json", tasks)
    write_jsonl(args.output_dir / f"{args.phase}_contracts.jsonl", contracts)
    report = summary(specs)
    write_json(args.output_dir / f"{args.phase}_generation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
