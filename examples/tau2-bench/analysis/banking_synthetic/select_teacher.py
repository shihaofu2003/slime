#!/usr/bin/env python3
"""Select valid autonomous zero-success tasks for node-guided teaching."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.documents import read_json, read_jsonl, write_json  # type: ignore
    from banking_synthetic.protocol import protocol_failure_reason  # type: ignore
else:
    from .documents import read_json, read_jsonl, write_json
    from .protocol import protocol_failure_reason


def select_teacher_tasks(
    contracts: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    split: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    contract_by_id = {row["task_id"]: row for row in contracts}
    if len(contract_by_id) != len(contracts):
        raise ValueError("contract task ids are not unique")
    valid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for simulation in payload.get("simulations") or []:
            task_id = simulation["task_id"]
            if task_id not in contract_by_id:
                continue
            if protocol_failure_reason(simulation) is None:
                valid[task_id].append(simulation)
    unresolved = {
        task_id: len(valid.get(task_id, []))
        for task_id in contract_by_id
        if len(valid.get(task_id, [])) not in {1, 4}
    }
    if unresolved:
        raise ValueError(
            "teacher selection requires one or four valid autonomous trials; "
            f"examples: {list(sorted(unresolved.items()))[:5]}"
        )
    selected = sorted(
        task_id
        for task_id, rows in valid.items()
        if split is None or contract_by_id[task_id]["split"] == split
        if not any(
            float((row.get("reward_info") or {}).get("reward") or 0.0) == 1.0
            for row in rows
        )
    )
    selected_contracts = [contract_by_id[task_id] for task_id in selected]
    report = {
        "autonomous_task_count": len(valid),
        "teacher_task_count": len(selected),
        "by_split": dict(
            sorted(Counter(row["split"] for row in selected_contracts).items())
        ),
        "by_category": dict(
            sorted(
                Counter(
                    row["business_category"] for row in selected_contracts
                ).items()
            )
        ),
        "by_form": dict(
            sorted(Counter(row["task_form"] for row in selected_contracts).items())
        ),
    }
    return selected, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "dev", "challenge"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected, report = select_teacher_tasks(
        read_jsonl(args.contracts),
        [read_json(path) for path in args.results],
        args.split,
    )
    write_json(args.output, selected)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
