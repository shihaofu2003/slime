#!/usr/bin/env python3
"""Merge disjoint synthetic artifact shards without changing their contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.documents import read_json, read_jsonl, write_json, write_jsonl  # type: ignore
else:
    from .documents import read_json, read_jsonl, write_json, write_jsonl


def merge_artifacts(
    task_groups: list[list[dict[str, Any]]],
    spec_groups: list[list[dict[str, Any]]],
    contract_groups: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not (len(task_groups) == len(spec_groups) == len(contract_groups)):
        raise ValueError("tasks/specs/contracts shard counts differ")
    tasks = [row for group in task_groups for row in group]
    specs = [row for group in spec_groups for row in group]
    contracts = [row for group in contract_groups for row in group]
    task_ids = [row["id"] for row in tasks]
    contract_ids = [row["task_id"] for row in contracts]
    scenario_ids = [row["scenario_id"] for row in specs]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task shards contain duplicate task ids")
    if len(contract_ids) != len(set(contract_ids)):
        raise ValueError("contract shards contain duplicate task ids")
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("spec shards contain duplicate scenario ids")
    if set(task_ids) != set(contract_ids):
        raise ValueError("merged task and contract ids differ")
    if {row["scenario_id"] for row in contracts} != set(scenario_ids):
        raise ValueError("merged contract and spec scenario ids differ")
    return tasks, specs, contracts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--specs", type=Path, action="append", required=True)
    parser.add_argument("--contracts", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks, specs, contracts = merge_artifacts(
        [read_json(path) for path in args.tasks],
        [read_jsonl(path) for path in args.specs],
        [read_jsonl(path) for path in args.contracts],
    )
    write_json(args.output_dir / f"{args.prefix}_tasks.json", tasks)
    write_jsonl(args.output_dir / f"{args.prefix}_scenario_specs.jsonl", specs)
    write_jsonl(args.output_dir / f"{args.prefix}_contracts.jsonl", contracts)
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "scenario_count": len(specs),
                "shard_count": len(args.tasks),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
