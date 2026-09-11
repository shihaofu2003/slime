#!/usr/bin/env python3
"""Stage a retrieval-consistent Tau2 Banking curriculum evaluation shard."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVICE_AGENT_ROOT = PROJECT_ROOT.parent
DEFAULT_TASKS = (
    PROJECT_ROOT
    / "output/experiments/tau2-banking-task-curriculum/expanded_tasks.json"
)
DEFAULT_CONTRACTS = (
    PROJECT_ROOT
    / "output/experiments/tau2-banking-task-curriculum/expanded_contracts.jsonl"
)
DEFAULT_BANKING_DATA = (
    SERVICE_AGENT_ROOT / "tau2-bench/data/tau2/domains/banking_knowledge"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_tasks(
    tasks: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    *,
    retrieval_variant: str,
    shard_index: int,
    num_shards: int,
    source_limit: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be within [0, num_shards)")

    task_by_id = {task["id"]: task for task in tasks}
    contract_by_id = {item["task_id"]: item for item in contracts}
    if len(task_by_id) != len(tasks) or len(contract_by_id) != len(contracts):
        raise ValueError("task and contract ids must be unique")
    if set(task_by_id) != set(contract_by_id):
        raise ValueError("task and contract ids differ")

    eligible = [
        item
        for item in contracts
        if item["retrieval_variant"] == retrieval_variant
    ]
    if task_ids:
        requested_task_ids = set(task_ids)
        eligible = [
            item for item in eligible if item["task_id"] in requested_task_ids
        ]
        selected_task_ids = {item["task_id"] for item in eligible}
        missing_task_ids = sorted(requested_task_ids - selected_task_ids)
        if missing_task_ids:
            raise ValueError(
                f"task ids are absent from {retrieval_variant}: {missing_task_ids}"
            )
    sources = sorted({item["source_task_id"] for item in eligible})
    selected_sources = sources[shard_index::num_shards]
    if source_limit is not None:
        if source_limit < 1:
            raise ValueError("source_limit must be positive")
        selected_sources = selected_sources[:source_limit]
    selected_source_set = set(selected_sources)
    selected_contract_by_id = {
        item["task_id"]: item
        for item in eligible
        if item["source_task_id"] in selected_source_set
    }
    selected_tasks = [
        task for task in tasks if task["id"] in selected_contract_by_id
    ]
    selected_contracts = [
        selected_contract_by_id[task["id"]] for task in selected_tasks
    ]
    if not selected_tasks:
        raise ValueError("selection produced no tasks")
    return selected_tasks, selected_contracts


def build_manifest(
    tasks: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    *,
    retrieval_variant: str,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    return {
        "retrieval_variant": retrieval_variant,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "task_count": len(tasks),
        "source_count": len({item["source_task_id"] for item in contracts}),
        "tasks_by_variant": dict(Counter(item["variant"] for item in contracts)),
        "tasks_by_level": dict(Counter(item["level"] for item in contracts)),
        "tasks_by_category": dict(
            Counter(item["business_category"] for item in contracts)
        ),
        "task_ids": [task["id"] for task in tasks],
    }


def stage_tasks(args: argparse.Namespace) -> dict[str, Any]:
    tasks = read_json(args.tasks)
    contracts = read_jsonl(args.contracts)
    selected_tasks, selected_contracts = select_tasks(
        tasks,
        contracts,
        retrieval_variant=args.retrieval_variant,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        source_limit=args.source_limit,
        task_ids=args.task_ids,
    )

    domain_dir = args.data_root / "tau2/domains/banking_knowledge"
    if domain_dir.exists():
        raise FileExistsError(f"staging directory already exists: {domain_dir}")
    domain_dir.mkdir(parents=True)
    shutil.copy2(args.banking_data / "db.json", domain_dir / "db.json")
    shutil.copytree(args.banking_data / "documents", domain_dir / "documents")
    shutil.copytree(args.banking_data / "prompts", domain_dir / "prompts")
    shutil.copytree(
        args.banking_data.parent.parent / "user_simulator",
        args.data_root / "tau2/user_simulator",
    )
    task_dir = domain_dir / "tasks"
    task_dir.mkdir()
    for index, task in enumerate(selected_tasks):
        (task_dir / f"task_{index:04d}.json").write_text(
            json.dumps(task, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = build_manifest(
        selected_tasks,
        selected_contracts,
        retrieval_variant=args.retrieval_variant,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    (domain_dir / "curriculum_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--banking-data", type=Path, default=DEFAULT_BANKING_DATA)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--retrieval-variant",
        choices=("bm25", "golden_retrieval"),
        required=True,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--source-limit", type=int, default=None)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    return parser


def main() -> None:
    manifest = stage_tasks(build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
