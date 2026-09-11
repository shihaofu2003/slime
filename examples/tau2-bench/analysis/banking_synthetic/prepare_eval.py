#!/usr/bin/env python3
"""Stage an isolated synthetic Banking evaluation shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.documents import read_json, read_jsonl, stage_runtime  # type: ignore
else:
    from .documents import read_json, read_jsonl, stage_runtime


def select(
    tasks: list[dict],
    contracts: list[dict],
    *,
    retrieval_variant: str,
    split: str | None = None,
    shard_index: int,
    num_shards: int,
    limit: int | None,
    task_ids: set[str] | None,
) -> tuple[list[dict], list[dict]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be within [0, num_shards)")
    task_by_id = {task["id"]: task for task in tasks}
    eligible = [
        contract
        for contract in contracts
        if contract["retrieval_variant"] == retrieval_variant
        and (split is None or contract.get("split") == split)
        and (task_ids is None or contract["task_id"] in task_ids)
    ]
    eligible = eligible[shard_index::num_shards]
    if limit is not None:
        eligible = eligible[:limit]
    if not eligible:
        raise ValueError("selection produced no tasks")
    selected_tasks = [task_by_id[contract["task_id"]] for contract in eligible]
    return selected_tasks, eligible


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--banking-data", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--retrieval-variant", choices=("bm25", "golden_retrieval"), required=True)
    parser.add_argument("--split", choices=("train", "dev", "challenge"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--task-ids-file", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_ids = set(args.task_ids or [])
    if args.task_ids_file:
        task_ids.update(read_json(args.task_ids_file))
    tasks, contracts = select(
        read_json(args.tasks),
        read_jsonl(args.contracts),
        retrieval_variant=args.retrieval_variant,
        split=args.split,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
        task_ids=task_ids or None,
    )
    manifest = stage_runtime(
        tasks=tasks,
        contracts=contracts,
        allowlist_path=args.allowlist,
        source_documents_dir=args.banking_data / "documents",
        source_prompts_dir=args.banking_data / "prompts",
        source_user_simulator_dir=args.banking_data.parent.parent / "user_simulator",
        data_root=args.data_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
