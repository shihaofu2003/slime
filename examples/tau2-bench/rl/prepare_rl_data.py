#!/usr/bin/env python3
"""Prepare AReaL tau2 RL tasks for slime rollout.

The source data is close to tau2's Task schema, but slime's Dataset loader only
passes extra fields through a single metadata column. This script wraps each
task into metadata and keeps the prompt as a small stable task id.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tau2.data_model.tasks import Task


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
DEFAULT_INPUT = SERVICE_AGENT_ROOT / "datasets/AReaL-tau2-data/tau2_rl_train.jsonl"
DEFAULT_OUTPUT = (
    SERVICE_AGENT_ROOT
    / "slime/output/datasets/tau2-bench-rl/areal_tau2_rl_train.jsonl"
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if line:
                yield row_index, json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def parse_evaluation_criteria(raw: Any, *, task_id: str) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"task {task_id}: evaluation_criteria is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"task {task_id}: evaluation_criteria must be an object")
    return raw


def domain_from_task_id(task_id: str) -> str:
    domain = task_id.split("_", 1)[0]
    if domain not in {"airline", "retail", "telecom"}:
        raise ValueError(f"task {task_id}: unsupported domain prefix {domain!r}")
    return domain


def convert_row(row: dict[str, Any], *, source_row: int, data_root: Path) -> dict[str, Any]:
    task_id = row.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"row {source_row}: missing task id")

    domain = domain_from_task_id(task_id)
    db_path_raw = row.get("db_path")
    if not isinstance(db_path_raw, str) or not db_path_raw:
        raise ValueError(f"task {task_id}: missing db_path")
    db_path = Path(db_path_raw)
    if not db_path.is_absolute():
        db_path = data_root / db_path
    if not db_path.exists():
        raise FileNotFoundError(f"task {task_id}: db_path does not exist: {db_path}")

    task = dict(row)
    task.pop("db_path", None)
    task["evaluation_criteria"] = parse_evaluation_criteria(
        task.get("evaluation_criteria"), task_id=task_id
    )
    Task.model_validate(task)

    return {
        "prompt": task_id,
        "metadata": {
            "source_row": source_row,
            "domain": domain,
            "db_path": str(db_path),
            "task": task,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to convert for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.input.parent
    rows: list[dict[str, Any]] = []
    domains: Counter[str] = Counter()
    db_paths: Counter[str] = Counter()

    for row_index, row in read_jsonl(args.input):
        if args.limit is not None and len(rows) >= args.limit:
            break
        converted = convert_row(row, source_row=row_index, data_root=data_root)
        rows.append(converted)
        metadata = converted["metadata"]
        domains[metadata["domain"]] += 1
        db_paths[metadata["db_path"]] += 1

    if not rows:
        raise SystemExit(f"No rows converted from {args.input}")

    write_jsonl(args.output, rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "domains": dict(sorted(domains.items())),
        "db_paths": dict(sorted(db_paths.items())),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
