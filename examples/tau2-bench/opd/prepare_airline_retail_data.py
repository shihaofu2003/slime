#!/usr/bin/env python3
"""Merge processed Tau2 Airline/Retail tasks, optionally adding Telecom/Banking."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
PROJECT_ROOT = SERVICE_AGENT_ROOT / "slime-async-tau2"
DEFAULT_AIRLINE = PROJECT_ROOT / "output/experiments/tau2-domain-experts-sft4505-b128/data/airline_train.jsonl"
DEFAULT_RETAIL = PROJECT_ROOT / "output/experiments/tau2-domain-experts-sft4505-b128/data/retail_train.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "output/experiments/tau2-opd-airline-retail-pilot/data/airline_retail_train.jsonl"


def validate_row(row: dict[str, Any], *, source: Path, line_number: int) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}:{line_number}: metadata must be an object")
    domain = metadata.get("domain")
    if domain not in {"airline", "retail", "telecom", "banking"}:
        raise ValueError(f"{source}:{line_number}: unsupported domain {domain!r}")
    task = metadata.get("task")
    task_id = task.get("id") if isinstance(task, dict) else None
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"{source}:{line_number}: metadata.task.id is required")
    if task_id.split("_", 1)[0] != domain:
        raise ValueError(f"{source}:{line_number}: domain does not match task id {task_id!r}")
    db_path = metadata.get("db_path")
    if not isinstance(db_path, str) or not db_path:
        raise ValueError(f"{source}:{line_number}: metadata.db_path is required")
    if not Path(db_path).exists():
        raise FileNotFoundError(f"{source}:{line_number}: db_path does not exist: {db_path}")
    return domain


def _read_rows(source: Path) -> Iterable[tuple[dict[str, Any], str]]:
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: row must be an object")
            yield row, str(line_number)


def merge_sources(
    sources: list[Path], output: Path, *, required_domains: tuple[str, ...] = ("airline", "retail")
) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        for row, line_number in _read_rows(source):
            domain = validate_row(row, source=source, line_number=int(line_number))
            rows.append(row)
            counts[domain] += 1
    missing = [domain for domain in required_domains if not counts[domain]]
    if missing:
        raise ValueError(f"merged OPD data is missing requested domains: {', '.join(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    return dict(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airline", type=Path, default=DEFAULT_AIRLINE)
    parser.add_argument("--retail", type=Path, default=DEFAULT_RETAIL)
    parser.add_argument("--telecom", type=Path)
    parser.add_argument("--banking", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [args.airline, args.retail]
    domains = ["airline", "retail"]
    for domain in ("telecom", "banking"):
        path = getattr(args, domain)
        if path is not None:
            sources.append(path)
            domains.append(domain)
    counts = merge_sources(sources, args.output, required_domains=tuple(domains))
    print(json.dumps({"output": str(args.output), "rows": sum(counts.values()), "domains": counts}, indent=2))


if __name__ == "__main__":
    main()
