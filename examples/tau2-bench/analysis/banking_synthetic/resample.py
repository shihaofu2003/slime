#!/usr/bin/env python3
"""Generate fresh replacement scenarios for unmet train category/form cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import BUSINESS_CATEGORIES, TRAIN_FORM_QUOTAS  # type: ignore
    from banking_synthetic.documents import DocumentCatalog, read_json, write_json, write_jsonl  # type: ignore
    from banking_synthetic.models import scenario_to_task  # type: ignore
    from banking_synthetic.templates import build_scenario  # type: ignore
else:
    from .constants import BUSINESS_CATEGORIES, TRAIN_FORM_QUOTAS
    from .documents import DocumentCatalog, read_json, write_json, write_jsonl
    from .models import scenario_to_task
    from .templates import build_scenario


def generate_replacements(
    requests: list[dict[str, Any]], catalog: DocumentCatalog, serial_base: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if serial_base < 400000:
        raise ValueError("replacement serial base must be at least 400000")
    specs = []
    tasks = []
    contracts = []
    for offset, request in enumerate(requests):
        category = request["business_category"]
        form = request["task_form"]
        if category not in BUSINESS_CATEGORIES or form not in TRAIN_FORM_QUOTAS:
            raise ValueError(f"invalid replacement request: {request}")
        spec = build_scenario(
            split="train",
            serial=serial_base + offset,
            category=category,
            form=form,
            catalog=catalog,
        )
        task, contract = scenario_to_task(spec)
        specs.append(spec.to_dict())
        tasks.append(task)
        contracts.append(contract.to_dict())
    return specs, tasks, contracts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--serial-base", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requests = read_json(args.requests)
    if args.start < 0 or args.start > len(requests):
        raise ValueError("--start is outside the replacement request range")
    requests = requests[args.start :]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        requests = requests[: args.limit]
    specs, tasks, contracts = generate_replacements(
        requests,
        DocumentCatalog.load(args.documents_dir, args.allowlist),
        args.serial_base + args.start,
    )
    write_jsonl(args.output_dir / f"{args.prefix}_scenario_specs.jsonl", specs)
    write_json(args.output_dir / f"{args.prefix}_tasks.json", tasks)
    write_jsonl(args.output_dir / f"{args.prefix}_contracts.jsonl", contracts)
    print(
        json.dumps(
            {
                "replacement_count": len(tasks),
                "serial_base": args.serial_base + args.start,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
