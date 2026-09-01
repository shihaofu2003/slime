#!/usr/bin/env python3
"""Partition local judge payloads and merge independently written review shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sft_quality import read_jsonl, write_jsonl


def shard_index(dialog_id: str, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    digest = hashlib.sha256(dialog_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def record_map(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    counts = Counter(str(row.get("dialog_id") or "") for row in rows)
    duplicates = sorted(dialog_id for dialog_id, count in counts.items() if count > 1)
    missing = counts.get("")
    if missing or duplicates:
        raise ValueError(
            f"{label}: invalid dialog IDs; missing={missing or 0} duplicates={duplicates[:5]}"
        )
    return {str(row["dialog_id"]): row for row in rows}


def partition(args: argparse.Namespace) -> None:
    payloads = [
        row
        for path in args.input
        for row in read_jsonl(path)
    ]
    payload_counts = Counter(str(row.get("_dialog_id") or "") for row in payloads)
    duplicate_payloads = sorted(
        dialog_id for dialog_id, count in payload_counts.items() if count > 1
    )
    if payload_counts.get("") or duplicate_payloads:
        raise ValueError(
            "payload union has missing/duplicate dialog IDs: "
            f"{duplicate_payloads[:5]}"
        )
    existing_rows = list(read_jsonl(args.existing)) if args.existing.is_file() else []
    existing = record_map(existing_rows, label=str(args.existing))
    expected_ids = set(payload_counts)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "num_shards": args.num_shards,
        "payload_rows": len(payloads),
        "existing_rows_in_scope": len(set(existing) & expected_ids),
        "shards": {},
    }
    for index in range(args.num_shards):
        shard_payloads = sorted(
            (
                row
                for row in payloads
                if shard_index(str(row["_dialog_id"]), args.num_shards) == index
            ),
            key=lambda row: str(row["_dialog_id"]),
        )
        shard_ids = {str(row["_dialog_id"]) for row in shard_payloads}
        review_path = args.shard_dir / f"reviews_{index:02d}.jsonl"
        prior_shard_rows = list(read_jsonl(review_path)) if review_path.is_file() else []
        prior_shard = record_map(prior_shard_rows, label=str(review_path))
        seeds = {
            dialog_id: record
            for dialog_id, record in existing.items()
            if dialog_id in shard_ids
        }
        seeds.update(
            {
                dialog_id: record
                for dialog_id, record in prior_shard.items()
                if dialog_id in shard_ids
            }
        )
        payload_path = args.shard_dir / f"payloads_{index:02d}.jsonl"
        write_jsonl(payload_path, shard_payloads)
        write_jsonl(
            review_path,
            (seeds[dialog_id] for dialog_id in sorted(seeds)),
        )
        manifest["shards"][str(index)] = {
            "payload_rows": len(shard_payloads),
            "seed_review_rows": len(seeds),
            "payload_path": str(payload_path),
            "review_path": str(review_path),
        }
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def merge(args: argparse.Namespace) -> None:
    expected_payloads = [
        row
        for path in args.input
        for row in read_jsonl(path)
    ]
    expected_ids = {str(row["_dialog_id"]) for row in expected_payloads}
    if len(expected_ids) != len(expected_payloads):
        raise ValueError("expected payload inputs contain duplicate dialog IDs")
    merged = {}
    for index in range(args.num_shards):
        review_path = args.shard_dir / f"reviews_{index:02d}.jsonl"
        rows = list(read_jsonl(review_path))
        shard_reviews = record_map(rows, label=str(review_path))
        wrong_shard = sorted(
            dialog_id
            for dialog_id in shard_reviews
            if shard_index(dialog_id, args.num_shards) != index
        )
        if wrong_shard:
            raise ValueError(f"{review_path}: rows assigned to another shard {wrong_shard[:5]}")
        overlap = set(merged) & set(shard_reviews)
        if overlap:
            raise ValueError(f"duplicate cross-shard reviews: {sorted(overlap)[:5]}")
        merged.update(shard_reviews)
    missing = sorted(expected_ids - set(merged))
    unexpected = sorted(set(merged) - expected_ids)
    if missing or unexpected:
        raise ValueError(
            f"review coverage mismatch; missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    write_jsonl(args.output, (merged[dialog_id] for dialog_id in sorted(merged)))
    result = {
        "status": "pass",
        "num_shards": args.num_shards,
        "payload_rows": len(expected_ids),
        "merged_review_rows": len(merged),
        "output": str(args.output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    partition_parser = subparsers.add_parser("partition")
    partition_parser.add_argument("--input", type=Path, action="append", required=True)
    partition_parser.add_argument("--existing", type=Path, required=True)
    partition_parser.add_argument("--shard-dir", type=Path, required=True)
    partition_parser.add_argument("--manifest", type=Path, required=True)
    partition_parser.add_argument("--num-shards", type=int, required=True)
    partition_parser.set_defaults(func=partition)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--input", type=Path, action="append", required=True)
    merge_parser.add_argument("--shard-dir", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--num-shards", type=int, required=True)
    merge_parser.set_defaults(func=merge)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
