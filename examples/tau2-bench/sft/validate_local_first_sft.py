#!/usr/bin/env python3
"""Fail-fast validation for a built local-first tau2 Agent SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from build_local_first_sft import (  # noqa: E402
    BANNED_SINGLE_CALL_RE,
    FILTER_VERSION,
    PROTOCOL_INSERTION,
    TRAINING_SCHEDULE_VERSION,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield line_number, row


def validate_row_structure(
    row: dict[str, Any],
    *,
    line_number: int,
    expected_schedule_index: int | None,
) -> tuple[str, int, str]:
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"line {line_number}: row must end at an assistant target")
    masks = [message.get("step_loss_mask") for message in messages]
    if masks[-1] != 1 or any(mask != 0 for mask in masks[:-1]):
        raise ValueError(
            f"line {line_number}: only the final target may have step_loss_mask=1"
        )
    systems = [message for message in messages if message.get("role") == "system"]
    if len(systems) != 1:
        raise ValueError(f"line {line_number}: expected exactly one system message")
    system = str(systems[0].get("content") or "")
    if system.count(PROTOCOL_INSERTION) != 1:
        raise ValueError(
            f"line {line_number}: dependency-safe protocol count is not one"
        )
    if BANNED_SINGLE_CALL_RE.search(system):
        raise ValueError(f"line {line_number}: single-call-only text remains")

    metadata = row.get("metadata") or {}
    source_row = metadata.get("source_row")
    dialog_id = str(metadata.get("source_dialog_id") or "")
    domain = str(metadata.get("domain") or "")
    if not isinstance(source_row, int) or not dialog_id or not domain:
        raise ValueError(f"line {line_number}: incomplete source provenance")
    quality_filter = metadata.get("quality_filter") or {}
    if (
        quality_filter.get("version") != FILTER_VERSION
        or quality_filter.get("target_only_loss") is not True
        or not quality_filter.get("decision_sha256")
    ):
        raise ValueError(f"line {line_number}: invalid quality_filter metadata")
    schedule = metadata.get("training_schedule")
    if expected_schedule_index is not None:
        if (
            not isinstance(schedule, dict)
            or schedule.get("version") != TRAINING_SCHEDULE_VERSION
            or schedule.get("schedule_index") != expected_schedule_index
            or schedule.get("filter_decision_sha256")
            != quality_filter.get("decision_sha256")
        ):
            raise ValueError(f"line {line_number}: invalid training schedule metadata")
    return domain, source_row, str(quality_filter["decision_sha256"])


def validate(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    expected_artifact = (summary.get("artifacts") or {}).get(args.data.name)
    if not isinstance(expected_artifact, dict):
        raise ValueError(f"{args.summary}: missing artifact entry for {args.data.name}")
    actual_sha256 = file_sha256(args.data)
    if actual_sha256 != expected_artifact.get("sha256"):
        raise ValueError(f"{args.data}: SHA256 does not match filter_summary.json")
    if expected_artifact.get("rows") != args.expected_rows:
        raise ValueError("summary row count does not match --expected-rows")

    mask_generator = None
    if not args.skip_tokenization:
        from transformers import AutoTokenizer

        from slime.utils.mask_utils import MultiTurnLossMaskGenerator

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=True,
        )
        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")

    domains: Counter[str] = Counter()
    source_rows: Counter[int] = Counter()
    decisions: Counter[str] = Counter()
    max_total_tokens = 0
    row_count = 0
    for line_number, row in read_jsonl(args.data):
        domain, source_row, decision = validate_row_structure(
            row,
            line_number=line_number,
            expected_schedule_index=row_count,
        )
        domains[domain] += 1
        source_rows[source_row] += 1
        decisions[decision] += 1
        if mask_generator is not None:
            token_ids, loss_mask = mask_generator.get_loss_mask(row["messages"])
            if len(token_ids) != len(loss_mask) or not any(loss_mask):
                raise ValueError(f"line {line_number}: invalid token/loss mask")
            if len(token_ids) > args.max_total_tokens:
                raise ValueError(
                    f"line {line_number}: {len(token_ids)} tokens exceeds "
                    f"{args.max_total_tokens}"
                )
            max_total_tokens = max(max_total_tokens, len(token_ids))
        row_count += 1
    if row_count != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {row_count}")
    if len(decisions) != (summary.get("selected") or {}).get("canonical_rows"):
        raise ValueError("unique scheduled decisions do not match canonical row count")
    return {
        "status": "pass",
        "data": str(args.data),
        "sha256": actual_sha256,
        "rows": row_count,
        "unique_source_rows": len(source_rows),
        "unique_filter_decisions": len(decisions),
        "rows_by_domain": dict(sorted(domains.items())),
        "max_total_tokens": max_total_tokens if mask_generator is not None else None,
        "max_allowed_tokens": args.max_total_tokens,
        "tokenization_checked": mask_generator is not None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=19318)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--skip-tokenization", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = validate(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
