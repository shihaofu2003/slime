#!/usr/bin/env python3
"""Select scale tasks that need seeds 301-303 after a valid first trial."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import (  # type: ignore
        FOLLOWUP_TRAIN_COUNT,
        FRONTIER_FORM_PRIORITY,
    )
    from banking_synthetic.documents import read_json, read_jsonl, write_json  # type: ignore
    from banking_synthetic.protocol import protocol_failure_reason  # type: ignore
else:
    from .constants import FOLLOWUP_TRAIN_COUNT, FRONTIER_FORM_PRIORITY
    from .documents import read_json, read_jsonl, write_json
    from .protocol import protocol_failure_reason


def select_followup(
    contracts: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    train_count: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
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
    invalid_counts = {
        task_id: len(valid.get(task_id, []))
        for task_id in contract_by_id
        if len(valid.get(task_id, [])) != 1
    }
    if invalid_counts:
        examples = list(sorted(invalid_counts.items()))[:5]
        raise ValueError(
            "follow-up selection requires exactly one valid first trial per task; "
            f"examples: {examples}"
        )

    train = [row for row in contracts if row["split"] == "train"]
    if not 0 <= train_count <= len(train):
        raise ValueError("train_count is outside the available train range")
    buckets: dict[tuple[str, str, bool], deque[dict[str, Any]]] = defaultdict(deque)
    for contract in sorted(train, key=lambda row: row["task_id"]):
        simulation = valid[contract["task_id"]][0]
        succeeded = float((simulation.get("reward_info") or {}).get("reward") or 0.0) == 1.0
        buckets[
            (contract["business_category"], contract["task_form"], succeeded)
        ].append(contract)

    selected_train: list[dict[str, Any]] = []
    available_forms = {row["task_form"] for row in train}
    form_order = [form for form in FRONTIER_FORM_PRIORITY if form in available_forms]
    form_order.extend(sorted(available_forms - set(form_order)))
    for form in form_order:
        bucket_keys = sorted(
            (key for key in buckets if key[1] == form),
            key=lambda key: (key[0], not key[2]),
        )
        while len(selected_train) < train_count:
            added = False
            for key in bucket_keys:
                if buckets[key] and len(selected_train) < train_count:
                    selected_train.append(buckets[key].popleft())
                    added = True
            if not added:
                break
        if len(selected_train) == train_count:
            break
    selected = selected_train + [
        row for row in contracts if row["split"] in {"dev", "challenge"}
    ]
    by_variant = {"bm25": [], "golden_retrieval": []}
    for contract in selected:
        by_variant[contract["retrieval_variant"]].append(contract["task_id"])
    for task_ids in by_variant.values():
        task_ids.sort()

    selected_successes = sum(
        float((valid[row["task_id"]][0].get("reward_info") or {}).get("reward") or 0.0)
        == 1.0
        for row in selected_train
    )
    report = {
        "first_trial_task_count": len(valid),
        "first_trial_success_count": sum(
            float((rows[0].get("reward_info") or {}).get("reward") or 0.0) == 1.0
            for rows in valid.values()
        ),
        "selected_train_count": len(selected_train),
        "selected_train_first_success_count": selected_successes,
        "selected_train_first_failure_count": len(selected_train) - selected_successes,
        "selected_train_by_form": dict(
            sorted(Counter(row["task_form"] for row in selected_train).items())
        ),
        "selected_dev_count": sum(row["split"] == "dev" for row in selected),
        "selected_challenge_count": sum(
            row["split"] == "challenge" for row in selected
        ),
        "selected_total_count": len(selected),
        "by_variant": {
            name: len(task_ids) for name, task_ids in by_variant.items()
        },
    }
    return by_variant, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--train-count", type=int, default=FOLLOWUP_TRAIN_COUNT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected, report = select_followup(
        read_jsonl(args.contracts),
        [read_json(path) for path in args.results],
        args.train_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, task_ids in selected.items():
        write_json(args.output_dir / f"{variant}_task_ids.json", task_ids)
    write_json(args.output_dir / "followup_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
