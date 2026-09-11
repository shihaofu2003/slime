#!/usr/bin/env python3
"""Select only missing or protocol-failed autonomous tasks for recovery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def recovery_selection(
    payloads: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    recovery_payloads: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    contract_by_id = {contract["task_id"]: contract for contract in contracts}
    expected_attempts: Counter[str] = Counter()
    first_observed_attempts: Counter[str] = Counter()
    valid_attempts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()

    def record_simulations(
        selected_payloads: list[dict[str, Any]], *, first_attempt: bool
    ) -> None:
        for payload in selected_payloads:
            for simulation in payload.get("simulations") or []:
                task_id = simulation["task_id"]
                if task_id not in contract_by_id:
                    continue
                if first_attempt:
                    first_observed_attempts[task_id] += 1
                reason = protocol_failure_reason(simulation)
                if reason is None:
                    valid_attempts[task_id] += 1
                else:
                    reasons[reason] += 1

    for payload in payloads:
        expected_attempts.update(
            task["id"]
            for task in payload.get("tasks") or []
            if task["id"] in contract_by_id
        )
    record_simulations(payloads, first_attempt=True)
    recovery_payloads = recovery_payloads or []
    record_simulations(recovery_payloads, first_attempt=False)
    for task_id, expected_count in expected_attempts.items():
        if task_id not in contract_by_id:
            raise ValueError(f"expected task absent from contracts: {task_id}")
        missing_count = expected_count - first_observed_attempts[task_id]
        if missing_count > 0:
            reasons["missing_simulation"] += missing_count

    deficits = {
        task_id: expected_count - valid_attempts[task_id]
        for task_id, expected_count in expected_attempts.items()
        if expected_count > valid_attempts[task_id]
    }
    round_task_ids = []
    for round_index in range(max(deficits.values(), default=0)):
        by_variant = {"bm25": [], "golden_retrieval": []}
        for task_id, deficit in sorted(deficits.items()):
            if deficit > round_index:
                variant = contract_by_id[task_id]["retrieval_variant"]
                by_variant[variant].append(task_id)
        round_task_ids.append(by_variant)
    result = (
        round_task_ids[0]
        if round_task_ids
        else {"bm25": [], "golden_retrieval": []}
    )
    report = {
        "first_attempt_task_count": len(expected_attempts),
        "first_attempt_expected_simulation_count": sum(expected_attempts.values()),
        "first_attempt_simulation_count": sum(first_observed_attempts.values()),
        "recovery_simulation_count": sum(
            len(payload.get("simulations") or []) for payload in recovery_payloads
        ),
        "recovery_task_count": sum(len(task_ids) for task_ids in result.values()),
        "recovery_attempt_count": sum(deficits.values()),
        "recovery_round_count": len(round_task_ids),
        "by_variant": {name: len(task_ids) for name, task_ids in result.items()},
        "by_reason": dict(sorted(reasons.items())),
        "round_task_ids": round_task_ids,
    }
    return result, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", default=[])
    parser.add_argument("--recovery-results", type=Path, action="append", default=[])
    parser.add_argument("--missing-run-manifest", type=Path, action="append", default=[])
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.results and not args.missing_run_manifest:
        raise ValueError("at least one result or missing-run manifest is required")
    payloads = [read_json(path) for path in args.results]
    payloads.extend(
        {
            "tasks": [
                {"id": task_id}
                for task_id in read_json(path).get("task_ids") or []
            ],
            "simulations": [],
        }
        for path in args.missing_run_manifest
    )
    selected, report = recovery_selection(
        payloads,
        read_jsonl(args.contracts),
        [read_json(path) for path in args.recovery_results],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, task_ids in selected.items():
        write_json(args.output_dir / f"{variant}_task_ids.json", task_ids)
    for round_index, by_variant in enumerate(report["round_task_ids"], 1):
        for variant, task_ids in by_variant.items():
            write_json(
                args.output_dir
                / f"round_{round_index}_{variant}_task_ids.json",
                task_ids,
            )
    write_json(args.output_dir / "recovery_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
