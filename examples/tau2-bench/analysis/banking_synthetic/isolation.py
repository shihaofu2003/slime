#!/usr/bin/env python3
"""Build the 480-document allowlist and audit synthetic artifacts.

This is the only synthetic-v1 module allowed to read sealed benchmark tasks.
Generation and teacher modules consume the resulting allowlist and synthetic
specifications, never benchmark examples.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import DATA_ORIGIN, TASK_ID_PREFIX  # type: ignore
    from banking_synthetic.documents import read_json, read_jsonl, write_json  # type: ignore
else:
    from .constants import DATA_ORIGIN, TASK_ID_PREFIX
    from .documents import read_json, read_jsonl, write_json


def load_benchmark_tasks(tasks_dir: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted(tasks_dir.glob("task_*.json"))]


def load_documents(documents_dir: Path) -> dict[str, dict[str, Any]]:
    documents = {}
    for path in sorted(documents_dir.glob("*.json")):
        document = read_json(path)
        documents[document["id"]] = document
    return documents


def build_allowlist(
    benchmark_tasks: list[dict[str, Any]], documents: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    required = {
        document_id
        for task in benchmark_tasks
        for document_id in task.get("required_documents") or []
    }
    allowed = sorted(set(documents) - required)
    if len(benchmark_tasks) != 97:
        raise ValueError(f"expected 97 sealed benchmark tasks, found {len(benchmark_tasks)}")
    if len(documents) != 698:
        raise ValueError(f"expected 698 Banking documents, found {len(documents)}")
    if len(required) != 218 or len(allowed) != 480:
        raise ValueError(
            f"expected 218 excluded and 480 allowed documents, found {len(required)} and {len(allowed)}"
        )
    return {
        "data_origin": DATA_ORIGIN,
        "allowed_document_count": 480,
        "excluded_required_document_count": 218,
        "allowed_document_ids": allowed,
    }


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _tokens(value: str) -> set[str]:
    return {token for token in _normalized_text(value).split() if len(token) >= 4}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _benchmark_entity_values(tasks: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    identity_fields = {
        "user_id",
        "account_id",
        "card_id",
        "transaction_id",
        "credit_card_account_id",
        "email",
        "phone_number",
        "name",
        "customer_name",
        "address",
    }
    for task in tasks:
        initial = ((task.get("initial_state") or {}).get("initialization_data") or {})
        for side in ("agent_data", "user_data"):
            data = initial.get(side) or {}
            for table in data.values():
                for record_id, record in (table.get("data") or {}).items():
                    if len(str(record_id)) >= 5:
                        values.add(str(record_id))
                    if isinstance(record, dict):
                        for key, value in record.items():
                            if key in identity_fields and isinstance(value, (str, int)):
                                values.add(str(value))
        for action in ((task.get("evaluation_criteria") or {}).get("actions") or []):
            for key, value in (action.get("arguments") or {}).items():
                if key in identity_fields and isinstance(value, (str, int)):
                    values.add(str(value))
    return {value for value in values if len(value) >= 5}


def _initial_records(tasks: list[dict[str, Any]]) -> set[str]:
    records = set()
    for task in tasks:
        initial = ((task.get("initial_state") or {}).get("initialization_data") or {})
        for side in ("agent_data", "user_data"):
            for table in (initial.get(side) or {}).values():
                for record in (table.get("data") or {}).values():
                    records.add(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return records


def audit_artifacts(
    *,
    tasks: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    allowlist: dict[str, Any],
    benchmark_tasks: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    allowed = set(allowlist["allowed_document_ids"])
    benchmark_required_sets = {
        frozenset(task.get("required_documents") or []) for task in benchmark_tasks
    }
    benchmark_requests = {
        _normalized_text(str((task.get("user_scenario") or {}).get("instructions") or ""))
        for task in benchmark_tasks
    }
    benchmark_request_tokens = [
        (task["id"], _tokens(str((task.get("user_scenario") or {}).get("instructions") or "")))
        for task in benchmark_tasks
    ]
    benchmark_entities = _benchmark_entity_values(benchmark_tasks)
    benchmark_initial_records = _initial_records(benchmark_tasks)

    task_by_id = {task.get("id"): task for task in tasks}
    contract_by_id = {contract.get("task_id"): contract for contract in contracts}
    spec_by_scenario = {spec.get("scenario_id"): spec for spec in specs}
    if len(task_by_id) != len(tasks):
        errors.append("task ids are not unique")
    if len(contract_by_id) != len(contracts):
        errors.append("contract task ids are not unique")
    if set(task_by_id) != set(contract_by_id):
        errors.append("task and contract ids differ")

    nearest_score = 0.0
    nearest_pair: tuple[str, str] | None = None
    for task_id, task in task_by_id.items():
        if not isinstance(task_id, str) or not task_id.startswith(TASK_ID_PREFIX):
            errors.append(f"invalid synthetic task id: {task_id}")
            continue
        contract = contract_by_id.get(task_id, {})
        if contract.get("data_origin") != DATA_ORIGIN:
            errors.append(f"{task_id}: invalid data_origin")
        if "source_task_id" in contract:
            errors.append(f"{task_id}: source_task_id is prohibited")
        scenario_id = contract.get("scenario_id")
        spec = spec_by_scenario.get(scenario_id)
        if spec is None:
            errors.append(f"{task_id}: scenario spec is missing")

        required = set(task.get("required_documents") or [])
        if not required <= allowed:
            errors.append(f"{task_id}: contains document outside the 480-document allowlist")
        if frozenset(required) in benchmark_required_sets:
            errors.append(f"{task_id}: reuses a benchmark required-document set")

        request = str((task.get("user_scenario") or {}).get("instructions") or "")
        normalized = _normalized_text(request)
        if normalized in benchmark_requests:
            errors.append(f"{task_id}: reuses a complete benchmark user request")
        request_tokens = _tokens(request)
        for benchmark_id, benchmark_tokens in benchmark_request_tokens:
            score = _jaccard(request_tokens, benchmark_tokens)
            if score > nearest_score:
                nearest_score = score
                nearest_pair = (task_id, benchmark_id)
            if score >= 0.72:
                errors.append(
                    f"{task_id}: probable benchmark-instance paraphrase of {benchmark_id} (token Jaccard {score:.3f})"
                )
                break

        for value in _walk_strings({"task": task, "spec": spec or {}, "contract": contract}):
            if value in benchmark_entities:
                errors.append(f"{task_id}: reuses benchmark entity value {value!r}")
                break
        synthetic_records = _initial_records([task])
        if synthetic_records & benchmark_initial_records:
            errors.append(f"{task_id}: reuses a benchmark initialization record")

    report = {
        "data_origin": DATA_ORIGIN,
        "task_count": len(tasks),
        "scenario_count": len(spec_by_scenario),
        "allowed_document_count": len(allowed),
        "benchmark_task_count": len(benchmark_tasks),
        "benchmark_entity_reuse_count": sum("entity value" in error for error in errors),
        "benchmark_request_reuse_count": sum("user request" in error for error in errors),
        "benchmark_document_set_reuse_count": sum("required-document set" in error for error in errors),
        "benchmark_initial_record_reuse_count": sum(
            "initialization record" in error for error in errors
        ),
        "nearest_request_pair": list(nearest_pair) if nearest_pair else None,
        "nearest_request_token_jaccard": nearest_score,
        "error_count": len(errors),
    }
    return errors, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    allowlist = subparsers.add_parser("build-allowlist")
    allowlist.add_argument("--benchmark-tasks-dir", type=Path, required=True)
    allowlist.add_argument("--documents-dir", type=Path, required=True)
    allowlist.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--benchmark-tasks-dir", type=Path, required=True)
    audit.add_argument("--tasks", type=Path, required=True)
    audit.add_argument("--specs", type=Path, required=True)
    audit.add_argument("--contracts", type=Path, required=True)
    audit.add_argument("--allowlist", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--failures", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    benchmark_tasks = load_benchmark_tasks(args.benchmark_tasks_dir)
    if args.command == "build-allowlist":
        manifest = build_allowlist(benchmark_tasks, load_documents(args.documents_dir))
        write_json(args.output, manifest)
        print(json.dumps({key: value for key, value in manifest.items() if key != "allowed_document_ids"}, ensure_ascii=False))
        return
    errors, report = audit_artifacts(
        tasks=read_json(args.tasks),
        specs=read_jsonl(args.specs),
        contracts=read_jsonl(args.contracts),
        allowlist=read_json(args.allowlist),
        benchmark_tasks=benchmark_tasks,
    )
    write_json(args.output, report)
    write_json(args.failures, errors)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
