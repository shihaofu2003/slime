#!/usr/bin/env python3
"""Build a diagnostic manifest for tasks derived from the Banking benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACTS = (
    PROJECT_ROOT
    / "output/experiments/tau2-banking-task-curriculum/expanded_contracts.jsonl"
)

KNOWN_VARIANTS = {
    "retrieval_only",
    "decision_only",
    "evidence_given",
    "single_action",
    "full_composition",
    "two_skill_composition",
}
BENCHMARK_DERIVED_POOL = "benchmark_derived_diagnostic"
TRAINING_EXCLUSION_REASON = (
    "Derived from a Banking benchmark evaluation task; excluded from SFT, GRPO, and OPD."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_pool_manifest(
    results_paths: list[Path],
    contracts: list[dict[str, Any]],
    *,
    replacement_results_paths: Optional[list[Path]] = None,
    expected_task_count: Optional[int] = None,
) -> dict[str, Any]:
    contract_by_id = {item["task_id"]: item for item in contracts}
    if len(contract_by_id) != len(contracts):
        raise ValueError("contract ids must be unique")

    simulations: dict[str, tuple[dict[str, Any], Path]] = {}
    models = set()
    for path in results_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = (((payload.get("info") or {}).get("agent_info") or {}).get("llm"))
        if model:
            models.add(model)
        for simulation in payload.get("simulations") or []:
            task_id = simulation["task_id"]
            if task_id in simulations:
                raise ValueError(f"duplicate evaluated task: {task_id}")
            simulations[task_id] = (simulation, path)

    replacement_task_ids = set()
    for path in replacement_results_paths or []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = (((payload.get("info") or {}).get("agent_info") or {}).get("llm"))
        if model:
            models.add(model)
        for simulation in payload.get("simulations") or []:
            task_id = simulation["task_id"]
            if task_id not in simulations:
                raise ValueError(f"replacement task was not in base results: {task_id}")
            simulations[task_id] = (simulation, path)
            replacement_task_ids.add(task_id)

    if expected_task_count is not None and len(simulations) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} evaluated tasks, found {len(simulations)}"
        )
    if len(models) > 1:
        raise ValueError(f"results contain multiple Agent models: {sorted(models)}")

    records = []
    for task_id, (simulation, path) in sorted(simulations.items()):
        if task_id not in contract_by_id:
            raise ValueError(f"missing curriculum contract: {task_id}")
        item = contract_by_id[task_id]
        variant = item["variant"]
        if variant not in KNOWN_VARIANTS:
            raise ValueError(f"unknown curriculum variant: {variant}")
        termination_reason = simulation.get("termination_reason")
        reward = float((simulation.get("reward_info") or {}).get("reward") or 0.0)
        task_pool = (
            "infrastructure_recovery"
            if termination_reason == "infrastructure_error"
            else BENCHMARK_DERIVED_POOL
        )
        records.append(
            {
                "task_id": task_id,
                "source_task_id": item["source_task_id"],
                "variant": variant,
                "level": item["level"],
                "business_category": item["business_category"],
                "retrieval_variant": item["retrieval_variant"],
                "task_pool": task_pool,
                "benchmark_derived": True,
                "training_eligible": False,
                "training_exclusion_reason": TRAINING_EXCLUSION_REASON,
                "sft_opd_candidate": False,
                "reward": reward,
                "termination_reason": termination_reason,
                "simulation_id": simulation.get("id"),
                "results_file": str(path),
                "is_replacement": task_id in replacement_task_ids,
            }
        )

    return {
        "data_status": "benchmark_derived_diagnostic_only",
        "training_eligible": False,
        "training_exclusion_reason": TRAINING_EXCLUSION_REASON,
        "model": next(iter(models), None),
        "tasks": len(records),
        "verified_success_trajectories": sum(item["reward"] == 1.0 for item in records),
        "training_eligible_trajectories": 0,
        "task_pool_counts": dict(
            sorted(Counter(item["task_pool"] for item in records).items())
        ),
        "variant_counts": dict(
            sorted(Counter(item["variant"] for item in records).items())
        ),
        "replacement_task_ids": sorted(replacement_task_ids),
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--replacement-results", type=Path, nargs="+", default=None)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_pool_manifest(
        args.results,
        read_jsonl(args.contracts),
        replacement_results_paths=args.replacement_results,
        expected_task_count=args.expected_task_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "data_status",
                    "tasks",
                    "verified_success_trajectories",
                    "training_eligible_trajectories",
                    "task_pool_counts",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
