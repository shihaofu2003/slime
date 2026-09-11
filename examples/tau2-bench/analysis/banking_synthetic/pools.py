#!/usr/bin/env python3
"""Build overlapping SFT/GRPO/anchor/challenge pool manifests."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import DATA_ORIGIN, MODEL_PATH  # type: ignore
    from banking_synthetic.documents import read_json, read_jsonl, write_json, write_jsonl  # type: ignore
    from banking_synthetic.protocol import protocol_failure_reason  # type: ignore
else:
    from .constants import DATA_ORIGIN, MODEL_PATH
    from .documents import read_json, read_jsonl, write_json, write_jsonl
    from .protocol import protocol_failure_reason


def build_pools(
    *,
    contracts: list[dict[str, Any]],
    result_payloads: list[dict[str, Any]],
    teacher_task_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    simulations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_simulation_ids = set()
    for payload in result_payloads:
        for simulation in payload.get("simulations") or []:
            simulation_id = simulation.get("id")
            if simulation_id and simulation_id in seen_simulation_ids:
                continue
            if simulation_id:
                seen_simulation_ids.add(simulation_id)
            simulations[simulation["task_id"]].append(simulation)
    pools: dict[str, list[str]] = {
        "sft_candidate": [],
        "grpo_frontier": [],
        "behavior_anchor": [],
        "hard_sft_only": [],
        "synthetic_challenge": [],
    }
    successful_trajectories = []
    failures = []
    valid_task_ids = []
    four_trial_task_ids = []
    for contract in contracts:
        task_id = contract["task_id"]
        is_train = contract["split"] == "train"
        rows = simulations.get(task_id, [])
        classified = [(row, protocol_failure_reason(row)) for row in rows]
        protocol_failures = [row for row, reason in classified if reason is not None]
        valid = [row for row, reason in classified if reason is None]
        successes = [
            row
            for row in valid
            if float((row.get("reward_info") or {}).get("reward") or 0.0) == 1.0
        ]
        if len(valid) in {1, 4}:
            valid_task_ids.append(task_id)
        if len(valid) == 4:
            four_trial_task_ids.append(task_id)
        if is_train and (successes or task_id in teacher_task_ids):
            pools["sft_candidate"].append(task_id)
        if is_train and len(valid) == 4 and 1 <= len(successes) <= 3:
            pools["grpo_frontier"].append(task_id)
        if is_train and len(valid) == 4 and len(successes) == 4:
            pools["behavior_anchor"].append(task_id)
        if (
            is_train
            and len(valid) == 4
            and not successes
            and task_id in teacher_task_ids
        ):
            pools["hard_sft_only"].append(task_id)
        if contract["split"] == "challenge":
            pools["synthetic_challenge"].append(task_id)
        if is_train:
            for row in successes:
                trajectory = copy.deepcopy(row)
                trajectory["data_origin"] = DATA_ORIGIN
                trajectory["training_eligible"] = True
                successful_trajectories.append(trajectory)
        if len(valid) not in {1, 4}:
            failures.append(
                f"{task_id}: valid={len(valid)} protocol_failures={len(protocol_failures)}"
            )
    manifest = {
        "task_count": len(contracts),
        "pool_counts": {name: len(task_ids) for name, task_ids in pools.items()},
        "pools": pools,
        "autonomous_success_trajectories": len(successful_trajectories),
        "valid_autonomous_task_ids": valid_task_ids,
        "four_trial_task_ids": four_trial_task_ids,
        "unresolved_count": len(failures),
    }
    return manifest, successful_trajectories, failures


def teacher_trajectories_for_contracts(
    trajectories: list[dict[str, Any]], contract_task_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep teacher rows for the candidate or final task set being pooled."""

    return [row for row in trajectories if row.get("task_id") in contract_task_ids]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--model-manifest", type=Path, action="append", required=True)
    parser.add_argument("--teacher-manifest", type=Path)
    parser.add_argument("--teacher-trajectories", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successful-trajectories", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    contracts = read_jsonl(args.contracts)
    contract_by_id = {row["task_id"]: row for row in contracts}
    if len(args.results) != len(args.model_manifest):
        raise ValueError("each result file needs its runtime model manifest")
    result_payloads = [read_json(path) for path in args.results]
    model_manifests = [read_json(path) for path in args.model_manifest]
    for payload, model_manifest in zip(result_payloads, model_manifests):
        info = payload.get("info") or {}
        agent_llm = str((info.get("agent_info") or {}).get("llm") or "")
        user_llm = str((info.get("user_info") or {}).get("llm") or "")
        if model_manifest.get("agent_model_path") != MODEL_PATH or model_manifest.get(
            "user_model_path"
        ) != MODEL_PATH:
            raise ValueError("result runtime model path differs from Qwen3.8-27B")
        if not agent_llm.endswith(
            str(model_manifest.get("agent_served_model") or "missing")
        ):
            raise ValueError("result Agent model differs from its runtime manifest")
        if not user_llm.endswith(
            str(model_manifest.get("user_served_model") or "missing")
        ):
            raise ValueError("result User model differs from its runtime manifest")
    contract_task_ids = set(contract_by_id)
    teacher_ids = set()
    if args.teacher_manifest:
        teacher_ids = set(
            read_json(args.teacher_manifest).get("successful_task_ids") or []
        ) & contract_task_ids
    all_teacher_trajectories = [
        row for path in args.teacher_trajectories for row in read_jsonl(path)
    ]
    for row in all_teacher_trajectories:
        if row.get("data_origin") != DATA_ORIGIN:
            raise ValueError("teacher trajectory has invalid data origin")
        if row.get("model_path") != MODEL_PATH:
            raise ValueError("teacher trajectory did not use Qwen3.8-27B")
        if float((row.get("reward_info") or {}).get("reward") or 0.0) != 1.0:
            raise ValueError("teacher trajectory is not reward-one")
    teacher_trajectories = teacher_trajectories_for_contracts(
        all_teacher_trajectories, contract_task_ids
    )
    trajectory_teacher_ids = {row["task_id"] for row in teacher_trajectories}
    if teacher_ids and trajectory_teacher_ids != teacher_ids:
        raise ValueError("teacher manifest and trajectory task ids differ")
    teacher_ids.update(trajectory_teacher_ids)
    manifest, trajectories, failures = build_pools(
        contracts=contracts,
        result_payloads=result_payloads,
        teacher_task_ids=teacher_ids,
    )
    manifest["result_model_names_verified"] = True
    manifest["result_file_count"] = len(result_payloads)
    autonomous_success_ids = {row["task_id"] for row in trajectories}
    for row in teacher_trajectories:
        task_id = row["task_id"]
        if contract_by_id[task_id]["split"] != "train":
            continue
        if task_id not in autonomous_success_ids:
            trajectory = copy.deepcopy(row)
            trajectory["training_eligible"] = True
            trajectories.append(trajectory)
    manifest["teacher_success_trajectories"] = len(teacher_trajectories)
    manifest["successful_training_trajectories"] = len(trajectories)
    write_json(args.output, manifest)
    write_jsonl(args.successful_trajectories, trajectories)
    write_json(args.failures, failures)
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key != "pools" and not key.endswith("_task_ids")
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
