#!/usr/bin/env python3
"""Combine a selected pilot train pool with separately evaluated held-out tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.documents import read_json, read_jsonl, write_json, write_jsonl  # type: ignore
else:
    from .documents import read_json, read_jsonl, write_json, write_jsonl


TRAINING_POOLS = (
    "sft_candidate",
    "grpo_frontier",
    "behavior_anchor",
    "hard_sft_only",
)
ACTION_FORMS = {
    "l3_single_action",
    "l4_two_skill",
    "l5_medium_workflow",
    "l6_full_workflow",
}


def combine_final_pool(
    *,
    contracts: list[dict[str, Any]],
    base_manifest: dict[str, Any],
    base_trajectories: list[dict[str, Any]],
    eval_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    all_ids = {row["task_id"] for row in contracts}
    train_ids = {row["task_id"] for row in contracts if row["split"] == "train"}
    eval_ids = all_ids - train_ids
    challenge_ids = {
        row["task_id"] for row in contracts if row["split"] == "challenge"
    }
    base_pools = base_manifest.get("pools") or {}
    pools = {
        name: sorted(train_ids & set(base_pools.get(name) or []))
        for name in TRAINING_POOLS
    }
    pools["synthetic_challenge"] = sorted(challenge_ids)

    valid_ids = sorted(
        all_ids
        & (
            set(base_manifest.get("valid_autonomous_task_ids") or [])
            | set(eval_manifest.get("valid_autonomous_task_ids") or [])
        )
    )
    four_trial_ids = sorted(
        all_ids
        & (
            set(base_manifest.get("four_trial_task_ids") or [])
            | set(eval_manifest.get("four_trial_task_ids") or [])
        )
    )
    valid_set = set(valid_ids)
    four_trial_set = set(four_trial_ids)
    required_four = eval_ids | {
        row["task_id"]
        for row in contracts
        if row["split"] == "train" and row.get("task_form") in ACTION_FORMS
    } | set(pools["grpo_frontier"])
    failures = []
    if missing := sorted(all_ids - valid_set):
        failures.append(f"{len(missing)} tasks lack a valid autonomous trajectory")
    if missing := sorted(required_four - four_trial_set):
        failures.append(f"{len(missing)} stratified tasks lack four valid trials")

    trajectories = [
        row for row in base_trajectories if row.get("task_id") in train_ids
    ]
    manifest = {
        "task_count": len(contracts),
        "pool_counts": {name: len(task_ids) for name, task_ids in pools.items()},
        "pools": pools,
        "autonomous_success_trajectories": len(trajectories),
        "valid_autonomous_task_ids": valid_ids,
        "four_trial_task_ids": four_trial_ids,
        "unresolved_count": len(failures),
        "result_model_names_verified": bool(
            base_manifest.get("result_model_names_verified")
            and eval_manifest.get("result_model_names_verified")
        ),
        "result_file_count": int(base_manifest.get("result_file_count") or 0)
        + int(eval_manifest.get("result_file_count") or 0),
        "teacher_success_trajectories": 0,
        "successful_training_trajectories": len(trajectories),
    }
    return manifest, trajectories, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-trajectories", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successful-trajectories", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, trajectories, failures = combine_final_pool(
        contracts=read_jsonl(args.contracts),
        base_manifest=read_json(args.base_manifest),
        base_trajectories=read_jsonl(args.base_trajectories),
        eval_manifest=read_json(args.eval_manifest),
    )
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
