#!/usr/bin/env python3
"""Check every Banking synthetic v1 acceptance condition on frozen artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import (  # type: ignore
        CHALLENGE_TASK_COUNT,
        DATA_ORIGIN,
        DEV_TASK_COUNT,
        MIN_GRPO_FRONTIER,
        MIN_TRAIN_SCENARIOS,
        MODEL_PATH,
        TRAIN_CATEGORY_QUOTAS,
        TRAIN_FORM_QUOTAS,
        TRAIN_TASK_COUNT,
    )
    from banking_synthetic.documents import read_json, read_jsonl, write_json  # type: ignore
    from banking_synthetic.validate import static_errors  # type: ignore
else:
    from .constants import (
        CHALLENGE_TASK_COUNT,
        DATA_ORIGIN,
        DEV_TASK_COUNT,
        MIN_GRPO_FRONTIER,
        MIN_TRAIN_SCENARIOS,
        MODEL_PATH,
        TRAIN_CATEGORY_QUOTAS,
        TRAIN_FORM_QUOTAS,
        TRAIN_TASK_COUNT,
    )
    from .documents import read_json, read_jsonl, write_json
    from .validate import static_errors


EXPECTED_SPLIT_COUNTS = {
    "train": TRAIN_TASK_COUNT,
    "dev": DEV_TASK_COUNT,
    "challenge": CHALLENGE_TASK_COUNT,
}


def acceptance_errors(
    *,
    tasks: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    allowed_document_ids: set[str],
    pool_manifest: dict[str, Any],
    reference_report: dict[str, Any],
    isolation_report: dict[str, Any],
    model_manifests: list[dict[str, Any]],
    successful_trajectories: list[dict[str, Any]] | None = None,
) -> list[str]:
    errors = static_errors(tasks, specs, contracts, allowed_document_ids)
    contracts_by_split = {
        split: [row for row in contracts if row["split"] == split]
        for split in EXPECTED_SPLIT_COUNTS
    }
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        if len(contracts_by_split[split]) != expected:
            errors.append(
                f"{split}: expected {expected} tasks, found {len(contracts_by_split[split])}"
            )
    scenario_sets = {
        split: {row["scenario_id"] for row in rows}
        for split, rows in contracts_by_split.items()
    }
    for left, right in (("train", "dev"), ("train", "challenge"), ("dev", "challenge")):
        if scenario_sets[left] & scenario_sets[right]:
            errors.append(f"scenario ids overlap between {left} and {right}")
    if len(scenario_sets["train"]) < MIN_TRAIN_SCENARIOS:
        errors.append(
            f"train has fewer than {MIN_TRAIN_SCENARIOS} distinct scenarios"
        )
    train_categories = Counter(
        row["business_category"] for row in contracts_by_split["train"]
    )
    train_forms = Counter(row["task_form"] for row in contracts_by_split["train"])
    if train_categories != Counter(TRAIN_CATEGORY_QUOTAS):
        errors.append("train business-category quotas differ from the declared v1 quotas")
    if train_forms != Counter(TRAIN_FORM_QUOTAS):
        errors.append("train task-form quotas differ from the declared v1 quotas")

    pools = pool_manifest.get("pools") or {}
    train_ids = {row["task_id"] for row in contracts_by_split["train"]}
    challenge_ids = {
        row["task_id"] for row in contracts_by_split["challenge"]
    }
    all_ids = {row["task_id"] for row in contracts}
    sft_ids = set(pools.get("sft_candidate") or [])
    pooled_ids = {
        task_id for task_ids in pools.values() for task_id in task_ids
    }
    if extra := pooled_ids - all_ids:
        errors.append(f"pool manifest contains {len(extra)} unknown task ids")
    training_pool_ids = {
        task_id
        for name in (
            "sft_candidate",
            "grpo_frontier",
            "behavior_anchor",
            "hard_sft_only",
        )
        for task_id in pools.get(name) or []
    }
    if leaked := training_pool_ids - train_ids:
        errors.append(f"training pools contain {len(leaked)} non-train task ids")
    if leaked := set(pools.get("synthetic_challenge") or []) - challenge_ids:
        errors.append(
            f"synthetic challenge pool contains {len(leaked)} tasks outside challenge"
        )
    if missing := train_ids - sft_ids:
        errors.append(f"{len(missing)} train tasks lack a successful Qwen3.8 trajectory")
    successful_trajectories = successful_trajectories or []
    invalid_task_ids = sum(
        row.get("task_id") not in train_ids for row in successful_trajectories
    )
    invalid_origins = sum(
        row.get("data_origin") != DATA_ORIGIN for row in successful_trajectories
    )
    ineligible = sum(
        row.get("training_eligible") is not True for row in successful_trajectories
    )
    non_reward_one = sum(
        float((row.get("reward_info") or {}).get("reward") or 0.0) != 1.0
        for row in successful_trajectories
    )
    trajectory_ids = [row.get("id") for row in successful_trajectories if row.get("id")]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        errors.append("successful trajectory ids are not unique")
    if invalid_task_ids:
        errors.append(
            f"{invalid_task_ids} successful trajectories have non-train task ids"
        )
    if invalid_origins:
        errors.append(
            f"{invalid_origins} successful trajectories have invalid data origin"
        )
    if ineligible:
        errors.append(
            f"{ineligible} successful trajectories are not training eligible"
        )
    if non_reward_one:
        errors.append(f"{non_reward_one} successful trajectories are not reward-one")
    verified_trajectory_ids = {
        row["task_id"]
        for row in successful_trajectories
        if row.get("task_id") in train_ids
        and row.get("data_origin") == DATA_ORIGIN
        and row.get("training_eligible") is True
        and float((row.get("reward_info") or {}).get("reward") or 0.0) == 1.0
    }
    if missing := train_ids - verified_trajectory_ids:
        errors.append(
            f"{len(missing)} train tasks lack a verified successful training trajectory"
        )
    if pool_manifest.get("successful_training_trajectories") != len(
        successful_trajectories
    ):
        errors.append("pool successful trajectory count differs from its artifact")
    valid_autonomous = set(pool_manifest.get("valid_autonomous_task_ids") or [])
    if missing := all_ids - valid_autonomous:
        errors.append(f"{len(missing)} tasks lack a valid autonomous trajectory")
    four_trial = set(pool_manifest.get("four_trial_task_ids") or [])
    train_frontier = train_ids & set(pools.get("grpo_frontier") or [])
    action_train_ids = {
        row["task_id"]
        for row in contracts_by_split["train"]
        if row["task_form"]
        in {
            "l3_single_action",
            "l4_two_skill",
            "l5_medium_workflow",
            "l6_full_workflow",
        }
    }
    if missing := action_train_ids - four_trial:
        errors.append(
            f"{len(missing)} action-bearing train tasks lack four valid trials"
        )
    if len(train_frontier) < MIN_GRPO_FRONTIER:
        errors.append(
            f"train GRPO frontier has {len(train_frontier)} tasks, fewer than "
            f"{MIN_GRPO_FRONTIER}"
        )
    required_four = (
        {row["task_id"] for row in contracts_by_split["dev"]}
        | {row["task_id"] for row in contracts_by_split["challenge"]}
        | action_train_ids
        | set(pools.get("grpo_frontier") or [])
    )
    if missing := required_four - four_trial:
        errors.append(f"{len(missing)} stratified tasks lack four valid trials")
    if pool_manifest.get("unresolved_count"):
        errors.append(f"pool manifest has {pool_manifest['unresolved_count']} unresolved tasks")

    expected_total = sum(EXPECTED_SPLIT_COUNTS.values())
    if pool_manifest.get("task_count") != expected_total:
        errors.append(f"pool manifest task count is not {expected_total}")
    if pool_manifest.get("result_model_names_verified") is not True:
        errors.append("pool result model names were not verified")
    if reference_report.get("task_count") != expected_total:
        errors.append(
            f"combined reference report task count is not {expected_total}"
        )
    if reference_report.get("schema_pass_count") != expected_total:
        errors.append("combined schema validation is not 100%")
    if reference_report.get("reference_replay_count") != expected_total:
        errors.append(
            f"combined reference replay count is not {expected_total}"
        )
    if reference_report.get("reference_reward_one_count") != expected_total:
        errors.append("combined reference replay is not 100% reward-one")
    if reference_report.get("error_count") != 0:
        errors.append("combined reference report has errors")
    if reference_report.get("bm25_top10_hit_count") != reference_report.get(
        "bm25_query_count"
    ):
        errors.append("combined reference BM25 top-10 validation is incomplete")
    if isolation_report.get("task_count") != expected_total:
        errors.append(
            f"combined isolation report task count is not {expected_total}"
        )
    for field in (
        "error_count",
        "benchmark_entity_reuse_count",
        "benchmark_request_reuse_count",
        "benchmark_document_set_reuse_count",
        "benchmark_initial_record_reuse_count",
    ):
        if isolation_report.get(field) != 0:
            errors.append(f"combined isolation report has nonzero {field}")
    if isolation_report.get("allowed_document_count") != 480:
        errors.append("isolated document count is not 480")

    if not model_manifests:
        errors.append("runtime model manifests are missing")
    for manifest in model_manifests:
        if manifest.get("agent_model_path") != MODEL_PATH:
            errors.append("an Agent runtime did not use Qwen3.8-27B")
        if manifest.get("user_model_path") != MODEL_PATH:
            errors.append("a User runtime did not use Qwen3.8-27B")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in EXPECTED_SPLIT_COUNTS:
        parser.add_argument(f"--{split}-tasks", type=Path, required=True)
        parser.add_argument(f"--{split}-specs", type=Path, required=True)
        parser.add_argument(f"--{split}-contracts", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--successful-trajectories", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--isolation-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = []
    specs = []
    contracts = []
    for split in EXPECTED_SPLIT_COUNTS:
        tasks.extend(read_json(getattr(args, f"{split}_tasks")))
        specs.extend(read_jsonl(getattr(args, f"{split}_specs")))
        contracts.extend(read_jsonl(getattr(args, f"{split}_contracts")))
    pool_manifest = read_json(args.pool_manifest)
    errors = acceptance_errors(
        tasks=tasks,
        specs=specs,
        contracts=contracts,
        allowed_document_ids=set(read_json(args.allowlist)["allowed_document_ids"]),
        pool_manifest=pool_manifest,
        reference_report=read_json(args.reference_report),
        isolation_report=read_json(args.isolation_report),
        model_manifests=[read_json(path) for path in args.model_manifest],
        successful_trajectories=read_jsonl(args.successful_trajectories),
    )
    report = {
        "accepted": not errors,
        "task_count": len(tasks),
        "split_counts": {
            split: sum(row["split"] == split for row in contracts)
            for split in EXPECTED_SPLIT_COUNTS
        },
        "distinct_scenario_count": len({row["scenario_id"] for row in contracts}),
        "pool_counts": pool_manifest.get("pool_counts") or {},
        "error_count": len(errors),
    }
    write_json(args.output, report)
    write_json(args.failures, errors)
    print(json.dumps(report, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
