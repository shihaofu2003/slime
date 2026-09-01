#!/usr/bin/env python3
"""Prepare, calibrate, summarize, and filter the tau2 SFT quality audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from sft_quality import (
    DIMENSIONS,
    PROMPT_VERSION,
    audit_source_dialogs,
    build_samples,
    classify_dialog,
    read_jsonl,
    write_jsonl,
)
from trajectory_patterns import load_tool_catalog


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
DEFAULT_SOURCE = SERVICE_AGENT_ROOT / "datasets/AReaL-tau2-data/tau2_sft_train.jsonl"
DEFAULT_TRAINING = (
    SERVICE_AGENT_ROOT
    / "slime/output/datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking_max8192.jsonl"
)
DEFAULT_TAU2_ROOT = SERVICE_AGENT_ROOT / "tau2-bench"
DEFAULT_OUT = SERVICE_AGENT_ROOT / "slime/output/experiments/tau2-sft-quality-audit"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rate(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2%}" if denominator else "n/a"


def aggregate_inventory(features: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    label_combinations = Counter(
        (record["source_correct"], record["source_reward"]) for record in features
    )
    label_training_rows = Counter()
    for record in features:
        label_training_rows[(record["source_correct"], record["source_reward"])] += record[
            "included_row_count"
        ]
    static_issues = Counter(
        issue for record in features for issue in record["static"]["hard_issues"]
    )
    hard_issues = Counter(issue for record in features for issue in record["hard_issues"])
    domains = Counter(record["domain"] for record in features)
    training_rows_by_domain = Counter()
    positive_labels_by_domain = Counter()
    clean_success_by_domain = Counter()
    clean_success_rows_by_domain = Counter()
    clean_success_full_by_domain = Counter()
    clean_success_state_changing_by_domain = Counter()
    multitool_by_domain = Counter()
    exposures = Counter()
    exposures_by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    exposures_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    behavior_flags = Counter()
    for record in features:
        domain = record["domain"]
        training_rows_by_domain[domain] += record["included_row_count"]
        positive_labels_by_domain[domain] += int(
            record["source_correct"] == 1 and record["source_reward"] == 1
        )
        clean_success_by_domain[domain] += int(record["deterministic_clean_success"])
        is_clean = int(record["deterministic_clean_success"])
        clean_success_rows_by_domain[domain] += is_clean * record["included_row_count"]
        clean_success_full_by_domain[domain] += is_clean * int(
            record["included_row_count"] == record["source_row_count"]
        )
        clean_success_state_changing_by_domain[domain] += is_clean * int(
            record["workflow_type"] == "state_changing"
        )
        exposures.update(record["exposures"])
        exposures_by_domain[domain].update(record["exposures"])
        label_group = (
            "positive"
            if record["source_correct"] == 1 and record["source_reward"] == 1
            else "nonpositive_or_conflict"
        )
        exposures_by_label[label_group].update(record["exposures"])
        static = record["static"]
        multitool_by_domain[domain] += int(bool(static["multitool_turns"]))
        behavior_flags["multitool_dialogs"] += int(bool(static["multitool_turns"]))
        behavior_flags["multitool_with_write_dialogs"] += int(bool(static["multiwrite_turns"]))
        behavior_flags["repeat_dialogs"] += int(
            static["adjacent_repeated_calls"] + static["nonadjacent_repeated_calls"] > 0
        )
        behavior_flags["fanout_dialogs"] += int(bool(static["fanout_names"]))
        behavior_flags["tool_error_dialogs"] += int(bool(static["tool_error_turns"]))
        behavior_flags["state_changing_dialogs"] += int(record["workflow_type"] == "state_changing")
    dialog_retention_by_domain = {}
    for domain in sorted(domains):
        domain_records = [record for record in features if record["domain"] == domain]
        retention_values = [
            record["included_row_count"] / record["source_row_count"]
            for record in domain_records
            if record["source_row_count"]
        ]
        dialog_retention_by_domain[domain] = {
            "training_dialogs": len(domain_records),
            "fully_retained_dialogs": sum(
                record["included_row_count"] == record["source_row_count"]
                for record in domain_records
            ),
            "median_target_retention": (
                statistics.median(retention_values) if retention_values else None
            ),
            "median_source_rows": statistics.median(
                record["source_row_count"] for record in domain_records
            ),
            "median_training_rows": statistics.median(
                record["included_row_count"] for record in domain_records
            ),
        }
    payload = dict(base)
    payload.update(
        {
            "dialogs_by_domain": dict(sorted(domains.items())),
            "training_rows_by_domain": dict(sorted(training_rows_by_domain.items())),
            "positive_label_dialogs_by_domain": dict(sorted(positive_labels_by_domain.items())),
            "deterministic_clean_success_dialogs_by_domain": dict(
                sorted(clean_success_by_domain.items())
            ),
            "deterministic_clean_success_rows_by_domain": dict(
                sorted(clean_success_rows_by_domain.items())
            ),
            "deterministic_clean_success_full_dialogs_by_domain": dict(
                sorted(clean_success_full_by_domain.items())
            ),
            "deterministic_clean_success_state_changing_dialogs_by_domain": dict(
                sorted(clean_success_state_changing_by_domain.items())
            ),
            "multitool_dialogs_by_domain": dict(sorted(multitool_by_domain.items())),
            "dialog_retention_by_domain": dialog_retention_by_domain,
            "label_combinations": [
                {
                    "correct": key[0],
                    "reward": key[1],
                    "dialogs": count,
                    "training_rows": label_training_rows[key],
                }
                for key, count in sorted(label_combinations.items(), key=lambda item: str(item[0]))
            ],
            "dialogs_with_positive_labels": sum(
                record["source_correct"] == 1 and record["source_reward"] == 1
                for record in features
            ),
            "deterministic_clean_success_dialogs": sum(
                record["deterministic_clean_success"] for record in features
            ),
            "static_issue_dialogs": sum(bool(record["static"]["hard_issues"]) for record in features),
            "static_issues": dict(sorted(static_issues.items())),
            "all_hard_issues": dict(sorted(hard_issues.items())),
            "loss_exposures": dict(sorted(exposures.items())),
            "loss_exposures_by_domain": {
                domain: dict(sorted(counts.items()))
                for domain, counts in sorted(exposures_by_domain.items())
            },
            "loss_exposures_by_label": {
                label: dict(sorted(counts.items()))
                for label, counts in sorted(exposures_by_label.items())
            },
            "behavior_flags": dict(sorted(behavior_flags.items())),
        }
    )
    return payload


def deterministic_report(inventory: dict[str, Any]) -> str:
    labels = inventory["label_combinations"]
    nonpositive_training_rows = sum(
        row["training_rows"]
        for row in labels
        if row["correct"] != 1 or row["reward"] != 1
    )
    label_rows = "\n".join(
        f"| {row['correct']} | {row['reward']} | {row['dialogs']} | {row['training_rows']} |"
        for row in labels
    )
    issues = inventory["static_issues"]
    issue_rows = "\n".join(
        f"| `{name}` | {count} | {rate(count, inventory['training_dialogs'])} |"
        for name, count in issues.items()
    ) or "| none | 0 | 0% |"
    exposures = inventory["loss_exposures"]
    behaviors = inventory["behavior_flags"]
    retention_rows = "\n".join(
        f"| {domain} | {source_count} | {inventory['training_rows_by_domain'].get(domain, 0)} | "
        f"{rate(inventory['training_rows_by_domain'].get(domain, 0), source_count)} |"
        for domain, source_count in inventory["source_rows_by_domain"].items()
    )
    dialog_retention_rows = "\n".join(
        f"| {domain} | {values['training_dialogs']:,} | "
        f"{values['fully_retained_dialogs']:,} "
        f"({rate(values['fully_retained_dialogs'], values['training_dialogs'])}) | "
        f"{values['median_target_retention']:.2%} | "
        f"{values['median_source_rows']:g} | {values['median_training_rows']:g} |"
        for domain, values in inventory["dialog_retention_by_domain"].items()
    )
    domain_quality_rows = "\n".join(
        f"| {domain} | {dialogs} | "
        f"{inventory['positive_label_dialogs_by_domain'].get(domain, 0)} | "
        f"{inventory['multitool_dialogs_by_domain'].get(domain, 0)} "
        f"({rate(inventory['multitool_dialogs_by_domain'].get(domain, 0), dialogs)}) | "
        f"{inventory['deterministic_clean_success_dialogs_by_domain'].get(domain, 0)} | "
        f"{inventory['deterministic_clean_success_rows_by_domain'].get(domain, 0)} | "
        f"{inventory['deterministic_clean_success_full_dialogs_by_domain'].get(domain, 0)} | "
        f"{inventory['deterministic_clean_success_state_changing_dialogs_by_domain'].get(domain, 0)} |"
        for domain, dialogs in inventory["dialogs_by_domain"].items()
    )
    domain_exposure_rows = "\n".join(
        f"| {domain} | {counts.get('target_assistant_rows', 0):,} | "
        f"{counts.get('target_multitool_rows', 0):,} "
        f"({rate(counts.get('target_multitool_rows', 0), counts.get('target_assistant_rows', 0))}) | "
        f"{counts.get('training_rows_with_multitool_prefix', 0):,} "
        f"({rate(counts.get('training_rows_with_multitool_prefix', 0), counts.get('target_assistant_rows', 0))}) | "
        f"{counts.get('multitool_turn_exposures', 0):,} |"
        for domain, counts in inventory["loss_exposures_by_domain"].items()
    )
    label_exposure_rows = "\n".join(
        f"| `{label}` | {counts.get('target_assistant_rows', 0):,} | "
        f"{counts.get('target_multitool_rows', 0):,} "
        f"({rate(counts.get('target_multitool_rows', 0), counts.get('target_assistant_rows', 0))}) | "
        f"{counts.get('training_rows_with_multitool_prefix', 0):,} "
        f"({rate(counts.get('training_rows_with_multitool_prefix', 0), counts.get('target_assistant_rows', 0))}) |"
        for label, counts in inventory["loss_exposures_by_label"].items()
    )
    return f"""# tau2 SFT 数据质量：确定性审计

实验名：`tau2-sft-quality-audit`。对象是实际训练文件中的
{inventory['training_rows']:,} 条 prefix 样本，对应 {inventory['training_dialogs']:,}
个去重 source dialog；source 全量为 {inventory['source_rows']:,} 行、
{inventory['source_dialogs']:,} 个 dialog。

## 训练保留率

| domain | source rows | trained rows | retention |
|---|---:|---:|---:|
{retention_rows}

| domain | training dialogs | full dialog retained | median per-dialog retention | median source targets | median kept targets |
|---|---:|---:|---:|---:|---:|
{dialog_retention_rows}

转换形状与 8192-token 上限共同排除了
{inventory['source_rows'] - inventory['training_rows']:,} 行；实际训练数据的领域比例
因此不同于 source，尤其需要单独检查 telecom 的能力覆盖。

## Source 标签

| correct | reward | dialogs | training rows |
|---:|---:|---:|---:|
{label_rows}

正标签 dialog 为 {inventory['dialogs_with_positive_labels']:,}；同时满足正标签且未命中
确定性硬错误的候选为 {inventory['deterministic_clean_success_dialogs']:,}。`reward=1`
只说明 source verifier 的最终结果，不能替代 policy、过程和可模仿性判断。实际训练
文件中有 {nonpositive_training_rows:,} 行来自非正或冲突标签 dialog。

## 确定性问题

| issue | dialogs | rate |
|---|---:|---:|
{issue_rows}

| domain | dialogs | positive label | same-turn multi | rule-clean dialogs | rule-clean rows | full | state-changing |
|---|---:|---:|---:|---:|---:|---:|---:|
{domain_quality_rows}

所有转换后样本都注入了允许 `one or more tool calls` 的输出协议，而 tau2 三个领域
policy 要求每轮最多一个调用；这是数据协议冲突，不由最终 reward 自动发现。

同轮多调用 dialog 中有 {behaviors.get('multitool_with_write_dialogs', 0):,} 个把写操作
提前放进 batch；另有 {behaviors.get('fanout_dialogs', 0):,} 个 dialog 对同一工具使用
至少三组参数，{behaviors.get('repeat_dialogs', 0):,} 个含完全重复调用。这些模式会让
模型学到“提高调用宽度/覆盖”，但不保证参数、依赖或最终状态正确。

## Loss 暴露

- 训练 target 行：{exposures.get('target_assistant_rows', 0):,}
- target 本身为同轮多调用：{exposures.get('target_multitool_rows', 0):,}
- prefix 已含同轮多调用的训练行：{exposures.get('training_rows_with_multitool_prefix', 0):,}
- assistant turn 暴露：{exposures.get('assistant_turn_exposures', 0):,}
- tool call 暴露：{exposures.get('tool_call_exposures', 0):,}
- 同轮多调用 turn 暴露：{exposures.get('multitool_turn_exposures', 0):,}
- 写调用暴露：{exposures.get('write_call_exposures', 0):,}

| domain | target rows | multi target | prefix 已含 multi | multi turn exposure |
|---|---:|---:|---:|---:|
{domain_exposure_rows}

| source label group | target rows | multi target | prefix 已含 multi |
|---|---:|---:|---:|
{label_exposure_rows}

这里统计的是 SFT loss 实际看到的重复 prefix，而不是去重 dialog 数。后续模型评审
在 dialog 层进行，最终筛选再映射回原始训练行。

## 下一步

用 `calibration_sample.jsonl` 对本地 judge 做双次盲审校准；校准通过后审查
`judge_payloads.jsonl` 全量。低置信、重复审查不一致、规则冲突或格式失败的样本
才升级到 API。最终由 `summarize` 生成 `keep/review/drop` manifest 和训练 JSONL。
"""


def prepare(args: argparse.Namespace) -> None:
    catalog = load_tool_catalog(args.tau2_root)
    features, payloads, base_inventory = audit_source_dialogs(
        args.source,
        args.training,
        catalog,
    )
    inventory = aggregate_inventory(features, base_inventory)
    success_sample, calibration_sample = build_samples(
        features,
        payloads,
        success_limit=args.success_sample_size,
        calibration_limit=args.calibration_sample_size,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "features.jsonl", features)
    write_jsonl(args.out / "judge_payloads.jsonl", payloads)
    write_jsonl(args.out / "success_sample.jsonl", success_sample)
    write_jsonl(args.out / "calibration_sample.jsonl", calibration_sample)
    write_jsonl(
        args.out / "calibration_payloads.jsonl",
        (
            {
                **item["judge_payload"],
                "_calibration_category": item["features"].get("calibration_category"),
            }
            for item in calibration_sample
        ),
    )
    write_json(args.out / "inventory.json", inventory)
    (args.out / "DETERMINISTIC_REPORT.md").write_text(
        deterministic_report(inventory), encoding="utf-8"
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


def calibration_metrics(
    samples: list[dict[str, Any]], reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    sample_by_id = {item["features"]["dialog_id"]: item["features"] for item in samples}
    review_by_id = {record["dialog_id"]: record for record in reviews}
    verdict_matches = []
    dimension_differences = []
    hard_issue_kept = []
    confidences = []
    missing = []
    incomplete_primary_pairs = []
    unresolved = 0
    for dialog_id, feature in sample_by_id.items():
        record = review_by_id.get(dialog_id)
        if not record:
            missing.append(dialog_id)
            continue
        primary = record.get("primary_reviews") or []
        if len(primary) >= 2:
            verdict_matches.append(primary[0].get("verdict") == primary[1].get("verdict"))
            for name in DIMENSIONS:
                first = primary[0].get("dimensions", {}).get(name, {}).get("score")
                second = primary[1].get("dimensions", {}).get(name, {}).get("score")
                if isinstance(first, int) and isinstance(second, int):
                    dimension_differences.append(abs(first - second))
        else:
            incomplete_primary_pairs.append(dialog_id)
        local_review = primary[0] if primary else {}
        if isinstance(local_review.get("confidence"), (int, float)):
            confidences.append(float(local_review["confidence"]))
        if feature["static"]["hard_issues"]:
            hard_issue_kept.append(local_review.get("verdict") == "keep")
        unresolved += int(bool(record.get("unresolved_escalation")))
    completed = len(sample_by_id) - len(missing)
    metrics = {
        "expected": len(sample_by_id),
        "completed": completed,
        "completion_rate": completed / len(sample_by_id) if sample_by_id else 0,
        "missing_dialog_ids": missing,
        "complete_primary_pairs": len(sample_by_id) - len(missing) - len(incomplete_primary_pairs),
        "incomplete_primary_pair_dialog_ids": incomplete_primary_pairs,
        "repeated_verdict_agreement": (
            statistics.mean(verdict_matches) if verdict_matches else 0
        ),
        "mean_dimension_absolute_difference": (
            statistics.mean(dimension_differences) if dimension_differences else None
        ),
        "hard_issue_keep_rate": statistics.mean(hard_issue_kept) if hard_issue_kept else 0,
        "median_confidence": statistics.median(confidences) if confidences else 0,
        "unresolved_escalations": unresolved,
    }
    metrics["passed"] = bool(
        metrics["completion_rate"] >= 0.99
        and metrics["complete_primary_pairs"] >= math.ceil(0.99 * len(sample_by_id))
        and metrics["repeated_verdict_agreement"] >= 0.85
        and metrics["mean_dimension_absolute_difference"] is not None
        and metrics["mean_dimension_absolute_difference"] <= 0.5
        and metrics["hard_issue_keep_rate"] <= 0.05
        and metrics["median_confidence"] >= 0.75
    )
    return metrics


def calibrate(args: argparse.Namespace) -> None:
    metrics = calibration_metrics(
        list(read_jsonl(args.sample)),
        list(read_jsonl(args.reviews)),
    )
    write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.require_pass and not metrics["passed"]:
        raise SystemExit(2)


def load_review_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    reviews = [
        record
        for record in read_jsonl(path)
        if record.get("prompt_version") == PROMPT_VERSION
    ]
    duplicates = [dialog_id for dialog_id, count in Counter(r["dialog_id"] for r in reviews).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate judge records: {duplicates[:5]}")
    return {record["dialog_id"]: record for record in reviews}


def mechanical_review_contradictions(
    feature: dict[str, Any], review: dict[str, Any]
) -> list[str]:
    """Find judge claims that exact parser facts can disprove."""

    issues = set(review.get("critical_issues") or [])
    static = feature.get("static") or {}
    contradictions = []
    if review.get("verdict") == "keep" and static.get("hard_issues"):
        contradictions.extend(
            f"keep_despite_{issue}" for issue in static["hard_issues"]
        )
    if "multi_tool_policy_violation" in issues and not static.get("multitool_turns"):
        contradictions.append("false_multi_tool_policy_violation")
    if "tool_namespace_confusion" in issues and not static.get("namespace_confusion_turns"):
        contradictions.append("false_tool_namespace_confusion")
    return contradictions


def adjudicate_review_record(
    feature: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Prefer an uncontradicted review and keep any override conservative."""

    adjudicated = dict(record)
    selected = record.get("selected_review")
    if not isinstance(selected, dict):
        adjudicated["adjudication_overrides"] = []
        return adjudicated
    contradictions = mechanical_review_contradictions(feature, selected)
    if not contradictions:
        adjudicated["adjudication_overrides"] = []
        return adjudicated

    candidates = []
    primary_provider = record.get("primary_provider")
    for review in record.get("primary_reviews") or []:
        if isinstance(review, dict):
            candidates.append((primary_provider, review))
    fallback = record.get("fallback_review")
    if isinstance(fallback, dict):
        candidates.append((record.get("fallback_provider"), fallback))
    replacement = next(
        (
            (provider, review)
            for provider, review in candidates
            if review is not selected
            and not mechanical_review_contradictions(feature, review)
        ),
        None,
    )
    replacement_provider, replacement_review = replacement or (None, None)
    adjudicated.update(
        {
            "selected_provider": replacement_provider,
            "selected_review": replacement_review,
            "unresolved_escalation": True,
            "adjudication_overrides": [
                {
                    "reason": "judge_contradicts_mechanical_facts",
                    "rejected_provider": record.get("selected_provider"),
                    "contradictions": contradictions,
                    "replacement_provider": replacement_provider,
                }
            ],
        }
    )
    return adjudicated


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "dialog_id",
        "domain",
        "tier",
        "quality_score",
        "selected_provider",
        "source_correct",
        "source_reward",
        "source_row_count",
        "included_row_count",
        "fully_retained",
        "deterministic_clean_success",
        "workflow_type",
        "length_bucket",
        "reasons",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "reasons": ";".join(row["reasons"]),
                }
            )


def filtered_rows(training: Path, kept_dialogs: set[str]) -> Iterable[dict[str, Any]]:
    for row in read_jsonl(training):
        dialog_id = str((row.get("metadata") or {}).get("source_dialog_id") or "")
        if dialog_id in kept_dialogs:
            metadata = dict(row.get("metadata") or {})
            metadata["sft_quality_tier"] = "keep"
            row = dict(row)
            row["metadata"] = metadata
            yield row


def judge_diagnostics(
    features: list[dict[str, Any]], review_map: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    feature_by_id = {feature["dialog_id"]: feature for feature in features}
    selected = {
        dialog_id: record["selected_review"]
        for dialog_id, record in review_map.items()
        if record.get("selected_review") is not None
    }
    positive = {
        dialog_id
        for dialog_id in selected
        if feature_by_id.get(dialog_id, {}).get("source_correct") == 1
        and feature_by_id.get(dialog_id, {}).get("source_reward") == 1
    }
    multitool = {
        dialog_id
        for dialog_id in selected
        if (feature_by_id.get(dialog_id, {}).get("static") or {}).get("multitool_turns")
    }
    multiwrite = {
        dialog_id
        for dialog_id in selected
        if (feature_by_id.get(dialog_id, {}).get("static") or {}).get("multiwrite_turns")
    }
    fully_retained = {
        dialog_id
        for dialog_id in selected
        if feature_by_id.get(dialog_id, {}).get("included_row_count")
        == feature_by_id.get(dialog_id, {}).get("source_row_count")
        and feature_by_id.get(dialog_id, {}).get("source_row_count") is not None
    }
    airline = {
        dialog_id
        for dialog_id in selected
        if feature_by_id.get(dialog_id, {}).get("domain") == "airline"
    }
    scopes = {
        "all_reviewed": set(selected),
        "positive_rule_clean": {
            dialog_id
            for dialog_id in selected
            if feature_by_id.get(dialog_id, {}).get("deterministic_clean_success")
        },
        "positive_same_turn_multi_read_only": (positive & multitool) - multiwrite,
        "positive_same_turn_multi_with_write": positive & multitool & multiwrite,
        "positive_fully_retained": positive & fully_retained,
        "positive_truncated_training_view": positive - fully_retained,
        "positive_airline_fully_retained": positive & airline & fully_retained,
        "positive_airline_truncated_training_view": (positive & airline) - fully_retained,
        "nonpositive_or_conflict": set(selected) - positive,
        **{
            f"domain_{domain}": {
                dialog_id
                for dialog_id in selected
                if feature_by_id.get(dialog_id, {}).get("domain") == domain
            }
            for domain in ("airline", "retail", "telecom")
        },
    }
    dimension_quality = {}
    for scope, dialog_ids in scopes.items():
        dimensions = {}
        for name in DIMENSIONS:
            scores = [
                selected[dialog_id].get("dimensions", {}).get(name, {}).get("score")
                for dialog_id in dialog_ids
            ]
            scores = [score for score in scores if isinstance(score, int) and not isinstance(score, bool)]
            below_three = sum(score < 3 for score in scores)
            dimensions[name] = {
                "scored": len(scores),
                "mean_score": round(statistics.mean(scores), 4) if scores else None,
                "below_3": below_three,
                "below_3_rate": below_three / len(scores) if scores else None,
            }
        dimension_quality[scope] = dimensions
    local_fallback_pairs = [
        (record["primary_reviews"][0], record["fallback_review"])
        for record in review_map.values()
        if record.get("primary_reviews") and isinstance(record.get("fallback_review"), dict)
    ]
    label_groups = {
        "positive": positive,
        "nonpositive_or_conflict": set(selected) - positive,
    }
    label_alignment = {
        label: {
            "reviewed": len(dialog_ids),
            "verdicts": dict(
                sorted(Counter(selected[dialog_id].get("verdict") for dialog_id in dialog_ids).items())
            ),
            "outcomes": dict(
                sorted(
                    Counter(
                        selected[dialog_id].get("apparent_task_outcome")
                        for dialog_id in dialog_ids
                    ).items()
                )
            ),
        }
        for label, dialog_ids in label_groups.items()
    }
    local_fallback_dimension_delta = {}
    for name in DIMENSIONS:
        differences = [
            fallback["dimensions"][name]["score"] - local["dimensions"][name]["score"]
            for local, fallback in local_fallback_pairs
        ]
        local_fallback_dimension_delta[name] = {
            "mean_fallback_minus_local": (
                round(statistics.mean(differences), 4) if differences else None
            ),
            "mean_absolute_difference": (
                round(statistics.mean(abs(value) for value in differences), 4)
                if differences
                else None
            ),
        }
    return {
        "selected_reviews": len(selected),
        "cohort_sizes": {scope: len(dialog_ids) for scope, dialog_ids in scopes.items()},
        "selected_provider_counts": dict(
            sorted(
                Counter(
                    str(record.get("selected_provider") or "none")
                    for record in review_map.values()
                ).items()
            )
        ),
        "verdicts": dict(
            sorted(Counter(review.get("verdict") for review in selected.values()).items(), key=str)
        ),
        "outcomes": dict(
            sorted(
                Counter(review.get("apparent_task_outcome") for review in selected.values()).items(),
                key=str,
            )
        ),
        "escalated_dialogs": sum(bool(record.get("escalation_reasons")) for record in review_map.values()),
        "unresolved_escalations": sum(
            bool(record.get("unresolved_escalation")) for record in review_map.values()
        ),
        "records_with_errors": sum(bool(record.get("errors")) for record in review_map.values()),
        "local_fallback_pairs": len(local_fallback_pairs),
        "local_fallback_verdict_agreement": sum(
            local.get("verdict") == fallback.get("verdict")
            for local, fallback in local_fallback_pairs
        ),
        "local_fallback_outcome_agreement": sum(
            local.get("apparent_task_outcome") == fallback.get("apparent_task_outcome")
            for local, fallback in local_fallback_pairs
        ),
        "local_fallback_dimension_delta": local_fallback_dimension_delta,
        "source_label_alignment": label_alignment,
        "mechanical_adjudication_overrides": sum(
            bool(record.get("adjudication_overrides")) for record in review_map.values()
        ),
        "dimension_quality": dimension_quality,
    }


def model_report(
    inventory: dict[str, Any],
    manifest: list[dict[str, Any]],
    reviewed: int,
    kept_rows: int,
    judge_usage: dict[str, dict[str, int]],
    diagnostics: dict[str, Any],
) -> str:
    tiers = Counter(row["tier"] for row in manifest)
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_domain_training_rows: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest:
        by_domain[row["domain"]][row["tier"]] += 1
        by_domain_training_rows[row["domain"]][row["tier"]] += row["included_row_count"]
    domain_rows = "\n".join(
        f"| {domain} | {counts['keep']} | {by_domain_training_rows[domain]['keep']} | "
        f"{counts['review']} | {by_domain_training_rows[domain]['review']} | "
        f"{counts['drop']} | {by_domain_training_rows[domain]['drop']} |"
        for domain, counts in sorted(by_domain.items())
    )
    reasons = Counter(reason for row in manifest for reason in row["reasons"])
    reason_rows = "\n".join(
        f"| `{reason}` | {count} |"
        for reason, count in reasons.most_common(15)
    ) or "| none | 0 |"
    usage_rows = "\n".join(
        f"| `{provider}` | {usage['calls']} | {usage['prompt_tokens']} | "
        f"{usage['completion_tokens']} | {usage['total_tokens']} |"
        for provider, usage in sorted(judge_usage.items())
    ) or "| none | 0 | 0 | 0 | 0 |"
    all_dimensions = diagnostics["dimension_quality"]["all_reviewed"]
    clean_dimensions = diagnostics["dimension_quality"]["positive_rule_clean"]

    def dimension_value(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    dimension_rows = "\n".join(
        f"| `{name}` | {dimension_value(all_dimensions[name]['mean_score'])} | "
        f"{rate(all_dimensions[name]['below_3'], all_dimensions[name]['scored'])} | "
        f"{dimension_value(clean_dimensions[name]['mean_score'])} | "
        f"{rate(clean_dimensions[name]['below_3'], clean_dimensions[name]['scored'])} |"
        for name in DIMENSIONS
    )
    cohort_labels = (
        ("positive_rule_clean", "positive + rule-clean"),
        ("positive_fully_retained", "positive + 完整保留"),
        ("positive_truncated_training_view", "positive + 截断视图"),
        ("positive_airline_fully_retained", "airline positive + 完整保留"),
        ("positive_airline_truncated_training_view", "airline positive + 截断视图"),
        ("positive_same_turn_multi_read_only", "positive multi（batch 无写）"),
        ("positive_same_turn_multi_with_write", "positive multi（batch 含写）"),
        ("nonpositive_or_conflict", "非正/冲突 source 标签"),
        ("domain_airline", "airline"),
        ("domain_retail", "retail"),
        ("domain_telecom", "telecom"),
    )
    cohort_dimension_rows = "\n".join(
        f"| {label} | {diagnostics['cohort_sizes'][scope]:,} | "
        f"{dimension_value(diagnostics['dimension_quality'][scope]['task_completion']['mean_score'])} | "
        f"{dimension_value(diagnostics['dimension_quality'][scope]['policy_compliance']['mean_score'])} | "
        f"{dimension_value(diagnostics['dimension_quality'][scope]['tool_choice']['mean_score'])} | "
        f"{dimension_value(diagnostics['dimension_quality'][scope]['argument_grounding']['mean_score'])} | "
        f"{dimension_value(diagnostics['dimension_quality'][scope]['workflow_and_dependencies']['mean_score'])} | "
        f"{dimension_value(diagnostics['dimension_quality'][scope]['recovery']['mean_score'])} |"
        for scope, label in cohort_labels
    )
    label_alignment_rows = "\n".join(
        f"| `{label}` | {values['reviewed']:,} | "
        f"{values['verdicts'].get('keep', 0):,} / "
        f"{values['verdicts'].get('review', 0):,} / "
        f"{values['verdicts'].get('drop', 0):,} | "
        f"{values['outcomes'].get('success', 0):,} / "
        f"{values['outcomes'].get('partial', 0):,} / "
        f"{values['outcomes'].get('failure', 0):,} / "
        f"{values['outcomes'].get('uncertain', 0):,} |"
        for label, values in diagnostics["source_label_alignment"].items()
    )
    provider_delta_rows = "\n".join(
        f"| `{name}` | "
        f"{dimension_value(values['mean_fallback_minus_local'])} | "
        f"{dimension_value(values['mean_absolute_difference'])} |"
        for name, values in diagnostics["local_fallback_dimension_delta"].items()
    )
    provider_summary = ", ".join(
        f"`{provider}` {count:,}"
        for provider, count in diagnostics["selected_provider_counts"].items()
    ) or "none"
    return f"""# tau2 SFT 数据质量审计

实验名：`tau2-sft-quality-audit`。审计实际训练文件的
{inventory['training_rows']:,} 条样本（{inventory['training_dialogs']:,} 个去重
dialog），并将 dialog 结论映射回训练行。

## 结论

- 完成模型盲审：{reviewed:,}/{inventory['training_dialogs']:,} dialogs。
- `keep/review/drop`：{tiers['keep']:,} / {tiers['review']:,} / {tiers['drop']:,} dialogs。
- Judge 与确定性事实冲突并被保守覆盖：
  {diagnostics['mechanical_adjudication_overrides']:,} dialogs；这些冲突不会自动进入 `keep`。
- 严格筛选后训练行：{kept_rows:,}/{inventory['training_rows']:,}
  ({rate(kept_rows, inventory['training_rows'])})。
- `keep` 要求 source `correct=reward=1`、无确定性硬错误、judge verdict 为 `keep`，
  outcome 为成功或仅因 target-tool 边界而不确定，置信度至少 0.75、总分至少
  3.25/4，且任务完成、policy、工具、参数、流程五项均至少 3/4。

## 分领域

| domain | keep dialogs | keep rows | review dialogs | review rows | drop dialogs | drop rows |
|---|---:|---:|---:|---:|---:|---:|
{domain_rows}

## 主要筛除原因

| reason | dialogs |
|---|---:|
{reason_rows}

## 模型维度诊断

| criterion | all mean | all score < 3 | positive + rule-clean mean | clean score < 3 |
|---|---:|---:|---:|---:|
{dimension_rows}

`positive + rule-clean` 是 source 正标签且未命中确定性硬错误的候选；该列用于判断
排除协议/标签污染后，是否仍存在工具选择、参数、流程或沟通质量问题。

| cohort | n | task | policy | tool | args | workflow | recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
{cohort_dimension_rows}

同轮 multi 两行只包含 source 正标签；`batch 无写/含写` 由确定性工具集合划分。
由于 multi 在 tau2 中必然违反协议，判断它是否还教坏“如何调用”时应重点比较
tool、args 与 workflow，而不能只看 policy 分数。`n` 是确定性裁决后仍有可用
selected review 的数量。

“完整保留/截断视图”用于分离轨迹本身的语义质量与 max-length/转换造成的缺尾；
即使 prompt 要求不惩罚正常 prefix 边界，两组分差仍应作为 judge 敏感性和训练覆盖
偏差共同解释。

| source label group | reviewed | judge keep / review / drop | outcome success / partial / failure / uncertain |
|---|---:|---:|---:|
{label_alignment_rows}

Source 标签是完整 dialog 的结果，而 judge 看的是实际保留下来的最长训练 prefix；
二者不一致不能直接叫“错标”，但能识别成功 dialog 中不值得模仿的步骤，以及失败
dialog 中可能仍可回收的早期步骤。

## Judge 用量

最终选用 provider：{provider_summary}；触发升级
{diagnostics['escalated_dialogs']:,} 条，未解决 {diagnostics['unresolved_escalations']:,}
条，调用错误记录 {diagnostics['records_with_errors']:,} 条。本地/API 配对
{diagnostics['local_fallback_pairs']:,} 条，verdict 一致
{rate(diagnostics['local_fallback_verdict_agreement'], diagnostics['local_fallback_pairs'])}，
outcome 一致
{rate(diagnostics['local_fallback_outcome_agreement'], diagnostics['local_fallback_pairs'])}。

| dimension | fallback - local mean | paired mean absolute difference |
|---|---:|---:|
{provider_delta_rows}

配对集合由本地低置信、不确定或规则冲突触发，存在选择偏差；差值用于暴露 judge
敏感性，不能解释成 provider 全局优劣。

| provider | calls | prompt tokens | completion tokens | total tokens |
|---|---:|---:|---:|---:|
{usage_rows}

## 产物

- `selection_manifest.jsonl/csv`：每个 dialog 的可审计决定。
- `filtered_sft_keep.jsonl`：只含 `keep` dialog 的原训练行，未改写对话。
- `SUCCESS_CASES.md`：高质量成功轨迹的模型归因与证据轮次。
- `features.jsonl` / `judge_reviews.jsonl`：规则事实与模型原始结构化判断。
- `judge_adjudication_overrides.jsonl`：模型判断与机器可判事实冲突的保守裁决。

## 限制

SFT source 不含可重放的 task DB 与完整 evaluator criteria，因此模型只能结合 policy、
工具 schema、对话和 observation 判断“可模仿质量”；source reward 仍作为独立证据，
不能证明每一步或最终 DB 都正确。筛选结果应通过同配置 SFT 重训和 held-out official
eval 做最终因果验证。`filtered_sft_keep.jsonl` 为了可审计仍保留原 system prompt，
包括允许 `one or more tool calls` 的冲突文本；它是选择产物，不是可直接训练的
`tau2_official_single` 数据。正式重训应重新生成或回放协议一致的单调用轨迹。
"""


def success_cases(manifest: list[dict[str, Any]], review_map: dict[str, dict[str, Any]]) -> str:
    kept = [row for row in manifest if row["tier"] == "keep"]
    selected = []
    for domain in ("airline", "retail", "telecom"):
        candidates = sorted(
            (row for row in kept if row["domain"] == domain),
            key=lambda row: (-(row["quality_score"] or 0), row["dialog_id"]),
        )
        domain_selected = []
        for fully_retained, workflow_type in (
            (True, "state_changing"),
            (True, "read_only"),
            (False, "state_changing"),
            (False, "read_only"),
        ):
            candidate = next(
                (
                    row
                    for row in candidates
                    if row.get("fully_retained") is fully_retained
                    and row.get("workflow_type") == workflow_type
                    and row not in domain_selected
                ),
                None,
            )
            if candidate is not None:
                domain_selected.append(candidate)
        domain_selected.extend(
            row for row in candidates if row not in domain_selected
        )
        selected.extend(domain_selected[:4])
    lines = [
        "# 高质量成功轨迹画像",
        "",
        "以下案例通过 source 正标签、确定性规则和模型盲审三道门槛。证据轮次对应",
        "`judge_payloads.jsonl` 的 conversation index。",
        "部分训练 prefix 恰好结束于正确的 target tool call；此时 `keep/uncertain` 表示",
        "可见动作值得模仿，不表示缺失 observation 之后的任务成功已被直接观察。",
        "",
        "## 代表案例",
        "",
        "| domain | dialog | score | 成功原因 |",
        "|---|---|---:|---|",
    ]
    for row in selected:
        review = review_map[row["dialog_id"]]["selected_review"]
        strengths = review.get("strengths") or []
        if strengths:
            reason = "; ".join(
                f"{item.get('pattern')}: {item.get('explanation')} @ "
                f"{item.get('evidence_turns')}"
                for item in strengths[:3]
            )
        else:
            evidence = []
            for dimension in (
                "task_completion",
                "tool_choice",
                "workflow_and_dependencies",
                "authorization_and_confirmation",
                "recovery",
                "communication_and_closeout",
            ):
                finding = (review.get("dimensions") or {}).get(dimension) or {}
                if finding.get("score") == 4 and finding.get("rationale"):
                    evidence.append(
                        f"{dimension}: {finding['rationale']} @ "
                        f"{finding.get('evidence_turns', [])}"
                    )
                if len(evidence) == 3:
                    break
            reason = "; ".join(evidence) or review.get("summary", "")
        reason = str(reason).replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {row['domain']} | `{row['dialog_id']}` | {row['quality_score']:.2f} | {reason} |"
        )
    if not selected:
        lines.append("| - | - | - | 尚无通过全部门槛的模型评审 |")
    counterexamples = []
    for domain in ("airline", "retail", "telecom"):
        candidates = sorted(
            (
                row
                for row in manifest
                if row["domain"] == domain
                and row["tier"] != "keep"
                and row.get("deterministic_clean_success")
                and isinstance(
                    (review_map.get(row["dialog_id"]) or {}).get("selected_review"),
                    dict,
                )
            ),
            key=lambda row: (
                not bool(
                    review_map[row["dialog_id"]]["selected_review"].get(
                        "critical_issues"
                    )
                ),
                row["quality_score"] if row["quality_score"] is not None else -1,
                row["dialog_id"],
            ),
        )
        counterexamples.extend(candidates[:2])
    if counterexamples:
        lines.extend(
            [
                "",
                "## 规则未覆盖的语义反例",
                "",
                "这些 dialog 有 source 正标签且通过机械规则，但模型认为可见步骤仍不值得直接模仿。",
                "",
                "| domain | dialog | tier/score | 语义问题 |",
                "|---|---|---:|---|",
            ]
        )
        for row in counterexamples:
            review = review_map[row["dialog_id"]]["selected_review"]
            weaknesses = review.get("weaknesses") or []
            reason = "; ".join(
                f"{item.get('pattern')}: {item.get('explanation')} @ "
                f"{item.get('evidence_turns')}"
                for item in weaknesses[:3]
            )
            if not reason:
                dimensions = review.get("dimensions") or {}
                weak_dimensions = sorted(
                    (
                        (finding.get("score", 4), name, finding)
                        for name, finding in dimensions.items()
                        if isinstance(finding, dict) and finding.get("score", 4) < 3
                    ),
                    key=lambda item: (item[0], item[1]),
                )
                reason = "; ".join(
                    f"{name}: {finding.get('rationale', '')} @ "
                    f"{finding.get('evidence_turns', [])}"
                    for _, name, finding in weak_dimensions[:3]
                )
            reason = reason or review.get("summary", "")
            critical = ", ".join(review.get("critical_issues") or [])
            if critical:
                reason = f"{critical}: {reason}"
            reason = str(reason).replace("\n", " ").replace("|", "\\|")
            score = row["quality_score"]
            score_text = "n/a" if score is None else f"{score:.2f}"
            lines.append(
                f"| {row['domain']} | `{row['dialog_id']}` | "
                f"{row['tier']}/{score_text} | {reason} |"
            )
    lines.extend(
        [
            "",
            "## 共性定义",
            "",
            "高质量不是只有最终 reward=1；还要求意图完整、读后再写、工具与参数有 observation",
            "依据、写操作经过当轮确认、错误后能改变策略、最终陈述与工具结果一致，并严格遵守",
            "tau2 每轮最多一个 tool call 的领域协议。",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> None:
    features = list(read_jsonl(args.features))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    raw_reviews = load_review_map(args.reviews)
    reviews = {
        feature["dialog_id"]: adjudicate_review_record(
            feature, raw_reviews[feature["dialog_id"]]
        )
        for feature in features
        if feature["dialog_id"] in raw_reviews
    }
    diagnostics = judge_diagnostics(features, reviews)
    manifest = []
    for feature in features:
        decision = classify_dialog(
            feature,
            reviews.get(feature["dialog_id"]),
            confidence_threshold=args.confidence_threshold,
            score_threshold=args.score_threshold,
        )
        manifest.append(
            {
                "dialog_id": feature["dialog_id"],
                "domain": feature["domain"],
                "source_correct": feature["source_correct"],
                "source_reward": feature["source_reward"],
                "source_row_count": feature["source_row_count"],
                "included_row_count": feature["included_row_count"],
                "fully_retained": (
                    feature["included_row_count"] == feature["source_row_count"]
                ),
                "deterministic_clean_success": feature["deterministic_clean_success"],
                "workflow_type": feature["workflow_type"],
                "length_bucket": feature["length_bucket"],
                **decision,
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "selection_manifest.jsonl", manifest)
    write_manifest_csv(args.out / "selection_manifest.csv", manifest)
    write_jsonl(
        args.out / "judge_adjudication_overrides.jsonl",
        (
            {
                "dialog_id": dialog_id,
                "adjudication_overrides": record["adjudication_overrides"],
            }
            for dialog_id, record in reviews.items()
            if record.get("adjudication_overrides")
        ),
    )
    kept_dialogs = {row["dialog_id"] for row in manifest if row["tier"] == "keep"}
    kept_rows = write_jsonl(
        args.out / "filtered_sft_keep.jsonl",
        filtered_rows(args.training, kept_dialogs),
    )
    reviewed = sum(feature["dialog_id"] in reviews for feature in features)
    judge_usage: dict[str, Counter[str]] = defaultdict(Counter)
    for record in raw_reviews.values():
        for call in record.get("failed_judge_calls") or []:
            provider = str(call.get("provider") or "unknown")
            usage = call.get("usage") or {}
            judge_usage[provider]["calls"] += 1
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(field)
                if isinstance(value, int):
                    judge_usage[provider][field] += value
        completed_reviews = list(record.get("primary_reviews") or [])
        if record.get("fallback_review"):
            completed_reviews.append(record["fallback_review"])
        for review in completed_reviews:
            for call in review.get("judge_calls") or []:
                provider = str(call.get("provider") or "unknown")
                usage = call.get("usage") or {}
                judge_usage[provider]["calls"] += 1
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(field)
                    if isinstance(value, int):
                        judge_usage[provider][field] += value
    serialized_usage = {
        provider: {
            field: counts.get(field, 0)
            for field in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
        }
        for provider, counts in sorted(judge_usage.items())
    }
    (args.out / "REPORT.md").write_text(
        model_report(inventory, manifest, reviewed, kept_rows, serialized_usage, diagnostics),
        encoding="utf-8",
    )
    (args.out / "SUCCESS_CASES.md").write_text(
        success_cases(manifest, reviews), encoding="utf-8"
    )
    summary = {
        "dialogs": len(manifest),
        "reviewed_dialogs": reviewed,
        "tiers": dict(sorted(Counter(row["tier"] for row in manifest).items())),
        "kept_training_rows": kept_rows,
        "source_training_rows": inventory["training_rows"],
        "confidence_threshold": args.confidence_threshold,
        "score_threshold": args.score_threshold,
        "judge_usage": serialized_usage,
        "judge_diagnostics": diagnostics,
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Reconstruct and audit SFT dialogs")
    prepare_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare_parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    prepare_parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    prepare_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    prepare_parser.add_argument("--success-sample-size", type=int, default=48)
    prepare_parser.add_argument("--calibration-sample-size", type=int, default=72)
    prepare_parser.add_argument("--seed", type=int, default=20260802)
    prepare_parser.set_defaults(func=prepare)

    calibrate_parser = subparsers.add_parser("calibrate", help="Validate repeated local judge reviews")
    calibrate_parser.add_argument(
        "--sample", type=Path, default=DEFAULT_OUT / "calibration_sample.jsonl"
    )
    calibrate_parser.add_argument(
        "--reviews", type=Path, default=DEFAULT_OUT / "calibration_reviews.jsonl"
    )
    calibrate_parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUT / "calibration_report.json"
    )
    calibrate_parser.add_argument("--require-pass", action="store_true")
    calibrate_parser.set_defaults(func=calibrate)

    summarize_parser = subparsers.add_parser("summarize", help="Apply quality gates and filter SFT rows")
    summarize_parser.add_argument(
        "--features", type=Path, default=DEFAULT_OUT / "features.jsonl"
    )
    summarize_parser.add_argument(
        "--inventory", type=Path, default=DEFAULT_OUT / "inventory.json"
    )
    summarize_parser.add_argument(
        "--reviews", type=Path, default=DEFAULT_OUT / "judge_reviews.jsonl"
    )
    summarize_parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    summarize_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    summarize_parser.add_argument("--confidence-threshold", type=float, default=0.75)
    summarize_parser.add_argument("--score-threshold", type=float, default=3.25)
    summarize_parser.set_defaults(func=summarize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
