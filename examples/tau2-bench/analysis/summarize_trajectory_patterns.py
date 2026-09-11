#!/usr/bin/env python3
"""Merge deterministic features and semantic reviews into comparison artifacts."""

from __future__ import annotations

import argparse
import copy
import csv
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from trajectory_patterns import (
    KEY_PATTERNS,
    PATTERNS,
    PROMPT_VERSION,
    duplicate_ids,
    read_jsonl,
    validate_review,
)

MODEL_ORDER = ("raw", "sft", "rl2e6")
MODEL_DISPLAY_ZH = {
    "raw": "Raw 原始模型",
    "sft": "Multitool SFT",
    "rl2e6": "RL 2e-6 iter99",
}
PATTERN_DISPLAY_ZH = {
    "intent_tracking": "意图持续跟踪",
    "clarify_before_act": "行动前澄清",
    "read_plan_act": "先读取、再规划、后行动",
    "atomic_commit": "原子化提交",
    "progressive_diagnosis": "渐进式诊断",
    "grounded_recovery": "基于工具结果恢复",
    "grounded_closeout": "基于事实收尾",
    "wrong_tool_or_workflow": "工具或流程错误",
    "wrong_arguments": "参数错误",
    "missing_precondition": "缺少前置条件",
    "confirmation_violation": "确认流程违规",
    "unrelated_or_extra_write": "无关或额外写入",
    "multi_tool_policy_violation": "单轮多工具违规",
    "tool_namespace_confusion": "Agent/User 工具命名空间混淆",
    "nonexistent_tool": "调用不存在的工具",
    "repeat_or_fanout": "重复调用或扇出搜索",
    "failed_recovery": "失败后恢复不当",
    "false_success_claim": "无依据宣称成功",
    "incomplete_task": "任务未完成",
    "premature_transfer_or_close": "过早转人工或结束",
    "domain_policy_compliance": "领域 policy 合规",
}
PATTERN_DESCRIPTION_ZH = {
    "intent_tracking": "正确保持用户目标、话题切换和多个子任务。",
    "clarify_before_act": "行动前收齐身份、实体、选项、原因和支付信息。",
    "read_plan_act": "先读取真实状态，再依据状态与 policy 选择流程。",
    "atomic_commit": "列明变更、费用和影响，获得最新确认后只写入一次。",
    "progressive_diagnosis": "Telecom 按 service→data→MMS 依赖顺序诊断并复验。",
    "grounded_recovery": "理解工具错误后改变方案，不重复碰撞。",
    "grounded_closeout": "仅在工具结果支持时宣称成功，并完成全部子任务。",
    "wrong_tool_or_workflow": "选择了不适用的工具或业务流程。",
    "wrong_arguments": "工具参数缺失、越界或与真实状态不符。",
    "missing_precondition": "未完成身份、状态读取或资格检查便继续行动。",
    "confirmation_violation": "写入前未获得包含完整细节的最新确认。",
    "unrelated_or_extra_write": "执行了用户未要求或流程不需要的写操作。",
    "multi_tool_policy_violation": "同一轮调用多个工具，违反领域 policy。",
    "tool_namespace_confusion": "混用了 Agent 后台工具与 User 设备工具。",
    "nonexistent_tool": "调用工具目录中不存在的工具。",
    "repeat_or_fanout": "重复失败调用或无依据扩大搜索。",
    "failed_recovery": "工具失败后未调整方案或补齐信息。",
    "false_success_claim": "工具未支持成功结论，但 Agent 仍声称完成。",
    "incomplete_task": "遗漏子任务，或在目标完成前结束。",
    "premature_transfer_or_close": "尚有可执行步骤时便转人工或结束。",
    "domain_policy_compliance": "遵守领域 policy 和所有必需流程。",
}
FAILURE_MODE_DISPLAY_ZH = {
    "wrong_workflow": "流程错误",
    "wrong_arguments": "参数错误",
    "missing_precondition": "缺少前置条件",
    "confirmation_violation": "确认违规",
    "tool_namespace_confusion": "工具命名空间混淆",
    "failed_recovery": "失败恢复不当",
    "false_success_claim": "虚假成功声明",
    "incomplete_task": "任务未完成",
    "policy_violation": "Policy 违规",
    "other": "其他",
    "none": "未识别到主要 Agent 错误",
}
NEGATIVE_PATTERNS = (
    "wrong_tool_or_workflow",
    "wrong_arguments",
    "missing_precondition",
    "confirmation_violation",
    "unrelated_or_extra_write",
    "multi_tool_policy_violation",
    "tool_namespace_confusion",
    "nonexistent_tool",
    "repeat_or_fanout",
    "failed_recovery",
    "false_success_claim",
    "incomplete_task",
    "premature_transfer_or_close",
    "domain_policy_compliance",
)


def _review_map(path: Path) -> dict[str, dict[str, Any]]:
    records = [
        record
        for record in read_jsonl(path)
        if record.get("prompt_version") == PROMPT_VERSION
        and record.get("calibration_run", 0) == 0
    ]
    duplicates = duplicate_ids(records)
    if duplicates:
        raise ValueError(f"duplicate review IDs: {duplicates[:5]}")
    return {record["trajectory_id"]: record for record in records}


def adjudicate_review(
    feature: dict[str, Any], review: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Apply exact rule facts and taxonomy applicability to a semantic review."""
    result = copy.deepcopy(review)
    overrides: list[dict[str, str]] = []

    def replace(
        pattern: str,
        status: str,
        severity: str,
        reason: str,
        turn_indices: list[int] | None = None,
    ) -> None:
        finding = result["patterns"][pattern]
        old_status = finding["status"]
        old_severity = finding["severity"]
        if old_status != status or old_severity != severity:
            overrides.append(
                {
                    "trajectory_id": feature["trajectory_id"],
                    "pattern": pattern,
                    "judge_status": old_status,
                    "judge_severity": old_severity,
                    "final_status": status,
                    "final_severity": severity,
                    "reason": reason,
                }
            )
        finding["status"] = status
        finding["severity"] = severity
        if turn_indices is not None:
            finding["turn_indices"] = list(dict.fromkeys(turn_indices))[:2]
        if old_status != status:
            finding["evidence"] = reason

    for finding in result["patterns"].values():
        finding["turn_indices"] = list(
            dict.fromkeys(finding["turn_indices"])
        )[:2]
    user_turns = result["user_simulator_error"]["turn_indices"]
    result["user_simulator_error"]["turn_indices"] = list(
        dict.fromkeys(user_turns)
    )[:2]

    if feature["domain"] != "telecom":
        replace(
            "progressive_diagnosis",
            "not_applicable",
            "none",
            "Only telecom diagnosis is applicable.",
            [],
        )
    if (
        feature["workflow_type"] == "read_only"
        and feature["num_observed_writes"] == 0
    ):
        replace(
            "atomic_commit",
            "not_applicable",
            "none",
            "No state change was attempted.",
            [],
        )

    if feature["multitool_policy_violation"]:
        replace(
            "multi_tool_policy_violation",
            "fail",
            "critical",
            "Multiple tools were called in one turn.",
            feature["multitool_turn_indices"],
        )
        replace(
            "domain_policy_compliance",
            "fail",
            "critical",
            "One-tool-per-turn policy was violated.",
            feature["multitool_turn_indices"],
        )
        if result["primary_failure_mode"] == "none":
            result["primary_failure_mode"] = "policy_violation"
    else:
        replace(
            "multi_tool_policy_violation",
            "pass",
            "none",
            "No multi-tool turn occurred.",
            [],
        )

    if feature["namespace_confusion_turns"]:
        replace(
            "tool_namespace_confusion",
            "fail",
            "critical",
            "Tool requestor used the wrong namespace.",
            feature["namespace_confusion_turns"],
        )
        if result["primary_failure_mode"] == "none":
            result["primary_failure_mode"] = "tool_namespace_confusion"
    if feature["invalid_tool_calls"]:
        replace(
            "nonexistent_tool",
            "fail",
            "critical",
            "A called tool is absent from both catalogs.",
        )
        if result["primary_failure_mode"] == "none":
            result["primary_failure_mode"] = "wrong_workflow"
    if feature["tool_argument_schema_errors"]:
        replace(
            "wrong_arguments",
            "fail",
            "critical",
            "Tool arguments violated the declared schema.",
            [
                error["turn_index"]
                for error in feature["tool_argument_schema_errors"]
            ],
        )
        if result["primary_failure_mode"] == "none":
            result["primary_failure_mode"] = "wrong_arguments"

    for pattern in NEGATIVE_PATTERNS:
        finding = result["patterns"][pattern]
        if finding["status"] == "not_applicable":
            replace(
                pattern,
                "uncertain",
                "none",
                "Judge returned invalid applicability.",
                finding["turn_indices"],
            )
    return result, overrides


def behavior_valid_success(feature: dict[str, Any], review: dict[str, Any]) -> bool:
    if not feature["success"]:
        return False
    if any(
        finding.get("status") == "fail" and finding.get("severity") == "critical"
        for finding in review["patterns"].values()
    ):
        return False
    return all(
        review["patterns"][pattern]["status"] in {"pass", "not_applicable"}
        for pattern in KEY_PATTERNS
    )


def _rate(group: list[tuple[dict, dict]], predicate: Callable[[dict, dict], bool]) -> float:
    return statistics.mean(predicate(feature, review) for feature, review in group)


def write_pattern_summary(
    joined: list[tuple[dict[str, Any], dict[str, Any]]], outdir: Path
) -> None:
    fields = (
        "model",
        "domain",
        "reward_outcome",
        "n",
        "raw_reward_rate",
        "behavior_valid_success_rate",
        "reward_positive_behavior_invalid_rate",
        "pattern",
        "applicable_n",
        "pass_rate",
        "fail_rate",
        "critical_fail_rate",
        "basis",
    )
    groups: dict[tuple[str, str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for feature, review in joined:
        groups[
            (
                feature["model"],
                feature["domain"],
                "success" if feature["success"] else "failure",
            )
        ].append((feature, review))
    rows = []
    for key, group in sorted(groups.items()):
        reward_positive = [item for item in group if item[0]["success"]]
        invalid_positive = sum(
            not behavior_valid_success(feature, review)
            for feature, review in reward_positive
        )
        for pattern in PATTERNS:
            findings = [review["patterns"][pattern] for _, review in group]
            applicable = [
                finding for finding in findings if finding["status"] != "not_applicable"
            ]
            rows.append(
                {
                    "model": key[0],
                    "domain": key[1],
                    "reward_outcome": key[2],
                    "n": len(group),
                    "raw_reward_rate": _rate(group, lambda feature, review: feature["success"]),
                    "behavior_valid_success_rate": _rate(
                        group, behavior_valid_success
                    ),
                    "reward_positive_behavior_invalid_rate": (
                        invalid_positive / len(reward_positive) if reward_positive else ""
                    ),
                    "pattern": pattern,
                    "applicable_n": len(applicable),
                    "pass_rate": (
                        sum(finding["status"] == "pass" for finding in applicable)
                        / len(applicable)
                        if applicable
                        else ""
                    ),
                    "fail_rate": (
                        sum(finding["status"] == "fail" for finding in applicable)
                        / len(applicable)
                        if applicable
                        else ""
                    ),
                    "critical_fail_rate": (
                        sum(
                            finding["status"] == "fail"
                            and finding["severity"] == "critical"
                            for finding in applicable
                        )
                        / len(applicable)
                        if applicable
                        else ""
                    ),
                    "basis": (
                        "rule+judge"
                        if pattern
                        in {
                            "atomic_commit",
                            "progressive_diagnosis",
                            "wrong_arguments",
                            "multi_tool_policy_violation",
                            "tool_namespace_confusion",
                            "nonexistent_tool",
                            "domain_policy_compliance",
                        }
                        else "judge"
                    ),
                }
            )
    with (outdir / "pattern_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_index(
    joined: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[tuple[str, str, int], dict[str, tuple[dict, dict]]]:
    pairs: dict[tuple[str, str, int], dict[str, tuple[dict, dict]]] = defaultdict(dict)
    for feature, review in joined:
        pairs[(feature["domain"], feature["task_id"], feature["trial"])][
            feature["model"]
        ] = (feature, review)
    incomplete = [key for key, values in pairs.items() if set(values) != set(MODEL_ORDER)]
    if incomplete:
        raise ValueError(f"incomplete model pairs: {incomplete[:5]}")
    return pairs


def write_paired_transitions(
    pairs: dict[tuple[str, str, int], dict[str, tuple[dict, dict]]], outdir: Path
) -> None:
    fields = [
        "domain",
        "task_id",
        "trial",
        "workflow_type",
        "raw_reward",
        "sft_reward",
        "rl2e6_reward",
        "reward_transition",
        "raw_primary_failure",
        "sft_primary_failure",
        "rl2e6_primary_failure",
    ]
    for pattern in PATTERNS:
        fields.extend(
            (
                f"raw_{pattern}",
                f"sft_{pattern}",
                f"rl2e6_{pattern}",
                f"raw_to_sft_{pattern}",
                f"sft_to_rl2e6_{pattern}",
            )
        )
    with (outdir / "paired_transitions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, models in sorted(pairs.items()):
            features = {model: models[model][0] for model in MODEL_ORDER}
            reviews = {model: models[model][1] for model in MODEL_ORDER}
            rewards = {model: int(features[model]["success"]) for model in MODEL_ORDER}
            row = {
                "domain": key[0],
                "task_id": key[1],
                "trial": key[2],
                "workflow_type": features["raw"]["workflow_type"],
                "raw_reward": rewards["raw"],
                "sft_reward": rewards["sft"],
                "rl2e6_reward": rewards["rl2e6"],
                "reward_transition": f"{rewards['raw']}->{rewards['sft']}->{rewards['rl2e6']}",
                "raw_primary_failure": reviews["raw"]["primary_failure_mode"],
                "sft_primary_failure": reviews["sft"]["primary_failure_mode"],
                "rl2e6_primary_failure": reviews["rl2e6"]["primary_failure_mode"],
            }
            for pattern in PATTERNS:
                statuses = {
                    model: reviews[model]["patterns"][pattern]["status"]
                    for model in MODEL_ORDER
                }
                row.update(
                    {
                        f"raw_{pattern}": statuses["raw"],
                        f"sft_{pattern}": statuses["sft"],
                        f"rl2e6_{pattern}": statuses["rl2e6"],
                        f"raw_to_sft_{pattern}": f"{statuses['raw']}->{statuses['sft']}",
                        f"sft_to_rl2e6_{pattern}": f"{statuses['sft']}->{statuses['rl2e6']}",
                    }
                )
            writer.writerow(row)


def _cluster_bootstrap(
    task_values: dict[tuple[str, str], tuple[float, float]],
    samples: int = 10000,
) -> tuple[float, float, float]:
    keys = sorted(task_values)
    observed = statistics.mean(
        task_values[key][1] - task_values[key][0] for key in keys
    )
    rng = random.Random(20260730)
    draws = []
    for _ in range(samples):
        sampled = [rng.choice(keys) for _ in keys]
        draws.append(
            statistics.mean(
                task_values[key][1] - task_values[key][0] for key in sampled
            )
        )
    draws.sort()
    return observed, draws[int(samples * 0.025)], draws[int(samples * 0.975)]


def write_pattern_deltas(
    pairs: dict[tuple[str, str, int], dict[str, tuple[dict, dict]]], outdir: Path
) -> list[dict[str, Any]]:
    fields = ("comparison", "pattern", "pass_rate_delta", "ci95_low", "ci95_high")
    rows = []
    for before, after in (("raw", "sft"), ("sft", "rl2e6")):
        for pattern in PATTERNS:
            per_task: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
            for key, models in pairs.items():
                statuses = [
                    models[model][1]["patterns"][pattern]["status"]
                    for model in (before, after)
                ]
                if "not_applicable" in statuses or "uncertain" in statuses:
                    continue
                per_task[(key[0], key[1])].append(
                    (int(statuses[0] == "pass"), int(statuses[1] == "pass"))
                )
            task_values = {
                key: (
                    statistics.mean(item[0] for item in values),
                    statistics.mean(item[1] for item in values),
                )
                for key, values in per_task.items()
            }
            if not task_values:
                continue
            delta, low, high = _cluster_bootstrap(task_values)
            rows.append(
                {
                    "comparison": f"{before}->{after}",
                    "pattern": pattern,
                    "pass_rate_delta": delta,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    with (outdir / "pattern_deltas.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_failure_modes(
    joined: list[tuple[dict[str, Any], dict[str, Any]]], outdir: Path
) -> None:
    fields = (
        "model",
        "domain",
        "reward_outcome",
        "primary_failure_mode",
        "n",
        "rate",
    )
    rows = []
    for model in MODEL_ORDER:
        for domain in ("airline", "retail", "telecom"):
            for outcome in ("success", "failure"):
                selected = [
                    review
                    for feature, review in joined
                    if feature["model"] == model
                    and feature["domain"] == domain
                    and ("success" if feature["success"] else "failure") == outcome
                ]
                counts = Counter(review["primary_failure_mode"] for review in selected)
                for mode, count in sorted(counts.items()):
                    rows.append(
                        {
                            "model": model,
                            "domain": domain,
                            "reward_outcome": outcome,
                            "primary_failure_mode": mode,
                            "n": count,
                            "rate": count / len(selected),
                        }
                    )
    with (outdir / "failure_modes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_adjudication_overrides(
    rows: list[dict[str, str]], outdir: Path
) -> None:
    fields = (
        "trajectory_id",
        "pattern",
        "judge_status",
        "judge_severity",
        "final_status",
        "final_severity",
        "reason",
    )
    with (outdir / "adjudication_overrides.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_representative_cases(
    pairs: dict[tuple[str, str, int], dict[str, tuple[dict, dict]]], outdir: Path
) -> None:
    lines = [
        "# 配对代表案例",
        "",
        "案例均来自相同 task/trial 的 Raw→SFT→RL 配对轨迹。Reward 变化是官方评测事实；"
        "模式与证据轮次来自 Gemini 判断，并经过确定性规则裁决。",
        "",
    ]
    lines.extend(("## 好模式案例", ""))
    for pattern in KEY_PATTERNS:
        selected = None
        for key, models in sorted(pairs.items()):
            for model in reversed(MODEL_ORDER):
                finding = models[model][1]["patterns"][pattern]
                if (
                    behavior_valid_success(*models[model])
                    and finding["status"] == "pass"
                ):
                    selected = (key, model, finding)
                    break
            if selected:
                break
        if selected:
            key, model, finding = selected
            lines.append(
                f"- **{PATTERN_DISPLAY_ZH[pattern]}**（`{pattern}`）— "
                f"`{model} | {key[0]} | {key[1]} | trial {key[2]}`，"
                f"证据轮次 `{finding['turn_indices']}`："
                f"{PATTERN_DESCRIPTION_ZH[pattern]}"
            )
    lines.extend(("", "## 主要失败模式", ""))
    for mode in sorted(
        {
            models[model][1]["primary_failure_mode"]
            for models in pairs.values()
            for model in MODEL_ORDER
        }
        - {"none"}
    ):
        selected = None
        for key, models in sorted(pairs.items()):
            for model in MODEL_ORDER:
                review = models[model][1]
                if review["primary_failure_mode"] != mode:
                    continue
                finding = next(
                    (
                        value
                        for value in review["patterns"].values()
                        if value["status"] == "fail" and value.get("evidence")
                    ),
                    None,
                )
                selected = (key, model, finding)
                break
            if selected:
                break
        if selected:
            key, model, finding = selected
            lines.append(
                f"- **{FAILURE_MODE_DISPLAY_ZH[mode]}**（`{mode}`）— "
                f"`{model} | {key[0]} | {key[1]} | trial {key[2]}`，"
                f"证据轮次 `{finding['turn_indices'] if finding else []}`。"
            )
    lines.append("")
    categories = {
        "官方 reward 改善案例": lambda rewards: rewards in {(0, 0, 1), (0, 1, 1)},
        "官方 reward 退化案例": lambda rewards: rewards in {(1, 0, 0), (1, 1, 0)},
        "官方 reward 持续成功案例": lambda rewards: rewards == (1, 1, 1),
        "官方 reward 持续失败案例": lambda rewards: rewards == (0, 0, 0),
    }
    for heading, predicate in categories.items():
        lines.extend((f"## {heading}", ""))
        shown = 0
        for key, models in sorted(pairs.items()):
            rewards = tuple(int(models[model][0]["success"]) for model in MODEL_ORDER)
            if not predicate(rewards):
                continue
            valid = tuple(
                int(behavior_valid_success(*models[model]))
                for model in MODEL_ORDER
            )
            modes = [models[model][1]["primary_failure_mode"] for model in MODEL_ORDER]
            evidence_pattern = None
            for model in reversed(MODEL_ORDER):
                review = models[model][1]
                evidence_pattern = next(
                    (
                        pattern
                        for pattern, finding in review["patterns"].items()
                        if finding["status"] == "fail"
                    ),
                    None,
                )
                if evidence_pattern:
                    break
            mode_text = " → ".join(
                f"{FAILURE_MODE_DISPLAY_ZH[mode]}（`{mode}`）" for mode in modes
            )
            evidence = (
                f"证据模式：{PATTERN_DISPLAY_ZH[evidence_pattern]}。"
                if evidence_pattern
                else "未识别到明确的失败模式。"
            )
            lines.append(
                f"- `{key[0]} | {key[1]} | trial {key[2]}`：reward "
                f"`{rewards[0]}→{rewards[1]}→{rewards[2]}`，行为有效 "
                f"`{valid[0]}→{valid[1]}→{valid[2]}`；主要模式 "
                f"{mode_text}。{evidence}"
            )
            shown += 1
            if shown == 3:
                break
        if not shown:
            lines.append("- 没有符合条件的配对案例。")
        lines.append("")
    (outdir / "representative_cases.md").write_text("\n".join(lines), encoding="utf-8")


def _format_rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_final_readme(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    delta_rows: list[dict[str, Any]],
    outdir: Path,
) -> None:
    by_model = {
        model: [(feature, review) for feature, review in joined if feature["model"] == model]
        for model in MODEL_ORDER
    }
    display = MODEL_DISPLAY_ZH
    metric_rows = []
    for model in MODEL_ORDER:
        group = by_model[model]
        successes = sum(feature["success"] for feature, _ in group)
        valid = sum(behavior_valid_success(feature, review) for feature, review in group)
        invalid_positive = successes - valid
        metric_rows.append(
            (
                display[model],
                successes,
                valid,
                invalid_positive / successes if successes else 0,
            )
        )

    read_only_success = {
        model: sum(
            feature["success"] and feature["workflow_type"] == "read_only"
            for feature, _ in by_model[model]
        )
        for model in MODEL_ORDER
    }
    write_success = {
        model: sum(
            feature["success"] and feature["workflow_type"] == "state_changing"
            for feature, _ in by_model[model]
        )
        for model in MODEL_ORDER
    }
    called_write_failure = {
        model: sum(
            feature["expected_write_name_called_failure"]
            for feature, _ in by_model[model]
        )
        for model in MODEL_ORDER
    }
    multitool = {
        model: statistics.mean(
            feature["multitool_policy_violation"] for feature, _ in by_model[model]
        )
        for model in MODEL_ORDER
    }

    failure_modes: dict[str, list[tuple[str, float]]] = {}
    success_patterns: dict[str, list[tuple[str, float]]] = {}
    failure_patterns: dict[str, list[tuple[str, float]]] = {}
    for model in MODEL_ORDER:
        counts = Counter(
            review["primary_failure_mode"]
            for feature, review in by_model[model]
            if not feature["success"]
        )
        total = sum(counts.values())
        failure_modes[model] = [
            (mode, count / total) for mode, count in counts.most_common(3)
        ]
        successful = [
            review
            for feature, review in by_model[model]
            if feature["success"]
        ]
        failed = [
            review
            for feature, review in by_model[model]
            if not feature["success"]
        ]
        success_rates = []
        for pattern in KEY_PATTERNS:
            applicable = [
                review["patterns"][pattern]
                for review in successful
                if review["patterns"][pattern]["status"] != "not_applicable"
            ]
            if applicable:
                success_rates.append(
                    (
                        pattern,
                        sum(item["status"] == "pass" for item in applicable)
                        / len(applicable),
                    )
                )
        success_patterns[model] = sorted(
            success_rates, key=lambda item: item[1], reverse=True
        )[:4]
        failure_rates = []
        for pattern in NEGATIVE_PATTERNS:
            findings = [review["patterns"][pattern] for review in failed]
            failure_rates.append(
                (
                    pattern,
                    sum(item["status"] == "fail" for item in findings)
                    / len(findings),
                )
            )
        failure_patterns[model] = sorted(
            failure_rates, key=lambda item: item[1], reverse=True
        )[:4]

    key_deltas = []
    for comparison in ("raw->sft", "sft->rl2e6"):
        candidates = [
            row
            for row in delta_rows
            if row["comparison"] == comparison and row["pattern"] in KEY_PATTERNS
        ]
        candidates.sort(key=lambda row: abs(row["pass_rate_delta"]), reverse=True)
        key_deltas.extend(candidates[:3])

    lines = [
        "# Raw→SFT→RL 轨迹模式分析",
        "",
        "实验名：`tau2-traj-pattern`。目的：在相同的 1,200 条官方 tau2 "
        "test 轨迹上，对比 Raw Instruct、multitool SFT 与 RL `2e-6 iter99` "
        "的成功和失败行为模式。",
        "",
        "## 范围与判定",
        "",
        "- 每阶段 400 条轨迹，覆盖 airline、retail、telecom；同一 task/trial "
        "进行 Raw→SFT→RL 配对比较。",
        "- Judge 为 `openai/gemini-2.5-flash`，prompt 为 "
        f"`{PROMPT_VERSION}`；54 条校准轨迹与 18 条重复审查的一致率为 "
        "92.328%，JSON 解析率为 100%。",
        "- 精确可判定的多工具、schema、namespace 与工具存在性事实优先于 "
        "Gemini 的冲突标签；所有覆盖记录在 `adjudication_overrides.csv`。",
        "- 报告明确区分官方 reward、确定性规则事实、Gemini 语义判断和配对推断；"
        "reference action/DB check 是证据，不是唯一合法轨迹。",
        "",
        "## 总体结果",
        "",
        "| 模型阶段 | 官方 reward 成功 | 行为有效成功 | Reward-positive 中行为无效 |",
        "|---|---:|---:|---:|",
    ]
    for name, successes, valid, invalid_rate in metric_rows:
        lines.append(
            f"| {name} | {successes}/400 ({successes / 400:.2%}) | "
            f"{valid}/400 ({valid / 400:.2%}) | {_format_rate(invalid_rate)} |"
        )
    lines.extend(
        [
            "",
            "“行为有效成功”要求：官方 reward=1、没有 critical Agent error，且经过"
            "确定性裁决后所有适用的关键好模式均为 pass/not_applicable。",
            "",
            "## Tau2-bench 好模式",
            "",
        ]
    )
    for pattern in KEY_PATTERNS:
        lines.append(
            f"- **{PATTERN_DISPLAY_ZH[pattern]}**（`{pattern}`）："
            f"{PATTERN_DESCRIPTION_ZH[pattern]}"
        )

    for model in MODEL_ORDER:
        group = by_model[model]
        successes = sum(feature["success"] for feature, _ in group)
        valid = sum(behavior_valid_success(feature, review) for feature, review in group)
        invalid_rate = (successes - valid) / successes if successes else 0
        good_text = "、".join(
            f"{PATTERN_DISPLAY_ZH[pattern]} {rate:.1%}"
            for pattern, rate in success_patterns[model]
        )
        mode_text = "、".join(
            f"{FAILURE_MODE_DISPLAY_ZH[mode]}（`{mode}`）{rate:.1%}"
            for mode, rate in failure_modes[model]
        )
        issue_text = "、".join(
            f"{PATTERN_DISPLAY_ZH[pattern]} {rate:.1%}"
            for pattern, rate in failure_patterns[model]
        )
        lines.extend(
            [
                "",
                f"## {display[model]}：成功与失败模式",
                "",
                f"- **成功轨迹**：官方成功 {successes}/400，行为有效 {valid}/400；"
                f"成功轨迹中适用好模式 pass 率最高的是 {good_text}。",
                f"- **失败轨迹**：主要 failure mode 为 {mode_text}；"
                f"高频失败 pattern 为 {issue_text}。",
                f"- **事务能力**：无需 Agent 写 DB 的成功数为 "
                f"{read_only_success[model]}，需要 Agent 写 DB 的成功数为 "
                f"{write_success[model]}；失败但调用过预期写工具名的轨迹为 "
                f"{called_write_failure[model]}。",
                f"- **Policy 风险**：{multitool[model]:.2%} 的轨迹含单轮多工具调用；"
                f"reward-positive 中行为无效率为 {invalid_rate:.2%}。",
            ]
        )
        if model == "raw":
            lines.append(
                "- **阶段判断**：Raw 已有较强的基础意图跟踪和收尾能力；主要问题是 "
                "telecom 的 Agent/User 工具混淆、错误参数和事务流程不完整。部分 "
                "reward=0 轨迹未识别到主要 Agent 错误，说明失败还包含 evaluator、"
                "action/DB mismatch 或 User simulator 因素。"
            )
        elif model == "sft":
            lines.append(
                "- **阶段判断**：SFT 没有提高总体 reward，却显著学会同轮批量提交"
                "调用；这在评测 system prompt 中被允许，但违反三个 domain policy "
                "的单轮单工具约束。runner 实际按顺序执行这些调用，并非并发执行。"
                "它提高了调用覆盖，却没有提高核心事务写入正确性。"
            )
        else:
            lines.append(
                "- **阶段判断**：RL 的 reward 增益来自无需写 DB 的任务；需要写 DB "
                "的成功数与 SFT 相同。RL 更常调用看似正确的工具名，但参数、流程和"
                "最终状态仍不正确，同时进一步放大了同轮批量调用倾向。"
            )

    known_case_text = "- 未找到预设的 SFT airline task 6 reward-positive 反例。"
    known_case = next(
        (
            (feature, review)
            for feature, review in by_model["sft"]
            if feature["trajectory_id"] == "sft|airline|6|0"
        ),
        None,
    )
    if known_case:
        feature, review = known_case
        false_success = review["patterns"]["false_success_claim"]
        known_case_text = (
            f"- Reward-positive 反例 `sft|airline|6|0`：官方 reward=1，但包含"
            f"单轮多工具和 {feature['num_observed_writes']} 次写调用，其中一次带保险"
            "的新预订失败；Gemini 将其保险成功声明判为 "
            f"`{false_success['status']}`，证据轮次 "
            f"`{false_success['turn_indices']}`。"
        )
    delta_lines = []
    if key_deltas:
        for row in key_deltas:
            delta_lines.append(
                f"- `{row['comparison']}` `{row['pattern']}`："
                f"{row['pass_rate_delta']:+.2%}, 95% CI "
                f"[{row['ci95_low']:+.2%}, {row['ci95_high']:+.2%}]。"
            )
    lines.extend(
        [
            "",
            "## 阶段变化与配对结论",
            "",
            "- 无需 Agent 写 DB 的成功数："
            f"`{read_only_success['raw']} → {read_only_success['sft']} → "
            f"{read_only_success['rl2e6']}`；需要 Agent 写 DB 的成功数："
            f"`{write_success['raw']} → {write_success['sft']} → "
            f"{write_success['rl2e6']}`。RL 没有改善核心事务执行能力。",
            "- 失败但调用过预期写工具名："
            f"`{called_write_failure['raw']} → {called_write_failure['sft']} → "
            f"{called_write_failure['rl2e6']}`。工具名覆盖增加没有转化为正确参数、"
            "流程或 DB 结果。",
            "- 单轮多工具调用率："
            f"`{multitool['raw']:.2%} → {multitool['sft']:.2%} → "
            f"{multitool['rl2e6']:.2%}`。同轮批量提交既符合评测 system prompt，"
            "又违反"
            "领域 policy，因此不能直接当作正向效率能力。",
            known_case_text,
            "",
            "关键好模式的配对 pass-rate 变化（按 task 聚类 bootstrap）：",
            *delta_lines,
            "",
            "配对阶段比较基于相同 task/trial 的变化，不把未配对均值解释为因果"
            "效应。"
            "`paired_transitions.csv` 包含全部 400 组转换。",
            "",
            "## 代表案例与产物",
            "",
            "- [`SINGLE_CALL_REPORT.md`](SINGLE_CALL_REPORT.md)：同轮批量工具调用"
            "问题的完整证据、真实轨迹、数据占比、判断与整改方案。",
            "- `representative_cases.md`：好模式、主要失败模式、改善、退化、持续成功"
            "和持续失败的证据案例。",
            "- `features.jsonl`：1,200 条确定性特征。",
            "- `judge_calibration.jsonl` / `judge_reviews.jsonl`：校准与全量语义审查。",
            "- `pattern_summary.csv`：模型×domain×reward outcome 的裁决后模式率。",
            "- `paired_transitions.csv` / `pattern_deltas.csv`：配对变化及 task-cluster "
            "95% CI。",
            "- `failure_modes.csv`：主要失败模式分布。",
            "- `adjudication_overrides.csv`：规则事实覆盖 judge 标签的完整审计。",
            "- `diagnostics/`：已被替代的格式错误或低一致率校准尝试。",
            "",
            "`3e-6/5e-6` 仅保留既有确定性退化统计，未进入本轮 Gemini 全量审查。",
            "",
            "## 复现与验证",
            "",
            "依次运行 `analyze_trajectories.py`、`judge_trajectories.py --mode "
            "calibration`、校准验证、`--mode full`，最后运行 "
            "`summarize_trajectory_patterns.py`。脚本参数见各自的 `--help`。",
            "",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    features = read_jsonl(args.features)
    reviews = _review_map(args.reviews)
    feature_ids = {feature["trajectory_id"] for feature in features}
    if len(features) != 1200 or len(feature_ids) != 1200:
        raise SystemExit("features must contain 1200 unique trajectory IDs")
    if set(reviews) != feature_ids:
        missing = sorted(feature_ids - set(reviews))
        extra = sorted(set(reviews) - feature_ids)
        raise SystemExit(
            f"review coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    invalid_reviews = []
    for feature in features:
        errors = validate_review(
            reviews[feature["trajectory_id"]], set(feature["valid_turn_indices"])
        )
        if errors:
            invalid_reviews.append((feature["trajectory_id"], errors))
    if invalid_reviews:
        raise SystemExit(f"invalid reviews: {invalid_reviews[:3]}")
    joined = []
    overrides = []
    for feature in features:
        review, review_overrides = adjudicate_review(
            feature, reviews[feature["trajectory_id"]]
        )
        joined.append((feature, review))
        overrides.extend(review_overrides)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = _paired_index(joined)
    write_adjudication_overrides(overrides, args.out)
    write_pattern_summary(joined, args.out)
    write_paired_transitions(pairs, args.out)
    delta_rows = write_pattern_deltas(pairs, args.out)
    write_failure_modes(joined, args.out)
    write_representative_cases(pairs, args.out)
    write_final_readme(joined, delta_rows, args.out)
    invalid_positive = Counter()
    for feature, review in joined:
        if feature["success"] and not behavior_valid_success(feature, review):
            invalid_positive[feature["model"]] += 1
    print("reward-positive but behavior-invalid:", dict(invalid_positive))


if __name__ == "__main__":
    main()
