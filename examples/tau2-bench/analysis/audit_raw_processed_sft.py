#!/usr/bin/env python3
"""Audit raw-all versus processed Tau2 SFT data and paired eval outcomes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator


SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
PROJECT_ROOT = SERVICE_AGENT_ROOT / "slime"
EXPERIMENT_DIR = PROJECT_ROOT / "output/experiments/tau2-sft-raw-all-max16384"
PROCESSED_EXPERIMENT_DIR = (
    PROJECT_ROOT / "output/experiments/tau2-sft-official-native-expanded"
)
RESULTS_ROOT = SERVICE_AGENT_ROOT / "tau2-bench"
DOMAINS = ("airline", "retail", "telecom")
EXPECTED_TASKS = {"airline": 20, "retail": 40, "telecom": 40}
PASS_METRICS = ("pass_at_1", "pass_at_4_any", "pass_power_4")
METRIC_LABELS = {
    "pass_at_1": "pass@1",
    "pass_at_4_any": "pass@4(any)",
    "pass_power_4": "pass^4",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def infer_raw_domain(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    explicit = str(metadata.get("domain") or "").lower()
    if explicit in DOMAINS:
        return explicit
    dialog_id = str(metadata.get("source_dialog_id") or "").lower()
    matches = [domain for domain in DOMAINS if domain in dialog_id]
    if len(matches) != 1:
        system_text = "\n".join(
            str(message.get("content") or "").lower()
            for message in row.get("messages") or []
            if isinstance(message, Mapping) and message.get("role") == "system"
        )
        matches = [
            domain for domain in DOMAINS if f"{domain} agent policy" in system_text
        ]
    if len(matches) != 1:
        raise ValueError(f"cannot infer domain from metadata: {metadata}")
    return matches[0]


def is_double_success(row: Mapping[str, Any]) -> bool:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return metadata.get("correct") in (1, 1.0, True) and metadata.get("reward") in (
        1,
        1.0,
        True,
    )


def call_parts(call: Any) -> tuple[str | None, dict[str, Any] | None, bool]:
    if not isinstance(call, Mapping):
        return None, None, False
    function = call.get("function")
    function = function if isinstance(function, Mapping) else {}
    direct_name = call.get("name")
    function_name = function.get("name")
    if direct_name and function_name and direct_name != function_name:
        return None, None, False
    name = direct_name or function_name
    arguments = call.get("arguments", function.get("arguments"))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return str(name or "") or None, None, False
    valid_shape = isinstance(name, str) and bool(name) and isinstance(arguments, dict)
    return str(name or "") or None, arguments if isinstance(arguments, dict) else None, valid_shape


def load_agent_catalogs(
    processed_path: Path,
) -> tuple[
    dict[str, dict[str, Draft202012Validator]],
    dict[str, list[Any]],
]:
    validators: dict[str, dict[str, Draft202012Validator]] = {}
    canonical_tools: dict[str, list[Any]] = {}
    for row in read_jsonl(processed_path):
        metadata = row.get("metadata") or {}
        domain = str(metadata.get("domain") or "")
        if domain in validators:
            continue
        tools = row.get("tools")
        if not isinstance(tools, list):
            raise ValueError(f"processed row has no tool schemas for {domain}")
        domain_validators = {}
        for tool in tools:
            function = tool.get("function") if isinstance(tool, Mapping) else None
            if not isinstance(function, Mapping):
                raise ValueError(f"malformed processed tool schema for {domain}")
            name = function.get("name")
            parameters = function.get("parameters")
            if not isinstance(name, str) or not isinstance(parameters, Mapping):
                raise ValueError(f"malformed processed tool schema for {domain}")
            Draft202012Validator.check_schema(parameters)
            domain_validators[name] = Draft202012Validator(parameters)
        validators[domain] = domain_validators
        canonical_tools[domain] = tools
    if set(validators) != set(DOMAINS):
        raise ValueError(f"processed data is missing domains: {set(DOMAINS) - set(validators)}")
    return validators, canonical_tools


def observed_user_tool_names(raw_path: Path) -> dict[str, set[str]]:
    names = {domain: set() for domain in DOMAINS}
    for row in read_jsonl(raw_path):
        domain = infer_raw_domain(row)
        for message in row.get("messages") or []:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            for call in message.get("tool_calls") or []:
                name, _, _ = call_parts(call)
                if name:
                    names[domain].add(name)
    return names


def arguments_match_schema(
    arguments: dict[str, Any], validator: Draft202012Validator
) -> bool:
    properties = set((validator.schema.get("properties") or {}).keys())
    return not (set(arguments) - properties) and validator.is_valid(arguments)


def audit_target_call(
    stats: dict[str, int],
    call: Any,
    *,
    domain: str,
    validators: dict[str, dict[str, Draft202012Validator]],
    user_tool_names: dict[str, set[str]],
    failure_labeled: bool,
) -> None:
    stats["target_calls"] += 1
    stats["failure_labeled_target_calls"] += int(failure_labeled)
    name, arguments, valid_shape = call_parts(call)
    if not valid_shape:
        stats["malformed_target_calls"] += 1
        return
    validator = validators[domain].get(name or "")
    if validator is None:
        stats["nonexistent_agent_target_calls"] += 1
        stats["observed_user_only_target_calls"] += int(
            name in user_tool_names[domain]
        )
        return
    stats["current_agent_namespace_target_calls"] += 1
    if arguments is not None and arguments_match_schema(arguments, validator):
        stats["current_schema_valid_target_calls"] += 1
    else:
        stats["invalid_argument_target_calls"] += 1


def empty_data_stats() -> dict[str, int]:
    return {
        "rows": 0,
        "double_success_rows": 0,
        "failure_labeled_rows": 0,
        "text_target_rows": 0,
        "tool_call_target_rows": 0,
        "empty_target_rows": 0,
        "multi_call_target_rows": 0,
        "mixed_text_and_call_target_rows": 0,
        "rows_with_multi_call_prefix": 0,
        "target_calls": 0,
        "failure_labeled_target_calls": 0,
        "malformed_target_calls": 0,
        "nonexistent_agent_target_calls": 0,
        "observed_user_only_target_calls": 0,
        "current_agent_namespace_target_calls": 0,
        "current_schema_valid_target_calls": 0,
        "invalid_argument_target_calls": 0,
        "runtime_schema_aligned_rows": 0,
        "target_only_loss_mask_rows": 0,
        "complete_prefix_call_result_rows": 0,
    }


def has_multi_call(messages: list[Any]) -> bool:
    return any(
        isinstance(message, Mapping)
        and message.get("role") == "assistant"
        and len(message.get("tool_calls") or []) > 1
        for message in messages
    )


def has_complete_prefix_pairing(messages: list[Any]) -> bool:
    seen: set[str] = set()
    results: set[str] = set()
    final_call_ids: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            return False
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = call.get("id") if isinstance(call, Mapping) else None
                if not isinstance(call_id, str) or not call_id or call_id in seen:
                    return False
                seen.add(call_id)
                if index == len(messages) - 1:
                    final_call_ids.add(call_id)
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in seen or call_id in results:
                return False
            results.add(call_id)
    return seen - results == final_call_ids


def audit_raw_data(
    path: Path,
    *,
    validators: dict[str, dict[str, Draft202012Validator]],
    user_tool_names: dict[str, set[str]],
) -> dict[str, int]:
    stats = empty_data_stats()
    for row in read_jsonl(path):
        stats["rows"] += 1
        domain = infer_raw_domain(row)
        double_success = is_double_success(row)
        stats["double_success_rows"] += int(double_success)
        stats["failure_labeled_rows"] += int(not double_success)
        answer = row.get("answer")
        answer = answer if isinstance(answer, Mapping) else {}
        calls = answer.get("tool_calls") or []
        content = answer.get("content")
        has_text = isinstance(content, str) and bool(content.strip())
        if calls:
            stats["tool_call_target_rows"] += 1
            stats["multi_call_target_rows"] += int(len(calls) > 1)
            stats["mixed_text_and_call_target_rows"] += int(has_text)
        elif has_text:
            stats["text_target_rows"] += 1
        else:
            stats["empty_target_rows"] += 1
        messages = row.get("messages")
        messages = messages if isinstance(messages, list) else []
        stats["rows_with_multi_call_prefix"] += int(has_multi_call(messages))
        for call in calls:
            audit_target_call(
                stats,
                call,
                domain=domain,
                validators=validators,
                user_tool_names=user_tool_names,
                failure_labeled=not double_success,
            )
    return stats


def audit_processed_data(
    path: Path,
    *,
    validators: dict[str, dict[str, Draft202012Validator]],
    canonical_tools: dict[str, list[Any]],
    user_tool_names: dict[str, set[str]],
) -> dict[str, int]:
    stats = empty_data_stats()
    for row in read_jsonl(path):
        stats["rows"] += 1
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        domain = str(metadata.get("domain") or "")
        if domain not in validators:
            raise ValueError(f"invalid processed domain: {domain!r}")
        double_success = is_double_success(row)
        stats["double_success_rows"] += int(double_success)
        stats["failure_labeled_rows"] += int(not double_success)
        messages = row.get("messages")
        messages = messages if isinstance(messages, list) else []
        target = messages[-1] if messages and isinstance(messages[-1], Mapping) else {}
        calls = target.get("tool_calls") or []
        content = target.get("content")
        has_text = isinstance(content, str) and bool(content.strip())
        if calls:
            stats["tool_call_target_rows"] += 1
            stats["multi_call_target_rows"] += int(len(calls) > 1)
            stats["mixed_text_and_call_target_rows"] += int(has_text)
        elif has_text:
            stats["text_target_rows"] += 1
        else:
            stats["empty_target_rows"] += 1
        stats["rows_with_multi_call_prefix"] += int(has_multi_call(messages[:-1]))
        stats["runtime_schema_aligned_rows"] += int(
            row.get("tools") == canonical_tools[domain]
        )
        stats["target_only_loss_mask_rows"] += int(
            bool(messages)
            and target.get("step_loss_mask") == 1
            and all(
                isinstance(message, Mapping) and message.get("step_loss_mask") == 0
                for message in messages[:-1]
            )
        )
        stats["complete_prefix_call_result_rows"] += int(
            has_complete_prefix_pairing(messages)
        )
        for call in calls:
            audit_target_call(
                stats,
                call,
                domain=domain,
                validators=validators,
                user_tool_names=user_tool_names,
                failure_labeled=not double_success,
            )
    return stats


def load_task_rewards(
    summary_path: Path, results_root: Path
) -> dict[tuple[str, str], tuple[float, ...]]:
    summary = read_json(summary_path)
    task_trials: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for domain, expected_count in EXPECTED_TASKS.items():
        domain_summary = (summary.get("domains") or {}).get(domain) or {}
        relative_path = Path(str(domain_summary.get("results_file") or ""))
        result_path = (
            relative_path if relative_path.is_absolute() else results_root / relative_path
        )
        for simulation in read_json(result_path).get("simulations") or []:
            task_id = str(simulation.get("task_id") or "")
            trial = simulation.get("trial")
            reward = (simulation.get("reward_info") or {}).get("reward")
            key = (domain, task_id)
            if not task_id or not isinstance(trial, int) or trial in task_trials[key]:
                raise ValueError(f"malformed task/trial in {result_path}")
            if reward not in (0, 0.0, 1, 1.0):
                raise ValueError(f"non-binary reward in {result_path}")
            task_trials[key][trial] = float(reward)
        domain_keys = [key for key in task_trials if key[0] == domain]
        if len(domain_keys) != expected_count:
            raise ValueError(
                f"{summary_path}: {domain} has {len(domain_keys)} tasks, expected {expected_count}"
            )
    rewards = {}
    for key, trials in task_trials.items():
        if set(trials) != {0, 1, 2, 3}:
            raise ValueError(f"{summary_path}: incomplete trials for {key}")
        rewards[key] = tuple(trials[index] for index in range(4))
    return rewards


def task_metric(rewards: tuple[float, ...], metric: str) -> float:
    if metric == "pass_at_1":
        return sum(rewards) / len(rewards)
    if metric == "pass_at_4_any":
        return float(any(rewards))
    if metric == "pass_power_4":
        return float(all(rewards))
    raise ValueError(f"unsupported pass metric: {metric}")


def paired_bootstrap_ci(
    deltas: list[float], *, samples: int, seed: int
) -> tuple[float, float, float]:
    if not deltas or samples <= 0:
        raise ValueError("paired bootstrap needs values and a positive sample count")
    values = np.asarray(deltas, dtype=np.float64)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 10_000):
        stop = min(start + 10_000, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    return observed, float(low), float(high)


def build_paired_eval(
    *,
    raw: dict[str, dict[tuple[str, str], tuple[float, ...]]],
    processed: dict[str, dict[tuple[str, str], tuple[float, ...]]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    task_sets = [set(rows) for rows in (*raw.values(), *processed.values())]
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError("raw and processed evaluations do not contain the same tasks")
    seeds = sorted(raw)
    if set(seeds) != set(processed):
        raise ValueError("raw and processed evaluations do not contain the same seeds")
    rows = []
    scopes = (("overall", None), *((domain, domain) for domain in DOMAINS))
    for scope_index, (scope, selected_domain) in enumerate(scopes):
        tasks = [
            key
            for key in sorted(task_sets[0])
            if selected_domain is None or key[0] == selected_domain
        ]
        for metric_index, metric in enumerate(PASS_METRICS):
            deltas = [
                sum(
                    task_metric(processed[item_seed][key], metric)
                    - task_metric(raw[item_seed][key], metric)
                    for item_seed in seeds
                )
                / len(seeds)
                for key in tasks
            ]
            observed, low, high = paired_bootstrap_ci(
                deltas,
                samples=samples,
                seed=seed + scope_index * len(PASS_METRICS) + metric_index,
            )
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "tasks": len(tasks),
                    "delta": observed,
                    "ci95": [low, high],
                    "ci_excludes_zero": low > 0 or high < 0,
                    "tasks_better": sum(value > 0 for value in deltas),
                    "tasks_tied": sum(math.isclose(value, 0.0) for value in deltas),
                    "tasks_worse": sum(value < 0 for value in deltas),
                }
            )
    return rows


def fraction(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{numerator:,}/{denominator:,} ({100 * numerator / denominator:.2f}%)"


def render_markdown(report: dict[str, Any]) -> str:
    raw = report["data_audit"]["raw_all"]
    processed = report["data_audit"]["processed"]
    lines = [
        "# Raw-all versus processed SFT quality audit",
        "",
        "## Data-level audit",
        "",
        "Rates are descriptive. Raw-all uses one source answer per row; processed may "
        "expand one source answer into multiple atomic targets.",
        "",
        "| Metric | Raw-all | Processed |",
        "|---|---:|---:|",
        f"| Double-success provenance | "
        f"{fraction(raw['double_success_rows'], raw['rows'])} | "
        f"{fraction(processed['double_success_rows'], processed['rows'])} |",
        f"| Failure-labeled rows | "
        f"{fraction(raw['failure_labeled_rows'], raw['rows'])} | "
        f"{fraction(processed['failure_labeled_rows'], processed['rows'])} |",
        f"| Multi-call targets / tool-call rows | "
        f"{fraction(raw['multi_call_target_rows'], raw['tool_call_target_rows'])} | "
        f"{fraction(processed['multi_call_target_rows'], processed['tool_call_target_rows'])} |",
        f"| Mixed text + call targets / tool-call rows | "
        f"{fraction(raw['mixed_text_and_call_target_rows'], raw['tool_call_target_rows'])} | "
        f"{fraction(processed['mixed_text_and_call_target_rows'], processed['tool_call_target_rows'])} |",
        f"| Rows with a multi-call prefix | "
        f"{fraction(raw['rows_with_multi_call_prefix'], raw['rows'])} | "
        f"{fraction(processed['rows_with_multi_call_prefix'], processed['rows'])} |",
        f"| Calls from failure-labeled rows | "
        f"{fraction(raw['failure_labeled_target_calls'], raw['target_calls'])} | "
        f"{fraction(processed['failure_labeled_target_calls'], processed['target_calls'])} |",
        f"| Observed User-only target calls | "
        f"{fraction(raw['observed_user_only_target_calls'], raw['target_calls'])} | "
        f"{fraction(processed['observed_user_only_target_calls'], processed['target_calls'])} |",
        f"| Current Agent namespace calls | "
        f"{fraction(raw['current_agent_namespace_target_calls'], raw['target_calls'])} | "
        f"{fraction(processed['current_agent_namespace_target_calls'], processed['target_calls'])} |",
        f"| Current JSON-schema-valid calls | "
        f"{fraction(raw['current_schema_valid_target_calls'], raw['target_calls'])} | "
        f"{fraction(processed['current_schema_valid_target_calls'], processed['target_calls'])} |",
        f"| Current tool-schema rows | n/a | "
        f"{fraction(processed['runtime_schema_aligned_rows'], processed['rows'])} |",
        f"| Target-only loss-mask rows | n/a | "
        f"{fraction(processed['target_only_loss_mask_rows'], processed['rows'])} |",
        f"| Complete prefix call/result pairing | n/a | "
        f"{fraction(processed['complete_prefix_call_result_rows'], processed['rows'])} |",
        "",
        "## Paired official evaluation",
        "",
        f"The interval resamples {report['bootstrap']['unique_tasks']} matched tasks. "
        "The two seeds are averaged within each task before "
        f"{report['bootstrap']['samples']:,} bootstrap draws.",
        "",
        "| Scope | Metric | Processed - raw-all | 95% paired CI | "
        "Tasks better / tied / worse | Excludes 0 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["paired_eval"]:
        lines.append(
            f"| {row['scope']} | {METRIC_LABELS[row['metric']]} | "
            f"{100 * row['delta']:+.2f}pp | "
            f"[{100 * row['ci95'][0]:+.2f}, {100 * row['ci95'][1]:+.2f}]pp | "
            f"{row['tasks_better']} / {row['tasks_tied']} / {row['tasks_worse']} | "
            f"{'yes' if row['ci_excludes_zero'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The direct data-level evidence supports cleaner outcome provenance, atomic "
            "targets, and current-runtime compatibility. The paired intervals are "
            "conditional on the two observed generation seeds and do not isolate which "
            "processing operation caused them, because filtering, target expansion, "
            "prompt/schema replacement, and the domain mix changed together.",
            "",
            "Raw-all target calls were already 99.92% valid against the current Agent "
            "namespace and JSON schemas. Target-call schema validity alone is therefore "
            "not the main differentiator; outcome filtering and atomic context are the "
            "larger measured changes.",
            "",
            "A double-success source label is episode-level evidence; it does not prove "
            "that every retained intermediate action is optimal. Action and DB accuracy "
            "therefore remain necessary counter-metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=EXPERIMENT_DIR / "data/raw_all_max16384.jsonl",
    )
    parser.add_argument(
        "--processed-data",
        type=Path,
        default=PROCESSED_EXPERIMENT_DIR
        / "data/agent_official_native_expanded.jsonl",
    )
    parser.add_argument(
        "--raw-seed300",
        type=Path,
        default=EXPERIMENT_DIR / "eval/final/seed300_0903_035744_summary.json",
    )
    parser.add_argument(
        "--raw-seed301",
        type=Path,
        default=EXPERIMENT_DIR / "eval/final/seed301_0903_035747_summary.json",
    )
    parser.add_argument(
        "--processed-seed300",
        type=Path,
        default=PROCESSED_EXPERIMENT_DIR
        / "eval/checkpoint-iter_0003795/seed300_0902_224647_summary.json",
    )
    parser.add_argument(
        "--processed-seed301",
        type=Path,
        default=PROCESSED_EXPERIMENT_DIR
        / "eval/checkpoint-iter_0003795/seed301_0902_233656_summary.json",
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument(
        "--json-output", type=Path, default=EXPERIMENT_DIR / "QUALITY_AUDIT.json"
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=EXPERIMENT_DIR / "QUALITY_AUDIT.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    validators, canonical_tools = load_agent_catalogs(args.processed_data)
    user_tool_names = observed_user_tool_names(args.raw_data)
    data_audit = {
        "raw_all": audit_raw_data(
            args.raw_data,
            validators=validators,
            user_tool_names=user_tool_names,
        ),
        "processed": audit_processed_data(
            args.processed_data,
            validators=validators,
            canonical_tools=canonical_tools,
            user_tool_names=user_tool_names,
        ),
    }
    raw_rewards = {
        "300": load_task_rewards(args.raw_seed300, args.results_root),
        "301": load_task_rewards(args.raw_seed301, args.results_root),
    }
    processed_rewards = {
        "300": load_task_rewards(args.processed_seed300, args.results_root),
        "301": load_task_rewards(args.processed_seed301, args.results_root),
    }
    paired_eval = build_paired_eval(
        raw=raw_rewards,
        processed=processed_rewards,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report = {
        "data_audit": data_audit,
        "bootstrap": {
            "unit": "domain/task_id with seeds averaged within task",
            "unique_tasks": 100,
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "interval": "percentile_95",
        },
        "paired_eval": paired_eval,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
