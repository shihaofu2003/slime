#!/usr/bin/env python3
"""Summarize autonomous Qwen3.8 synthetic Banking evaluations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import MODEL_PATH  # type: ignore
    from banking_synthetic.documents import read_json, read_jsonl, write_json  # type: ignore
    from banking_synthetic.protocol import protocol_failure_reason  # type: ignore
else:
    from .constants import MODEL_PATH
    from .documents import read_json, read_jsonl, write_json
    from .protocol import protocol_failure_reason


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _message_tool_stats(messages: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str]]:
    totals: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    call_requestor: dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            requestor = call.get("requestor") or message.get("role")
            call_requestor[call.get("id", "")] = requestor
        if message.get("role") != "tool":
            continue
        requestor = message.get("requestor") or call_requestor.get(message.get("id", ""), "assistant")
        label = "agent" if requestor == "assistant" else "user"
        totals[label] += 1
        content = str(message.get("content") or "")
        if not message.get("error") and not content.lstrip().startswith(("Error:", "Failed to ")):
            successes[label] += 1
    return totals, successes


def _component(reward_info: dict[str, Any], name: str) -> float | None:
    breakdown = reward_info.get("reward_breakdown") or {}
    value = breakdown.get(name)
    return float(value) if value is not None else None


def _compact_row(
    simulation: dict[str, Any],
    contract: dict[str, Any],
    attempt_kind: str,
) -> dict[str, Any]:
    messages = simulation.get("messages") or []
    prompt_tokens = sum(
        int(((message.get("usage") or {}).get("prompt_tokens") or 0))
        for message in messages
    )
    completion_tokens = sum(
        int(((message.get("usage") or {}).get("completion_tokens") or 0))
        for message in messages
    )
    failure_reason = protocol_failure_reason(simulation)
    component_values = {}
    tool_totals: Counter[str] = Counter()
    tool_successes: Counter[str] = Counter()
    bm25_queries = 0
    bm25_hits = 0
    if failure_reason is None:
        reward_info = simulation.get("reward_info") or {}
        for component in ("DB", "ACTION", "COMMUNICATE"):
            value = _component(reward_info, component)
            if value is not None:
                component_values[component] = value
        tool_totals, tool_successes = _message_tool_stats(messages)
        required = set(contract.get("required_documents") or [])
        for index, message in enumerate(messages):
            for call in message.get("tool_calls") or []:
                if call.get("name") != "KB_search":
                    continue
                bm25_queries += 1
                result = messages[index + 1] if index + 1 < len(messages) else {}
                ranked = re.findall(
                    r"\bID:\s+([^\s]+)", str(result.get("content") or "")
                )
                bm25_hits += int(bool(required & set(ranked[:10])))
    return {
        "task_id": simulation["task_id"],
        "attempt_kind": attempt_kind,
        "breakdowns": {
            field: contract.get(field)
            for field in ("business_category", "difficulty", "template_id")
        },
        "termination_reason": str(simulation.get("termination_reason")),
        "turns": sum(
            message.get("role") in {"assistant", "user"} for message in messages
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "protocol_failure_reason": failure_reason,
        "success": float(
            (simulation.get("reward_info") or {}).get("reward") or 0.0
        )
        == 1.0,
        "component_values": component_values,
        "tool_totals": dict(tool_totals),
        "tool_successes": dict(tool_successes),
        "bm25_queries": bm25_queries,
        "bm25_hits": bm25_hits,
    }


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_successes: dict[str, list[bool]] = defaultdict(list)
    observed_task_ids: set[str] = set()
    component_values: dict[str, list[float]] = defaultdict(list)
    terminations: Counter[str] = Counter()
    tool_totals: Counter[str] = Counter()
    tool_successes: Counter[str] = Counter()
    turns = []
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    bm25_queries = 0
    bm25_hits = 0
    protocol_failures = 0
    protocol_failure_reasons: Counter[str] = Counter()
    for source_row in rows:
        row = (
            _compact_row(
                source_row["simulation"],
                source_row["contract"],
                source_row.get("attempt_kind", "first"),
            )
            if "simulation" in source_row
            else source_row
        )
        observed_task_ids.add(row["task_id"])
        terminations[row["termination_reason"]] += 1
        turns.append(row["turns"])
        prompt_tokens.append(row["prompt_tokens"])
        completion_tokens.append(row["completion_tokens"])
        total_tokens.append(row["prompt_tokens"] + row["completion_tokens"])
        failure_reason = row["protocol_failure_reason"]
        if failure_reason is not None:
            protocol_failures += 1
            protocol_failure_reasons[failure_reason] += 1
            continue

        task_successes[row["task_id"]].append(row["success"])
        for component, value in row["component_values"].items():
            component_values[component].append(value)
        tool_totals.update(row["tool_totals"])
        tool_successes.update(row["tool_successes"])
        bm25_queries += row["bm25_queries"]
        bm25_hits += row["bm25_hits"]
    successes = sum(sum(values) for values in task_successes.values())
    valid_simulations = sum(len(values) for values in task_successes.values())
    attempts = len(rows)
    four_trial = [values for values in task_successes.values() if len(values) == 4]
    return {
        "tasks": len(observed_task_ids),
        "valid_tasks": len(task_successes),
        "simulations": attempts,
        "valid_simulations": valid_simulations,
        "pass_at_1": _ratio(successes, valid_simulations),
        "pass_at_4_any": _ratio(sum(any(values) for values in four_trial), len(four_trial)),
        "pass_pow_4": _ratio(sum(all(values) for values in four_trial), len(four_trial)),
        "tasks_with_four_trials": len(four_trial),
        "component_success": {
            name: _ratio(sum(values), len(values)) for name, values in component_values.items()
        },
        "bm25_top10_hit_rate": _ratio(bm25_hits, bm25_queries),
        "bm25_queries": bm25_queries,
        "agent_action_success_rate": _ratio(tool_successes["agent"], tool_totals["agent"]),
        "user_action_success_rate": _ratio(tool_successes["user"], tool_totals["user"]),
        "termination_reasons": dict(terminations),
        "mean_turns": mean(turns) if turns else None,
        "mean_prompt_tokens": mean(prompt_tokens) if prompt_tokens else None,
        "mean_completion_tokens": (
            mean(completion_tokens) if completion_tokens else None
        ),
        "mean_total_tokens": mean(total_tokens) if total_tokens else None,
        "protocol_failure_rate": _ratio(protocol_failures, attempts),
        "first_protocol_failure_rate": _ratio(protocol_failures, attempts),
        "protocol_failure_count": protocol_failures,
        "protocol_failure_reasons": dict(protocol_failure_reasons),
    }


def summarize(
    result_files: list[Path], contracts: list[dict[str, Any]], model_manifests: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(result_files) != len(model_manifests):
        raise ValueError("each result file needs its own runtime model manifest")
    contract_by_id = {contract["task_id"]: contract for contract in contracts}
    rows = []
    for result_file, manifest in zip(result_files, model_manifests):
        payload = read_json(result_file)
        info = payload.get("info") or {}
        agent_llm = str((info.get("agent_info") or {}).get("llm") or "")
        user_llm = str((info.get("user_info") or {}).get("llm") or "")
        if not agent_llm.endswith(str(manifest.get("agent_served_model") or "missing")):
            raise ValueError(f"result Agent model differs from manifest: {agent_llm}")
        if not user_llm.endswith(str(manifest.get("user_served_model") or "missing")):
            raise ValueError(f"result User model differs from manifest: {user_llm}")
        for simulation in payload.get("simulations") or []:
            task_id = simulation["task_id"]
            if task_id not in contract_by_id:
                continue
            rows.append(
                _compact_row(
                    simulation,
                    contract_by_id[task_id],
                    manifest.get("attempt_kind", "first"),
                )
            )
    for manifest in model_manifests:
        if manifest.get("agent_model_path") != MODEL_PATH or manifest.get("user_model_path") != MODEL_PATH:
            raise ValueError("Agent and User must both use /mnt/afs/models/Qwen3.8-27B")

    first_rows = [row for row in rows if row["attempt_kind"] == "first"]
    recovery_rows = [row for row in rows if row["attempt_kind"] == "recovery"]
    breakdowns = {}
    for field in ("business_category", "difficulty", "template_id"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in first_rows:
            grouped[str(row["breakdowns"][field])].append(row)
        breakdowns[field] = {key: _group_metrics(value) for key, value in sorted(grouped.items())}
    return {
        "model_path_verified": True,
        "result_file_count": len(result_files),
        "overall": _group_metrics(first_rows),
        "recovery": _group_metrics(recovery_rows) if recovery_rows else None,
        "all_attempts": _group_metrics(rows),
        "breakdowns": breakdowns,
    }


def _markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    percent = lambda value: "n/a" if value is None else f"{100 * value:.2f}%"
    lines = [
        "# Banking synthetic evaluation metrics",
        "",
        f"- Tasks/attempts/valid simulations: {overall['tasks']}/{overall['simulations']}/{overall['valid_simulations']}",
        f"- pass@1: {percent(overall['pass_at_1'])}",
        f"- pass@4(any): {percent(overall['pass_at_4_any'])}",
        f"- pass^4: {percent(overall['pass_pow_4'])}",
        f"- BM25 top-10 hit rate: {percent(overall['bm25_top10_hit_rate'])}",
        f"- Agent/User action success: {percent(overall['agent_action_success_rate'])} / {percent(overall['user_action_success_rate'])}",
        f"- First protocol failure rate: {percent(overall['first_protocol_failure_rate'])}",
        f"- Terminations: `{json.dumps(overall['termination_reasons'], sort_keys=True)}`",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = summarize(
        args.results,
        read_jsonl(args.contracts),
        [read_json(path) for path in args.model_manifest],
    )
    write_json(args.output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["overall"], sort_keys=True))


if __name__ == "__main__":
    main()
