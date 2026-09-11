#!/usr/bin/env python3
"""Contrastive atomic-capability analysis for completed tau2 evaluations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVICE_AGENT_ROOT = PROJECT_ROOT.parent
DEFAULT_ANALYSIS_DIR = (
    PROJECT_ROOT / "output/experiments/tau2-qwen3-qwen35-atomic-gap-analysis"
)
DEFAULT_BASE_A_SUMMARY = (
    PROJECT_ROOT
    / "output/experiments/tau2-eval-official-native-full/eval/four-domain-full-bm25"
    / "seed300_0902_044845_summary.json"
)
DEFAULT_BASE_B_SUMMARY = (
    PROJECT_ROOT
    / "output/experiments/tau2-eval-official-native-full/eval/four-domain-full-bm25"
    / "seed300_0902_060330_summary.json"
)
DEFAULT_STRONG_SUMMARY = (
    PROJECT_ROOT
    / "output/experiments/tau2-eval-qwen35-official-native-four-domain/eval"
    / "four-domain-full-bm25-nonthinking/seed300_0902_054605_summary.json"
)
DEFAULT_SIMULATIONS_ROOT = SERVICE_AGENT_ROOT / "tau2-bench/data/simulations"
DEFAULT_JUDGE_TOKENIZER = SERVICE_AGENT_ROOT / "models/Qwen3.6-27B"


OUTCOME_PRECEDENCE = {
    "stable_gap": 0,
    "shared_hard": 1,
    "base_unstable": 2,
    "reverse_control": 3,
    "all_success": 4,
}

ATOMIC_GAPS = {
    "goal_constraint_tracking",
    "evidence_sufficiency",
    "entity_state_binding",
    "workflow_precondition_authorization",
    "action_selection",
    "argument_composition",
    "multi_subtask_coverage_stop",
    "observation_grounded_recovery",
    "none",
    "unclear",
}
PREFERRED_TRACES = {"A", "B", "tie", "unclear"}
EXTERNAL_FACTORS = {"agent", "retrieval", "user", "evaluator", "unclear"}
CONFIDENCE_LEVELS = {"clear", "uncertain"}
DISCOVERY_TOOLS = {
    "list_discoverable_agent_tools",
    "call_discoverable_agent_tool",
    "unlock_discoverable_agent_tool",
    "give_discoverable_user_tool",
    "call_discoverable_user_tool",
}
DOCUMENT_ID_RE = re.compile(r"(?m)^\s*ID:\s*([^\s]+)\s*$")
AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()


def classify_outcome(base_a: int, base_b: int, strong: int) -> str:
    """Classify one task/trial cell across two baselines and one contrast run."""

    if not base_a and not base_b:
        return "stable_gap" if strong else "shared_hard"
    if bool(base_a) != bool(base_b):
        return "base_unstable"
    if not strong:
        return "reverse_control"
    return "all_success"


def _success(simulation: dict[str, Any]) -> int:
    reward_info = simulation.get("reward_info") or {}
    return int(float(reward_info.get("reward") or 0.0) > 0.0)


def index_simulations(
    domain: str,
    run_label: str,
    results: dict[str, Any],
    source_path: str,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Index a results payload by its stable scenario-cell key."""

    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for simulation_index, simulation in enumerate(results.get("simulations") or []):
        task_id = str(simulation["task_id"])
        trial = int(simulation["trial"])
        key = (domain, task_id, trial)
        if key in indexed:
            raise ValueError(f"duplicate scenario cell {domain}/{task_id}/{trial} in {run_label}")
        indexed[key] = {
            "run_label": run_label,
            "source_path": source_path,
            "simulation_index": simulation_index,
            "simulation": simulation,
        }
    return indexed


def pair_run_indexes(
    base_a: dict[tuple[str, str, int], dict[str, Any]],
    base_b: dict[tuple[str, str, int], dict[str, Any]],
    strong: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair three indexed runs after confirming their scenario cells match."""

    expected_keys = set(base_a)
    if set(base_b) != expected_keys or set(strong) != expected_keys:
        raise ValueError("scenario cells differ across base_a, base_b, and strong runs")

    rows = []
    for domain, task_id, trial in sorted(
        expected_keys,
        key=lambda key: (key[0], key[1], key[2]),
    ):
        runs = {
            "base_a": base_a[(domain, task_id, trial)],
            "base_b": base_b[(domain, task_id, trial)],
            "strong": strong[(domain, task_id, trial)],
        }
        rewards = {label: _success(record["simulation"]) for label, record in runs.items()}
        rows.append(
            {
                "case_id": f"{domain}/{task_id}/{trial}",
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "rewards": rewards,
                "outcome_class": classify_outcome(
                    rewards["base_a"],
                    rewards["base_b"],
                    rewards["strong"],
                ),
                "runs": runs,
            }
        )
    return rows


def summarize_tasks(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate paired outcomes at task level rather than trajectory level."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[(row["domain"], row["task_id"])].append(row)

    summaries = []
    for (domain, task_id), rows in sorted(grouped.items()):
        baseline_successes = sum(
            row["rewards"][label]
            for row in rows
            for label in ("base_a", "base_b")
        )
        strong_successes = sum(row["rewards"]["strong"] for row in rows)
        summaries.append(
            {
                "domain": domain,
                "task_id": task_id,
                "baseline_successes": baseline_successes,
                "strong_successes": strong_successes,
                "high_confidence_gap": (
                    baseline_successes <= 2 and strong_successes >= 3
                ),
                "canonical_case_id": select_canonical_case(rows)["case_id"],
            }
        )
    return summaries


def select_canonical_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the most informative deterministic trial for one task."""

    if not rows:
        raise ValueError("cannot select a canonical case from no rows")
    return min(
        rows,
        key=lambda row: (
            OUTCOME_PRECEDENCE[row["outcome_class"]],
            int(row["trial"]),
        ),
    )


def _normalise_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def _tool_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = {
        message.get("id"): message
        for message in messages
        if message.get("role") == "tool" and message.get("id")
    }
    events = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            result = results.get(call.get("id"), {})
            events.append(
                {
                    "turn": message.get("turn_idx", message_index),
                    "name": call.get("name"),
                    "arguments": _normalise_arguments(call.get("arguments")),
                    "requestor": call.get("requestor", "assistant"),
                    "result": result.get("content"),
                    "error": bool(result.get("error")),
                }
            )
    return events


def extract_trace_features(
    domain: str,
    simulation: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    """Extract dependency-free facts shared by every tau2 domain."""

    criteria = task.get("evaluation_criteria") or {}
    reward_info = simulation.get("reward_info") or {}
    reference_actions = criteria.get("actions")
    if reference_actions is None:
        reference_actions = [
            check.get("action") or {}
            for check in reward_info.get("action_checks") or []
        ]
    expected_agent_actions = [
        action
        for action in reference_actions or []
        if action.get("requestor", "assistant") == "assistant"
    ]
    action_checks = reward_info.get("action_checks") or []
    db_check = reward_info.get("db_check") or {}
    events = _tool_events(simulation.get("messages") or [])
    return {
        "domain": domain,
        "task_id": str(simulation.get("task_id")),
        "trial": int(simulation.get("trial") or 0),
        "termination_reason": simulation.get("termination_reason"),
        "expected_agent_actions": expected_agent_actions,
        "environment_assertions": criteria.get("env_assertions") or [],
        "natural_language_assertions": criteria.get("nl_assertions") or [],
        "observed_tools": events,
        "tool_errors": sum(event["error"] for event in events),
        "golden_total": len(action_checks),
        "golden_matched": sum(bool(check.get("action_match")) for check in action_checks),
        "exact_action_match": bool(action_checks)
        and all(bool(check.get("action_match")) for check in action_checks),
        "db_match": db_check.get("db_match") if isinstance(db_check, dict) else None,
        "valid_turns": [
            message.get("turn_idx", index)
            for index, message in enumerate(simulation.get("messages") or [])
            if isinstance(message.get("turn_idx", index), int)
        ],
    }


def extract_banking_features(
    simulation: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    """Extract Banking retrieval and dynamic-tool evidence."""

    events = _tool_events(simulation.get("messages") or [])
    retrieval_events = []
    discovery_events = []
    retrieved_documents: set[str] = set()
    for event in events:
        if event["name"] == "KB_search":
            document_ids = DOCUMENT_ID_RE.findall(str(event.get("result") or ""))
            retrieved_documents.update(document_ids)
            retrieval_events.append(
                {
                    "turn": event["turn"],
                    "query": str(event["arguments"].get("query") or ""),
                    "document_ids": document_ids,
                    "error": event["error"],
                }
            )
        if event["name"] in DISCOVERY_TOOLS:
            discovery_events.append(
                {
                    "turn": event["turn"],
                    "name": event["name"],
                    "arguments": event["arguments"],
                    "error": event["error"],
                }
            )

    required = {str(item) for item in task.get("required_documents") or []}
    return {
        "retrieval_events": retrieval_events,
        "discovery_events": discovery_events,
        "required_documents": sorted(required),
        "required_document_hits": sorted(required & retrieved_documents),
        "missing_required_documents": sorted(required - retrieved_documents),
    }


def _conversation_view(simulation: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = []
    for index, message in enumerate(simulation.get("messages") or []):
        item = {
            "turn": message.get("turn_idx", index),
            "role": message.get("role"),
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            item["tool_calls"] = message["tool_calls"]
        if message.get("role") == "tool":
            item["error"] = bool(message.get("error"))
        conversation.append(item)
    return conversation


def build_case_packet(
    paired_row: dict[str, Any],
    task: dict[str, Any],
    *,
    ordinal: int,
) -> dict[str, Any]:
    """Build one identity-blinded pairwise review packet."""

    ordered_labels = ("base_b", "strong") if ordinal % 2 == 0 else ("strong", "base_b")
    trace_labels = {"A": ordered_labels[0], "B": ordered_labels[1]}
    traces = {}
    internal_sources = {}
    for blind_label, run_label in trace_labels.items():
        record = paired_row["runs"][run_label]
        simulation = record["simulation"]
        features = extract_trace_features(paired_row["domain"], simulation, task)
        if paired_row["domain"] == "banking_knowledge":
            features["banking"] = extract_banking_features(simulation, task)
        traces[blind_label] = {
            "features": features,
            "conversation": _conversation_view(simulation),
        }
        internal_sources[run_label] = {
            "source_path": record["source_path"],
            "simulation_index": record["simulation_index"],
        }

    criteria = task.get("evaluation_criteria") or {}
    judge_payload = {
        "case_id": paired_row["case_id"],
        "domain": paired_row["domain"],
        "task": {
            "description": task.get("description"),
            "user_scenario": task.get("user_scenario"),
            "reference_actions": criteria.get("actions"),
            "environment_assertions": criteria.get("env_assertions"),
            "natural_language_assertions": criteria.get("nl_assertions"),
            "required_documents": task.get("required_documents"),
        },
        "traces": traces,
    }
    return {
        "case_id": paired_row["case_id"],
        "domain": paired_row["domain"],
        "task_id": paired_row["task_id"],
        "trial": paired_row["trial"],
        "outcome_class": paired_row["outcome_class"],
        "run_success": {
            "base_b": paired_row["rewards"]["base_b"],
            "strong": paired_row["rewards"]["strong"],
        },
        "trace_labels": trace_labels,
        "sources": internal_sources,
        "judge_payload": judge_payload,
    }


def route_judge_payload(
    judge_payload: dict[str, Any],
    token_counter,
    *,
    context_tokens: int = 32768,
    output_tokens: int = 2048,
) -> dict[str, Any]:
    """Route complete payloads without truncating evidence."""

    messages = build_judge_messages(judge_payload)
    input_tokens = int(token_counter(messages))
    route = "local" if input_tokens + output_tokens <= context_tokens else "agent"
    return {"route": route, "input_tokens": input_tokens}


def validate_atomic_review(
    review: dict[str, Any],
    valid_turns: set[int],
) -> list[str]:
    """Validate the small public review schema used by both model tiers."""

    errors = []
    if review.get("preferred_trace") not in PREFERRED_TRACES:
        errors.append("invalid preferred_trace")
    if review.get("primary_gap") not in ATOMIC_GAPS:
        errors.append("invalid primary_gap")
    if review.get("external_factor") not in EXTERNAL_FACTORS:
        errors.append("invalid external_factor")
    if review.get("confidence") not in CONFIDENCE_LEVELS:
        errors.append("invalid confidence")

    divergence = review.get("first_divergence")
    if not isinstance(divergence, dict):
        errors.append("first_divergence must be an object")
    else:
        for trace in ("A", "B"):
            turn = divergence.get(trace)
            if turn is not None and (not isinstance(turn, int) or turn not in valid_turns):
                errors.append(f"first_divergence.{trace}: invalid turn")

    evidence = review.get("evidence")
    if not isinstance(evidence, list):
        errors.append("evidence must be a list")
    else:
        for index, item in enumerate(evidence):
            if not isinstance(item, dict) or item.get("trace") not in {"A", "B"}:
                errors.append(f"evidence[{index}]: invalid trace")
                continue
            turn = item.get("turn")
            if not isinstance(turn, int) or turn not in valid_turns:
                errors.append(f"evidence[{index}]: invalid turn")
    return errors


def resolve_results_path(results_file: str, simulations_root: Path) -> Path:
    """Resolve the standard results_file emitted by the official evaluator."""

    path = Path(results_file)
    if path.is_absolute():
        return path
    prefix = Path("data/simulations")
    try:
        relative = path.relative_to(prefix)
    except ValueError as exc:
        raise ValueError(
            f"results_file must be absolute or below data/simulations: {results_file}"
        ) from exc
    return simulations_root / relative


def _load_evaluation_run(
    summary_path: Path,
    run_label: str,
    simulations_root: Path,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    tasks: dict[tuple[str, str], dict[str, Any]] = {}
    for domain, domain_summary in sorted((summary.get("domains") or {}).items()):
        result_path = resolve_results_path(
            str(domain_summary["results_file"]),
            simulations_root,
        )
        results = json.loads(result_path.read_text(encoding="utf-8"))
        for task in results.get("tasks") or []:
            tasks[(domain, str(task["id"]))] = task
        domain_index = index_simulations(
            domain,
            run_label,
            results,
            str(result_path),
        )
        overlap = set(indexed) & set(domain_index)
        if overlap:
            domain_name, task_id, trial = min(overlap)
            raise ValueError(
                f"duplicate scenario cell {domain_name}/{task_id}/{trial} across domains"
            )
        indexed.update(domain_index)
    return indexed, tasks


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")


def mandatory_agent_review_reasons(
    paired_row: dict[str, Any],
    *,
    high_confidence_gap: bool,
    local_route: bool,
) -> list[str]:
    """Return the small set of evidence-backed reasons for mandatory adjudication."""

    reasons = []
    if high_confidence_gap:
        reasons.append("high_confidence_gap")
    if not local_route:
        reasons.append("over_context")
    if paired_row["outcome_class"] == "reverse_control":
        reasons.append("reverse_control")
    if (
        paired_row["domain"] == "banking_knowledge"
        and paired_row["rewards"]["base_b"] == 0
        and paired_row["rewards"]["strong"] == 1
    ):
        reasons.append("banking_positive_contrast")
    return sorted(reasons)


def prepare_analysis(
    *,
    base_a_summary: Path,
    base_b_summary: Path,
    strong_summary: Path,
    simulations_root: Path,
    output_dir: Path,
    token_counter,
    context_tokens: int = 32768,
    output_tokens: int = 2048,
) -> dict[str, Any]:
    """Prepare deterministic comparison tables and mixed-review queues."""

    base_a, _ = _load_evaluation_run(base_a_summary, "base_a", simulations_root)
    base_b, tasks = _load_evaluation_run(base_b_summary, "base_b", simulations_root)
    strong, _ = _load_evaluation_run(strong_summary, "strong", simulations_root)
    paired_rows = pair_run_indexes(base_a, base_b, strong)
    task_rows = summarize_tasks(paired_rows)
    paired_by_id = {row["case_id"]: row for row in paired_rows}

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_outcomes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "domain",
                "task_id",
                "trial",
                "base_a_success",
                "base_b_success",
                "strong_success",
                "outcome_class",
            ),
        )
        writer.writeheader()
        for row in paired_rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "domain": row["domain"],
                    "task_id": row["task_id"],
                    "trial": row["trial"],
                    "base_a_success": row["rewards"]["base_a"],
                    "base_b_success": row["rewards"]["base_b"],
                    "strong_success": row["rewards"]["strong"],
                    "outcome_class": row["outcome_class"],
                }
            )

    with (output_dir / "task_outcomes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(task_rows[0]) if task_rows else ())
        if task_rows:
            writer.writeheader()
            writer.writerows(task_rows)

    packets = []
    packet_by_case: dict[str, dict[str, Any]] = {}
    local_payloads = []
    agent_queue_by_case: dict[str, dict[str, Any]] = {}
    for ordinal, task_row in enumerate(task_rows):
        paired_row = paired_by_id[task_row["canonical_case_id"]]
        task_key = (task_row["domain"], task_row["task_id"])
        if task_key not in tasks:
            raise ValueError(f"task definition missing for {task_key[0]}/{task_key[1]}")
        packet = build_case_packet(paired_row, tasks[task_key], ordinal=ordinal)
        route = route_judge_payload(
            packet["judge_payload"],
            token_counter,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
        )
        packet.update(route)
        packet["high_confidence_gap"] = task_row["high_confidence_gap"]
        packets.append(packet)
        packet_by_case[packet["case_id"]] = packet
        if route["route"] == "local":
            local_payloads.append(packet["judge_payload"])
        reasons = mandatory_agent_review_reasons(
            paired_row,
            high_confidence_gap=task_row["high_confidence_gap"],
            local_route=route["route"] == "local",
        )
        if reasons:
            queued = dict(packet)
            queued["review_reasons"] = reasons
            agent_queue_by_case[packet["case_id"]] = queued

    high_confidence_keys = {
        (row["domain"], row["task_id"]): row["high_confidence_gap"]
        for row in task_rows
    }
    for ordinal, paired_row in enumerate(paired_rows, start=len(task_rows)):
        if paired_row["case_id"] in packet_by_case:
            continue
        task_key = (paired_row["domain"], paired_row["task_id"])
        reasons_without_route = mandatory_agent_review_reasons(
            paired_row,
            high_confidence_gap=False,
            local_route=True,
        )
        if not reasons_without_route:
            continue
        packet = packet_by_case.get(paired_row["case_id"])
        if packet is None:
            packet = build_case_packet(paired_row, tasks[task_key], ordinal=ordinal)
            route = route_judge_payload(
                packet["judge_payload"],
                token_counter,
                context_tokens=context_tokens,
                output_tokens=output_tokens,
            )
            packet.update(route)
            packet["high_confidence_gap"] = high_confidence_keys[task_key]
        reasons = mandatory_agent_review_reasons(
            paired_row,
            high_confidence_gap=False,
            local_route=packet["route"] == "local",
        )
        queued = dict(packet)
        queued["review_reasons"] = reasons
        agent_queue_by_case[packet["case_id"]] = queued

    calibration_payloads = []
    packets_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        if packet["route"] == "local":
            packets_by_domain[packet["domain"]].append(packet)
    for domain in sorted(packets_by_domain):
        ranked = sorted(
            packets_by_domain[domain],
            key=lambda packet: (
                OUTCOME_PRECEDENCE[packet["outcome_class"]],
                packet["task_id"],
            ),
        )
        selected = []
        seen_outcomes = set()
        for packet in ranked:
            if packet["outcome_class"] in seen_outcomes:
                continue
            selected.append(packet)
            seen_outcomes.add(packet["outcome_class"])
            if len(selected) == 3:
                break
        if len(selected) < 3:
            selected_ids = {packet["case_id"] for packet in selected}
            selected.extend(
                packet
                for packet in ranked
                if packet["case_id"] not in selected_ids
            )
        calibration_payloads.extend(
            packet["judge_payload"] for packet in selected[:3]
        )

    agent_queue = [
        agent_queue_by_case[case_id]
        for case_id in sorted(agent_queue_by_case)
    ]

    _write_jsonl(output_dir / "case_packets.jsonl", packets)
    _write_jsonl(output_dir / "judge_payloads.jsonl", local_payloads)
    _write_jsonl(output_dir / "agent_review_queue.jsonl", agent_queue)
    _write_jsonl(output_dir / "calibration_payloads.jsonl", calibration_payloads)

    outcome_counts: dict[str, int] = defaultdict(int)
    for row in paired_rows:
        outcome_counts[row["outcome_class"]] += 1
    route_counts: dict[str, int] = defaultdict(int)
    for packet in packets:
        route_counts[packet["route"]] += 1
    high_confidence_by_domain: dict[str, int] = defaultdict(int)
    for row in task_rows:
        if row["high_confidence_gap"]:
            high_confidence_by_domain[row["domain"]] += 1
    by_domain = {}
    for domain in sorted({row["domain"] for row in paired_rows}):
        domain_rows = [row for row in paired_rows if row["domain"] == domain]
        domain_outcomes: dict[str, int] = defaultdict(int)
        for row in domain_rows:
            domain_outcomes[row["outcome_class"]] += 1
        by_domain[domain] = {
            "cells": len(domain_rows),
            "tasks": sum(row["domain"] == domain for row in task_rows),
            "base_a_successes": sum(row["rewards"]["base_a"] for row in domain_rows),
            "base_b_successes": sum(row["rewards"]["base_b"] for row in domain_rows),
            "strong_successes": sum(row["rewards"]["strong"] for row in domain_rows),
            "outcomes": dict(sorted(domain_outcomes.items())),
        }
    inventory = {
        "scenario_cells": len(paired_rows),
        "tasks": len(task_rows),
        "domains": sorted({row["domain"] for row in paired_rows}),
        "by_domain": by_domain,
        "outcomes": dict(sorted(outcome_counts.items())),
        "routes": dict(sorted(route_counts.items())),
        "high_confidence_tasks": sum(high_confidence_by_domain.values()),
        "high_confidence_by_domain": dict(sorted(high_confidence_by_domain.items())),
        "mandatory_agent_reviews": len(agent_queue),
        "calibration_cases": len(calibration_payloads),
        "inputs": {
            "base_a_summary": str(base_a_summary),
            "base_b_summary": str(base_b_summary),
            "strong_summary": str(strong_summary),
            "simulations_root": str(simulations_root),
        },
    }
    (output_dir / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from a plain or fenced model response."""

    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            stripped = stripped[first_newline + 1 : last_fence].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must be an object")
    return parsed


def build_judge_messages(judge_payload: dict[str, Any]) -> list[dict[str, str]]:
    """Build the identity-blind atomic-gap review request."""

    system = (
        "You compare two customer-service trajectories for the same task. "
        "Find the first causal divergence, identify exactly one primary atomic "
        "capability gap in the weaker trace, and explain the minimal corrective "
        "mechanism in the better trace. Trace A and Trace B identities are hidden. "
        "Prefer task criteria, executed tool observations, and environment state over "
        "style. More calls are not inherently better. Use none or unclear when the "
        "evidence does not support a gap. Return only the requested JSON object."
    )
    user = (
        "Allowed primary_gap values: "
        + ", ".join(sorted(ATOMIC_GAPS))
        + ".\nAllowed external_factor values: "
        + ", ".join(sorted(EXTERNAL_FACTORS))
        + ".\nAnalyze this case:\n"
        + json.dumps(judge_payload, ensure_ascii=False, sort_keys=True)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def atomic_review_response_schema(valid_turns: set[int]) -> dict[str, Any]:
    """Return a structured-output schema bound to turns in one case."""

    turn_schema: dict[str, Any] = {"type": ["integer", "null"]}
    if valid_turns:
        turn_schema["enum"] = [None, *sorted(valid_turns)]
    evidence_turn: dict[str, Any] = {"type": "integer"}
    if valid_turns:
        evidence_turn["enum"] = sorted(valid_turns)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "case_id": {"type": "string"},
            "preferred_trace": {"type": "string", "enum": sorted(PREFERRED_TRACES)},
            "primary_gap": {"type": "string", "enum": sorted(ATOMIC_GAPS)},
            "first_divergence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"A": turn_schema, "B": turn_schema},
                "required": ["A", "B"],
            },
            "expected_next_action": {"type": "string"},
            "observed_failure": {"type": "string"},
            "corrective_mechanism": {"type": "string"},
            "external_factor": {"type": "string", "enum": sorted(EXTERNAL_FACTORS)},
            "confidence": {"type": "string", "enum": sorted(CONFIDENCE_LEVELS)},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "trace": {"type": "string", "enum": ["A", "B"]},
                        "turn": evidence_turn,
                        "reason": {"type": "string"},
                    },
                    "required": ["trace", "turn", "reason"],
                },
            },
        },
        "required": [
            "case_id",
            "preferred_trace",
            "primary_gap",
            "first_divergence",
            "expected_next_action",
            "observed_failure",
            "corrective_mechanism",
            "external_factor",
            "confidence",
            "evidence",
        ],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tau2_atomic_capability_gap",
            "strict": True,
            "schema": schema,
        },
    }


def _review_one_payload(
    payload: dict[str, Any],
    complete,
    pass_index: int,
) -> dict[str, Any]:
    valid_turns = {
        turn
        for trace in (payload.get("traces") or {}).values()
        for turn in (trace.get("features") or {}).get("valid_turns", [])
        if isinstance(turn, int)
    }
    try:
        raw_text = complete(
            build_judge_messages(payload),
            atomic_review_response_schema(valid_turns),
        )
        review = parse_json_object(raw_text)
    except Exception as exc:
        return {
            "case_id": payload.get("case_id"),
            "pass_index": pass_index,
            "status": "agent_review",
            "review": None,
            "errors": [f"invalid JSON response: {exc}"],
        }

    errors = validate_atomic_review(review, valid_turns)
    if review.get("case_id") != payload.get("case_id"):
        errors.append("case_id does not match the request")
    return {
        "case_id": payload.get("case_id"),
        "pass_index": pass_index,
        "status": "valid" if not errors else "agent_review",
        "review": review,
        "errors": errors,
    }


def run_judge_reviews(
    payloads: list[dict[str, Any]],
    *,
    complete,
    primary_passes: int = 1,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Review payloads concurrently while preserving deterministic output order."""

    work = [
        (order, payload, pass_index)
        for order, payload in enumerate(payloads)
        for pass_index in range(primary_passes)
    ]
    if not work:
        return []
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_keys = {
            executor.submit(_review_one_payload, payload, complete, pass_index): (
                order,
                pass_index,
            )
            for order, payload, pass_index in work
        }
        for future in as_completed(future_keys):
            completed[future_keys[future]] = future.result()
    return [completed[key] for key in sorted(completed)]


def extend_agent_review_queue(
    *,
    base_queue: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    qwen_reviews: list[dict[str, Any]],
    calibration_reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge local uncertainty and calibration disagreement into the agent queue."""

    packet_by_case = {packet["case_id"]: packet for packet in packets}
    queued: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in base_queue:
        case_id = row["case_id"]
        queued[case_id] = dict(row)
        reasons[case_id].update(row.get("review_reasons") or [])
        packet_by_case.setdefault(case_id, row)

    for row in qwen_reviews:
        if row.get("pass_index") != 0:
            continue
        case_id = row["case_id"]
        if row.get("status") != "valid":
            reasons[case_id].add("local_invalid")
        elif (row.get("review") or {}).get("confidence") != "clear":
            reasons[case_id].add("local_uncertain")

    calibration_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_reviews:
        calibration_by_case[row["case_id"]].append(row)
    for case_id, rows in calibration_by_case.items():
        if any(row.get("status") != "valid" for row in rows):
            reasons[case_id].add("calibration_invalid")
            continue
        decisions = {
            (
                (row.get("review") or {}).get("primary_gap"),
                (row.get("review") or {}).get("preferred_trace"),
            )
            for row in rows
        }
        if len(decisions) > 1:
            reasons[case_id].add("calibration_disagreement")

    for case_id, case_reasons in reasons.items():
        if not case_reasons:
            continue
        if case_id not in packet_by_case:
            raise ValueError(f"review references unknown case {case_id}")
        row = dict(queued.get(case_id) or packet_by_case[case_id])
        row["review_reasons"] = sorted(case_reasons)
        queued[case_id] = row
    return [queued[case_id] for case_id in sorted(queued)]


def normalize_adjudication(
    packet: dict[str, Any],
    review: dict[str, Any],
    *,
    reviewer: str,
) -> dict[str, Any]:
    """Map blinded trace citations back to the two evaluated run roles."""

    trace_labels = packet["trace_labels"]
    preferred_trace = review["preferred_trace"]
    preferred_run = trace_labels.get(preferred_trace, preferred_trace)
    divergence = {
        trace_labels[trace]: turn
        for trace, turn in review["first_divergence"].items()
    }
    evidence = [
        {
            **item,
            "run": trace_labels[item["trace"]],
        }
        for item in review.get("evidence") or []
    ]
    return {
        "case_id": packet["case_id"],
        "domain": packet["domain"],
        "task_id": packet["task_id"],
        "trial": packet["trial"],
        "reviewer": reviewer,
        "preferred_run": preferred_run,
        "primary_gap": review["primary_gap"],
        "first_divergence": divergence,
        "expected_next_action": review["expected_next_action"],
        "observed_failure": review["observed_failure"],
        "corrective_mechanism": review["corrective_mechanism"],
        "external_factor": review["external_factor"],
        "confidence": review["confidence"],
        "evidence": evidence,
        "input_tokens": packet.get("input_tokens"),
        "high_confidence_gap": bool(packet.get("high_confidence_gap")),
        "run_success": packet.get("run_success") or {},
        "sources": packet.get("sources") or {},
    }


def _eligible_for_replay(row: dict[str, Any]) -> bool:
    if row.get("confidence") != "clear":
        return False
    if row.get("primary_gap") in {None, "none", "unclear"}:
        return False
    divergence = row.get("first_divergence") or {}
    if not all(isinstance(divergence.get(run), int) for run in ("base_b", "strong")):
        return False
    if row.get("domain") == "banking_knowledge":
        success = row.get("run_success") or {}
        return success.get("base_b") == 0 and success.get("strong") == 1
    return bool(row.get("high_confidence_gap"))


def select_replay_cases(
    adjudicated_rows: list[dict[str, Any]],
    *,
    per_domain: int = 3,
) -> list[dict[str, Any]]:
    """Select diverse, short, clear cases for the bounded replay experiment."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in adjudicated_rows:
        if _eligible_for_replay(row):
            grouped[row["domain"]].append(row)

    selected = []
    for domain in sorted(grouped):
        candidates = grouped[domain]
        gap_counts: dict[str, int] = defaultdict(int)
        for row in candidates:
            gap_counts[row["primary_gap"]] += 1
        ranked_gaps = sorted(gap_counts, key=lambda gap: (-gap_counts[gap], gap))
        domain_selected = []
        used_tasks = set()
        for gap in ranked_gaps[:per_domain]:
            choices = [
                row
                for row in candidates
                if row["primary_gap"] == gap and row["task_id"] not in used_tasks
            ]
            if not choices:
                continue
            choice = min(
                choices,
                key=lambda row: (int(row.get("input_tokens") or 0), row["case_id"]),
            )
            domain_selected.append(choice)
            used_tasks.add(choice["task_id"])
        if len(domain_selected) < per_domain:
            remaining = sorted(
                (row for row in candidates if row["task_id"] not in used_tasks),
                key=lambda row: (int(row.get("input_tokens") or 0), row["case_id"]),
            )
            domain_selected.extend(remaining[: per_domain - len(domain_selected)])
        selected.extend(domain_selected)
    return selected


def build_replay_manifest(
    selected_cases: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...] = (300, 301, 302, 303),
) -> list[dict[str, Any]]:
    """Expand selected cases into the fixed 2-model × 2-prefix design."""

    manifest = []
    for case in selected_cases:
        for model in ("qwen3", "qwen35"):
            for prefix in ("weak", "strong"):
                source_run = "base_b" if prefix == "weak" else "strong"
                source = case["sources"][source_run]
                for seed in seeds:
                    manifest.append(
                        {
                            "replay_id": (
                                f"{case['case_id']}/{model}/{prefix}/seed{seed}"
                            ),
                            "case_id": case["case_id"],
                            "domain": case["domain"],
                            "task_id": case["task_id"],
                            "primary_gap": case["primary_gap"],
                            "expected_next_action": case.get("expected_next_action"),
                            "model": model,
                            "prefix": prefix,
                            "source_run": source_run,
                            "source_path": source["source_path"],
                            "simulation_index": source["simulation_index"],
                            "prefix_turn": case["first_divergence"][source_run],
                            "seed": seed,
                        }
                    )
    return manifest


def _assistant_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": call["id"],
            "name": call.get("name"),
            "type": "function",
            "function": {
                "name": call.get("name"),
                "arguments": json.dumps(
                    _normalise_arguments(call.get("arguments")),
                    ensure_ascii=False,
                ),
            },
        }
        for call in message.get("tool_calls") or []
        if call.get("requestor", "assistant") == "assistant"
    ]


def build_official_replay_request(
    simulation: dict[str, Any],
    *,
    prefix_turn: int,
    tool_schemas: list[dict[str, Any]],
    model: str,
    seed: int,
    max_tokens: int,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    """Reconstruct one official-native Agent request immediately before a turn."""

    system_prompt = (
        "<instructions>\n"
        f"{AGENT_INSTRUCTION}\n"
        "</instructions>\n"
        "<policy>\n"
        f"{simulation.get('policy') or ''}\n"
        "</policy>"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]
    assistant_call_ids: set[str] = set()
    for index, message in enumerate(simulation.get("messages") or []):
        turn = message.get("turn_idx", index)
        if isinstance(turn, int) and turn >= prefix_turn:
            break
        role = message.get("role")
        if role == "user":
            if message.get("tool_calls"):
                continue
            content = message.get("content")
            if content:
                messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tool_calls = _assistant_tool_calls(message)
            assistant_call_ids.update(
                call["id"] for call in tool_calls if isinstance(call.get("id"), str)
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls or None,
                }
            )
        elif role == "tool" and message.get("id") in assistant_call_ids:
            messages.append(
                {
                    "role": "tool",
                    "content": message.get("content"),
                    "tool_call_id": message.get("id"),
                }
            )

    request = {
        "model": model,
        "messages": messages,
        "tools": tool_schemas,
        "tool_choice": "auto",
        "temperature": 0.6,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if enable_thinking is not None:
        request["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return request


def classify_replay_evidence(condition_scores: dict[str, int]) -> str:
    """Interpret the two directional effects without a scoring threshold network."""

    qwen3_weak = condition_scores["qwen3_weak"]
    model_effect = condition_scores["qwen35_weak"] > qwen3_weak
    prefix_effect = condition_scores["qwen3_strong"] > qwen3_weak
    if model_effect and prefix_effect:
        return "compound"
    if model_effect:
        return "local_decision"
    if prefix_effect:
        return "upstream_evidence_state"
    return "uncertain"


def request_chat_completion(
    base_url: str,
    request_body: dict[str, Any],
    *,
    api_key: str = "EMPTY",
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Send one concrete OpenAI-compatible chat request."""

    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise ValueError("chat completion response has no assistant message")
    return choices[0]["message"]


def make_chat_token_counter(tokenizer, *, enable_thinking: bool):
    """Count the exact rendered tokens used by the local review request."""

    def count(messages: list[dict[str, Any]]) -> int:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return len(token_ids)

    return count


def make_replay_request_fitter(tokenizer, *, context_tokens: int):
    """Check a complete replay request against the model context without truncation."""

    def fits(request: dict[str, Any]) -> tuple[bool, int]:
        template_kwargs = dict(request.get("chat_template_kwargs") or {})
        render_messages = copy.deepcopy(request["messages"])
        for message in render_messages:
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                if "arguments" in function:
                    function["arguments"] = _normalise_arguments(
                        function["arguments"]
                    )
        token_ids = tokenizer.apply_chat_template(
            render_messages,
            tools=request.get("tools") or None,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        input_tokens = len(token_ids)
        return input_tokens + int(request["max_tokens"]) <= context_tokens, input_tokens

    return fits


def load_official_tool_schemas(
    domains: list[str],
    *,
    retrieval_config: str,
    registry=None,
) -> dict[str, list[dict[str, Any]]]:
    """Load the exact tool schemas exposed by official tau2 environments."""

    if registry is None:
        from tau2.registry import registry as tau2_registry

        registry = tau2_registry
    schemas = {}
    for domain in domains:
        environment_kwargs = {}
        if domain == "banking_knowledge":
            environment_kwargs["retrieval_variant"] = retrieval_config
        environment = registry.get_env_constructor(domain)(**environment_kwargs)
        schemas[domain] = [tool.openai_schema for tool in environment.get_tools()]
    return schemas


def _valid_turns_for_packet(packet: dict[str, Any]) -> set[int]:
    return {
        turn
        for trace in (packet.get("judge_payload") or {}).get("traces", {}).values()
        for turn in (trace.get("features") or {}).get("valid_turns", [])
        if isinstance(turn, int)
    }


def merge_adjudications(
    packets: list[dict[str, Any]],
    qwen_reviews: list[dict[str, Any]],
    *,
    agent_reviews: list[dict[str, Any]],
    mandatory_agent_case_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Prefer agent adjudication and use only clear, valid local reviews otherwise."""

    packet_by_case = {packet["case_id"]: packet for packet in packets}
    agent_by_case = {review["case_id"]: review for review in agent_reviews}
    qwen_by_case = {
        row["case_id"]: row
        for row in qwen_reviews
        if row.get("pass_index") == 0
    }
    normalized = []
    missing = []
    for packet in packets:
        case_id = packet["case_id"]
        agent_review = agent_by_case.get(case_id)
        if agent_review is not None:
            errors = validate_atomic_review(agent_review, _valid_turns_for_packet(packet))
            if errors:
                raise ValueError(f"invalid agent review for {case_id}: {errors}")
            normalized.append(
                normalize_adjudication(packet, agent_review, reviewer="agent")
            )
            continue
        if case_id in mandatory_agent_case_ids:
            missing.append(case_id)
            continue
        qwen_row = qwen_by_case.get(case_id)
        if (
            qwen_row
            and qwen_row.get("status") == "valid"
            and (qwen_row.get("review") or {}).get("confidence") == "clear"
        ):
            normalized.append(
                normalize_adjudication(
                    packet,
                    qwen_row["review"],
                    reviewer="qwen36",
                )
            )
        else:
            missing.append(case_id)
    unknown_agent_cases = sorted(set(agent_by_case) - set(packet_by_case))
    if unknown_agent_cases:
        raise ValueError(f"agent reviews reference unknown cases: {unknown_agent_cases}")
    return normalized, missing


def _assistant_action(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw_arguments = function.get("arguments")
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except (TypeError, ValueError):
            arguments = raw_arguments
        tool_calls.append(
            {
                "name": function.get("name") or call.get("name"),
                "arguments": arguments,
            }
        )
    return {
        "content": message.get("content"),
        "tool_calls": tool_calls,
    }


def run_replays(
    manifest: list[dict[str, Any]],
    *,
    model_label: str,
    served_model: str,
    tool_schemas_by_domain: dict[str, list[dict[str, Any]]],
    max_tokens: int,
    enable_thinking: bool | None,
    request_fits,
    complete,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Execute one model arm of the fixed one-step replay manifest."""

    selected = [row for row in manifest if row["model"] == model_label]
    result_cache: dict[str, dict[str, Any]] = {}
    prepared = []
    for order, row in enumerate(selected):
        source_path = row["source_path"]
        if source_path not in result_cache:
            result_cache[source_path] = json.loads(
                Path(source_path).read_text(encoding="utf-8")
            )
        results = result_cache[source_path]
        simulation = results["simulations"][int(row["simulation_index"])]
        tasks = {str(task["id"]): task for task in results.get("tasks") or []}
        request = build_official_replay_request(
            simulation,
            prefix_turn=int(row["prefix_turn"]),
            tool_schemas=tool_schemas_by_domain[row["domain"]],
            model=served_model,
            seed=int(row["seed"]),
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        fits, input_tokens = request_fits(request)
        prepared.append(
            (
                order,
                row,
                request,
                bool(fits),
                int(input_tokens),
                tasks.get(str(row["task_id"])),
            )
        )

    def execute(item):
        order, row, request, fits, input_tokens, task = item
        record = {
            **row,
            "input_tokens": input_tokens,
            "task": task,
        }
        if not fits:
            record.update({"status": "over_context", "action": None})
            return order, record
        try:
            message = complete(request)
            record.update(
                {
                    "status": "generated",
                    "assistant_message": message,
                    "action": _assistant_action(message),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "generation_error",
                    "action": None,
                    "error": str(exc),
                }
            )
        return order, record

    completed = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(execute, item): item[0] for item in prepared}
        for future in as_completed(futures):
            order, record = future.result()
            completed[order] = record
    return [completed[index] for index in sorted(completed)]


def build_replay_review_queue(
    replay_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose every generated next action for semantic acceptability review."""

    return [
        {
            "replay_id": row["replay_id"],
            "case_id": row["case_id"],
            "domain": row["domain"],
            "model": row["model"],
            "prefix": row["prefix"],
            "seed": row["seed"],
            "expected_next_action": row.get("expected_next_action"),
            "task": row.get("task"),
            "action": row.get("action"),
            "source_path": row["source_path"],
            "simulation_index": row["simulation_index"],
            "prefix_turn": row["prefix_turn"],
        }
        for row in replay_records
        if row.get("status") == "generated"
    ]


def summarize_replay_adjudications(
    replay_records: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Report exact acceptable/evaluated counts and the simple causal pattern."""

    record_by_id = {row["replay_id"]: row for row in replay_records}
    decision_by_id = {row["replay_id"]: row for row in decisions}
    unknown = sorted(set(decision_by_id) - set(record_by_id))
    if unknown:
        raise ValueError(f"replay adjudications reference unknown rows: {unknown}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replay_records:
        grouped[row["case_id"]].append(row)
    summaries = []
    for case_id in sorted(grouped):
        rows = grouped[case_id]
        conditions = {}
        for model in ("qwen3", "qwen35"):
            for prefix in ("weak", "strong"):
                name = f"{model}_{prefix}"
                condition_rows = [
                    row
                    for row in rows
                    if row["model"] == model and row["prefix"] == prefix
                ]
                evaluated = [
                    decision_by_id[row["replay_id"]]
                    for row in condition_rows
                    if row["replay_id"] in decision_by_id
                ]
                conditions[name] = {
                    "acceptable": sum(
                        bool(decision.get("acceptable_next_action"))
                        for decision in evaluated
                    ),
                    "evaluated": len(evaluated),
                    "generation_errors": sum(
                        row.get("status") != "generated" for row in condition_rows
                    ),
                }
        required = ("qwen3_weak", "qwen35_weak", "qwen3_strong")
        if all(conditions[name]["evaluated"] == 4 for name in required):
            causal_pattern = classify_replay_evidence(
                {name: conditions[name]["acceptable"] for name in required}
            )
        else:
            causal_pattern = "uncertain"
        summaries.append(
            {
                "case_id": case_id,
                "domain": rows[0]["domain"],
                "conditions": conditions,
                "causal_pattern": causal_pattern,
            }
        )
    return summaries


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _unblind_trace_text(value: Any, trace_labels: dict[str, str]) -> str:
    text = str(value or "")
    display_names = {"base_b": "Qwen3", "strong": "Qwen3.5"}
    for trace, run in trace_labels.items():
        text = re.sub(
            rf"\bTrace {re.escape(trace)}\b",
            display_names.get(run, run),
            text,
        )
    return text


def write_formal_report(
    *,
    adjudicated_cases: list[dict[str, Any]],
    canonical_packets: list[dict[str, Any]],
    replay_records: list[dict[str, Any]],
    replay_decisions: list[dict[str, Any]],
    inventory: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write task-weighted tables and the experiment's concise formal report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_by_id = {packet["case_id"]: packet for packet in canonical_packets}
    adjudicated_by_id = {row["case_id"]: row for row in adjudicated_cases}
    canonical_rows = [
        adjudicated_by_id[case_id]
        for case_id in canonical_by_id
        if case_id in adjudicated_by_id
    ]
    for row in canonical_rows:
        packet = canonical_by_id[row["case_id"]]
        row["high_confidence_gap"] = bool(packet.get("high_confidence_gap"))

    gap_rows = []
    scopes = {
        "high_confidence": [
            row for row in canonical_rows if row["high_confidence_gap"]
        ],
        "exploratory_canonical": canonical_rows,
    }
    for scope, rows in scopes.items():
        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_domain[row["domain"]].append(row)
        for domain in sorted(by_domain):
            denominator = len(by_domain[domain])
            counts: dict[str, int] = defaultdict(int)
            for row in by_domain[domain]:
                counts[row["primary_gap"]] += 1
            for gap, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                gap_rows.append(
                    {
                        "scope": scope,
                        "domain": domain,
                        "primary_gap": gap,
                        "tasks": count,
                        "denominator": denominator,
                        "share": f"{count / denominator:.6f}",
                    }
                )

    with (output_dir / "atomic_gap_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = (
            "scope",
            "domain",
            "primary_gap",
            "tasks",
            "denominator",
            "share",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gap_rows)

    replay_summary = summarize_replay_adjudications(
        replay_records,
        replay_decisions,
    )
    replay_totals = {
        name: {
            "acceptable": sum(
                row["conditions"][name]["acceptable"] for row in replay_summary
            ),
            "evaluated": sum(
                row["conditions"][name]["evaluated"] for row in replay_summary
            ),
        }
        for name in ("qwen3_weak", "qwen35_weak", "qwen3_strong", "qwen35_strong")
    }
    replay_pattern_counts: dict[str, int] = defaultdict(int)
    for row in replay_summary:
        replay_pattern_counts[row["causal_pattern"]] += 1
    (output_dir / "replay_summary.json").write_text(
        json.dumps(replay_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Qwen3–Qwen3.5 Atomic Capability Gap Analysis",
        "",
        "Experiment name: `tau2-qwen3-qwen35-atomic-gap-analysis`. Purpose: explain "
        "Qwen3-4B-Instruct-2507 failures across the four-domain official benchmark "
        "using deterministic evidence, blinded contrastive review, and bounded prefix replay.",
        "",
        "Qwen3.5 non-thinking is contrastive evidence rather than a ground-truth teacher.",
        "",
        "## Inputs",
        "",
        f"- Scenario cells: {inventory.get('scenario_cells', 0)} across "
        f"{inventory.get('tasks', 0)} tasks.",
        "- Runs: two Qwen3-4B-Instruct-2507 seed-300 evaluations and one matched "
        "Qwen3.5-4B non-thinking evaluation; official-native, four trials, BM25 Banking retrieval.",
        f"- Canonical cases adjudicated: {len(canonical_rows)}/{len(canonical_packets)}.",
        "",
        "## Outcome matrix",
        "",
        "| Outcome | Scenario cells |",
        "|---|---:|",
    ]
    for outcome in (
        "stable_gap",
        "shared_hard",
        "base_unstable",
        "reverse_control",
        "all_success",
    ):
        lines.append(f"| {outcome} | {(inventory.get('outcomes') or {}).get(outcome, 0)} |")

    domain_inventory = inventory.get("by_domain") or {}
    if domain_inventory:
        lines.extend(
            [
                "",
                "## Domain outcome rates",
                "",
                "Success counts use the same scenario-cell denominator within each domain.",
                "",
                "| Domain | Tasks | Qwen3 base A | Qwen3 base B | Qwen3.5 | Stable gap | Shared hard |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for domain, domain_row in sorted(domain_inventory.items()):
            cells = int(domain_row["cells"])
            outcomes = domain_row.get("outcomes") or {}
            lines.append(
                f"| {domain} | {domain_row['tasks']} | "
                f"{domain_row['base_a_successes']}/{cells} | "
                f"{domain_row['base_b_successes']}/{cells} | "
                f"{domain_row['strong_successes']}/{cells} | "
                f"{outcomes.get('stable_gap', 0)} | "
                f"{outcomes.get('shared_hard', 0)} |"
            )

    high_gap_rows = [row for row in gap_rows if row["scope"] == "high_confidence"]
    lines.extend(
        [
            "",
            "## Atomic gaps",
            "",
            "Formal prevalence uses only task-weighted, adjudicated high-confidence contrasts.",
            "A task is high confidence when Qwen3 succeeds at most 2/8 times across both "
            "repeats and Qwen3.5 succeeds at least 3/4 times.",
            "",
            "| Domain | Primary gap | Tasks | Share |",
            "|---|---|---:|---:|",
        ]
    )
    if high_gap_rows:
        for row in high_gap_rows:
            lines.append(
                f"| {row['domain']} | {row['primary_gap']} | {row['tasks']} | "
                f"{float(row['share']):.1%} |"
            )
    else:
        lines.append("| — | — | 0 | — |")

    lines.extend(
        [
            "",
            "## Domain findings",
            "",
            "Banking is reported as shared difficulty because it has no high-confidence "
            "task-level contrast and Qwen3.5 succeeds only sparsely there.",
            "",
            "| Domain | High-confidence tasks | Rows summarized | Leading adjudicated gaps |",
            "|---|---:|---:|---|",
        ]
    )
    domains = sorted(
        set((inventory.get("high_confidence_by_domain") or {}))
        | {row["domain"] for row in canonical_rows}
    )
    for domain in domains:
        high_total = (inventory.get("high_confidence_by_domain") or {}).get(domain, 0)
        if high_total:
            domain_rows = [
                row
                for row in high_gap_rows
                if row["domain"] == domain
            ]
            reviewed = sum(int(row["tasks"]) for row in domain_rows)
        else:
            domain_rows = [
                row
                for row in gap_rows
                if row["scope"] == "exploratory_canonical"
                and row["domain"] == domain
            ]
            reviewed = len([row for row in canonical_rows if row["domain"] == domain])
        leading = ", ".join(
            f"{row['primary_gap']} ({row['tasks']})" for row in domain_rows[:3]
        ) or "not adjudicated"
        lines.append(f"| {domain} | {high_total} | {reviewed} | {leading} |")
    lines.extend(
        [
            "",
            "Candidate capability priorities (not training results):",
            "",
            "- Retrieve decisive evidence and bind it to the correct entity/state before acting.",
            "- Check workflow preconditions and authorization before irreversible writes.",
            "- Select the valid action and compose grounded arguments without inventing values.",
            "- Preserve multi-subtask coverage and execute the required stop or transfer condition.",
        ]
    )

    lines.extend(
        [
            "",
            "## Prefix replay",
            "",
            "Counts are acceptable next actions over four samples per model/prefix condition.",
            "",
            "| Case | Domain | Qwen3/weak | Qwen3.5/weak | Qwen3/strong | "
            "Qwen3.5/strong | Pattern |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    if replay_summary:
        for row in replay_summary:
            conditions = row["conditions"]
            values = []
            for name in ("qwen3_weak", "qwen35_weak", "qwen3_strong", "qwen35_strong"):
                condition = conditions[name]
                values.append(f"{condition['acceptable']}/{condition['evaluated']}")
            lines.append(
                f"| {row['case_id']} | {row['domain']} | "
                + " | ".join(values)
                + f" | {row['causal_pattern']} |"
            )
        lines.extend(
            [
                "",
                "Replay aggregate: "
                f"Qwen3 weak {replay_totals['qwen3_weak']['acceptable']}/"
                f"{replay_totals['qwen3_weak']['evaluated']}, "
                f"Qwen3.5 weak {replay_totals['qwen35_weak']['acceptable']}/"
                f"{replay_totals['qwen35_weak']['evaluated']}, "
                f"Qwen3 strong {replay_totals['qwen3_strong']['acceptable']}/"
                f"{replay_totals['qwen3_strong']['evaluated']}, and "
                f"Qwen3.5 strong {replay_totals['qwen35_strong']['acceptable']}/"
                f"{replay_totals['qwen35_strong']['evaluated']}.",
                "Directional case patterns: "
                + ", ".join(
                    f"{name} {count}"
                    for name, count in sorted(replay_pattern_counts.items())
                )
                + ".",
            ]
        )
    else:
        lines.append("| — | — | — | — | — | — | not run |")

    lines.extend(
        [
            "",
            "## Representative cases",
            "",
            "| Domain/task | Primary gap | First divergence (Qwen3/Qwen3.5) | "
            "Minimal corrective mechanism |",
            "|---|---|---|---|",
        ]
    )
    representative = []
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        run_success = row.get("run_success") or {}
        banking_positive = (
            row["domain"] == "banking_knowledge"
            and run_success.get("base_b") == 0
            and run_success.get("strong") == 1
        )
        if row.get("confidence") == "clear" and (
            row.get("high_confidence_gap") or banking_positive
        ):
            by_domain[row["domain"]].append(row)
    for domain in sorted(by_domain):
        candidates = sorted(
            by_domain[domain],
            key=lambda row: (
                not bool(row.get("high_confidence_gap")),
                row["primary_gap"],
                row["task_id"],
            ),
        )
        gap_counts: dict[str, int] = defaultdict(int)
        for row in candidates:
            gap_counts[row["primary_gap"]] += 1
        selected = []
        for gap in sorted(gap_counts, key=lambda name: (-gap_counts[name], name)):
            selected.append(
                min(
                    (row for row in candidates if row["primary_gap"] == gap),
                    key=lambda row: row["task_id"],
                )
            )
            if len(selected) == 2:
                break
        if len(selected) < 2:
            selected_ids = {row["case_id"] for row in selected}
            selected.extend(
                row for row in candidates if row["case_id"] not in selected_ids
            )
        representative.extend(selected[:2])
    for row in representative:
        divergence = row.get("first_divergence") or {}
        packet = canonical_by_id[row["case_id"]]
        corrective_mechanism = _unblind_trace_text(
            row.get("corrective_mechanism"),
            packet.get("trace_labels") or {},
        )
        lines.append(
            f"| {row['domain']}/{row['task_id']} | {row['primary_gap']} | "
            f"{divergence.get('base_b')}/{divergence.get('strong')} | "
            f"{_markdown_cell(corrective_mechanism)} |"
        )
    if not representative:
        lines.append("| — | — | — | — |")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Same task/trial trajectories stop being counterfactual after the first model action; "
            "offline labels are observational.",
            "- Causal wording is reserved for the directional one-step replay results.",
            "- Banking findings describe retrieval and shared-hard behavior, not a demonstrated "
            "Qwen3.5 capability.",
            "",
            "## Jobs",
            "",
        ]
    )
    job_logs = sorted(output_dir.glob("jobs/*/run_*.log"))
    if job_logs:
        for job_log in job_logs:
            relative = job_log.relative_to(output_dir)
            lines.append(f"- [{relative}]({relative})")
    else:
        lines.append("- No job log was found under this experiment directory.")

    (output_dir / "README.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return {
        "canonical_adjudicated": len(canonical_rows),
        "canonical_total": len(canonical_packets),
        "high_confidence_adjudicated": len(scopes["high_confidence"]),
        "replay_cases": len(replay_summary),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_enable_thinking(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze atomic capability gaps between completed tau2 evaluations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--base-a-summary", type=Path, default=DEFAULT_BASE_A_SUMMARY)
    prepare.add_argument("--base-b-summary", type=Path, default=DEFAULT_BASE_B_SUMMARY)
    prepare.add_argument("--strong-summary", type=Path, default=DEFAULT_STRONG_SUMMARY)
    prepare.add_argument(
        "--simulations-root", type=Path, default=DEFAULT_SIMULATIONS_ROOT
    )
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    prepare.add_argument("--tokenizer", type=Path, default=DEFAULT_JUDGE_TOKENIZER)
    prepare.add_argument("--context-tokens", type=int, default=32768)
    prepare.add_argument("--output-tokens", type=int, default=2048)
    prepare.add_argument("--expected-cells", type=int)
    prepare.add_argument("--expected-tasks", type=int)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--input", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--base-url", required=True)
    judge.add_argument("--model", required=True)
    judge.add_argument("--api-key", default="EMPTY")
    judge.add_argument("--max-tokens", type=int, default=2048)
    judge.add_argument("--concurrency", type=int, default=4)
    judge.add_argument("--primary-passes", type=int, default=1)
    judge.add_argument(
        "--enable-thinking", choices=("true", "false", "unset"), default="false"
    )
    judge.add_argument("--timeout", type=float, default=300.0)
    judge.add_argument("--case-packets", type=Path)
    judge.add_argument("--base-agent-queue", type=Path)
    judge.add_argument("--calibration-reviews", type=Path)
    judge.add_argument("--agent-queue-output", type=Path)

    select = subparsers.add_parser("select-replay")
    select.add_argument("--case-packets", type=Path, required=True)
    select.add_argument("--agent-queue", type=Path, required=True)
    select.add_argument("--qwen-reviews", type=Path, required=True)
    select.add_argument("--agent-reviews", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    select.add_argument("--per-domain", type=int, default=3)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--model-label", choices=("qwen3", "qwen35"), required=True)
    replay.add_argument("--served-model", required=True)
    replay.add_argument("--model-path", type=Path, required=True)
    replay.add_argument("--base-url", required=True)
    replay.add_argument("--api-key", default="EMPTY")
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--review-queue-output", type=Path, required=True)
    replay.add_argument("--max-tokens", type=int, required=True)
    replay.add_argument("--context-tokens", type=int, default=65536)
    replay.add_argument("--concurrency", type=int, default=4)
    replay.add_argument("--timeout", type=float, default=300.0)
    replay.add_argument("--retrieval-config", default="bm25")
    replay.add_argument(
        "--enable-thinking", choices=("true", "false", "unset"), default="unset"
    )

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--adjudicated-cases", type=Path, required=True)
    summarize.add_argument("--case-packets", type=Path, required=True)
    summarize.add_argument("--replay-qwen3", type=Path, required=True)
    summarize.add_argument("--replay-qwen35", type=Path, required=True)
    summarize.add_argument("--replay-adjudications", type=Path, required=True)
    summarize.add_argument("--inventory", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser


def _run_prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), trust_remote_code=True
    )
    inventory = prepare_analysis(
        base_a_summary=args.base_a_summary,
        base_b_summary=args.base_b_summary,
        strong_summary=args.strong_summary,
        simulations_root=args.simulations_root,
        output_dir=args.output_dir,
        token_counter=make_chat_token_counter(tokenizer, enable_thinking=False),
        context_tokens=args.context_tokens,
        output_tokens=args.output_tokens,
    )
    if args.expected_cells is not None and inventory["scenario_cells"] != args.expected_cells:
        raise SystemExit(
            f"expected {args.expected_cells} scenario cells, got {inventory['scenario_cells']}"
        )
    if args.expected_tasks is not None and inventory["tasks"] != args.expected_tasks:
        raise SystemExit(f"expected {args.expected_tasks} tasks, got {inventory['tasks']}")
    print(json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True))


def _run_judge(args: argparse.Namespace) -> None:
    payloads = _read_jsonl(args.input)
    enable_thinking = _parse_enable_thinking(args.enable_thinking)

    def complete(messages, response_format):
        request_body = {
            "model": args.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "response_format": response_format,
        }
        if enable_thinking is not None:
            request_body["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking
            }
        message = request_chat_completion(
            args.base_url,
            request_body,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        content = message.get("content")
        return content if isinstance(content, str) else json.dumps(content)

    reviews = run_judge_reviews(
        payloads,
        complete=complete,
        primary_passes=args.primary_passes,
        concurrency=args.concurrency,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output, reviews)

    queue_args = (
        args.case_packets,
        args.base_agent_queue,
        args.calibration_reviews,
        args.agent_queue_output,
    )
    if any(queue_args) and not all(queue_args):
        raise SystemExit(
            "case-packets, base-agent-queue, calibration-reviews, and "
            "agent-queue-output must be supplied together"
        )
    if all(queue_args):
        queue = extend_agent_review_queue(
            base_queue=_read_jsonl(args.base_agent_queue),
            packets=_read_jsonl(args.case_packets),
            qwen_reviews=reviews,
            calibration_reviews=_read_jsonl(args.calibration_reviews),
        )
        _write_jsonl(args.agent_queue_output, queue)
    print(
        json.dumps(
            {
                "reviews": len(reviews),
                "valid": sum(row["status"] == "valid" for row in reviews),
                "agent_review": sum(
                    row["status"] == "agent_review" for row in reviews
                ),
            },
            sort_keys=True,
        )
    )


def _run_select_replay(args: argparse.Namespace) -> None:
    canonical_packets = _read_jsonl(args.case_packets)
    agent_queue = _read_jsonl(args.agent_queue)
    packet_by_case = {packet["case_id"]: packet for packet in canonical_packets}
    for packet in agent_queue:
        packet_by_case.setdefault(packet["case_id"], packet)
    packets = list(packet_by_case.values())
    adjudicated, missing = merge_adjudications(
        packets,
        _read_jsonl(args.qwen_reviews),
        agent_reviews=_read_jsonl(args.agent_reviews),
        mandatory_agent_case_ids={packet["case_id"] for packet in agent_queue},
    )
    selected = select_replay_cases(adjudicated, per_domain=args.per_domain)
    manifest = build_replay_manifest(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "adjudicated_cases.jsonl", adjudicated)
    _write_jsonl(args.output_dir / "selected_replay_cases.jsonl", selected)
    _write_jsonl(args.output_dir / "replay_manifest.jsonl", manifest)
    (args.output_dir / "missing_adjudications.json").write_text(
        json.dumps({"case_ids": missing}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "adjudicated": len(adjudicated),
                "missing": len(missing),
                "selected_replay_cases": len(selected),
                "replay_requests": len(manifest),
            },
            sort_keys=True,
        )
    )


def _run_replay(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    manifest = _read_jsonl(args.manifest)
    model_rows = [row for row in manifest if row["model"] == args.model_label]
    domains = sorted({row["domain"] for row in model_rows})
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True
    )
    schemas = load_official_tool_schemas(
        domains,
        retrieval_config=args.retrieval_config,
    )
    records = run_replays(
        manifest,
        model_label=args.model_label,
        served_model=args.served_model,
        tool_schemas_by_domain=schemas,
        max_tokens=args.max_tokens,
        enable_thinking=_parse_enable_thinking(args.enable_thinking),
        request_fits=make_replay_request_fitter(
            tokenizer, context_tokens=args.context_tokens
        ),
        complete=lambda request: request_chat_completion(
            args.base_url,
            request,
            api_key=args.api_key,
            timeout=args.timeout,
        ),
        concurrency=args.concurrency,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output, records)
    _write_jsonl(args.review_queue_output, build_replay_review_queue(records))
    print(
        json.dumps(
            {
                "records": len(records),
                "generated": sum(row["status"] == "generated" for row in records),
                "over_context": sum(
                    row["status"] == "over_context" for row in records
                ),
                "generation_error": sum(
                    row["status"] == "generation_error" for row in records
                ),
            },
            sort_keys=True,
        )
    )


def _run_summarize(args: argparse.Namespace) -> None:
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    result = write_formal_report(
        adjudicated_cases=_read_jsonl(args.adjudicated_cases),
        canonical_packets=_read_jsonl(args.case_packets),
        replay_records=(
            _read_jsonl(args.replay_qwen3) + _read_jsonl(args.replay_qwen35)
        ),
        replay_decisions=_read_jsonl(args.replay_adjudications),
        inventory=inventory,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        _run_prepare(args)
    elif args.command == "judge":
        _run_judge(args)
    elif args.command == "select-replay":
        _run_select_replay(args)
    elif args.command == "replay":
        _run_replay(args)
    elif args.command == "summarize":
        _run_summarize(args)


if __name__ == "__main__":
    main()
