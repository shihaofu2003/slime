#!/usr/bin/env python3
"""Expand Banking benchmark sources into fact/action diagnostic probes."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from banking_task_curriculum import (
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_TASKS_DIR,
    DEFAULT_TOOLS_FILE,
    EXPANSION_EXCLUDED_SOURCES,
    action_summary,
    action_to_initialization,
    build_annotation,
    category_lookup,
    contract,
    document_fact_locator,
    effective_action,
    load_documents,
    load_runtime_tasks,
    make_pilot_task,
    parse_tool_types,
    read_json,
    read_jsonl,
    resolved_operation_goal,
    search_action,
    user_tools_for_actions,
    validate_environment,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "output/experiments/tau2-banking-task-curriculum-scale-v2"
)


def action_capabilities(action: dict[str, Any]) -> list[str]:
    name = action["name"]
    if name == "log_verification":
        return ["身份验证", "参数构造"]
    if name == "unlock_discoverable_agent_tool":
        return ["工具发现", "动作选择"]
    if name in {"give_discoverable_user_tool", "call_discoverable_user_tool"}:
        return ["用户工具协调", "参数构造"]
    return ["动作选择", "参数构造"]


def retrieval_anchor(document: dict[str, Any], statement: str) -> str:
    """Return a short indexed-body clue distinct from the requested fact."""
    normalized_statement = " ".join(statement.lower().split())
    for line in document["content"].splitlines():
        clean = " ".join(re.sub(r"^[#|\-\s]+|[|\s]+$", "", line).split())
        if len(clean.split()) < 6 or clean.lower() == normalized_statement:
            continue
        return " ".join(clean.split()[:20])
    return statement


def build_scaled_curriculum(
    runtime_tasks: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = []
    contracts = []
    for source_id, source in sorted(runtime_tasks.items()):
        if source_id in EXPANSION_EXCLUDED_SOURCES:
            continue
        annotation = annotations[source_id]
        category = annotation["business"]["primary_category"]
        source_actions = list(
            (source.get("evaluation_criteria") or {}).get("actions") or []
        )
        source_reward_basis = set(
            (source.get("evaluation_criteria") or {}).get("reward_basis") or []
        )
        final_name, final_arguments = effective_action(source_actions[-1])
        operation_goal = resolved_operation_goal(final_name, final_arguments)

        for fact in annotation["evidence"]["required_facts"]:
            fact_id = fact["id"]
            document_id = fact["source_document_ids"][0]
            statement = fact["statement"]
            fact_label = f"Fact: {statement}"
            locator = document_fact_locator(documents[document_id], statement)
            anchor = retrieval_anchor(documents[document_id], statement)
            base_id = f"banking_scale_v2_{source_id}_{fact_id}"

            task = make_pilot_task(
                source,
                variant="evidence_given",
                instructions=(
                    f"For this resolved customer goal, {operation_goal}. "
                    f"Use the provided document `{document_id}` and copy {locator} verbatim. "
                    f"Return exactly `{fact_label}`. Do not search or perform a banking operation."
                ),
                required_documents=[document_id],
                actions=None,
                communicate_info=[fact_label],
                reward_basis=["COMMUNICATE"],
            )
            task["id"] = f"{base_id}_evidence_given"
            tasks.append(task)
            contracts.append(
                contract(
                    task,
                    source_id=source_id,
                    category=category,
                    variant="evidence_given",
                    level="L1",
                    retrieval_variant="golden_retrieval",
                    capabilities=["单文档事实抽取"],
                    given_context=["decisive_evidence"],
                    required_documents=[document_id],
                    reference_steps=[],
                    reference_messages=[fact_label],
                    expected_changed_tables=[],
                )
            )

            retrieval = search_action(
                f"{base_id}_retrieval_only",
                f"{documents[document_id]['title']} {statement} {anchor}",
            )
            document_label = f"Document: {document_id}"
            task = make_pilot_task(
                source,
                variant="retrieval_only",
                instructions=(
                    f"For this resolved customer goal, {operation_goal}. Search the Banking "
                    f"knowledge base for the policy topic `{documents[document_id]['title']}`. "
                    f"Use the indexed-body search anchor `{anchor}`. "
                    f"Return exactly `{document_label}` and do not perform a banking operation."
                ),
                required_documents=[document_id],
                actions=[retrieval],
                communicate_info=[document_label],
                reward_basis=["ACTION", "COMMUNICATE"],
            )
            task["id"] = f"{base_id}_retrieval_only"
            tasks.append(task)
            contracts.append(
                contract(
                    task,
                    source_id=source_id,
                    category=category,
                    variant="retrieval_only",
                    level="L2",
                    retrieval_variant="bm25",
                    capabilities=["查询构造", "文档相关性判断"],
                    given_context=["business_question"],
                    required_documents=[document_id],
                    reference_steps=[retrieval],
                    reference_messages=[document_label],
                    expected_changed_tables=[],
                )
            )

        for index, action in enumerate(source_actions):
            payload = json.dumps(action_summary(action), ensure_ascii=False, sort_keys=True)
            requestor = action.get("requestor", "assistant")
            if requestor == "user":
                instruction = (
                    "All preceding source-task steps are complete. Ask the customer to execute "
                    f"the one remaining customer-owned operation with this resolved payload: `{payload}`."
                )
            else:
                instruction = (
                    "All preceding source-task steps are complete. Execute the one remaining "
                    f"reference operation with this resolved payload: `{payload}`."
                )
            reward_basis = ["ACTION"]
            if index == len(source_actions) - 1 and "DB" in source_reward_basis:
                reward_basis.insert(0, "DB")
            task = make_pilot_task(
                source,
                variant="single_action",
                instructions=instruction,
                required_documents=[],
                actions=[copy.deepcopy(action)],
                communicate_info=None,
                reward_basis=reward_basis,
                initialization_actions=[
                    action_to_initialization(item) for item in source_actions[:index]
                ]
                or None,
                user_tools=user_tools_for_actions(source, [action]),
            )
            task["id"] = f"banking_scale_v2_{source_id}_action_{action['action_id']}"
            tasks.append(task)
            contracts.append(
                contract(
                    task,
                    source_id=source_id,
                    category=category,
                    variant="single_action",
                    level="L3",
                    retrieval_variant="golden_retrieval",
                    capabilities=action_capabilities(action),
                    given_context=["completed_prerequisites", "resolved_action"],
                    required_documents=[],
                    reference_steps=[copy.deepcopy(action)],
                    reference_messages=[],
                    expected_changed_tables=[],
                )
            )
    return tasks, contracts


def validate_scaled_curriculum(
    tasks: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    runtime_tasks: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    clean_sources = set(runtime_tasks) - set(EXPANSION_EXCLUDED_SOURCES)
    expected_facts = sum(
        len(annotations[source_id]["evidence"]["required_facts"])
        for source_id in clean_sources
    )
    expected_actions = sum(
        len((runtime_tasks[source_id].get("evaluation_criteria") or {}).get("actions") or [])
        for source_id in clean_sources
    )
    expected_tasks = 2 * expected_facts + expected_actions
    task_by_id = {task["id"]: task for task in tasks}
    contract_by_id = {item["task_id"]: item for item in contracts}
    errors = []
    if len(tasks) != expected_tasks or len(task_by_id) != expected_tasks:
        errors.append(
            f"expected {expected_tasks} unique scaled tasks, found {len(tasks)}/{len(task_by_id)}"
        )
    if len(contracts) != expected_tasks or set(task_by_id) != set(contract_by_id):
        errors.append("scaled task and contract ids differ or contain duplicates")

    categories = category_lookup()
    for task_id, task in task_by_id.items():
        item = contract_by_id[task_id]
        source_id = item["source_task_id"]
        if source_id not in clean_sources:
            errors.append(f"{task_id}: invalid source {source_id}")
            continue
        if item["business_category"] != categories[source_id]:
            errors.append(f"{task_id}: category differs from source inventory")
        if set(task.get("required_documents") or []) - set(documents):
            errors.append(f"{task_id}: references a missing document")
        message_history = (task.get("initial_state") or {}).get("message_history") or []
        if len(message_history) != 1 or not message_history[0].get("content"):
            errors.append(f"{task_id}: expected one fixed initial User request")
        criteria = task.get("evaluation_criteria") or {}
        if not criteria.get("reward_basis"):
            errors.append(f"{task_id}: empty reward basis")
        expected_user_tools = {
            action["name"]
            for action in criteria.get("actions") or []
            if action.get("requestor", "assistant") == "user"
        }
        if set(task.get("user_tools") or []) != expected_user_tools:
            errors.append(f"{task_id}: exposed User tools differ from evaluated actions")
        for fact in item["success_contract"]["required_response_facts"]:
            if fact in message_history[0]["content"]:
                errors.append(f"{task_id}: initial request leaks `{fact}`")

        variant = item["variant"]
        action_count = len(criteria.get("actions") or [])
        document_count = len(task.get("required_documents") or [])
        if variant == "evidence_given" and (document_count != 1 or action_count):
            errors.append(f"{task_id}: invalid evidence task boundary")
        elif variant == "retrieval_only" and (
            document_count != 1
            or action_count != 1
            or criteria["actions"][0]["name"] != "KB_search"
        ):
            errors.append(f"{task_id}: invalid retrieval task boundary")
        elif variant == "single_action" and (document_count or action_count != 1):
            errors.append(f"{task_id}: invalid single-action task boundary")

    return errors, {
        "source_tasks": len(clean_sources),
        "excluded_sources": EXPANSION_EXCLUDED_SOURCES,
        "scaled_tasks": len(tasks),
        "scaled_contracts": len(contracts),
        "tasks_by_variant": dict(Counter(item["variant"] for item in contracts)),
        "tasks_by_level": dict(Counter(item["level"] for item in contracts)),
        "tasks_by_category": dict(Counter(item["business_category"] for item in contracts)),
    }


def build_inputs(args: argparse.Namespace):
    runtime_tasks = load_runtime_tasks(args.tasks_dir)
    documents = load_documents(args.documents_dir)
    categories = category_lookup()
    tool_types = parse_tool_types(args.tools_file)
    annotations = {
        source_id: build_annotation(task, categories[source_id], documents, tool_types)
        for source_id, task in runtime_tasks.items()
    }
    return runtime_tasks, documents, annotations


def build(args: argparse.Namespace) -> dict[str, Any]:
    runtime_tasks, documents, annotations = build_inputs(args)
    tasks, contracts = build_scaled_curriculum(runtime_tasks, annotations, documents)
    errors, summary = validate_scaled_curriculum(
        tasks, contracts, runtime_tasks, annotations, documents
    )
    if errors:
        raise ValueError("\n".join(errors))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "scaled_tasks.json", tasks)
    write_jsonl(args.output_dir / "scaled_contracts.jsonl", contracts)
    report = {"status": "pass", "static": summary, "environment": None, "errors": []}
    write_json(args.output_dir / "scaled_validation_report.json", report)
    return report


def validate(args: argparse.Namespace) -> dict[str, Any]:
    runtime_tasks, documents, annotations = build_inputs(args)
    tasks = read_json(args.output_dir / "scaled_tasks.json")
    contracts = read_jsonl(args.output_dir / "scaled_contracts.jsonl")
    errors, summary = validate_scaled_curriculum(
        tasks, contracts, runtime_tasks, annotations, documents
    )
    environment_summary = None
    if args.environment:
        environment_errors, environment_summary = validate_environment(
            runtime_tasks, tasks, contracts, documents
        )
        errors.extend(environment_errors)
    report = {
        "status": "pass" if not errors else "fail",
        "static": summary,
        "environment": environment_summary,
        "errors": errors,
    }
    write_json(args.output_dir / "scaled_validation_report.json", report)
    if errors:
        raise ValueError("\n".join(errors))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
        subparser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
        subparser.add_argument("--tools-file", type=Path, default=DEFAULT_TOOLS_FILE)
        subparser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        if command == "validate":
            subparser.add_argument("--environment", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build(args) if args.command == "build" else validate(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
