#!/usr/bin/env python3
"""Validate synthetic Banking schemas, reference actions, and evaluator reward."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from banking_synthetic.constants import (  # type: ignore
        ACTION_SUPPORT_DOCUMENTS,
        DATA_ORIGIN,
        DOCUMENT_FAMILY_PREFIXES,
        FORM_ACTION_BOUNDS,
        FORM_CASE_KINDS,
        FORM_DOCUMENT_BOUNDS,
        LOW_FORM_CATEGORIES,
        TASK_ID_PREFIX,
    )
    from banking_synthetic.documents import empty_db, read_json, read_jsonl, write_json  # type: ignore
else:
    from .constants import (
        ACTION_SUPPORT_DOCUMENTS,
        DATA_ORIGIN,
        DOCUMENT_FAMILY_PREFIXES,
        FORM_ACTION_BOUNDS,
        FORM_CASE_KINDS,
        FORM_DOCUMENT_BOUNDS,
        LOW_FORM_CATEGORIES,
        TASK_ID_PREFIX,
    )
    from .documents import empty_db, read_json, read_jsonl, write_json


def _valid_date(value: Any, date_format: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, date_format)
    except ValueError:
        return False
    return True


def static_errors(
    tasks: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    allowed_document_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    task_by_id = {task.get("id"): task for task in tasks}
    contract_by_id = {contract.get("task_id"): contract for contract in contracts}
    spec_by_scenario = {spec.get("scenario_id"): spec for spec in specs}
    if len(task_by_id) != len(tasks):
        errors.append("task ids are not unique")
    if len(contract_by_id) != len(contracts):
        errors.append("contract task ids are not unique")
    if len(spec_by_scenario) != len(specs):
        errors.append("scenario ids are not unique")
    if set(task_by_id) != set(contract_by_id):
        errors.append("task and contract ids differ")

    scenario_splits: dict[str, set[str]] = {}
    scenario_counts: Counter[str] = Counter()
    for task_id, contract in contract_by_id.items():
        task = task_by_id.get(task_id)
        if task is None:
            continue
        if not str(task_id).startswith(TASK_ID_PREFIX):
            errors.append(f"{task_id}: invalid task id prefix")
        if contract.get("data_origin") != DATA_ORIGIN:
            errors.append(f"{task_id}: invalid data origin")
        if "source_task_id" in contract:
            errors.append(f"{task_id}: source_task_id is prohibited")
        scenario_id = contract.get("scenario_id")
        scenario_counts[scenario_id] += 1
        scenario_splits.setdefault(scenario_id, set()).add(contract.get("split"))
        spec = spec_by_scenario.get(scenario_id)
        if spec is None:
            errors.append(f"{task_id}: missing scenario spec")
            continue
        if spec.get("data_origin") != DATA_ORIGIN:
            errors.append(f"{task_id}: spec has invalid data origin")
        if spec.get("split") != contract.get("split"):
            errors.append(f"{task_id}: split differs between spec and contract")
        if task.get("required_documents") != contract.get("required_documents"):
            errors.append(f"{task_id}: task/contract documents differ")
        task_initialization = (
            (((task.get("initial_state") or {}).get("initialization_data") or {}).get("agent_data"))
            or {}
        )
        if task_initialization != spec.get("initialization_data"):
            errors.append(f"{task_id}: task/spec initialization data differ")
        if task.get("user_tools") != contract.get("allowed_user_tools"):
            errors.append(f"{task_id}: task/contract user-tools differ")

        documents = task.get("required_documents") or []
        form = contract.get("task_form")
        if contract.get("case_kind") not in FORM_CASE_KINDS.get(form, ()):
            errors.append(f"{task_id}: case kind is incompatible with {form}")
        if (
            form in {"l1_evidence", "l2_retrieval"}
            and contract.get("business_category") not in LOW_FORM_CATEGORIES
        ):
            errors.append(f"{task_id}: low form is assigned to a category without an allowlisted document pool")
        min_docs, max_docs = FORM_DOCUMENT_BOUNDS.get(form, (-1, -1))
        if not min_docs <= len(documents) <= max_docs:
            errors.append(f"{task_id}: document count is outside {form} bounds")
        if not set(documents) <= allowed_document_ids:
            errors.append(f"{task_id}: document is outside allowlist")
        category_prefixes = DOCUMENT_FAMILY_PREFIXES.get(
            contract.get("business_category"), ()
        )
        if any(
            not document_id.startswith(category_prefixes)
            for document_id in documents
        ):
            errors.append(f"{task_id}: document is outside its business-category family")
        if form not in {"l1_evidence", "l2_retrieval", "l3_decision"}:
            support_documents = ACTION_SUPPORT_DOCUMENTS.get(
                contract.get("business_category"), ()
            )
            required_support = support_documents[: min(max_docs, len(support_documents))]
            if tuple(documents[: len(required_support)]) != required_support:
                errors.append(
                    f"{task_id}: action task lacks its ordered policy support documents"
                )

        actions = ((task.get("evaluation_criteria") or {}).get("actions") or [])
        graph_actions = [
            node.get("action")
            for node in contract.get("reference_action_graph") or []
        ]
        if actions != graph_actions:
            errors.append(f"{task_id}: task actions differ from the contract graph")
        min_actions, max_actions = FORM_ACTION_BOUNDS.get(form, (-1, -1))
        if not min_actions <= len(actions) <= max_actions:
            errors.append(f"{task_id}: action count is outside {form} bounds")
        action_ids = [action.get("action_id") for action in actions]
        if len(action_ids) != len(set(action_ids)):
            errors.append(f"{task_id}: action ids are not unique")
        allowed_user_tools = set(task.get("user_tools") or [])
        used_user_tools = {
            action["name"] for action in actions if action.get("requestor") == "user"
        }
        if not used_user_tools <= allowed_user_tools:
            errors.append(f"{task_id}: user action exceeds task user-tools boundary")
        instructions = str((task.get("user_scenario") or {}).get("instructions") or "")
        message_history = (task.get("initial_state") or {}).get("message_history") or []
        if (
            len(message_history) != 1
            or message_history[0].get("role") != "user"
            or not str(message_history[0].get("content") or "").strip()
        ):
            errors.append(f"{task_id}: task must have one visible initial User message")
            visible_opening = ""
        else:
            visible_opening = str(message_history[0]["content"])
        for marker_pattern in (
            r"\[Policy title: [^\]]+\]",
            r"<<Policy claim: .*?>>",
            r"<<Evidence locator: .*?>>",
            r"<<Requested outcome: .*?>>",
            r"<<Response requirement: .*?>>",
            r"<<Action clue: .*?>>",
            r"<<Case precondition: .*?>>",
        ):
            private_markers = set(re.findall(marker_pattern, instructions))
            visible_markers = set(re.findall(marker_pattern, visible_opening))
            if private_markers - visible_markers:
                errors.append(
                    f"{task_id}: initial User message omits a required visible marker"
                )
                break
        if used_user_tools and "<<User action protocol:" not in instructions:
            errors.append(f"{task_id}: user action lacks a private user-tool protocol")
        for action in actions:
            if action.get("requestor") != "user":
                continue
            if f"`{action['name']}`" not in instructions:
                errors.append(
                    f"{task_id}: private user-tool protocol omits {action['name']}"
                )
        action_clues = re.findall(r"<<Action clue: .*?>>", visible_opening)
        for action in actions:
            arguments = action.get("arguments") or {}
            if action.get("name") == "call_discoverable_agent_tool":
                label = arguments.get("agent_tool_name")
                encoded_arguments = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                visible = any(
                    "`call_discoverable_agent_tool`" in clue
                    and encoded_arguments in clue
                    for clue in action_clues
                )
            elif action.get("name") == "call_discoverable_user_tool":
                label = arguments.get("discoverable_tool_name")
                visible = any(f"`{label}`" in clue for clue in action_clues)
            elif action.get("requestor") == "user":
                label = action.get("name")
                visible = any(f"`{label}`" in clue for clue in action_clues)
            else:
                continue
            if not visible:
                errors.append(
                    f"{task_id}: action label {label} is not visible in the initial User message"
                )
        criteria = task.get("evaluation_criteria") or {}
        reward_basis = set(criteria.get("reward_basis") or [])
        if "DB" in reward_basis and not contract.get("expected_db_changes"):
            errors.append(f"{task_id}: DB-scored task has no expected state changes")
        if "COMMUNICATE" in reward_basis and not contract.get(
            "final_response_facts"
        ):
            errors.append(f"{task_id}: communication-scored task has no final facts")
        if criteria.get("communicate_info") != (
            contract.get("final_response_facts") or None
        ):
            errors.append(f"{task_id}: task/contract final response facts differ")

        init = (((task.get("initial_state") or {}).get("initialization_data") or {}).get("agent_data") or {})
        preverified = (
            form in {"l3_single_action", "l4_two_skill"}
            and bool(contract.get("expected_db_changes"))
            and contract.get("expected_db_changes")
            != [
                {
                    "table": "human_transfer_requests",
                    "scope": "synthetic scenario entities",
                }
            ]
        )
        if preverified and "<<Case precondition:" not in visible_opening:
            errors.append(f"{task_id}: preverified task omits its visible case precondition")
        if preverified and not ((init.get("verification_history") or {}).get("data") or {}):
            errors.append(f"{task_id}: preverified task lacks its initial verification record")
        users = (init.get("users") or {}).get("data") or {}
        accounts = (init.get("accounts") or {}).get("data") or {}
        cards = (init.get("debit_cards") or {}).get("data") or {}
        cc_accounts = (init.get("credit_card_accounts") or {}).get("data") or {}
        credit_transactions = (
            (init.get("credit_card_transaction_history") or {}).get("data") or {}
        )
        bank_transactions = (
            (init.get("bank_account_transaction_history") or {}).get("data") or {}
        )
        payments = (init.get("payment_history") or {}).get("data") or {}
        currency = re.compile(r"^-?\$\d+\.\d{2}$")
        decimal_amount = re.compile(r"^-?\d+\.\d{2}$")
        for user_id, user in users.items():
            if not _valid_date(user.get("date_of_birth"), "%m/%d/%Y"):
                errors.append(f"{task_id}: user {user_id} has invalid date_of_birth")
        for account_id, account in accounts.items():
            if account.get("user_id") not in users:
                errors.append(f"{task_id}: account {account_id} has missing user foreign key")
            if not _valid_date(account.get("date_opened"), "%m/%d/%Y"):
                errors.append(f"{task_id}: account {account_id} has invalid date_opened")
            if not decimal_amount.fullmatch(str(account.get("current_holdings") or "")):
                errors.append(f"{task_id}: account {account_id} has invalid holdings amount")
        for card_id, card in cards.items():
            if card.get("user_id") not in users or card.get("account_id") not in accounts:
                errors.append(f"{task_id}: debit card {card_id} has missing foreign key")
            if not _valid_date(card.get("expiration_date"), "%m/%y"):
                errors.append(f"{task_id}: debit card {card_id} has invalid expiration date")
        for account_id, account in cc_accounts.items():
            if account.get("user_id") not in users:
                errors.append(f"{task_id}: credit account {account_id} has missing user foreign key")
            for field in ("credit_limit", "current_balance"):
                if not currency.fullmatch(str(account.get(field) or "")):
                    errors.append(
                        f"{task_id}: credit account {account_id} has invalid {field} amount"
                    )
        for transaction_id, transaction in credit_transactions.items():
            if (
                transaction.get("user_id") not in users
                or transaction.get("credit_card_account_id") not in cc_accounts
            ):
                errors.append(
                    f"{task_id}: credit transaction {transaction_id} has missing foreign key"
                )
            if not _valid_date(transaction.get("transaction_date"), "%m/%d/%Y"):
                errors.append(
                    f"{task_id}: credit transaction {transaction_id} has invalid date"
                )
            if not currency.fullmatch(str(transaction.get("transaction_amount") or "")):
                errors.append(
                    f"{task_id}: credit transaction {transaction_id} has invalid amount"
                )
        for transaction_id, transaction in bank_transactions.items():
            if (
                transaction.get("user_id") not in users
                or transaction.get("account_id") not in accounts
                or transaction.get("card_id") not in cards
            ):
                errors.append(
                    f"{task_id}: bank transaction {transaction_id} has missing foreign key"
                )
            if not _valid_date(transaction.get("date"), "%m/%d/%Y"):
                errors.append(f"{task_id}: bank transaction {transaction_id} has invalid date")
            amount = transaction.get("amount")
            if not isinstance(amount, (int, float)) or not math.isfinite(amount):
                errors.append(f"{task_id}: bank transaction {transaction_id} has invalid amount")
        for payment_id, payment in payments.items():
            if payment.get("credit_card_account_id") not in cc_accounts:
                errors.append(f"{task_id}: payment {payment_id} has missing foreign key")
            if not _valid_date(payment.get("payment_date"), "%m/%d/%Y"):
                errors.append(f"{task_id}: payment {payment_id} has invalid date")
            if not currency.fullmatch(str(payment.get("amount") or "")):
                errors.append(f"{task_id}: payment {payment_id} has invalid amount")

    for scenario_id, count in scenario_counts.items():
        if count > 4:
            errors.append(f"{scenario_id}: contains more than four curriculum tasks")
    for scenario_id, splits in scenario_splits.items():
        if len(splits) != 1:
            errors.append(f"{scenario_id}: crosses splits")
    return errors


@lru_cache(maxsize=None)
def _load_knowledge_base(
    documents_dir: str, allowed_document_ids: tuple[str, ...]
) -> Any:
    from tau2.domains.banking_knowledge.data_model import Document, KnowledgeBase

    root = Path(documents_dir)
    documents = {
        document_id: Document.model_validate(
            read_json(root / f"{document_id}.json")
        )
        for document_id in allowed_document_ids
    }
    return KnowledgeBase(documents=documents)


def environment_constructor(
    documents_dir: Path,
    allowed_document_ids: tuple[str, ...],
    retrieval_variant: str,
    task: Any,
) -> Callable[..., Any]:
    from tau2.domains.banking_knowledge.data_model import TransactionalDB
    from tau2.domains.banking_knowledge.retrieval import build_policy, build_tools, resolve_variant
    from tau2.domains.banking_knowledge.tools import KnowledgeUserTools
    from tau2.environment.environment import Environment

    if len(allowed_document_ids) != 480 or len(set(allowed_document_ids)) != 480:
        raise ValueError("environment requires exactly 480 unique allowlisted documents")
    knowledge_base = _load_knowledge_base(
        str(documents_dir), tuple(sorted(allowed_document_ids))
    )

    def constructor(solo_mode: bool = False, **_: Any) -> Any:
        db = TransactionalDB.model_validate(empty_db())
        variant = resolve_variant(retrieval_variant)
        tools = build_tools(variant, db, knowledge_base)
        user_tools = KnowledgeUserTools(db)
        policy = build_policy(variant, knowledge_base, task)
        return Environment(
            domain_name="banking_knowledge",
            policy=policy,
            tools=tools,
            user_tools=user_tools,
            solo_mode=solo_mode,
        )

    return constructor


def replay_task(
    task_data: dict[str, Any],
    contract: dict[str, Any],
    documents_dir: Path,
    allowed_document_ids: tuple[str, ...],
) -> dict[str, Any]:
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.data_model.tasks import RewardType, Task
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = Task.model_validate(task_data)
    constructor = environment_constructor(
        documents_dir,
        allowed_document_ids,
        contract["retrieval_variant"],
        task,
    )
    live = constructor()
    initial = task.initial_state
    live.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=list(initial.message_history or []) if initial else [],
    )
    messages: list[Any] = list(initial.message_history or []) if initial else []
    outputs = []
    bm25_hits: list[bool] = []
    search_index = 0
    for index, action in enumerate(task.evaluation_criteria.actions or []):
        call = ToolCall(
            id=f"reference_{index:02d}",
            name=action.name,
            arguments=copy.deepcopy(action.arguments),
            requestor=action.requestor,
        )
        participant = AssistantMessage if action.requestor == "assistant" else UserMessage
        messages.append(participant(role=action.requestor, content=None, tool_calls=[call]))
        response = live.get_response(call)
        messages.append(response)
        content = response.content or ""
        outputs.append(content)
        if response.error or content.lstrip().startswith(("Error:", "Failed to ")):
            raise ValueError(f"reference action {action.name} failed: {content[:400]}")
        if action.name == "KB_search":
            target_ids = task.required_documents or []
            if search_index >= len(target_ids):
                raise ValueError("reference action graph has more searches than documents")
            ranked_ids = re.findall(r"\bID:\s+([^\s]+)", content)
            bm25_hits.append(target_ids[search_index] in ranked_ids[:10])
            search_index += 1

    final_text = "\n".join(task.evaluation_criteria.communicate_info or []) or "Task completed."
    messages.append(AssistantMessage.text(final_text))

    env_result = EnvironmentEvaluator.calculate_reward(
        environment_constructor=constructor,
        task=task,
        full_trajectory=messages,
        solo_mode=False,
    )
    action_result = ActionEvaluator.calculate_reward(task=task, full_trajectory=messages)
    communicate_result = CommunicateEvaluator.calculate_reward(task=task, full_trajectory=messages)
    component_rewards = {
        RewardType.DB: float((env_result.reward_breakdown or {}).get(RewardType.DB, 1.0)),
        RewardType.ENV_ASSERTION: float((env_result.reward_breakdown or {}).get(RewardType.ENV_ASSERTION, 1.0)),
        RewardType.ACTION: float(action_result.reward),
        RewardType.COMMUNICATE: float(communicate_result.reward),
    }
    reward = 1.0
    for basis in task.evaluation_criteria.reward_basis:
        reward *= component_rewards[basis]
    if reward != 1.0:
        raise ValueError(f"reference evaluator reward is {reward}: {component_rewards}")
    if bm25_hits and not all(bm25_hits):
        raise ValueError("reference BM25 query did not retrieve the target document in top 10")
    return {
        "task_id": task.id,
        "reward": reward,
        "reward_basis": [basis.value for basis in task.evaluation_criteria.reward_basis],
        "action_count": len(task.evaluation_criteria.actions or []),
        "bm25_queries": len(bm25_hits),
        "bm25_top10_hits": sum(bm25_hits),
    }


def validate_environment(
    tasks: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    documents_dir: Path,
    allowed_document_ids: tuple[str, ...],
    workers: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    contract_by_id = {contract["task_id"]: contract for contract in contracts}
    work = [
        (task, contract_by_id[task["id"]], documents_dir, allowed_document_ids)
        for task in tasks
    ]
    errors: list[str] = []
    results: list[dict[str, Any]] = []
    if workers == 1:
        completed = map(_replay_item, work)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        completed = executor.map(_replay_item, work, chunksize=4)
    try:
        for index, (error, result) in enumerate(completed, 1):
            if error:
                errors.append(error)
            elif result:
                results.append(result)
            if index % 100 == 0:
                print(f"validated {index}/{len(tasks)}", flush=True)
    finally:
        if workers != 1:
            executor.shutdown()
    return sorted(errors), results


def _replay_item(
    item: tuple[dict[str, Any], dict[str, Any], Path, tuple[str, ...]]
) -> tuple[str | None, dict[str, Any] | None]:
    task, contract, documents_dir, allowed_document_ids = item
    try:
        from loguru import logger

        logger.disable("tau2")
        return None, replay_task(
            task, contract, documents_dir, allowed_document_ids
        )
    except Exception as exc:
        return f"{task['id']}: {type(exc).__name__}: {exc}", None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--environment", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Reference replay worker processes (default: up to 32).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = read_json(args.tasks)
    specs = read_jsonl(args.specs)
    contracts = read_jsonl(args.contracts)
    allowed_document_ids = tuple(
        read_json(args.allowlist)["allowed_document_ids"]
    )
    allowed = set(allowed_document_ids)
    errors = static_errors(tasks, specs, contracts, allowed)
    static_passed = not errors
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    replay_results: list[dict[str, Any]] = []
    if args.environment and not errors:
        replay_errors, replay_results = validate_environment(
            tasks,
            contracts,
            args.documents_dir,
            allowed_document_ids,
            args.workers,
        )
        errors.extend(replay_errors)
    report = {
        "data_origin": DATA_ORIGIN,
        "task_count": len(tasks),
        "scenario_count": len({contract.get("scenario_id") for contract in contracts}),
        "schema_pass_count": len(tasks) if static_passed else 0,
        "reference_replay_count": len(replay_results),
        "reference_reward_one_count": sum(result["reward"] == 1.0 for result in replay_results),
        "bm25_query_count": sum(result["bm25_queries"] for result in replay_results),
        "bm25_top10_hit_count": sum(result["bm25_top10_hits"] for result in replay_results),
        "error_count": len(errors),
    }
    write_json(args.output, report)
    write_json(args.failures, errors)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
