#!/usr/bin/env python3
"""Replay simplified Banking traces and compare their final state to source."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/tau2-bench/analysis"))
from banking_synthetic.validate import environment_constructor
from simplify_banking_expert import SOURCE, document_bodies, read_rows


def matching_response(call_name, expected, actual):
    if bool(expected.error) != bool(actual.error):
        return False
    if call_name == "KB_search":
        expected_docs = document_bodies(expected.content or "")
        if expected_docs is not None:
            return expected_docs == document_bodies(actual.content or "")
    try:
        return json.loads(expected.content) == json.loads(actual.content)
    except (json.JSONDecodeError, TypeError):
        return expected.content == actual.content


def replay_candidate(task, constructor, source, candidate):
    """Replay *all* retained calls at their prefix, then recompute official bases.

    Unlike Environment.set_state, this also checks reads. This matters when a
    removed call supplied evidence or enabled a later discoverable tool.
    """
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    types = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    source = [types[m["role"]].model_validate(m) for m in source if m["role"] in types]
    candidate = [types[m["role"]].model_validate(m) for m in candidate if m["role"] in types]
    initial = task.initial_state
    initialization = {
        "initialization_data": initial.initialization_data if initial else None,
        "initialization_actions": initial.initialization_actions if initial else None,
    }
    original_env = constructor()
    original_env.set_state(**initialization, message_history=source)
    live = constructor()
    live.set_state(**initialization, message_history=list(initial.message_history or []) if initial else [])
    calls_checked = 0
    searches_checked = 0
    index = 0
    while index < len(candidate):
        message = candidate[index]
        if isinstance(message, ToolMessage):
            raise ValueError(f"Unpaired candidate result at message {index}")
        calls = getattr(message, "tool_calls", None) or []
        results = candidate[index + 1:index + len(calls) + 1]
        if len(results) != len(calls):
            raise ValueError("Candidate is missing tool results")
        for call, saved in zip(calls, results):
            if not isinstance(saved, ToolMessage) or saved.id != call.id:
                raise ValueError(f"Candidate result pairing changed for {call.id}")
            actual = live.get_response(call)
            if not matching_response(call.name, saved, actual):
                raise ValueError(f"Retained call result changed: {call.id} {call.name}; actual={str(actual.content)[:700]}")
            calls_checked += 1
            searches_checked += call.name == "KB_search"
        index += len(calls) + 1
    for owner in ("tools", "user_tools"):
        if getattr(original_env, owner).db.model_dump(mode="json") != getattr(live, owner).db.model_dump(mode="json"):
            raise ValueError(f"Final {owner} database changed")

    scores = []
    for messages in (source, candidate):
        env = EnvironmentEvaluator.calculate_reward(environment_constructor=constructor, task=task, full_trajectory=messages)
        components = {str(getattr(k, "value", k)): float(v) for k, v in (env.reward_breakdown or {}).items()}
        components["ACTION"] = float(ActionEvaluator.calculate_reward(task=task, full_trajectory=messages).reward)
        components["COMMUNICATE"] = float(CommunicateEvaluator.calculate_reward(task=task, full_trajectory=messages).reward)
        required = {str(getattr(basis, "value", basis)): components[str(getattr(basis, "value", basis))]
                    for basis in task.evaluation_criteria.reward_basis}
        scores.append(required)
    if any(value != 1 for value in scores[0].values()):
        raise ValueError(f"Original no longer passes required official bases: {scores[0]}")
    if scores[1] != scores[0]:
        raise ValueError(f"Required official scores changed: {scores[0]} -> {scores[1]}")
    return {"passed": True, "required_scores": scores[1], "verified_calls": calls_checked,
            "verified_search_results": searches_checked, "database_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--check-search-results", action="store_true", help="Also execute retained KB queries and compare their document bodies to saved results")
    args = parser.parse_args()
    from loguru import logger
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.data_model.tasks import Task
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    source_ids = {row["id"] for row in read_rows(args.data_dir / "simplified_trajectories.jsonl")}
    originals = {row["id"]: row for row in read_rows(SOURCE) if row["id"] in source_ids}
    tasks = {task["id"]: Task.model_validate(task) for task in json.loads((SOURCE.parent / "train_tasks.json").read_text())}
    allowed = tuple(json.loads((SOURCE.parent.parent / "allowed_documents.json").read_text())["allowed_document_ids"])
    message_types = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    stats = Counter()
    failures = []
    for row in read_rows(args.data_dir / "simplified_trajectories.jsonl"):
        task = tasks[row["task_id"]]
        constructor = environment_constructor(ROOT.parent / "tau2-bench/data/tau2/domains/banking_knowledge/documents", allowed, row["retrieval_variant"], task)
        original = originals[row["id"]]
        candidate_messages = [message_types[m["role"]].model_validate(m) for m in row["messages"] if m["role"] in message_types]
        source_messages = [message_types[m["role"]].model_validate(m) for m in original["messages"] if m["role"] in message_types]
        stats["trajectories"] += 1
        try:
            # Deleted searches are read-only; compare actual DB contents rather
            # than introducing a new digest/provenance scheme.
            states = []
            for messages in (source_messages, candidate_messages):
                env = constructor()
                initial = task.initial_state
                env.set_state(initialization_data=initial.initialization_data if initial else None,
                              initialization_actions=initial.initialization_actions if initial else None,
                              message_history=messages)
                states.append((env.tools.db.model_dump(mode="json"), env.user_tools.db.model_dump(mode="json")))
            if states[0] != states[1]:
                raise ValueError("Final Agent/User database changed")
            if args.check_search_results:
                search_env = constructor()
                saved = {m.id: m for m in candidate_messages if isinstance(m, ToolMessage)}
                for message in candidate_messages:
                    for call in getattr(message, "tool_calls", None) or []:
                        if call.name == "KB_search":
                            actual = search_env.get_response(call)
                            expected_docs = document_bodies(saved[call.id].content or "")
                            if not expected_docs or document_bodies(actual.content or "") != expected_docs:
                                raise ValueError(f"Search documents changed for {call.id}")
                            stats["verified_search_results"] += 1
            for evaluator, basis in ((ActionEvaluator, "ACTION"), (CommunicateEvaluator, "COMMUNICATE")):
                before = evaluator.calculate_reward(task=task, full_trajectory=source_messages)
                after = evaluator.calculate_reward(task=task, full_trajectory=candidate_messages)
                if before.reward != after.reward:
                    raise ValueError(f"{basis} score changed: {before.reward} -> {after.reward}")
                if basis in task.evaluation_criteria.reward_basis and after.reward != 1:
                    raise ValueError(f"Required {basis} score is no longer one")
            stats["passed"] += 1
        except (ValueError, KeyError, TypeError) as exc:
            failures.append({"id": row["id"], "task_id": row["task_id"], "reason": str(exc)[:1000]})
        if stats["trajectories"] % 50 == 0:
            print(dict(stats), flush=True)
    report = {**dict(stats), "failures": failures}
    (args.data_dir / "replay_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
