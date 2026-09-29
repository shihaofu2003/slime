"""Measure reward invariance under documented JSON argument representations.

This is an offline diagnostic of v2, with no model calls or job submission.
Only fields documented as encoded JSON are decoded; ordinary strings and IDs
are unchanged. Numeric variants also enter those encoded argument objects.
"""

import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from audit_reference_states import (
    ROOT, alternate_numbers, initialized, make_environment_constructor, task_from_metadata,
)

import reward
import reward_v2
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage


JSON_ARGUMENT_TOOLS = {
    "call_discoverable_agent_tool", "call_discoverable_user_tool", "give_discoverable_user_tool",
}


def transform(action, mode):
    arguments = dict(action.arguments)
    if action.name not in JSON_ARGUMENT_TOOLS or "arguments" not in arguments:
        return arguments
    decoded = json.loads(arguments["arguments"])
    if mode == "json_format":
        arguments["arguments"] = json.dumps(decoded, sort_keys=True, indent=1)
    else:
        alternate = alternate_numbers(decoded)
        if json.dumps(alternate) != json.dumps(decoded):
            arguments["arguments"] = json.dumps(alternate)
    return arguments


def replay(constructor, task, arguments):
    env = initialized(constructor, task)
    messages = list(task.initial_state.message_history or []) if task.initial_state else []
    errors = []
    for index, (action, values) in enumerate(zip(task.evaluation_criteria.actions or [], arguments)):
        call = ToolCall(id=f"call_{index}", name=action.name, requestor=action.requestor, arguments=values)
        role = AssistantMessage if action.requestor == "assistant" else UserMessage
        messages.append(role(role=action.requestor, tool_calls=[call]))
        response = env.get_response(call)
        messages.append(response)
        if response.error:
            errors.append(dict(action=action.name, message=response.content))
    if task.evaluation_criteria.communicate_info:
        messages.append(AssistantMessage(role="assistant", content="\n".join(task.evaluation_criteria.communicate_info)))
    result = reward.evaluate_simulation_with_constructor(
        simulation=SimpleNamespace(messages=messages, mode="half_duplex", termination_reason="user_stop"),
        task=task, domain="banking", environment_constructor=constructor,
    )
    return reward_v2.database_snapshot(env), dict(
        reward=result.reward, db_match=result.db_check.db_match,
        actions_match=all(check.action_match for check in result.action_checks), tool_errors=errors,
    )


def main():
    reward_v2.setup_worker()
    directory = Path(__file__).resolve().parent
    source = ROOT / "output/experiments/tau2-async-db-count-four-domain/data/train.jsonl"
    counts = {mode: Counter() for mode in ("json_format", "numeric_inside_json")}
    started = time.monotonic()
    with source.open() as data, (directory / "representation_equivalence.jsonl").open("w") as output:
        for line in data:
            metadata = json.loads(line)["metadata"]
            if metadata["domain"] != "banking":
                continue
            task = task_from_metadata(metadata)
            actions = task.evaluation_criteria.actions or []
            if not any(action.name in JSON_ARGUMENT_TOOLS for action in actions):
                continue
            variants = {
                mode: [transform(action, mode) for action in actions]
                for mode in counts
            }
            original = [action.arguments for action in actions]
            variants = {mode: arguments for mode, arguments in variants.items() if arguments != original}
            if not variants:
                continue
            constructor = make_environment_constructor(
                domain=metadata["domain"], db_path=Path(metadata["db_path"]), task=task,
            )
            baseline_state, baseline = replay(constructor, task, original)
            for mode, arguments in variants.items():
                state, result = replay(constructor, task, arguments)
                row = dict(
                    task=task.id, domain=metadata["domain"], mode=mode,
                    basis=[value.value for value in task.evaluation_criteria.reward_basis],
                    baseline=baseline, alternate=result, same_db_state=state == baseline_state,
                )
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                count = counts[mode]
                count["tasks"] += 1
                count["same_db_state"] += row["same_db_state"]
                count["action_match_changed"] += baseline["actions_match"] != result["actions_match"]
                count["reward_changed"] += baseline["reward"] != result["reward"]
                count["one_to_zero"] += baseline["reward"] == 1 and result["reward"] == 0
                count["tasks_with_tool_errors"] += bool(baseline["tool_errors"] or result["tool_errors"])
            completed = sum(count["tasks"] for count in counts.values())
            if completed % 25 == 0:
                print(completed, "pairs", round(time.monotonic() - started, 1), "seconds", flush=True)
    summary = dict(counts=counts, seconds=round(time.monotonic() - started, 1))
    (directory / "representation_equivalence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
