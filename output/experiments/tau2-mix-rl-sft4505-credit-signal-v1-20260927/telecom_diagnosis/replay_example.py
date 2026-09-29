"""Replay one recorded Telecom trajectory and inspect reference state fields.

No inference, training changes or database digests are used. The environment
modules are loaded without Tau2's optional runner/voice package initializer.
"""

import json
import os
import sys
import types
from pathlib import Path

from loguru import logger

logger.remove()

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
SOURCE = ROOT / "output/experiments/tau2-areal-async-rl/dependencies/tau2/src/tau2"
package = types.ModuleType("tau2")
package.__path__ = [str(SOURCE)]
sys.modules["tau2"] = package
os.environ["TAU2_DATA_DIR"] = str(ROOT.parent / "tau2-bench/data")

from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage, SystemMessage
from tau2.data_model.tasks import Task
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.environment import get_environment


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            result.extend(differences(left.get(key), right.get(key), path + "/" + str(key)))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [difference for i, (a, b) in enumerate(zip(left, right))
                for difference in differences(a, b, path + "/" + str(i))]
    return [] if left == right else [dict(field=path, reference=left, actual=right)]


def snapshot(env):
    return dict(assistant=env.tools.db.model_dump(mode="json"), user=env.user_tools.db.model_dump(mode="json"))


def main():
    trace = OUT.parent / "arms/async/20260927_credit-signal-v1-train/trajectories/train.jsonl"
    with trace.open("rb") as stream:
        row = next(json.loads(line) for line in stream if line.startswith(b'{"sample_index": 115,'))
    task = Task.model_validate(row["metadata"]["task"])
    initial = task.initial_state
    message_types = dict(assistant=AssistantMessage, user=UserMessage, tool=ToolMessage, system=SystemMessage)
    messages = [message_types[m["role"]].model_validate(m) for m in row["simulation"]["messages"]]

    def initialized(history):
        env = get_environment(db=TelecomDB.load(row["metadata"]["db_path"]))
        env.set_state(initialization_data=initial.initialization_data,
                      initialization_actions=initial.initialization_actions, message_history=history)
        return env

    predicted = initialized(messages)
    target = initialized(initial.message_history or [])
    for action in task.evaluation_criteria.actions:
        target.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
    actual = snapshot(predicted)
    before = differences(snapshot(target), actual)
    target.sync_tools()
    after = differences(snapshot(target), actual)
    result = dict(task=task.id, group=row["group_index"], sample=row["sample_index"],
                  recorded_reward=row["reward"],
                  actual_assertions=[predicted.run_env_assertion(a, raise_assertion_error=False)
                                     for a in task.evaluation_criteria.env_assertions],
                  differences_with_current_reference=before,
                  differences_after_final_reference_sync=after)
    affected = []
    for line in (OUT.parent / "data/train.jsonl").open():
        metadata = json.loads(line)["metadata"]
        if metadata["domain"] != "telecom" or "reward_basis" in metadata["task"]["evaluation_criteria"]:
            continue
        candidate = Task.model_validate(metadata["task"])
        state = candidate.initial_state
        env = get_environment(db=TelecomDB.load(metadata["db_path"]))
        env.set_state(initialization_data=state.initialization_data if state else None,
                      initialization_actions=state.initialization_actions if state else None,
                      message_history=(state.message_history or []) if state else [])
        for action in candidate.evaluation_criteria.actions:
            env.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
        unsynced = snapshot(env)
        env.sync_tools()
        changed = differences(unsynced, snapshot(env))
        if changed:
            affected.append(dict(task=candidate.id, changed=changed))
    result["default_db_reference_tasks_changed_by_final_sync"] = affected
    (OUT / "reference_state_example.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
