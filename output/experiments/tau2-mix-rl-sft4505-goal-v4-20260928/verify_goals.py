"""Verify restored evidence conditions and reference success on prepared Banking tasks."""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
RL = ROOT / "examples/tau2-bench/rl"
SOURCE = ROOT / "output/experiments/tau2-areal-async-rl/dependencies/tau2/src"
sys.path[:0] = [str(ROOT), str(RL), str(SOURCE), str(RL.parent / "analysis")]
os.environ.setdefault("TAU2_DATA_DIR", str(ROOT.parent / "tau2-bench/data"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from loguru import logger
logger.remove()
import goal_progress_v4 as goal
import progress
import reward
from tau2.data_model.message import AssistantMessage, UserMessage, ToolCall


def initialize(constructor, task):
    env = constructor()
    initial = task.initial_state
    env.set_state(initialization_data=initial.initialization_data if initial else None,
                  initialization_actions=initial.initialization_actions if initial else None,
                  message_history=(initial.message_history or []) if initial else [])
    return env


def main():
    goal.setup_worker()
    directory = Path(__file__).resolve().parent
    data = [json.loads(line)["metadata"] for line in (directory / "data/train.jsonl").open()]
    started = time.monotonic()
    rows = []
    with (directory / "banking_reference_checks.jsonl").open("w") as output:
        for metadata in data:
            if metadata["domain"] != "banking":
                continue
            task = goal.task_from_metadata(metadata)
            constructor = goal.make_environment_constructor(domain="banking", db_path=Path(metadata["db_path"]), task=task)
            env = initialize(constructor, task)
            messages = list(task.initial_state.message_history or []) if task.initial_state else []
            policy_has_sources = True
            if task.retrieval_variant == "golden_retrieval":
                for doc_id in task.required_documents:
                    doc = json.loads((goal.envs.BANKING_DOCUMENTS / f"{doc_id}.json").read_text())
                    policy_has_sources &= doc["content"] in env.get_policy()
            errors = []
            for index, action in enumerate(task.evaluation_criteria.actions or []):
                call = ToolCall(id=f"call_{index}", name=action.name, requestor=action.requestor, arguments=action.arguments)
                role = AssistantMessage if action.requestor == "assistant" else UserMessage
                messages.append(role(role=action.requestor, tool_calls=[call]))
                response = env.get_response(call)
                messages.append(response)
                if response.error or (response.content or "").lstrip().startswith(("Error:", "Failed to ")):
                    errors.append(dict(tool=action.name, content=response.content))
            if task.evaluation_criteria.communicate_info:
                messages.append(AssistantMessage(role="assistant", content="\n".join(task.evaluation_criteria.communicate_info)))
            simulation = SimpleNamespace(messages=messages, mode="half_duplex", termination_reason="user_stop")
            score = reward.evaluate_simulation_with_constructor(simulation=simulation, task=task, domain="banking", environment_constructor=constructor)
            potential = goal.StatePotential(task, constructor)
            potential.messages = messages
            value = potential.score(env)
            row = dict(task=task.id, evidence_mode=task.retrieval_variant, reward=score.reward, final_potential=value,
                       potential_error=potential.reason, declared_sources_present=policy_has_sources, errors=errors)
            rows.append(row)
            output.write(json.dumps(row) + "\n")
            output.flush()
            if len(rows) % 100 == 0:
                print(len(rows), "Banking tasks", round(time.monotonic() - started, 1), "seconds", flush=True)

    metadata = next(m for m in data if m["task"]["id"] == "retail_79")
    task = goal.task_from_metadata(metadata)
    constructor = goal.make_environment_constructor(domain="retail", db_path=Path(metadata["db_path"]), task=task)
    potential = goal.StatePotential(task, constructor)
    partial = []
    for count in range(4):
        env = initialize(constructor, task)
        for action in task.evaluation_criteria.actions:
            if action.name != "modify_pending_order_items":
                continue
            if count:
                args = {**action.arguments, "item_ids": action.arguments["item_ids"][:count], "new_item_ids": action.arguments["new_item_ids"][:count]}
                env.make_tool_call(action.name, requestor=action.requestor, **args)
                env.sync_tools()
        snapshot = progress.database_snapshot(env)
        partial.append(dict(modified_items=count, old_potential=-progress.tree_field_distance(snapshot, potential.target),
                            new_potential=potential.score(env)))
    summary = dict(tasks=len(rows), modes=dict(Counter(row["evidence_mode"] for row in rows)),
                   failed=[row for row in rows if row["reward"] != 1 or row["final_potential"] != 0 or row["errors"] or not row["declared_sources_present"]],
                   retail_partial_completion=partial, seconds=round(time.monotonic() - started, 1))
    (directory / "reference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
