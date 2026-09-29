"""Score fixed v1 Telecom actions with corrected references, without retraining.

Live tool implementations stay at v1. The changes assessed here are reference
synchronization, structured DB comparison and refueling-description equality.
Other declared reward components use their recorded values unchanged.
"""

import csv
import json
import re
import time
from pathlib import Path

from audit_reference_states import ROOT, make_environment_constructor, task_from_metadata
import reward_v2
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from tau2.data_model.tasks import RewardType


def main():
    directory = Path(__file__).resolve().parent
    run = ROOT / "output/experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/arms/async/20260927_credit-signal-v1-train"
    groups = {}
    for line in (run / "run.log").open(errors="replace"):
        match = re.search(r"tau2_credit_group ({.*})", line)
        if match:
            row = json.loads(match[1])
            groups[row["group"]] = row
    telecom = {key for key, row in groups.items() if row["domain"] == "telecom"}
    types = dict(assistant=AssistantMessage, user=UserMessage, tool=ToolMessage, system=SystemMessage)
    references, rows = {}, []
    started = time.monotonic()
    with (run / "trajectories/train.jsonl").open("rb") as source:
        for line in source:
            match = re.search(rb'"group_index": (\d+)', line[:200])
            if not match or int(match[1]) not in telecom:
                continue
            data = json.loads(line)
            metadata = data["metadata"]
            info = metadata["tau2_reward_info"]
            task = task_from_metadata(metadata)
            old = float(info["reward"])
            row = dict(group=data["group_index"], sample=data["sample_index"], task=task.id,
                       old_reward=old, corrected_reward=old, error="",
                       basis="+".join(b.value for b in task.evaluation_criteria.reward_basis))
            simulation = data.get("simulation")
            if (simulation and simulation["termination_reason"] in {"user_stop", "agent_stop"}
                    and RewardType.DB in task.evaluation_criteria.reward_basis):
                try:
                    constructor = make_environment_constructor(domain="telecom", db_path=Path(metadata["db_path"]), task=task)
                    if task.id not in references:
                        references[task.id] = reward_v2.reference_database(task, constructor)
                    messages = []
                    for message in simulation["messages"]:
                        message.pop("raw_data", None)
                        messages.append(types[message["role"]].model_validate(message))
                    predicted = constructor()
                    initial = task.initial_state
                    predicted.set_state(initialization_data=initial.initialization_data if initial else None,
                                        initialization_actions=initial.initialization_actions if initial else None,
                                        message_history=messages)
                    match = reward_v2.database_snapshot(predicted) == references[task.id]
                    value = float(match)
                    for name, component in info["reward_breakdown"].items():
                        if name != "DB":
                            value *= component
                    row["corrected_reward"] = value
                except Exception as error:
                    row["corrected_reward"] = None
                    row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            if len(rows) % 500 == 0:
                print(len(rows), "Telecom trajectories", round(time.monotonic() - started, 1), "seconds", flush=True)
    with (directory / "telecom_counterfactual.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("complete", len(rows), "trajectories", len(telecom), "groups", round(time.monotonic() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
