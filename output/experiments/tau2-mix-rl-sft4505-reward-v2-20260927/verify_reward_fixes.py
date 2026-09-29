"""Recheck every affected task with v2, using actual tool execution and replay."""

import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from audit_reference_states import ROOT, alternate_numbers, initialized, make_environment_constructor, task_from_metadata

import progress
import reward
import reward_v2
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.tasks import RewardType


def main():
    directory = Path(__file__).resolve().parent
    audit = [json.loads(line) for line in (directory / "reference_audit.jsonl").open()]
    affected = {r["task"] for r in audit if r["sync_differences"] or r["numeric_differences"]}
    # Include an unaffected example in every domain, including Retail.
    for domain in ("airline", "retail", "telecom", "banking"):
        affected.add(next(r["task"] for r in audit if r["domain"] == domain))
    reward_v2.setup_worker()
    counts, failed = Counter(), []
    started = time.monotonic()
    source = ROOT / "output/experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/data/train.jsonl"
    with source.open() as data, (directory / "verification.jsonl").open("w") as output:
        for line in data:
            metadata = json.loads(line)["metadata"]
            if metadata["task"]["id"] not in affected:
                continue
            task = task_from_metadata(metadata)
            constructor = make_environment_constructor(domain=metadata["domain"], db_path=Path(metadata["db_path"]), task=task)
            env = initialized(constructor, task)
            messages = list(task.initial_state.message_history or []) if task.initial_state else []
            for i, action in enumerate(task.evaluation_criteria.actions or []):
                call = ToolCall(id=f"call_{i}", name=action.name, requestor=action.requestor,
                                arguments=alternate_numbers(action.arguments))
                role = AssistantMessage if action.requestor == "assistant" else UserMessage
                messages.append(role(role=action.requestor, tool_calls=[call]))
                response = env.get_response(call)
                if response.error:
                    raise RuntimeError(f"{task.id}: {response.content}")
                messages.append(response)
            if task.evaluation_criteria.communicate_info:
                messages.append(AssistantMessage(role="assistant", content="\n".join(task.evaluation_criteria.communicate_info)))
            simulation = SimpleNamespace(messages=messages, mode="half_duplex", termination_reason="user_stop")
            result = reward.evaluate_simulation_with_constructor(
                simulation=simulation, task=task, domain=metadata["domain"], environment_constructor=constructor)
            potential = progress.StatePotential(task, constructor)
            score = potential.score(env)
            requires_db = RewardType.DB in task.evaluation_criteria.reward_basis
            okay = bool(result.db_check.db_match) and (not requires_db or score == 0.0)
            row = dict(domain=metadata["domain"], task=task.id, equivalent_db_match=result.db_check.db_match,
                       reward=result.reward, progress=score, progress_reason=potential.reason, check_passed=okay)
            output.write(json.dumps(row) + "\n")
            output.flush()
            counts[metadata["domain"]] += 1
            if not okay:
                failed.append(row)
            if sum(counts.values()) % 100 == 0:
                print(dict(counts), round(time.monotonic() - started, 1), "seconds", flush=True)
    print(json.dumps(dict(checked=dict(counts), failures=failed), indent=2))
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
