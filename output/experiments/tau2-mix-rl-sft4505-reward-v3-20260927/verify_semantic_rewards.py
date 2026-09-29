"""Replay all four-domain tasks and equivalent argument representations on CPU."""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "output/experiments/tau2-mix-rl-sft4505-reward-v2-20260927"
sys.path.insert(0, str(PREVIOUS))
from audit_reference_states import alternate_numbers, initialized, make_environment_constructor, task_from_metadata
from check_representation_equivalence import transform

import progress
import reward
import reward_v2
import reward_v3
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage


def replay(constructor, task, arguments, potential, domain):
    env = initialized(constructor, task)
    messages = list(task.initial_state.message_history or []) if task.initial_state else []
    scores = [potential.score(env)]
    errors = []
    for index, (action, values) in enumerate(zip(task.evaluation_criteria.actions or [], arguments)):
        call = ToolCall(id=f"call_{index}", name=action.name, requestor=action.requestor, arguments=values)
        role = AssistantMessage if action.requestor == "assistant" else UserMessage
        messages.append(role(role=action.requestor, tool_calls=[call]))
        response = env.get_response(call)
        messages.append(response)
        scores.append(potential.score(env))
        if response.error:
            errors.append(dict(action=action.name, message=response.content))
    if task.evaluation_criteria.communicate_info:
        messages.append(AssistantMessage(role="assistant", content="\n".join(task.evaluation_criteria.communicate_info)))
    result = reward.evaluate_simulation_with_constructor(
        simulation=SimpleNamespace(messages=messages, mode="half_duplex", termination_reason="user_stop"),
        task=task, domain=domain, environment_constructor=constructor,
    )
    return reward_v2.database_snapshot(env), dict(
        reward=result.reward, db_match=result.db_check.db_match,
        actions_match=all(check.action_match for check in result.action_checks or []),
        progress_scores=scores, progress_reason=potential.reason, tool_errors=errors,
    )


def check(metadata):
    task = task_from_metadata(metadata)
    domain = metadata["domain"]
    constructor = make_environment_constructor(domain=domain, db_path=Path(metadata["db_path"]), task=task)
    actions = task.evaluation_criteria.actions or []
    original = [action.arguments for action in actions]
    variants = {"numeric_arguments": [alternate_numbers(arguments) for arguments in original]}
    if any(action.name in reward_v3.JSON_ARGUMENT_TOOLS for action in actions):
        variants.update({mode: [transform(action, mode) for action in actions]
                         for mode in ("json_format", "numeric_inside_json")})
    variants = {mode: args for mode, args in variants.items() if json.dumps(args) != json.dumps(original)}
    potential = progress.StatePotential(task, constructor)
    state, baseline = replay(constructor, task, original, potential, domain)
    checks = []
    for mode, arguments in variants.items():
        alternate_state, result = replay(constructor, task, arguments, potential, domain)
        checks.append(dict(mode=mode, same_db_state=state == alternate_state,
                           same_progress=baseline["progress_scores"] == result["progress_scores"],
                           same_reward=baseline["reward"] == result["reward"], alternate=result))
    return dict(task=task.id, domain=domain, basis=[b.value for b in task.evaluation_criteria.reward_basis],
                baseline=baseline, variants=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    source = ROOT / "output/experiments/tau2-async-db-count-four-domain/data/train.jsonl"
    metadata = [json.loads(line)["metadata"] for line in source.open()]
    counts, variants, failures = Counter(), {}, []
    baseline_rewards = Counter()
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=reward_v3.setup_worker) as pool:
        with (directory / "semantic_verification.jsonl").open("w") as output:
            for row in pool.map(check, metadata, chunksize=4):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                counts[row["domain"]] += 1
                baseline_rewards[str(row["baseline"]["reward"])] += 1
                if not row["baseline"]["db_match"] or row["baseline"]["tool_errors"]:
                    failures.append(dict(task=row["task"], mode="reference"))
                for variant in row["variants"]:
                    count = variants.setdefault(variant["mode"], Counter())
                    count["tasks"] += 1
                    for key in ("same_db_state", "same_progress", "same_reward"):
                        count[key] += variant[key]
                    result = variant["alternate"]
                    count["actions_match"] += result["actions_match"]
                    if not all(variant[key] for key in ("same_db_state", "same_progress", "same_reward")) or result["tool_errors"]:
                        failures.append(dict(task=row["task"], mode=variant["mode"]))
                if sum(counts.values()) % 100 == 0:
                    print(dict(counts), round(time.monotonic() - started, 1), "seconds", flush=True)
    summary = dict(tasks=counts, variants=variants, baseline_rewards=baseline_rewards,
                   failures=failures, seconds=round(time.monotonic() - started, 1))
    (directory / "semantic_verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
