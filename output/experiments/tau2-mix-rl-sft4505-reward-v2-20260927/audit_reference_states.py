"""Offline reference-action and numeric-equivalence audit of all training tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
RL = ROOT / "examples/tau2-bench/rl"
SOURCE = ROOT / "output/experiments/tau2-areal-async-rl/dependencies/tau2/src/tau2"
# Load actual domain implementations without optional runner/voice imports.
package = types.ModuleType("tau2")
package.__path__ = [str(SOURCE)]
sys.modules["tau2"] = package
sys.path[:0] = [str(ROOT), str(RL), str(ROOT / "examples/tau2-bench/analysis")]
os.environ["TAU2_DATA_DIR"] = str(ROOT.parent / "tau2-bench/data")
logger.remove()

from envs import make_environment_constructor, task_from_metadata
from tau2.environment.environment import Environment


def snapshot(environment):
    result = {"assistant": environment.tools.db.model_dump(mode="json")}
    if environment.user_tools is not None:
        result["user"] = environment.user_tools.db.model_dump(mode="json")
    return result


def alternate_numbers(value):
    if isinstance(value, dict):
        return {k: alternate_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [alternate_numbers(v) for v in value]
    if type(value) is int:
        return float(value)
    if type(value) is float and value.is_integer():
        return int(value)
    return value


def fields_different(left, right, path="", numeric_types=False):
    if isinstance(left, dict) and isinstance(right, dict):
        return [difference for key in left.keys() | right.keys()
                for difference in fields_different(left.get(key), right.get(key), path + "/" + str(key), numeric_types)]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [difference for i, (a, b) in enumerate(zip(left, right))
                for difference in fields_different(a, b, path + "/" + str(i), numeric_types)]
    if numeric_types and type(left) is not type(right) and type(left) in (int, float) and type(right) in (int, float) and left == right:
        return [{"field": path, "left": left, "right": right, "numeric_equivalent": True}]
    return [] if left == right else [{"field": path, "left": left, "right": right}]


def initialized(constructor, task):
    env = constructor()
    state = task.initial_state
    env.set_state(initialization_data=state.initialization_data if state else None,
                  initialization_actions=state.initialization_actions if state else None,
                  message_history=(state.message_history or []) if state else [])
    return env


def audit(metadata):
    task = task_from_metadata(metadata)
    constructor = make_environment_constructor(domain=metadata["domain"], db_path=Path(metadata["db_path"]), task=task)
    actions = task.evaluation_criteria.actions or []
    row = dict(domain=metadata["domain"], task=task.id,
               basis=[b.value for b in task.evaluation_criteria.reward_basis], actions=len(actions),
               reference_error=None, sync_differences=[], numeric_differences=[], numeric_error=None)
    try:
        env = initialized(constructor, task)
        for action in actions:
            env.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
        final = snapshot(env)
        if type(env).sync_tools is not Environment.sync_tools:
            env.sync_tools()
            row["sync_differences"] = fields_different(final, snapshot(env))
        del env
    except Exception as error:
        row["reference_error"] = f"{type(error).__name__}: {error}"
        return row
    # One equivalent representation for actual numeric arguments; no random
    # inputs or changes to numeric strings, IDs, dates or boolean arguments.
    alternate = [alternate_numbers(a.arguments) for a in actions]
    if any(json.dumps(a.arguments) != json.dumps(b) for a, b in zip(actions, alternate)):
        try:
            env = initialized(constructor, task)
            for action, arguments in zip(actions, alternate):
                env.make_tool_call(action.name, requestor=action.requestor, **arguments)
            row["numeric_differences"] = fields_different(final, snapshot(env), numeric_types=True)
        except Exception as error:
            row["numeric_error"] = f"{type(error).__name__}: {error}"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "output/experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/data/train.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reference_audit.jsonl"))
    parser.add_argument("--domain", action="append")
    args = parser.parse_args()
    started = time.monotonic()
    counts = Counter()
    with args.data.open() as data, args.output.open("w") as output:
        for line in data:
            metadata = json.loads(line)["metadata"]
            if args.domain and metadata["domain"] not in args.domain:
                continue
            result = audit(metadata)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            counts[metadata["domain"]] += 1
            if sum(counts.values()) % 100 == 0:
                print(dict(counts), round(time.monotonic() - started, 1), "seconds", flush=True)
    print("done", dict(counts), round(time.monotonic() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
