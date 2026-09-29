"""Manual CPU benchmark in the tau2 job environment; no model generation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter

from envs import make_environment_constructor, task_from_metadata
from progress import StatePotential, database_snapshot, tree_field_distance


def old_flatten(value, path=()):
    if isinstance(value, dict) and value:
        return {key: leaf for name, child in value.items()
                for key, leaf in old_flatten(child, path + (name,)).items()}
    return {path: value}


def old_distance(current, target):
    missing = object()
    return sum(current.get(key, missing) != target.get(key, missing)
               for key in current.keys() | target.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    data = Path(__file__).resolve().parents[3].parent / "datasets/tau2/rl/areal_tasks_slime.jsonl"
    fixtures = {}
    for line in data.open():
        metadata = json.loads(line)["metadata"]
        task = task_from_metadata(metadata)
        if "DB" in task.evaluation_criteria.reward_basis:
            fixtures.setdefault(metadata["domain"], (metadata, task))
        if len(fixtures) == 3:
            break
    results = []
    for domain, (metadata, task) in fixtures.items():
        constructor = make_environment_constructor(domain=domain, db_path=Path(metadata["db_path"]))
        potential = StatePotential(task, constructor)
        assert potential.reason is None, potential.reason
        target = potential.target
        old_target = old_flatten(target)
        env = constructor()
        initial = task.initial_state
        env.set_state(initialization_data=initial.initialization_data if initial else None,
                      initialization_actions=initial.initialization_actions if initial else None,
                      message_history=(initial.message_history or []) if initial else [])

        def old_score(_):
            return -old_distance(old_flatten(database_snapshot(env)), old_target)

        def new_score(_):
            return -tree_field_distance(database_snapshot(env), target)

        for step in range(len(task.evaluation_criteria.actions or []) + 1):
            assert old_score(None) == new_score(None)
            if step < len(task.evaluation_criteria.actions or []):
                action = task.evaluation_criteria.actions[step]
                env.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
        # Benchmark the initial (nonterminal) state, where many fields differ.
        env = constructor()
        env.set_state(initialization_data=initial.initialization_data if initial else None,
                      initialization_actions=initial.initialization_actions if initial else None,
                      message_history=(initial.message_history or []) if initial else [])
        row = {"domain": domain, "task_id": task.id,
               "gold_boundaries_checked": len(task.evaluation_criteria.actions or []) + 1}
        key = (domain, str(Path(metadata["db_path"]).resolve()))
        for name in ("cold", "warm"):
            started = perf_counter()
            cached = StatePotential(task, constructor, reference_key=key)
            row[f"reference_{name}_seconds"] = perf_counter() - started
            assert cached.target == target
        for workers in (1, args.workers):
            count = args.repeats * workers
            for name, score in (("old", old_score), ("new", new_score)):
                started = perf_counter()
                if workers == 1:
                    values = [score(i) for i in range(count)]
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        values = list(pool.map(score, range(count)))
                row[f"{name}_{workers}_workers_seconds"] = perf_counter() - started
                expected = new_score(None)
                assert all(value == expected for value in values)
            row[f"speedup_{workers}_workers"] = (
                row[f"old_{workers}_workers_seconds"] / row[f"new_{workers}_workers_seconds"])
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({"repeats": args.repeats, "workers": args.workers,
                                      "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
