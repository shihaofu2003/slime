#!/usr/bin/env python3
"""CPU tests for record-level DB change counting."""

from copy import deepcopy
import pytest

from progress import database_records, field_distance


def test_record_changes_count_once_per_record_and_restoration_cancels():
    initial = {"assistant": {"orders": {
        "one": {"status": "pending", "items": [1, 2]},
        "two": {"status": "pending"},
    }}, "user": {"device": {"wifi": False}}}
    current = deepcopy(initial)
    reference = database_records(initial)
    current["assistant"]["orders"]["one"]["status"] = "done"
    current["assistant"]["orders"]["one"]["items"].append(3)
    assert field_distance(database_records(current), reference) == 1
    current["user"]["device"]["wifi"] = True
    assert field_distance(database_records(current), reference) == 2
    current["assistant"]["orders"]["one"] = deepcopy(initial["assistant"]["orders"]["one"])
    current["user"]["device"]["wifi"] = False
    assert field_distance(database_records(current), reference) == 0


def test_telecom_list_rows_are_keyed_by_record_id():
    initial = {"assistant": {"lines": [
        {"line_id": "a", "status": "active"},
        {"line_id": "b", "status": "active"},
    ]}}
    current = deepcopy(initial)
    current["assistant"]["lines"].reverse()
    assert field_distance(database_records(current), database_records(initial)) == 0
    current["assistant"]["lines"][0]["status"] = "closed"
    assert field_distance(database_records(current), database_records(initial)) == 1



@pytest.mark.parametrize("side", ["assistant", "user"])
def test_banking_counts_records_on_each_side(side):
    initial = {"assistant": {"accounts": {"data": {"a": {"balance": 5, "status": "open"}, "b": {"balance": 7}}, "notes": ""}},
               "user": {"accounts": {"data": {"a": {"balance": 5, "status": "open"}, "b": {"balance": 7}}, "notes": ""}}}
    current = deepcopy(initial)
    reference = database_records(initial)
    rows = current[side]["accounts"]["data"]
    rows["a"].update(balance=2, status="closed")
    assert field_distance(database_records(current), reference) == 1
    del rows["b"]
    rows["c"] = {"balance": 10}
    assert field_distance(database_records(current), reference) == 3
    current[side] = deepcopy(initial[side])
    assert field_distance(database_records(current), reference) == 0


@pytest.mark.parametrize("basis", ["DB", "ACTION", "COMMUNICATE"])
def test_real_banking_environment(basis):
    import json
    from pathlib import Path
    from envs import make_environment_constructor, task_from_metadata
    from progress import StatePotential, database_snapshot
    root = Path(__file__).resolve().parents[4]
    tasks = json.loads((root / "datasets/tau2/rl/banking_independent_synthetic/tasks/train.json").read_text())
    task = next(task_from_metadata({"task": t}) for t in tasks if basis in t["evaluation_criteria"]["reward_basis"])
    constructor = make_environment_constructor(domain="banking_knowledge", db_path=Path(__file__).parent.parent / "analysis/banking_synthetic/assets/empty_db.json", task=task)
    potential = StatePotential(task, constructor)
    env = constructor()
    initial = task.initial_state
    env.set_state(initialization_data=initial.initialization_data if initial else None,
                  initialization_actions=initial.initialization_actions if initial else None,
                  message_history=(initial.message_history or []) if initial else [])
    before = database_snapshot(env)
    assert database_records(before) is not None
    if basis == "DB":
        assert potential.reason is None
        for action in task.evaluation_criteria.actions or []:
            env.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
        assert potential.score(env) == 0
    else:
        assert potential.score(env) is None
    fresh = constructor()
    fresh.set_state(initialization_data=initial.initialization_data if initial else None,
                    initialization_actions=initial.initialization_actions if initial else None,
                    message_history=(initial.message_history or []) if initial else [])
    assert database_snapshot(fresh) == before



def test_fixed_four_domain_pool_preserves_task_database_metadata():
    import json
    from collections import Counter
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    original = [json.loads(line) for line in (root / "datasets/tau2/rl/areal_tasks_slime.jsonl").open()]
    prepared = [json.loads(line) for line in (Path(__file__).resolve().parents[3] /
        "output/experiments/tau2-async-db-count-four-domain/data/train.jsonl").open()]
    assert len(prepared) == 2524
    assert Counter(row["metadata"]["domain"] for row in prepared) == {
        "airline": 1148, "retail": 563, "telecom": 271, "banking": 542}
    for source, row in zip(original, prepared[:1982], strict=True):
        assert row["metadata"]["db_path"] == source["metadata"]["db_path"]
        assert row["metadata"]["task"] == source["metadata"]["task"]
    banking = json.loads((root / "datasets/tau2/rl/banking_independent_synthetic/tasks/train.json").read_text())
    assert [row["metadata"]["task"] for row in prepared[1982:]] == banking
    assert sum("DB" in task["evaluation_criteria"]["reward_basis"] for task in banking) == 338


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_cached_decisions_match_full_recomputation_after_writes(monkeypatch):
    from types import SimpleNamespace
    from progress import DBCountProgressOrchestrator, StatePotential, database_snapshot, tree_field_distance
    from tau2.orchestrator.orchestrator import Orchestrator, Role

    data = {"orders": {"a": {"value": 0}}}
    dumps = []
    def dump(**kwargs):
        dumps.append(1)
        return deepcopy(data)
    env = SimpleNamespace(tools=SimpleNamespace(db=SimpleNamespace(model_dump=dump)),
                          user_tools=None, _is_mutating_tool=lambda name: name != "read")
    initial = database_snapshot(env)
    potential = StatePotential.__new__(StatePotential)
    potential.reason, potential.assertions = None, None
    potential.seconds = 0.0
    potential.target = initial
    potential._score_snapshot = potential._snapshot_score = None
    obj = DBCountProgressOrchestrator.__new__(DBCountProgressOrchestrator)
    obj.environment, obj.progress_potential = env, potential
    obj.initial_records = database_records(initial)
    obj.progress_scores, obj.progress_messages = [], []
    obj.db_diff_counts, obj.db_diff_side_counts = [], []
    obj._reuse_snapshot, obj._current_snapshot = True, None
    operation = [lambda: None]
    monkeypatch.setattr(Orchestrator, "step", lambda self: operation[0]())

    def decision():
        obj.from_role, obj.to_role = Role.USER, Role.AGENT
        obj.message = SimpleNamespace()
        operation[0] = lambda: None
        obj.step()
        expected = {"assistant": deepcopy(data)}
        assert obj.progress_scores[-1] == -tree_field_distance(expected, initial)
        assert obj.db_diff_counts[-1] == field_distance(database_records(expected), database_records(initial))

    before = len(dumps)
    decision()
    decision()  # Plain conversation does not change the state.
    assert len(dumps) == before + 1
    for name, mutate in [
        ("read", lambda: None),
        ("write", lambda: data["orders"]["a"].update(value=1)),
        ("failed_write", lambda: data["orders"]["a"].update(value=2)),
        ("restore", lambda: data["orders"]["a"].update(value=0)),
        ("add", lambda: data["orders"].update(b={"value": 3})),
        ("delete", lambda: data["orders"].pop("b")),
    ]:
        obj.from_role, obj.to_role = Role.AGENT, Role.ENV
        obj.message = SimpleNamespace(tool_calls=[SimpleNamespace(name=name)])
        operation[0] = mutate
        count = len(dumps)
        obj.step()
        decision()
        decision()
        assert len(dumps) == count + (name != "read")


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_cached_database_bytes_preserve_full_state_and_isolate_mutations(domain):
    import json
    from pathlib import Path
    from envs import make_environment_constructor, FlightDB, RetailDB, TelecomDB

    data = Path(__file__).resolve().parents[4] / "datasets/tau2/rl/areal_tasks_slime.jsonl"
    with data.open() as stream:
        metadata = next(json.loads(line)["metadata"] for line in stream
                        if json.loads(line)["metadata"]["domain"] == domain)
    path = Path(metadata["db_path"])
    model = {"airline": FlightDB, "retail": RetailDB, "telecom": TelecomDB}[domain]
    expected = model.load(path).model_dump(mode="json")
    constructor = make_environment_constructor(domain=domain, db_path=path)
    first, second = constructor(), constructor()
    assert first.tools.db.model_dump(mode="json") == expected
    assert second.tools.db.model_dump(mode="json") == expected
    table = next(value for value in first.tools.db.__dict__.values() if isinstance(value, (dict, list)) and value)
    table.clear()
    assert second.tools.db.model_dump(mode="json") == expected
    assert constructor().tools.db.model_dump(mode="json") == expected
