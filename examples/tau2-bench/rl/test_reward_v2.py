"""Standalone CPU preflight for corrected reward/progress and worker routing."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[3]
RL = Path(__file__).resolve().parent
SOURCE = ROOT / "output/experiments/tau2-areal-async-rl/dependencies/tau2/src"
sys.path[:0] = [str(ROOT), str(RL), str(SOURCE), str(RL.parent / "analysis")]
os.environ.setdefault("TAU2_DATA_DIR", str(ROOT.parent / "tau2-bench/data"))

import progress
import reward
import reward_v2
from envs import make_environment_constructor, task_from_metadata
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import RewardType, Task


@pytest.fixture(scope="module")
def tasks():
    path = ROOT / "output/experiments/tau2-async-db-count-four-domain/data/train.jsonl"
    return [json.loads(line)["metadata"] for line in path.open()]


@pytest.fixture
def corrected(monkeypatch):
    from tau2.domains.banking_knowledge import tools as banking_tools
    from tau2.domains.retail.tools import RetailTools

    monkeypatch.setattr(reward, "EnvironmentEvaluator", reward_v2.EnvironmentEvaluator)
    monkeypatch.setattr(progress, "reference_database", reward_v2.reference_database)
    monkeypatch.setattr(progress, "database_snapshot", reward_v2.database_snapshot)
    monkeypatch.setattr(banking_tools, "generate_application_id", reward_v2.generate_application_id)
    monkeypatch.setattr(RetailTools, "modify_pending_order_items", reward_v2.modify_pending_order_items)
    progress.cached_reference_database.cache_clear()
    yield
    progress.cached_reference_database.cache_clear()


def constructor_and_task(metadata):
    task = task_from_metadata(metadata)
    return make_environment_constructor(domain=metadata["domain"], db_path=Path(metadata["db_path"]), task=task), task


def initialized(constructor, task, history=None):
    env = constructor()
    state = task.initial_state
    env.set_state(initialization_data=state.initialization_data if state else None,
                  initialization_actions=state.initialization_actions if state else None,
                  message_history=history if history is not None else ((state.message_history or []) if state else []))
    return env


def reference_simulation(constructor, task, alternate_numeric=False):
    env = initialized(constructor, task)
    messages = list(task.initial_state.message_history or []) if task.initial_state else []

    def alternate(value):
        if isinstance(value, dict):
            return {k: alternate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [alternate(v) for v in value]
        if type(value) is int:
            return float(value)
        if type(value) is float and value.is_integer():
            return int(value)
        return value

    for i, action in enumerate(task.evaluation_criteria.actions or []):
        call = ToolCall(id=f"call_{i}", name=action.name, requestor=action.requestor,
                        arguments=alternate(action.arguments) if alternate_numeric else action.arguments)
        message_class = AssistantMessage if action.requestor == "assistant" else UserMessage
        messages.append(message_class(role=action.requestor, tool_calls=[call]))
        response = env.get_response(call)
        assert not response.error, response.content
        messages.append(response)
    if task.evaluation_criteria.communicate_info:
        messages.append(AssistantMessage(role="assistant", content="\n".join(task.evaluation_criteria.communicate_info)))
    return SimpleNamespace(messages=messages, mode="half_duplex", termination_reason="user_stop"), env


@pytest.fixture
def recorded_telecom():
    data = json.loads((RL / "fixtures/reward_v2_telecom_118.json").read_text())
    constructor, task = constructor_and_task(data)
    simulation = SimulationRun.model_validate(data["simulation"])
    return constructor, task, simulation


def test_recorded_false_zero_now_succeeds_and_progress_reaches_target(corrected, recorded_telecom):
    constructor, task, simulation = recorded_telecom
    assert simulation.reward_info.reward == 0  # Actual historical observation.
    result = reward.evaluate_simulation_with_constructor(
        simulation=simulation, task=task, domain="telecom", environment_constructor=constructor)
    assert result.reward == 1 and result.db_check.db_match
    assert all(a.met for a in result.env_assertions)
    assert all(a.action_match for a in result.action_checks)
    assert result.reward_basis == task.evaluation_criteria.reward_basis
    actual = initialized(constructor, task, simulation.messages)
    potential = progress.StatePotential(task, constructor)
    assert potential.reason is None and potential.score(actual) == 0


def test_reference_sync_matches_real_tool_execution_for_all_telecom_tasks(tasks):
    changed = 0
    for metadata in tasks:
        if metadata["domain"] != "telecom":
            continue
        constructor, task = constructor_and_task(metadata)
        legacy = initialized(constructor, task)
        for action in task.evaluation_criteria.actions or []:
            legacy.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
        before = reward_v2.database_snapshot(legacy)
        legacy.sync_tools()
        changed += before != reward_v2.database_snapshot(legacy)
        _, actual = reference_simulation(constructor, task)
        assert reward_v2.reference_database(task, constructor) == reward_v2.database_snapshot(actual)
    assert changed >= 136  # Includes explicit-ENV tasks as well as the affected default-DB tasks.


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom", "banking"])
def test_each_domain_reference_replay_matches_reward_and_progress(corrected, tasks, domain):
    metadata = next(m for m in tasks if m["domain"] == domain
                    and "DB" in m["task"]["evaluation_criteria"].get("reward_basis", ["DB"]))
    constructor, task = constructor_and_task(metadata)
    simulation, actual = reference_simulation(constructor, task)
    result = reward.evaluate_simulation_with_constructor(
        simulation=simulation, task=task, domain=domain, environment_constructor=constructor)
    assert result.reward == 1 and result.db_check.db_match
    potential = progress.StatePotential(task, constructor)
    assert potential.reason is None and potential.score(actual) == 0


@pytest.mark.parametrize("task_id", ["airline_1", "telecom_118", "banking_syn_v1_train_000062_l3_single_action"])
def test_real_numeric_equivalent_calls_receive_identical_reward(corrected, tasks, task_id):
    metadata = next(m for m in tasks if m["task"]["id"] == task_id)
    constructor, task = constructor_and_task(metadata)
    normal, _ = reference_simulation(constructor, task)
    alternate, actual = reference_simulation(constructor, task, alternate_numeric=True)
    for simulation in (normal, alternate):
        result = reward.evaluate_simulation_with_constructor(
            simulation=simulation, task=task, domain=metadata["domain"], environment_constructor=constructor)
        assert result.reward == 1 and result.db_check.db_match
    assert progress.StatePotential(task, constructor).score(actual) == 0


def test_banking_application_identity_preserves_different_business_inputs():
    make = reward_v2.generate_application_id
    assert make("Card", "Customer", 82062) == make("Card", "Customer", 82062.0)
    assert make("Card", "Customer", 82062) != make("Card", "Customer", 82063)
    assert make("Card", "Customer", 82062) != make("Other card", "Customer", 82062)


@pytest.mark.parametrize("task_id", ["retail_14", "retail_27"])
def test_retail_item_variants_duplicates_and_price_match_catalog(corrected, tasks, task_id):
    metadata = next(m for m in tasks if m["task"]["id"] == task_id)
    constructor, task = constructor_and_task(metadata)
    snapshots = []
    for reverse in (False, True):
        env = initialized(constructor, task)
        for action in task.evaluation_criteria.actions:
            arguments = dict(action.arguments)
            if action.name == "modify_pending_order_items":
                if reverse:
                    arguments["item_ids"] = list(reversed(arguments["item_ids"]))
                    arguments["new_item_ids"] = list(reversed(arguments["new_item_ids"]))
                order = env.tools._get_order(arguments["order_id"])
                remaining = list(order.items)
                expected = []
                difference = 0.0
                for old, new in zip(arguments["item_ids"], arguments["new_item_ids"]):
                    item = next(item for item in remaining if item.item_id == old)
                    remaining.remove(item)
                    variant = env.tools._get_variant(item.product_id, new)
                    expected.append((item, new, variant))
                    difference += variant.price - item.price
            response = env.get_response(ToolCall(id="call", name=action.name, requestor=action.requestor, arguments=arguments))
            assert not response.error, response.content
            if action.name == "modify_pending_order_items":
                for item, new, variant in expected:
                    assert (item.item_id, item.price, item.options) == (new, variant.price, variant.options)
                assert order.payment_history[-1].amount == abs(round(difference, 2))
                assert order.status == "pending (item modified)"
        snapshots.append(reward_v2.database_snapshot(env))
    assert snapshots[0] == snapshots[1]


def test_retail_invalid_item_does_not_charge_or_mutate_order(corrected, tasks):
    metadata = next(m for m in tasks if m["task"]["id"] == "retail_14")
    constructor, task = constructor_and_task(metadata)
    env = initialized(constructor, task)
    action = next(a for a in task.evaluation_criteria.actions if a.name == "modify_pending_order_items")
    arguments = dict(action.arguments)
    arguments["new_item_ids"] = list(arguments["new_item_ids"])
    arguments["new_item_ids"][-1] = "missing_item"
    before = reward_v2.database_snapshot(env)
    with pytest.raises(ValueError):
        env.tools.modify_pending_order_items(**arguments)
    assert reward_v2.database_snapshot(env) == before


def test_normalization_does_not_mutate_live_state_or_hide_different_amounts(recorded_telecom):
    constructor, task, simulation = recorded_telecom
    actual = initialized(constructor, task, simulation.messages)
    original = actual.tools.db.model_dump_json()
    result = reward_v2.database_snapshot(actual)
    assert actual.tools.db.model_dump_json() == original
    item = result["assistant"]["bills"][2]["line_items"][0]
    assert item["description"] == "Data refueling: 2 GB at $2/GB"
    other = copy.deepcopy(task)
    action = next(a for a in other.evaluation_criteria.actions if a.name == "refuel_data")
    action.arguments["gb_amount"] = 3.0
    assert result != reward_v2.reference_database(other, constructor)
    score = reward_v2.EnvironmentEvaluator.calculate_reward(constructor, other, simulation.messages)
    assert score.reward == 0 and not score.db_check.db_match


def test_action_and_communication_requirements_still_gate_reward(corrected, recorded_telecom):
    constructor, task, simulation = recorded_telecom
    task = copy.deepcopy(task)
    # This valid read is missing from the recorded trajectory and does not
    # alter the DB. It must still fail when ACTION is explicitly required.
    extra = copy.deepcopy(task.evaluation_criteria.actions[0])
    extra.name = "get_details_by_id"
    extra.requestor = "assistant"
    extra.arguments = {"id": "P1003"}
    task.evaluation_criteria.actions.append(extra)
    task.evaluation_criteria.reward_basis = [RewardType.ENV_ASSERTION, RewardType.ACTION]
    result = reward.evaluate_simulation_with_constructor(
        simulation=simulation, task=task, domain="telecom", environment_constructor=constructor)
    assert all(a.met for a in result.env_assertions)
    assert result.reward == 0
    task.evaluation_criteria.reward_basis = [RewardType.ENV_ASSERTION, RewardType.COMMUNICATE]
    task.evaluation_criteria.communicate_info = ["Your case number is 12345."]
    result = reward.evaluate_simulation_with_constructor(
        simulation=simulation, task=task, domain="telecom", environment_constructor=constructor)
    assert result.reward == 0


def test_premature_termination_remains_zero(corrected, recorded_telecom):
    constructor, task, simulation = recorded_telecom
    simulation = simulation.model_copy(update={"termination_reason": "max_steps"})
    result = reward.evaluate_simulation_with_constructor(
        simulation=simulation, task=task, domain="telecom", environment_constructor=constructor)
    assert result.reward == 0


def test_ray_hook_reaches_nested_environment_workers():
    import ray

    paths = os.pathsep.join([str(ROOT), str(RL), str(SOURCE), str(RL.parent / "analysis")])
    ray_temp = tempfile.TemporaryDirectory(prefix="rw2-")
    try:
        ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False,
                 object_store_memory=128 * 1024 * 1024, _temp_dir=ray_temp.name,
                 runtime_env={"env_vars": {"PYTHONPATH": paths, "TAU2_DATA_DIR": os.environ["TAU2_DATA_DIR"]},
                              "worker_process_setup_hook": "reward_v2.setup_worker"})
        @ray.remote(num_cpus=0)
        class Leaf:
            def probe(self):
                import progress
                import reward
                from tau2.domains.banking_knowledge import tools as banking_tools
                from tau2.domains.retail.tools import RetailTools
                return (reward.EnvironmentEvaluator.__module__, progress.reference_database.__module__,
                        progress.database_snapshot.__module__, banking_tools.generate_application_id.__module__,
                        RetailTools.modify_pending_order_items.__globals__["__name__"])

        @ray.remote(num_cpus=0)
        class Parent:
            def probe(self):
                leaf = Leaf.options(runtime_env={"env_vars": {"REWARD_TEST_CHILD": "1"}}).remote()
                return ray.get(leaf.probe.remote())

        parent = Parent.options(runtime_env={"env_vars": {"REWARD_TEST_PARENT": "1"}}).remote()
        assert ray.get(parent.probe.remote(), timeout=90) == ("reward_v2",) * 5
    finally:
        ray.shutdown()
        ray_temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
